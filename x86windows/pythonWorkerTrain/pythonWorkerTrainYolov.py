import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


class FilteredYOLODataset(YOLODataset):
    def get_img_files(self, img_path):
        im_files = super().get_img_files(img_path)
        label_paths = img2label_paths(im_files)
        filtered = []
        skipped = 0
        for img, lbl in zip(im_files, label_paths):
            p = Path(lbl)
            if p.exists() and p.stat().st_size > 0:
                filtered.append(img)
            else:
                skipped += 1
        if skipped:
            print(f"[FilteredDataset] Пропущено {skipped} изображений без валидных labels")
        print(f"[FilteredDataset] Итого в датасете: {len(filtered)} пар image+label")
        return filtered


class DynamicTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return FilteredYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            cache=self.args.cache,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            batch_size=batch,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            data=self.data,
            fraction=self.args.fraction,
            task=self.args.task,
            classes=self.args.classes,
        )


class TrainingOrchestrator:
    def __init__(self, project_dir: Path, settings_path: Path,
                 label_check_interval: int = 5, param_stable_epochs: int = 5):
        self.project_dir = project_dir
        self.settings_path = settings_path
        self.label_check_interval = label_check_interval
        self.param_stable_epochs = param_stable_epochs

        self.applied_settings = self._load_settings()
        self.last_seen_settings = dict(self.applied_settings)
        self.stable_epoch_count = 0
        self.epoch_counter = 0
        self.last_label_count = self._count_labels()

        self.need_restart = False
        self.restart_reason = ""
        self.load_best_on_restart = False
        self.model_changed = False

    def _load_settings(self):
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Orchestrator] Ошибка чтения JSON: {e}. Используем предыдущие значения.")
            return dict(self.last_seen_settings)

    def _count_labels(self):
        labels_dir = self.project_dir / 'labels'
        if not labels_dir.exists():
            return 0
        return len([p for p in labels_dir.glob('*.txt') if p.stat().st_size > 0])

    def check_epoch(self, trainer):
        current = self._load_settings()
        self.epoch_counter += 1

        if current != self.last_seen_settings:
            self.last_seen_settings = current
            self.stable_epoch_count = 0
            print(f"[Orchestrator] Обнаружены изменения параметров (эпоха {trainer.epoch})")
        else:
            self.stable_epoch_count += 1

        if current != self.applied_settings and self.stable_epoch_count >= self.param_stable_epochs:
            self.model_changed = (
                current.get('modelSelected') != self.applied_settings.get('modelSelected')
            )
            self.need_restart = True
            self.restart_reason = (
                f"Применение стабильных параметров "
                f"({self.stable_epoch_count} эпох без изменений)"
            )
            self.applied_settings = dict(current)
            self.load_best_on_restart = False
            return

        if self.epoch_counter % self.label_check_interval == 0:
            current_label_count = self._count_labels()
            if current_label_count > self.last_label_count:
                print(f"[Orchestrator] Новые labels: {self.last_label_count} -> {current_label_count}")
                self.last_label_count = current_label_count
                self.need_restart = True
                self.restart_reason = "Добавлены новые размеченные изображения"
                self.model_changed = (
                    current.get('modelSelected') != self.applied_settings.get('modelSelected')
                )
                self.load_best_on_restart = not self.model_changed
                self.applied_settings = dict(current)
                return

    def reset_restart_flag(self):
        self.need_restart = False
        self.restart_reason = ""
        self.load_best_on_restart = False
        self.model_changed = False


def _export_checkpoints(project_dir: Path, model_name: str):
    weights_dir = project_dir / 'train_temp' / 'weights'
    best_src = weights_dir / 'best.pt'
    last_src = weights_dir / 'last.pt'

    if best_src.exists():
        shutil.copy2(best_src, project_dir / f'{model_name}_best.pt')
        print(f"[Export] -> {model_name}_best.pt")
    if last_src.exists():
        shutil.copy2(last_src, project_dir / f'{model_name}_last.pt')
        print(f"[Export] -> {model_name}_last.pt")

    results_csv = project_dir / 'train_temp' / 'results.csv'
    if not results_csv.exists():
        return

    target_csv = project_dir / f'{model_name}.csv'
    try:
        df_new = pd.read_csv(results_csv)
        if len(df_new) == 0:
            print("[Export] results.csv пуст — CSV не обновлён")
            return

        df_old = None
        if target_csv.exists():
            try:
                df_old = pd.read_csv(target_csv)
            except Exception as e:
                print(f"[Export] Не удалось прочитать {target_csv.name}, будет создан заново: {e}")

        if df_old is not None and len(df_old) > 0:
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(target_csv, index=False)
        added = len(df_combined) - (len(df_old) if df_old is not None else 0)
        print(f"[Export] -> {model_name}.csv (добавлено {added}, всего {len(df_combined)} строк)")
    except Exception as e:
        print(f"[Export] Ошибка записи CSV: {e}")


def _export_best_every_epoch(project_dir: Path, model_name: str):
    src = project_dir / 'train_temp' / 'weights' / 'best.pt'
    dst = project_dir / f'{model_name}_best.pt'
    if not src.exists():
        return
    try:
        shutil.copy2(src, dst)
        print(f"[Export] epoch-best -> {dst.name}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Export] epoch-best failed: {e}", file=sys.stderr, flush=True)


def _export_last_every_epoch(project_dir: Path, model_name: str):
    src = project_dir / 'train_temp' / 'weights' / 'last.pt'
    dst = project_dir / f'{model_name}_last.pt'
    if not src.exists():
        return
    try:
        shutil.copy2(src, dst)
    except Exception:
        pass


def run_training(project_dir: Path, settings_path: Path):
    orchestrator = TrainingOrchestrator(
        project_dir, settings_path,
        label_check_interval=55,
        param_stable_epochs=55,
    )

    while True:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        model_name = settings['modelSelected']
        imgsz = int(settings.get('imgsz', 640))
        lr = float(settings.get('lr', 0.0001))
        batch_size = int(settings.get('batchSize', 16))
        workers = int(settings.get('worker', 2))
        use_gpu = bool(settings.get('useGPU', False))
        device = 0 if use_gpu else 'cpu'

        best_weights = project_dir / f'{model_name}_best.pt'
        last_weights = project_dir / f'{model_name}_last.pt'

        if orchestrator.load_best_on_restart and best_weights.exists():
            print(f"[Main] Перезапуск с весов best: {best_weights}")
            model = YOLO(str(best_weights))
        elif not orchestrator.model_changed and last_weights.exists():
            print(f"[Main] Продолжение с весов last: {last_weights}")
            model = YOLO(str(last_weights))
        else:
            print(f"[Main] Загрузка предобученной модели: {model_name}.pt")
            model = YOLO(f'{model_name}.pt')

        train_temp_csv = project_dir / 'train_temp' / 'results.csv'
        try:
            train_temp_csv.unlink()
            print("[Main] Очищен train_temp/results.csv для нового запуска")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[Main] Не удалось удалить train_temp/results.csv: {e}")

        def epoch_start_callback(trainer):
            orchestrator.check_epoch(trainer)
            if orchestrator.need_restart:
                print(f"[Callback] Остановка: {orchestrator.restart_reason}")
                trainer.stop = True

        def epoch_end_callback(trainer):
            _export_best_every_epoch(project_dir, model_name)
            _export_last_every_epoch(project_dir, model_name)

        model.add_callback("on_train_epoch_start", epoch_start_callback)
        model.add_callback("on_fit_epoch_end", epoch_end_callback)

        try:
            model.train(
                data=str(project_dir / 'data.yaml'),
                epochs=1000,
                imgsz=imgsz,
                lr0=lr,
                batch=batch_size,
                workers=workers,
                device=device,
                project=str(project_dir),
                name='train_temp',
                exist_ok=True,
                plots=False,
                save=True,
                save_period=-1,
                val=True,
                verbose=False,
                optimizer='AdamW',
                trainer=DynamicTrainer,
            )
        except Exception as e:
            print(f"[Main] Обучение прервано: {e}")

        if orchestrator.need_restart:
            _export_checkpoints(project_dir, model_name)
            orchestrator.reset_restart_flag()
            time.sleep(1)
        else:
            print("[Main] Обучение завершено штатно.")
            _export_checkpoints(project_dir, model_name)
            break


def run_inference(project_dir: Path, settings_path: Path, image_path: Path):
    if not image_path.exists():
        print(f"[Inference] Изображение не найдено: {image_path}")
        sys.exit(1)

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    model_name = settings['modelSelected']
    imgsz = int(settings.get('imgsz', 640))
    use_gpu = bool(settings.get('useGPU', False))
    device = 0 if use_gpu else 'cpu'
    conf = float(settings.get('conf', 0.25))
    iou = float(settings.get('iou', 0.45))

    state = _load_yolo_state(project_dir, settings_path)
    weights_path = state['weights_path']

    print(f"[Inference] Модель: {weights_path}")
    print(f"[Inference] Изображение: {image_path}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    labels_dir = project_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    label_file = labels_dir / f'{image_path.stem}.txt'

    lines = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(cls)):
            x, y, w, h = xywhn[i]
            lines.append(f"{cls[i]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[Inference] Записано {len(lines)} объектов в {label_file}")
    print("[Inference] Готово. Выход.")


def _file_stamp(path: Path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _read_settings(settings_path: Path):
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _pick_weights(project_dir: Path, model_name: str):
    live_best = project_dir / 'train_temp' / 'weights' / 'best.pt'
    live_last = project_dir / 'train_temp' / 'weights' / 'last.pt'
    exp_best  = project_dir / f'{model_name}_best.pt'
    exp_last  = project_dir / f'{model_name}_last.pt'

    if live_best.exists():
        return live_best, 'live-best'
    if exp_best.exists():
        return exp_best, 'exported-best'
    if live_last.exists():
        return live_last, 'live-last'
    if exp_last.exists():
        return exp_last, 'exported-last'
    return None, 'pretrained'


def _weights_signature(project_dir: Path, model_name: str):
    paths = [
        project_dir / 'train_temp' / 'weights' / 'best.pt',
        project_dir / 'train_temp' / 'weights' / 'last.pt',
        project_dir / f'{model_name}_best.pt',
        project_dir / f'{model_name}_last.pt',
    ]
    return tuple(_file_stamp(p) for p in paths)


def _load_yolo_state(project_dir: Path, settings_path: Path):
    settings = _read_settings(settings_path)
    model_name = settings.get('modelSelected', 'yolo26n')
    imgsz = int(settings.get('imgsz', 640))
    use_gpu = bool(settings.get('useGPU', False))
    device = 0 if use_gpu else 'cpu'
    conf = float(settings.get('conf', 0.25))
    iou = float(settings.get('iou', 0.45))

    weights_path, kind = _pick_weights(project_dir, model_name)
    if weights_path is None:
        weights_path = Path(f'{model_name}.pt')

    return {
        'settings':      settings,
        'model_name':    model_name,
        'imgsz':         imgsz,
        'device':        device,
        'conf':          conf,
        'iou':           iou,
        'weights_path':  weights_path,
        'weights_kind':  kind,
    }


def _write_labels(project_dir: Path, image_path: Path, results):
    labels_dir = project_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    label_file = labels_dir / f'{image_path.stem}.txt'

    lines = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xywhn = r.boxes.xywhn.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(cls)):
            x, y, w, h = xywhn[i]
            lines.append(f"{cls[i]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return len(lines)


def run_serve(project_dir: Path, settings_path: Path):
    state = _load_yolo_state(project_dir, settings_path)
    settings_stamp = _file_stamp(settings_path)
    weights_sig    = _weights_signature(project_dir, state['model_name'])

    print(f"[Serve] Project: {project_dir}", file=sys.stderr, flush=True)
    print(f"[Serve] Loading {state['weights_path']} ({state['weights_kind']})",
          file=sys.stderr, flush=True)

    model = YOLO(str(state['weights_path']))
    print("[Serve] Ready", file=sys.stderr, flush=True)
    print("READY", flush=True)

    for line in sys.stdin:
        raw = line.rstrip('\r\n')
        if not raw:
            continue
        if raw == "__stop__":
            print("STOP", flush=True)
            print("[Serve] Stop command", file=sys.stderr, flush=True)
            break

        image_path = Path(raw)
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()

        reload_reason = None

        cur_settings_stamp = _file_stamp(settings_path)
        if cur_settings_stamp != settings_stamp:
            new_state = _load_yolo_state(project_dir, settings_path)
            if (new_state['model_name'] != state['model_name'] or
                    new_state['imgsz'] != state['imgsz'] or
                    new_state['device'] != state['device'] or
                    new_state['conf'] != state['conf'] or
                    new_state['iou'] != state['iou']):
                reload_reason = "settings changed"
            state = new_state
            settings_stamp = cur_settings_stamp
            weights_sig = _weights_signature(project_dir, state['model_name'])

        cur_sig = _weights_signature(project_dir, state['model_name'])
        if cur_sig != weights_sig:
            new_state = _load_yolo_state(project_dir, settings_path)
            if new_state['weights_path'] != state['weights_path']:
                reload_reason = (
                    f"weights changed -> {new_state['weights_path'].name} "
                    f"({new_state['weights_kind']})"
                )
            else:
                reload_reason = f"{state['weights_path'].name} updated"
            state = new_state
            weights_sig = cur_sig

        if reload_reason:
            print(f"[Serve] Reload: {reload_reason}", file=sys.stderr, flush=True)
            try:
                model = YOLO(str(state['weights_path']))
                print(f"[Serve] Reloaded {state['weights_path']} ({state['weights_kind']})",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[Serve] Reload failed: {e}", file=sys.stderr, flush=True)

        if not image_path.exists():
            print(f"ERR image not found: {image_path}", flush=True)
            continue

        try:
            results = model.predict(
                source=str(image_path),
                imgsz=state['imgsz'],
                device=state['device'],
                conf=state['conf'],
                iou=state['iou'],
                verbose=False,
            )
            n = _write_labels(project_dir, image_path, results)
            print(f"OK {n}", flush=True)
        except Exception as e:
            print(f"ERR {type(e).__name__}: {e}", flush=True)

    print("[Serve] Exit", file=sys.stderr, flush=True)


def prepare_data_yaml(project_dir: Path):
    coco_txt = project_dir / 'coco.txt'
    if not coco_txt.exists():
        raise FileNotFoundError(f"Не найден файл классов: {coco_txt}")

    with open(coco_txt, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    data_yaml = {
        'path': str(project_dir.absolute()),
        'train': 'images',
        'val': 'images',
        'names': {i: name for i, name in enumerate(classes)},
        'nc': len(classes),
    }

    yaml_path = project_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    print(f"[Setup] Создан data.yaml: {yaml_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Динамическое обучение / инференс / serve YOLO')
    parser.add_argument('--projectname', type=str, required=True,
                        help='Имя папки проекта внутри ..\\projects\\')
    parser.add_argument('--image', type=str, default=None,
                        help='Путь к изображению для инференса (one-shot).')
    parser.add_argument('--serve', action='store_true',
                        help='Постоянный воркер: читает пути из stdin, отвечает OK/ERR.')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    project_dir = (script_dir / '..' / 'projects' / args.projectname).resolve()
    settings_path = project_dir / 'modelSettings.json'

    if not project_dir.exists():
        print(f"[Error] Папка проекта не найдена: {project_dir}", file=sys.stderr)
        sys.exit(1)
    if not settings_path.exists():
        print(f"[Error] Файл настроек не найден: {settings_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Start] Проект: {project_dir}", file=sys.stderr, flush=True)

    if args.serve:
        run_serve(project_dir, settings_path)
    elif args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = (Path.cwd() / image_path).resolve()
        run_inference(project_dir, settings_path, image_path)
    else:
        prepare_data_yaml(project_dir)
        run_training(project_dir, settings_path)