markdown
# AnnoLab

Локальный редактор разметки изображений с авто-детектом и обучением YOLO в реальном времени.

---

## 🇷🇺 Русский

### О проекте

**AnnoLab** — это self-hosted веб-приложение для создания датасетов под задачи object detection. Размечайте изображения bounding-боксами прямо в браузере, обучайте YOLO-модель в фоне и получайте автоподсказки в процессе разметки. Никакого облака: изображения, аннотации и веса модели остаются на вашем сервере.

### Возможности

- **Редактор разметки** — рисование боксов ПКМ, 8 ручек для ресайза, перетаскивание ЛКМ, панорамирование, зум колёсиком относительно курсора
- **Авто-детект** — клавиша `A` запускает модель на текущей картинке. Если картинка уже размечена, разметка не перезаписывается
- **Обучение в фоне** — модель дообучается на новых размеченных картинках. Параметры (`modelSelected`, `imgsz`, `lr`, `batchSize`, `worker`, `useGPU`) меняются на лету
- **Живые графики** — `box_loss`, `cls_loss`, `dfl_loss`, `precision`, `recall`, `mAP@50`, `mAP@50-95` обновляются каждую секунду во время обучения
- **Экспорт датасета** — ZIP с `images/`, `labels/` и `classes.txt` в YOLO-формате
- **Скачивание модели** — `.pt` из папки проекта одним кликом
- **Загрузка картинок** — перетащил файлы, и они в датасете
- **Мультипроектность** — активные и архивные проекты, переключение статуса
- **Горячие клавиши** — `A` (авто-разметка), `Delete` (удалить бокс), `Escape` (снять выделение), `Enter` (пропустить пустую картинку)

### Технологический стек

| Слой | Технология |
|---|---|
| Frontend | Vanilla JavaScript, CSS, SVG — без фреймворков и сборки |
| Backend | Python + Flask |
| ML | Ultralytics YOLO (yolo26n / s / m / l / x) |
| Хранение | Файловая система |

### Структура проекта
.
├── index.html # Страница входа
├── mainMenu.html # Список проектов
├── editorYolovDetection.html # Редактор разметки
├── style.css # Стили входа
├── mainMenu.css # Стили списка проектов
├── editorYolovDetection.css # Стили редактора
├── mainPage.js # Логика входа
├── mainMenuJS/
│ └── main.js # Логика списка проектов
├── editorYolovDetection/
│ └── main.js # Логика редактора
├── train.py # Оркестратор обучения YOLO
├── app.py # Flask-сервер с API
└── projects/
└── <project>/
├── images/ # Исходные изображения
├── labels/ # YOLO-аннотации (.txt)
├── modelSettings.json # Текущие параметры обучения
├── <model>_best.pt # Лучший чекпоинт
├── <model>_last.pt # Последний чекпоинт
└── <model>.csv # История метрик

text

### Установка

```bash
# клонировать репозиторий
git clone https://github.com/<your-username>/annolab.git
cd annolab

# установить зависимости
pip install flask ultralytics pandas pyyaml pillow

# запустить сервер
python app.py
Приложение будет доступно по адресу http://localhost:5000/index.html.

Быстрый старт
Войдите в аккаунт (или создайте локального пользователя)

Нажмите «Новый проект», загрузите coco.txt со списком классов

Поставьте галочку «Активный проект», чтобы запустить обучение

Откройте проект в редакторе, размечайте картинки

Когда модель обучится, нажмите A — она предложит боксы на новых изображениях

Скачайте датасет (ZIP) или модель (.pt) из списка проектов

API
Метод	Endpoint	Назначение
GET	/api/isLogin	Проверка логина и пароля
GET	/api/checkprojectsstate	Список проектов для mainMenu
POST	/api/createproject	Создание проекта + запуск обучения
GET	/api/getprojectinfo	Параметры и метрики для сайдбара
POST	/api/updatetrainparams	Обновление параметров обучения
POST	/api/setprojectstate	Включить/выключить проект
GET	/api/getclasses	Список классов проекта
GET	/api/imagenav	Навигация по изображениям
GET	/api/rawimage	Отдача файла изображения
GET	/api/getannotations	Аннотации для текущей картинки
POST	/api/saveannotations	Сохранение аннотаций в .txt
POST	/api/autodetect	Авто-разметка одной картинки
POST	/api/uploadimage	Загрузка новых изображений
GET	/api/downloadmodel	Скачивание .pt-модели
GET	/api/downloaddataset	Скачивание датасета в ZIP
Подробное описание контрактов каждого эндпоинта — в папке docs/ (или в комментариях к app.py).

Лицензия
Проект распространяется под лицензией GNU AGPL-3.0, поскольку использует библиотеку Ultralytics YOLO, которая сама лицензирована под AGPL-3.0.

Если вы хотите использовать AnnoLab в проприетарном продукте без раскрытия исходного кода, вам необходимо приобрести коммерческую лицензию Ultralytics.

🇬🇧 English
About
AnnoLab is a self-hosted web application for building object detection datasets. Annotate images with bounding boxes right in the browser, train a YOLO model in the background, and get auto-detection suggestions while you work. No cloud: images, annotations and model weights stay on your own server.

Features
Annotation editor — right-click to draw boxes, 8 resize handles, left-click to drag, pan with drag, wheel zoom anchored to the cursor

Auto-detect — press A to run the model on the current image. If the image is already labeled, existing annotations are not overwritten

Background training — the model keeps fine-tuning on newly labeled images. Parameters (modelSelected, imgsz, lr, batchSize, worker, useGPU) can be changed live

Live charts — box_loss, cls_loss, dfl_loss, precision, recall, mAP@50, mAP@50-95 update every second during training

Dataset export — ZIP with images/, labels/ and classes.txt in YOLO format

Model download — grab the .pt file from the project folder in one click

Image upload — drop files in, they land in the dataset

Multi-project — active and archived projects, toggle status anytime

Keyboard shortcuts — A (auto-detect), Delete (remove box), Escape (deselect), Enter (skip empty image)

Tech stack
Layer	Technology
Frontend	Vanilla JavaScript, CSS, SVG — no frameworks, no build step
Backend	Python + Flask
ML	Ultralytics YOLO (yolo26n / s / m / l / x)
Storage	Filesystem
Project structure
text
.
├── index.html                      # Login page
├── mainMenu.html                   # Project list
├── editorYolovDetection.html       # Annotation editor
├── style.css                       # Login styles
├── mainMenu.css                    # Project list styles
├── editorYolovDetection.css        # Editor styles
├── mainPage.js                     # Login logic
├── mainMenuJS/
│   └── main.js                     # Project list logic
├── editorYolovDetection/
│   └── main.js                     # Editor logic
├── train.py                        # YOLO training orchestrator
├── app.py                          # Flask API server
└── projects/
    └── <project>/
        ├── images/                 # Source images
        ├── labels/                 # YOLO annotations (.txt)
        ├── modelSettings.json      # Current training params
        ├── <model>_best.pt         # Best checkpoint
        ├── <model>_last.pt         # Last checkpoint
        └── <model>.csv             # Metrics history
Installation
bash
# clone
git clone https://github.com/<your-username>/annolab.git
cd annolab

# dependencies
pip install flask ultralytics pandas pyyaml pillow

# run
python app.py
The app will be available at http://localhost:5000/index.html.

Quick start
Sign in (or create a local user)

Click "New project", upload a coco.txt with class names

Tick "Active project" to start training

Open the project in the editor and start labeling

Once the model is trained, press A — it will propose boxes on new images

Download the dataset (ZIP) or model (.pt) from the project list

API
Method	Endpoint	Purpose
GET	/api/isLogin	Credential check
GET	/api/checkprojectsstate	Project list for mainMenu
POST	/api/createproject	Create project + kick off training
GET	/api/getprojectinfo	Params and metrics for the sidebar
POST	/api/updatetrainparams	Update training params
POST	/api/setprojectstate	Enable/disable project
GET	/api/getclasses	Project class list
GET	/api/imagenav	Image navigation
GET	/api/rawimage	Serve image file
GET	/api/getannotations	Annotations for current image
POST	/api/saveannotations	Save annotations to .txt
POST	/api/autodetect	Auto-label one image
POST	/api/uploadimage	Upload new images
GET	/api/downloadmodel	Download .pt model
GET	/api/downloaddataset	Download dataset as ZIP
Detailed contracts for each endpoint live in docs/ (or inline in app.py).

License
This project is released under the GNU AGPL-3.0 license because it uses Ultralytics YOLO, which itself is AGPL-3.0.

If you want to use AnnoLab in a proprietary product without disclosing source code, you must obtain a commercial Ultralytics license.

Скриншоты / Screenshots
<!-- Добавьте скриншоты после первого релиза / Add screenshots after the first release -->
Разметка / Annotation	Проекты / Projects
coming soon	coming soon
Contributing
Pull requests are welcome. For major changes please open an issue first.

Форкните репозиторий

Создайте ветку (git checkout -b feature/amazing)

Закоммитьте изменения (git commit -am 'Add amazing feature')

Запушьте (git push origin feature/amazing)

Откройте Pull Request

Acknowledgements
Ultralytics — the YOLO framework that powers auto-detection and training

Flask — lightweight Python web framework

All contributors who report bugs and suggest improvements

text

### Пояснения к структуре

**Порядок секций** — сначала русский блок, потом английский. Русская аудитория (вы и большинство пользователей) сразу видит знакомый текст, а иностранцы прокручивают до `🇬🇧 English`. Если репозиторий будет больше ориентирован на международную аудиторию, поменяйте блоки местами — английский первым.

**Оглавление** — не добавлял автоматическое, потому что GitHub сам генерирует TOC в правой панели для `.md`-файлов с заголовками. Если хочется ручное, добавьте после названия:

```markdown
- [Русский](#-русский)
  - [О проекте](#о-проекте)
  - [Возможности](#возможности)
  ...
- [English](#-english)
  - [About](#about)
  ...
Скриншоты — закомментированная секция. После первого релиза залейте 2–3 скриншота в docs/screenshots/ и раскомментируйте блок. Это сильно повышает конверсию репозитория — люди видят, что проект работает.

Contributing — минимальный, без шаблонов для issue и PR. Если проект будет активно развиваться, добавьте .github/PULL_REQUEST_TEMPLATE.md и .github/ISSUE_TEMPLATE/bug_report.md.

Лицензия — продублирована в обоих языковых блоках, потому что это юридически важная информация, и пользователь должен увидеть её независимо от того, какой язык он читает.
