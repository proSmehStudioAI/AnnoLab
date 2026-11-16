document.addEventListener('DOMContentLoaded', () => {
  const getCookie = (name) => {
    const key = encodeURIComponent(name) + '=';
    const parts = document.cookie ? document.cookie.split('; ') : [];
    for (const part of parts) {
      if (part.indexOf(key) === 0) return decodeURIComponent(part.slice(key.length));
    }
    return null;
  };

  const escapeHtml = (str) =>
    String(str).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));

  const mulberry32 = (seed) => {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  };

  const generateClassColors = (count, seed = 42) => {
    const rand = mulberry32(seed);
    const startHue = rand() * 360;
    const golden = 137.508;
    const colors = [];
    for (let i = 0; i < count; i++) {
      const hue = (startHue + i * golden + rand() * 6) % 360;
      const sat = 65 + rand() * 20;
      const light = 55 + rand() * 15;
      colors.push(`hsl(${hue.toFixed(1)}, ${sat.toFixed(1)}%, ${light.toFixed(1)}%)`);
    }
    return colors;
  };

  const login = getCookie('login');
  const password = getCookie('password');
  if (login === null || password === null) {
    window.location.href = 'index.html';
    return;
  }

  const params = new URLSearchParams(window.location.search);
  const project = params.get('project') || '';
  if (!project) {
    window.location.href = 'mainMenu.html';
    return;
  }

  const projectNameEl = document.querySelector('.topbar__project');
  if (projectNameEl) projectNameEl.textContent = project;

  const backBtn = document.querySelector('.topbar__actions .btn-ghost');
  if (backBtn) backBtn.addEventListener('click', () => {
    window.location.href = 'mainMenu.html';
  });

  const wrapEl = document.querySelector('.canvas-wrap');
  const canvasImage = document.querySelector('.canvas-image');
  const boxesLayer = document.getElementById('boxes-layer');
  const imgEl = document.querySelector('.canvas-image__img');
  const bgEl = document.querySelector('.canvas-image__bg');
  const fileEl = document.querySelector('.statusbar__file');
  const dimsEl = document.querySelector('.statusbar__dims');
  const counterEl = document.querySelector('.statusbar__counter');
  const prevBtn = document.querySelector('.nav-arrow--prev');
  const nextBtn = document.querySelector('.nav-arrow--next');
  const progressFill = document.getElementById('progress-fill');
  const progressPct = document.getElementById('progress-pct');
  const imgszSelect = document.getElementById('imgsz-view');
  const zoomValueEl = document.querySelector('.zoom-value');
  const classesListEl = document.getElementById('classes-list');
  const warningModal = document.getElementById('modal-warning');
  const btnSkip = document.getElementById('btn-skip');

  const autoDetectEl = document.getElementById('auto-detect');

  const projectStateEl = document.getElementById('project-state');
  const trainModelEl = document.getElementById('train-model');
  const trainImgszEl = document.getElementById('train-imgsz');
  const trainLrEl = document.getElementById('train-lr');
  const trainBatchEl = document.getElementById('train-batch');
  const trainWorkersEl = document.getElementById('train-workers');
  const trainGpuEl = document.getElementById('train-gpu');
  const metricsEpochEl = document.getElementById('metrics-epoch');

  const btnUpload = document.getElementById('btn-upload');
  const uploadInput = document.getElementById('upload-input');
  const toastEl = document.getElementById('toast');

  const state = {
    position: 0,
    total: 0,
    assigned: 0,
    historySize: 1,
    filename: null,
    canPrev: false,
    canNext: false,
    ratio: 16 / 9,
  };

  let classes = [];
  let annotations = [];
  let activeClassIndex = -1;
  let selectedBoxIndex = -1;
  let scale = 1;
  let tx = 0;
  let ty = 0;
  const MIN_SCALE = 0.1;
  const MAX_SCALE = 10;

  let sidebarInfo = null;
  let paramsSaveTimer = null;
  let metricsTimer = null;
  let autoDetectRunning = false;

  let toastTimer = null;
  const showToast = (msg, type = '') => {
    if (!toastEl) return;
    toastEl.textContent = msg;
    toastEl.className = 'toast' + (type ? ' is-' + type : '');
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 3000);
  };

  const applyTransform = () => {
    if (!canvasImage) return;
    canvasImage.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    if (zoomValueEl) zoomValueEl.textContent = Math.round(scale * 100) + '%';
  };

  const resetTransform = () => { scale = 1; tx = 0; ty = 0; applyTransform(); };

  const applyCanvasSize = () => {
    if (!canvasImage) return;
    const ar = state.ratio || 16 / 9;
    const size = Number(imgszSelect && imgszSelect.value) || 640;
    let w, h;
    if (ar >= 1) { w = size; h = size / ar; }
    else { h = size; w = size * ar; }
    canvasImage.style.width = Math.floor(w) + 'px';
    canvasImage.style.height = Math.floor(h) + 'px';
    canvasImage.style.aspectRatio = 'unset';
  };

  const updateProgress = () => {
    const pct = state.total > 0 ? Math.round((state.assigned / state.total) * 100) : 0;
    if (progressFill) progressFill.style.width = pct + '%';
    if (progressPct) progressPct.textContent = pct + '%';
  };

  const updateNavButtons = () => {
    if (prevBtn) prevBtn.disabled = !state.canPrev;
    if (nextBtn) nextBtn.disabled = !state.canNext;
  };

  const renderClasses = () => {
    if (!classesListEl) return;
    if (classes.length === 0) {
      classesListEl.innerHTML = '<li class="classes__empty">Классы не найдены</li>';
      return;
    }
    classesListEl.innerHTML = classes.map((c, i) => `
      <li class="class-item${i === activeClassIndex ? ' is-active' : ''}" data-class-index="${i}">
        <span class="class-item__color" style="--class-color: ${c.color};"></span>
        <span class="class-item__name">${escapeHtml(c.name)}</span>
        <span class="class-item__count" data-count="${i}">0</span>
      </li>
    `).join('');

    classesListEl.querySelectorAll('.class-item').forEach((el) => {
      el.addEventListener('click', () => {
        const idx = Number(el.dataset.classIndex);
        activeClassIndex = activeClassIndex === idx ? -1 : idx;
        renderClasses();
        wrapEl.classList.toggle('is-class-active', activeClassIndex >= 0);
      });
    });

    updateClassCounts();
  };

  const updateClassCounts = () => {
    if (!classesListEl) return;
    const counts = new Array(classes.length).fill(0);
    annotations.forEach((a) => {
      if (a.classIndex >= 0 && a.classIndex < counts.length) counts[a.classIndex]++;
    });
    counts.forEach((c, i) => {
      const el = classesListEl.querySelector(`[data-count="${i}"]`);
      if (el) el.textContent = c;
    });
  };

  const renderBoxes = () => {
    if (!boxesLayer) return;
    boxesLayer.querySelectorAll('.canvas-box').forEach((el) => el.remove());

    annotations.forEach((a, i) => {
      const cls = classes[a.classIndex] || { name: '?', color: '#fff' };
      const el = document.createElement('div');
      el.className = 'canvas-box' + (i === selectedBoxIndex ? ' is-selected' : '');
      el.dataset.index = i;
      el.style.left = (a.x * 100) + '%';
      el.style.top = (a.y * 100) + '%';
      el.style.width = (a.w * 100) + '%';
      el.style.height = (a.h * 100) + '%';
      el.style.setProperty('--box-color', cls.color);
      el.innerHTML = `
        <span class="canvas-box__label">${escapeHtml(cls.name)}</span>
        ${['nw','n','ne','w','e','sw','s','se'].map((h) =>
          `<span class="canvas-box__handle" data-handle="${h}"></span>`).join('')}
      `;
      boxesLayer.appendChild(el);
    });
    updateClassCounts();
  };

  const loadClasses = async () => {
    const url = `/api/getclasses?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project)}`;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      if (!Array.isArray(data)) return;

      const colors = generateClassColors(data.length);
      classes = data.map((c, i) => ({
        name: typeof c === 'string' ? c : (c.name || '?'),
        color: colors[i],
      }));
      renderClasses();
    } catch (err) {
      console.error('getclasses error:', err);
    }
  };

  const loadImage = (filename) => new Promise((resolve) => {
    if (!imgEl) return resolve();
    const url = `/api/rawimage?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project)}` +
                `&file=${encodeURIComponent(filename)}`;

    imgEl.onload = () => {
      const w = imgEl.naturalWidth;
      const h = imgEl.naturalHeight;
      state.ratio = h > 0 ? w / h : 16 / 9;
      if (dimsEl) dimsEl.textContent = `${w} × ${h} px`;
      imgEl.hidden = false;
      if (bgEl) bgEl.style.display = 'none';
      applyCanvasSize();
      resetTransform();
      resolve();
    };
    imgEl.onerror = () => {
      imgEl.hidden = true;
      if (bgEl) bgEl.style.display = '';
      if (dimsEl) dimsEl.textContent = '—';
      resolve();
    };
    imgEl.src = url;
  });

  const loadAnnotations = async (filename) => {
    const url = `/api/getannotations?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project)}` +
                `&filename=${encodeURIComponent(filename)}`;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) {
        annotations = [];
        selectedBoxIndex = -1;
        renderBoxes();
        return;
      }
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      annotations = Array.isArray(data)
        ? data.map((a) => ({
            classIndex: Number.isFinite(a.class_index) ? a.class_index : 0,
            x: Number(a.x) || 0,
            y: Number(a.y) || 0,
            w: Number(a.w) || 0,
            h: Number(a.h) || 0,
          }))
        : [];
      selectedBoxIndex = -1;
      renderBoxes();
    } catch (err) {
      console.error('getannotations error:', err);
      annotations = [];
      renderBoxes();
    }
  };

  const autoDetectCurrent = async () => {
    if (autoDetectRunning) return;
    if (!state.filename) return;
    if (annotations.length > 0) {
      showToast('Изображение уже размечено', 'error');
      return;
    }

    autoDetectRunning = true;
    showToast('Авто-разметка…');

    try {
      const res = await fetch('/api/autodetect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ login, password, project, filename: state.filename }),
      });
      const data = await res.json();

      if (data === false) { window.location.href = 'index.html'; return; }
      if (data && data.error) {
        showToast(String(data.error), 'error');
        return;
      }

      if (!Array.isArray(data) || data.length === 0) {
        showToast('Модель не нашла объектов', 'error');
        return;
      }

      annotations = data.map((a) => ({
        classIndex: Number.isFinite(a.class_index) ? a.class_index : 0,
        x: Number(a.x) || 0,
        y: Number(a.y) || 0,
        w: Number(a.w) || 0,
        h: Number(a.h) || 0,
      }));
      selectedBoxIndex = -1;
      renderBoxes();
      scheduleSave();
      showToast(`Авто-разметка: ${data.length} объектов`, 'success');
    } catch (err) {
      console.error('autodetect error:', err);
      showToast('Не удалось выполнить авто-разметку', 'error');
    } finally {
      autoDetectRunning = false;
    }
  };

  if (autoDetectEl) {
    autoDetectEl.addEventListener('change', () => {
      if (autoDetectEl.checked && annotations.length === 0 && state.filename) {
        autoDetectCurrent();
      }
    });
  }

  const applyState = async (data) => {
    state.position = Number.isFinite(data.position) ? data.position : 0;
    state.total = Number.isFinite(data.totalImages) ? data.totalImages : 0;
    state.assigned = Number.isFinite(data.assignedCount) ? data.assignedCount : 0;
    state.historySize = Number.isFinite(data.historySize) ? data.historySize : 1;
    state.filename = data.filename || null;
    state.canPrev = !!data.canPrev;
    state.canNext = !!data.canNext;

    if (counterEl) counterEl.textContent = `${state.position + 1} / ${state.total}`;
    if (fileEl && state.filename) fileEl.textContent = state.filename;

    updateProgress();

    annotations = [];
    selectedBoxIndex = -1;
    renderBoxes();

    if (state.filename) {
      await loadImage(state.filename);
      await loadAnnotations(state.filename);

      if (autoDetectEl?.checked && annotations.length === 0) {
        autoDetectCurrent();
      }
    }
  };

  let saveTimer = null;

  const saveAnnotations = async () => {
    clearTimeout(saveTimer);
    if (!state.filename) return;
    const body = {
      login, password, project,
      filename: state.filename,
      annotations: annotations.map((a) => ({
        class_index: a.classIndex,
        x: a.x, y: a.y, w: a.w, h: a.h,
      })),
    };
    try {
      const res = await fetch('/api/saveannotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data === false) window.location.href = 'index.html';
    } catch (err) {
      console.error('saveannotations error:', err);
    }
  };

  const scheduleSave = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveAnnotations, 800);
  };

  let isNavigating = false;

  const nav = async (dir) => {
    if (isNavigating) return;
    isNavigating = true;
    if (prevBtn) prevBtn.disabled = true;
    if (nextBtn) nextBtn.disabled = true;

    const url = `/api/imagenav?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project)}` +
                `&dir=${dir}`;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      if (data && data.error) {
        console.warn('imagenav:', data.error);
        if (dir === 'current') {
          isNavigating = false;
          updateNavButtons();
          return nav('next');
        }
        return;
      }
      await applyState(data);
    } catch (err) {
      console.error('imagenav error:', err);
    } finally {
      isNavigating = false;
      updateNavButtons();
    }
  };

  const doNav = async (dir) => {
    clearTimeout(saveTimer);
    await saveAnnotations();
    nav(dir);
  };

  const openWarning = () => { if (warningModal) warningModal.hidden = false; };
  const closeWarning = () => { if (warningModal) warningModal.hidden = true; };

  const requestNext = () => {
    const wouldBeNew = state.position + 1 >= state.historySize;
    if (wouldBeNew && annotations.length === 0) {
      openWarning();
      return;
    }
    doNav('next');
  };

  if (warningModal) {
    warningModal.querySelectorAll('[data-warning-close]').forEach((el) => {
      el.addEventListener('click', closeWarning);
    });
  }
  if (btnSkip) {
    btnSkip.addEventListener('click', () => {
      closeWarning();
      doNav('next');
    });
  }

  if (prevBtn) prevBtn.addEventListener('click', () => doNav('prev'));
  if (nextBtn) nextBtn.addEventListener('click', requestNext);

  if (imgszSelect) {
    imgszSelect.addEventListener('change', () => {
      applyCanvasSize();
      resetTransform();
    });
  }

  let drawing = null;
  let drawingBoxEl = null;

  const updateDrawingPreview = () => {
    if (!drawing || !boxesLayer) return;
    if (!drawingBoxEl) {
      drawingBoxEl = document.createElement('div');
      drawingBoxEl.className = 'canvas-box canvas-box--preview';
      boxesLayer.appendChild(drawingBoxEl);
    }
    const x1 = Math.min(drawing.startX, drawing.currentX);
    const y1 = Math.min(drawing.startY, drawing.currentY);
    const x2 = Math.max(drawing.startX, drawing.currentX);
    const y2 = Math.max(drawing.startY, drawing.currentY);
    drawingBoxEl.style.left = (x1 * 100) + '%';
    drawingBoxEl.style.top = (y1 * 100) + '%';
    drawingBoxEl.style.width = ((x2 - x1) * 100) + '%';
    drawingBoxEl.style.height = ((y2 - y1) * 100) + '%';
    drawingBoxEl.style.setProperty('--box-color',
      classes[activeClassIndex]?.color || '#fff');
  };

  const removeDrawingPreview = () => {
    if (drawingBoxEl) { drawingBoxEl.remove(); drawingBoxEl = null; }
  };

  const finalizeBox = () => {
    if (!drawing) return;
    if (activeClassIndex < 0) { removeDrawingPreview(); return; }
    const x1 = Math.min(drawing.startX, drawing.currentX);
    const y1 = Math.min(drawing.startY, drawing.currentY);
    const x2 = Math.max(drawing.startX, drawing.currentX);
    const y2 = Math.max(drawing.startY, drawing.currentY);
    const w = x2 - x1;
    const h = y2 - y1;
    if (w < 0.005 || h < 0.005) { removeDrawingPreview(); return; }
    annotations.push({ classIndex: activeClassIndex, x: x1, y: y1, w, h });
    removeDrawingPreview();
    renderBoxes();
    scheduleSave();
  };

  let resizing = null;

  const startResize = (e, handleEl) => {
    const boxEl = handleEl.closest('.canvas-box');
    if (!boxEl) return;
    const idx = Number(boxEl.dataset.index);
    if (!Number.isFinite(idx) || idx < 0 || idx >= annotations.length) return;
    resizing = {
      idx,
      dir: handleEl.dataset.handle,
      startX: e.clientX,
      startY: e.clientY,
      orig: { ...annotations[idx] },
      rect: canvasImage.getBoundingClientRect(),
    };
    e.preventDefault();
    e.stopPropagation();
  };

  const doResize = (e) => {
    if (!resizing) return;
    const { dir, orig, rect } = resizing;
    const dx = (e.clientX - resizing.startX) / rect.width;
    const dy = (e.clientY - resizing.startY) / rect.height;

    let { x, y, w, h } = orig;
    if (dir.includes('w')) { x = orig.x + dx; w = orig.w - dx; }
    if (dir.includes('e')) { w = orig.w + dx; }
    if (dir.includes('n')) { y = orig.y + dy; h = orig.h - dy; }
    if (dir.includes('s')) { h = orig.h + dy; }

    if (w < 0.005) { w = 0.005; if (dir.includes('w')) x = orig.x + orig.w - w; }
    if (h < 0.005) { h = 0.005; if (dir.includes('n')) y = orig.y + orig.h - h; }

    x = Math.max(0, Math.min(1 - w, x));
    y = Math.max(0, Math.min(1 - h, y));
    w = Math.min(w, 1 - x);
    h = Math.min(h, 1 - y);

    annotations[resizing.idx] = { classIndex: orig.classIndex, x, y, w, h };
    renderBoxes();
  };

  let boxDrag = null;

  const startBoxDrag = (e, idx) => {
    const rect = canvasImage.getBoundingClientRect();
    boxDrag = {
      idx,
      startX: e.clientX,
      startY: e.clientY,
      orig: { ...annotations[idx] },
      rect,
    };
    e.preventDefault();
    e.stopPropagation();
  };

  const doBoxDrag = (e) => {
    if (!boxDrag) return;
    const { orig, rect } = boxDrag;
    const dx = (e.clientX - boxDrag.startX) / rect.width;
    const dy = (e.clientY - boxDrag.startY) / rect.height;

    let x = orig.x + dx;
    let y = orig.y + dy;
    x = Math.max(0, Math.min(1 - orig.w, x));
    y = Math.max(0, Math.min(1 - orig.h, y));

    annotations[boxDrag.idx] = { ...orig, x, y };
    renderBoxes();
  };

  let dragging = false;
  let startX = 0, startY = 0, startTx = 0, startTy = 0;

  const startPan = (e) => {
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startTx = tx;
    startTy = ty;
    wrapEl.classList.add('is-dragging');
    e.preventDefault();
  };

  if (wrapEl && canvasImage) {
    wrapEl.addEventListener('contextmenu', (e) => e.preventDefault());

    wrapEl.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale * factor));
      if (newScale === scale) return;
      const rect = canvasImage.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const mx = e.clientX - cx;
      const my = e.clientY - cy;
      const r = newScale / scale;
      tx = tx + mx * (1 - r);
      ty = ty + my * (1 - r);
      scale = newScale;
      applyTransform();
    }, { passive: false });

    wrapEl.addEventListener('mousedown', (e) => {
      if (e.button === 2) {
        if (activeClassIndex < 0) return;
        const rect = canvasImage.getBoundingClientRect();
        const px = (e.clientX - rect.left) / rect.width;
        const py = (e.clientY - rect.top) / rect.height;
        if (px < 0 || px > 1 || py < 0 || py > 1) return;
        drawing = { startX: px, startY: py, currentX: px, currentY: py };
        updateDrawingPreview();
        e.preventDefault();
        return;
      }

      if (e.button === 0) {
        const handle = e.target.closest('[data-handle]');
        if (handle) { startResize(e, handle); return; }

        const box = e.target.closest('.canvas-box:not(.canvas-box--preview)');
        if (box) {
          const idx = Number(box.dataset.index);
          if (Number.isFinite(idx)) {
            selectedBoxIndex = idx;
            renderBoxes();
            startBoxDrag(e, idx);
          }
          return;
        }

        if (selectedBoxIndex >= 0) {
          selectedBoxIndex = -1;
          renderBoxes();
        }
        startPan(e);
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (drawing) {
        const rect = canvasImage.getBoundingClientRect();
        drawing.currentX = (e.clientX - rect.left) / rect.width;
        drawing.currentY = (e.clientY - rect.top) / rect.height;
        updateDrawingPreview();
      } else if (resizing) {
        doResize(e);
      } else if (boxDrag) {
        doBoxDrag(e);
      } else if (dragging) {
        tx = startTx + (e.clientX - startX);
        ty = startTy + (e.clientY - startY);
        applyTransform();
      }
    });

    window.addEventListener('mouseup', () => {
      if (drawing) {
        finalizeBox();
        drawing = null;
      }
      if (resizing) {
        resizing = null;
        scheduleSave();
      }
      if (boxDrag) {
        boxDrag = null;
        scheduleSave();
      }
      if (dragging) {
        dragging = false;
        wrapEl.classList.remove('is-dragging');
      }
    });
  }

  document.querySelectorAll('.tool-btn').forEach((btn) => {
    const t = btn.getAttribute('title');
    if (t === 'Приблизить') {
      btn.addEventListener('click', () => {
        const ns = Math.min(MAX_SCALE, scale * 1.25);
        if (ns !== scale) { scale = ns; applyTransform(); }
      });
    } else if (t === 'Отдалить') {
      btn.addEventListener('click', () => {
        const ns = Math.max(MIN_SCALE, scale / 1.25);
        if (ns !== scale) { scale = ns; applyTransform(); }
      });
    }
  });

  const isTypingInField = (el) => {
    if (!el || !el.closest) return false;
    return !!el.closest([
      'textarea',
      '[contenteditable="true"]',
      'select',
      'input:not([type="checkbox"]):not([type="radio"]):not([type="button"]):not([type="submit"]):not([type="range"]):not([type="file"])',
    ].join(','));
  };

  document.addEventListener('keydown', (e) => {
    if (warningModal && !warningModal.hidden && e.key === 'Enter') {
      closeWarning();
      doNav('next');
      e.preventDefault();
      return;
    }

    if (e.key === 'Escape') {
      if (warningModal && !warningModal.hidden) { closeWarning(); return; }
      if (selectedBoxIndex >= 0) { selectedBoxIndex = -1; renderBoxes(); return; }
    }

    if (isTypingInField(e.target)) return;

    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedBoxIndex >= 0) {
      annotations.splice(selectedBoxIndex, 1);
      selectedBoxIndex = -1;
      renderBoxes();
      scheduleSave();
      e.preventDefault();
      return;
    }

    const code = e.code || '';
    const key = e.key || '';
    const isA = code === 'KeyA'
             || key === 'a' || key === 'A'
             || key === 'ф' || key === 'Ф';
    if (isA && !e.ctrlKey && !e.metaKey && !e.altKey) {
      autoDetectCurrent();
      e.preventDefault();
      return;
    }
  });

  if (btnUpload && uploadInput) {
    btnUpload.addEventListener('click', () => uploadInput.click());

    uploadInput.addEventListener('change', async () => {
      const files = Array.from(uploadInput.files || []);
      if (files.length === 0) return;

      btnUpload.disabled = true;
      btnUpload.classList.add('is-loading');

      let ok = 0;
      let fail = 0;

      for (const file of files) {
        const fd = new FormData();
        fd.append('login', login);
        fd.append('password', password);
        fd.append('project', project);
        fd.append('file', file, file.name);

        try {
          const res = await fetch('/api/uploadimage', {
            method: 'POST',
            headers: { 'Accept': 'application/json' },
            body: fd,
          });
          const data = await res.json();

          if (data === false) {
            window.location.href = 'index.html';
            return;
          }
          if (data && data.success) ok++;
          else fail++;
        } catch (err) {
          console.error('uploadimage error:', err);
          fail++;
        }
      }

      btnUpload.disabled = false;
      btnUpload.classList.remove('is-loading');
      uploadInput.value = '';

      if (fail === 0) {
        showToast(`Загружено: ${ok}`, 'success');
      } else if (ok === 0) {
        showToast(`Не удалось загрузить ${fail} файл(ов)`, 'error');
      } else {
        showToast(`Загружено ${ok}, ошибок ${fail}`, 'error');
      }
    });
  }

  const buildMetricsPath = (values) => {
    if (!Array.isArray(values) || values.length === 0) {
      return 'M0 32 L100 32';
    }
    const max = Math.max(...values, 1e-9);
    const min = Math.min(...values, 0);
    const range = Math.max(max - min, 1e-9);
    const n = values.length;
    const pts = values.map((v, i) => {
      const x = n === 1 ? 50 : (i / (n - 1)) * 100;
      const y = 30 - ((v - min) / range) * 28;
      return `${x.toFixed(2)} ${y.toFixed(2)}`;
    });
    return 'M' + pts.join(' L');
  };

  const updateMetricChart = (key, values) => {
    const lineEl = document.querySelector(`[data-metric-line="${key}"]`);
    if (lineEl) lineEl.setAttribute('d', buildMetricsPath(values));

    const valueEl = document.querySelector(`[data-metric-value="${key}"]`);
    if (valueEl) {
      const last = Array.isArray(values) && values.length
        ? values[values.length - 1]
        : null;
      valueEl.textContent = last === null ? '—' : Number(last).toFixed(3);
    }
  };

  const fillSidebar = (info, { full = true } = {}) => {
    sidebarInfo = info;

    if (projectStateEl && document.activeElement !== projectStateEl) {
      projectStateEl.checked = !!info.state;
    }
    if (metricsEpochEl) {
      metricsEpochEl.textContent = info.epoch > 0 ? `эп. ${info.epoch}` : 'эп. —';
    }

    if (full) {
      const p = info.params || {};
      if (trainModelEl && p.modelSelected && document.activeElement !== trainModelEl) {
        trainModelEl.value = p.modelSelected;
      }
      if (trainImgszEl && Number.isFinite(p.imgsz) && document.activeElement !== trainImgszEl) {
        trainImgszEl.value = p.imgsz;
      }
      if (trainLrEl && Number.isFinite(p.lr) && document.activeElement !== trainLrEl) {
        trainLrEl.value = p.lr;
      }
      if (trainBatchEl && Number.isFinite(p.batchSize) && document.activeElement !== trainBatchEl) {
        trainBatchEl.value = p.batchSize;
      }
      if (trainWorkersEl && Number.isFinite(p.worker) && document.activeElement !== trainWorkersEl) {
        trainWorkersEl.value = p.worker;
      }
      if (trainGpuEl && document.activeElement !== trainGpuEl) {
        trainGpuEl.checked = !!p.useGPU;
      }
    }

    const m = info.metrics || {};
    ['box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall', 'map50', 'map50_95']
      .forEach((key) => updateMetricChart(key, m[key]));
  };

  const loadProjectInfo = async (opts = { full: true }) => {
    const url = `/api/getprojectinfo?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project)}`;
    try {
      const res = await fetch(url, { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      if (data && data.error) { console.warn('getprojectinfo:', data.error); return; }
      fillSidebar(data, opts);
    } catch (err) {
      console.warn('getprojectinfo error:', err);
    }
  };

  const collectParams = () => ({
    useGPU: !!trainGpuEl?.checked,
    modelSelected: trainModelEl?.value || 'yolo26n',
    imgsz: Number(trainImgszEl?.value) || 640,
    lr: Number(trainLrEl?.value) || 0.0001,
    batchSize: Number(trainBatchEl?.value) || 16,
    worker: Number(trainWorkersEl?.value) || 2,
  });

  const saveTrainParams = async () => {
    clearTimeout(paramsSaveTimer);
    const params = collectParams();
    try {
      const res = await fetch('/api/updatetrainparams', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          login,
          password,
          project,
          useGPU: params.useGPU,
          modelSelected: params.modelSelected,
          imgsz: params.imgsz,
          lr: params.lr,
          batchSize: params.batchSize,
          worker: params.worker,
        }),
      });
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      if (data && data.error) {
        showToast(String(data.error), 'error');
        if (sidebarInfo) fillSidebar(sidebarInfo);
        return;
      }
      if (data && data.success) {
        sidebarInfo = { ...(sidebarInfo || {}), params: { ...params } };
        showToast('Параметры сохранены', 'success');
      }
    } catch (err) {
      console.error('updatetrainparams error:', err);
      showToast('Не удалось сохранить параметры', 'error');
    }
  };

  const scheduleParamsSave = () => {
    clearTimeout(paramsSaveTimer);
    paramsSaveTimer = setTimeout(saveTrainParams, 700);
  };

  const saveProjectState = async () => {
    const newState = !!projectStateEl?.checked;
    try {
      const res = await fetch('/api/setprojectstate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ login, password, project, state: newState }),
      });
      const data = await res.json();
      if (data === false) { window.location.href = 'index.html'; return; }
      if (data && data.error) {
        showToast(String(data.error), 'error');
        if (projectStateEl && sidebarInfo) projectStateEl.checked = !!sidebarInfo.state;
        return;
      }
      if (data && data.success) {
        if (sidebarInfo) sidebarInfo.state = !!data.state;
        showToast(newState ? 'Проект активирован' : 'Проект остановлен', 'success');
      }
    } catch (err) {
      console.error('setprojectstate error:', err);
      showToast('Не удалось изменить статус', 'error');
      if (projectStateEl && sidebarInfo) projectStateEl.checked = !!sidebarInfo.state;
    }
  };

  if (projectStateEl) {
    projectStateEl.addEventListener('change', saveProjectState);
  }

  [trainModelEl, trainImgszEl, trainLrEl, trainBatchEl, trainWorkersEl, trainGpuEl]
    .filter(Boolean)
    .forEach((el) => {
      el.addEventListener('change', scheduleParamsSave);
      el.addEventListener('input', scheduleParamsSave);
    });

  metricsTimer = setInterval(() => {
    if (document.visibilityState !== 'visible') return;
    loadProjectInfo({ full: false });
  }, 1000);

  window.addEventListener('beforeunload', () => {
    if (!state.filename) return;
    try {
      const body = JSON.stringify({
        login, password, project,
        filename: state.filename,
        annotations: annotations.map((a) => ({
          class_index: a.classIndex,
          x: a.x, y: a.y, w: a.w, h: a.h,
        })),
      });
      navigator.sendBeacon('/api/saveannotations',
        new Blob([body], { type: 'application/json' }));
    } catch (_) { }
  });

  (async () => {
    await loadClasses();
    await loadProjectInfo();
    nav('current');
  })();
});