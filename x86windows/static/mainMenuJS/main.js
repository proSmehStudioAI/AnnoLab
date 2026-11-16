document.addEventListener('DOMContentLoaded', () => {
  const getCookie = (name) => {
    const key = encodeURIComponent(name) + '=';
    const parts = document.cookie ? document.cookie.split('; ') : [];
    for (const part of parts) {
      if (part.indexOf(key) === 0) return decodeURIComponent(part.slice(key.length));
    }
    return null;
  };

  const deleteCookie = (name) => {
    document.cookie = `${encodeURIComponent(name)}=; Max-Age=0; Path=/; SameSite=Lax`;
  };

  const escapeHtml = (str) =>
    String(str).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));

  const toastEl = document.getElementById('toast');
  let toastTimer = null;
  const showToast = (msg, type = '') => {
    if (!toastEl) {
      console.warn(msg);
      return;
    }
    toastEl.textContent = msg;
    toastEl.className = 'toast' + (type ? ' is-' + type : '');
    toastEl.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 3500);
  };

  const login = getCookie('login');
  const password = getCookie('password');

  if (login === null || password === null) {
    window.location.href = 'index.html';
    return;
  }

  const userNameEl = document.getElementById('user-name');
  const userAvatarEl = document.getElementById('user-avatar');
  if (userNameEl) userNameEl.textContent = login;
  if (userAvatarEl) userAvatarEl.textContent = login.charAt(0).toUpperCase();

  const btnLogout = document.getElementById('btn-logout');
  if (btnLogout) {
    btnLogout.addEventListener('click', () => {
      deleteCookie('login');
      deleteCookie('password');
      window.location.href = 'index.html';
    });
  }

  const listEl = document.querySelector('.projects__list');

  const iconSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="5"/><rect x="8.5" y="8.5" width="7" height="7" rx="1.6"/></svg>';
  const typeSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15h.01M12 15h.01M17 15h.01"/></svg>';
  const downloadSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v12"/><path d="m7 11 5 5 5-5"/><path d="M5 20h14"/></svg>';
  const datasetSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>';
  const trashSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6M14 11v6"/></svg>';

  const updateStats = (projects) => {
    const values = document.querySelectorAll('.stat__value');
    if (values.length < 3) return;
    const total = projects.length;
    const active = projects.filter((p) => p.state === true).length;
    values[0].textContent = total;
    values[1].textContent = active;
    values[2].textContent = total - active;
  };

  const openProject = (project) => {
    if (!project || !project.name) return;
    window.location.href =
      `editorYolovDetection.html?project=${encodeURIComponent(project.name)}`;
  };

  const saveBlobAsFile = (blob, filename) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const filenameFromDisposition = (cd, fallback) => {
    const match = /filename\*?=(?:UTF-8'')?["']?([^";']+)/i.exec(cd || '');
    return match ? decodeURIComponent(match[1]) : fallback;
  };

  const downloadDataset = async (project) => {
    if (!project || !project.name) return;
    const url = `/api/downloaddataset?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project.name)}`;

    try {
      const res = await fetch(url);
      const ct = res.headers.get('Content-Type') || '';

      if (ct.includes('application/json')) {
        const data = await res.json();
        if (data === false) { window.location.href = 'index.html'; return; }
        if (data && data.error) {
          showToast(String(data.error), 'error');
          return;
        }
        showToast('Датасет пуст', 'error');
        return;
      }

      if (!res.ok) {
        showToast(`Сервер вернул ошибку ${res.status}`, 'error');
        return;
      }

      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition');
      const filename = filenameFromDisposition(cd, `${project.name}-dataset.zip`);
      saveBlobAsFile(blob, filename);
      showToast('Датасет скачивается', 'success');
    } catch (err) {
      console.error('downloaddataset error:', err);
      showToast('Не удалось скачать датасет', 'error');
    }
  };

  const downloadModel = async (project) => {
    if (!project || !project.name) return;
    const url = `/api/downloadmodel?login=${encodeURIComponent(login)}` +
                `&password=${encodeURIComponent(password)}` +
                `&project=${encodeURIComponent(project.name)}`;

    try {
      const res = await fetch(url);
      const ct = res.headers.get('Content-Type') || '';

      if (ct.includes('application/json')) {
        const data = await res.json();
        if (data === false) { window.location.href = 'index.html'; return; }
        if (data && data.error) {
          showToast(String(data.error), 'error');
          return;
        }
        showToast('Модель не найдена', 'error');
        return;
      }

      if (!res.ok) {
        showToast(`Сервер вернул ошибку ${res.status}`, 'error');
        return;
      }

      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition');
      const filename = filenameFromDisposition(cd, `${project.name}-model.pt`);
      saveBlobAsFile(blob, filename);
      showToast('Модель скачивается', 'success');
    } catch (err) {
      console.error('downloadmodel error:', err);
      showToast('Не удалось скачать модель', 'error');
    }
  };

  const deleteModal = document.getElementById('modal-delete');
  const deleteProjectNameEl = document.getElementById('delete-project-name');
  const deleteCodeEl = document.getElementById('delete-code');
  const deleteCodeInput = document.getElementById('delete-code-input');
  const deleteMsg = document.getElementById('delete-msg');
  const btnDeleteConfirm = document.getElementById('btn-delete-confirm');

  const deleteState = {
    project: null,
    code: null,
    submitting: false,
  };

  const generateCode = () => String(Math.floor(100000 + Math.random() * 900000));

  const resetDeleteMsg = () => {
    if (!deleteMsg) return;
    deleteMsg.className = 'form__msg';
    deleteMsg.textContent = '';
  };

  const showDeleteError = (text) => {
    if (!deleteMsg) return;
    deleteMsg.className = 'form__msg form__msg--error';
    deleteMsg.textContent = text;
  };

  const openDeleteModal = (project) => {
    if (!deleteModal || !project || !project.name) return;

    deleteState.project = project;
    deleteState.code = generateCode();
    deleteState.submitting = false;

    if (deleteProjectNameEl) deleteProjectNameEl.textContent = project.name;
    if (deleteCodeEl) deleteCodeEl.textContent = deleteState.code;
    if (deleteCodeInput) deleteCodeInput.value = '';
    if (btnDeleteConfirm) btnDeleteConfirm.disabled = false;
    resetDeleteMsg();

    deleteModal.hidden = false;
    document.body.style.overflow = 'hidden';
    if (deleteCodeInput) setTimeout(() => deleteCodeInput.focus(), 50);
  };

  const closeDeleteModal = () => {
    if (!deleteModal) return;
    deleteModal.hidden = true;
    document.body.style.overflow = '';
    deleteState.project = null;
    deleteState.code = null;
    deleteState.submitting = false;
  };

  if (deleteCodeEl) {
    ['copy', 'cut', 'contextmenu', 'dragstart'].forEach((evt) => {
      deleteCodeEl.addEventListener(evt, (e) => e.preventDefault());
    });
  }

  if (deleteCodeInput) {
    ['paste', 'copy', 'cut', 'drop', 'dragstart'].forEach((evt) => {
      deleteCodeInput.addEventListener(evt, (e) => e.preventDefault());
    });

    deleteCodeInput.addEventListener('input', () => {
      deleteCodeInput.value = deleteCodeInput.value.replace(/\D/g, '').slice(0, 6);
    });

    deleteCodeInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        deleteProject();
      }
    });
  }

  if (deleteModal) {
    deleteModal.querySelectorAll('[data-delete-close]').forEach((el) => {
      el.addEventListener('click', closeDeleteModal);
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && deleteModal && !deleteModal.hidden) {
      closeDeleteModal();
    }
  });

  const deleteProject = async () => {
    if (!deleteState.project || deleteState.submitting) return;

    const entered = (deleteCodeInput?.value || '').trim();
    if (entered !== deleteState.code) {
      showDeleteError('Неверный код');
      deleteCodeInput?.focus();
      deleteCodeInput?.select();
      return;
    }

    deleteState.submitting = true;
    if (btnDeleteConfirm) btnDeleteConfirm.disabled = true;
    resetDeleteMsg();

    try {
      const res = await fetch('/api/deleteproject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          login,
          password,
          project: deleteState.project.name,
        }),
      });

      const data = await res.json();

      if (data === false) {
        redirecting = true;
        window.location.href = 'index.html';
        return;
      }

      if (data && data.error) {
        showDeleteError(String(data.error));
        deleteState.submitting = false;
        if (btnDeleteConfirm) btnDeleteConfirm.disabled = false;
        return;
      }

      if (data && data.success) {
        closeDeleteModal();
        showToast('Проект удалён', 'success');
        await checkProjectsState();
        return;
      }

      showDeleteError('Не удалось удалить проект');
      deleteState.submitting = false;
      if (btnDeleteConfirm) btnDeleteConfirm.disabled = false;
    } catch (err) {
      console.error('deleteproject error:', err);
      showDeleteError('Не удалось связаться с сервером');
      deleteState.submitting = false;
      if (btnDeleteConfirm) btnDeleteConfirm.disabled = false;
    }
  };

  if (btnDeleteConfirm) {
    btnDeleteConfirm.addEventListener('click', deleteProject);
  }

  const renderProjects = (projects) => {
    if (!listEl) return;
    if (!Array.isArray(projects) || projects.length === 0) {
      listEl.innerHTML = '';
      updateStats([]);
      return;
    }

    listEl.innerHTML = projects.map((p, i) => {
      const isActive = p.state === true;
      const rowClass = isActive ? 'project project--active' : 'project project--inactive';
      const statusBadge = isActive
        ? '<span class="badge badge--active"><span class="badge__dot"></span>Активный</span>'
        : '<span class="badge badge--inactive"><span class="badge__dot"></span>Не активный</span>';
      const downloadDisabled = '';
      const datasetDisabled = (Number(p.countImages) || 0) > 0 ? '' : ' disabled';

      return `
        <li class="${rowClass}">
          <div class="project__main">
            <div class="project__icon">${iconSvg}</div>
            <div class="project__text">
              <div class="project__name">${escapeHtml(p.name)}</div>
              <div class="project__sub">${Number(p.countImages) || 0} изображений · ${Number(p.countClass) || 0} класса</div>
            </div>
          </div>
          <div class="project__type">
            <span class="badge badge--type">${typeSvg}${escapeHtml(p.type)}</span>
          </div>
          <div class="project__status">${statusBadge}</div>
          <div class="project__actions">
            <button class="btn-ghost" type="button"${datasetDisabled}
                    data-download-dataset="${i}" title="Скачать размеченный датасет">
              ${datasetSvg}Датасет
            </button>
            <button class="btn-ghost" type="button"${downloadDisabled}
                    data-download-model="${i}" title="Скачать обученную модель">
              ${downloadSvg}Модель
            </button>
            <button class="btn-ghost" type="button" data-open-index="${i}">Открыть</button>
            <button class="btn-ghost btn-ghost--icon" type="button"
                    data-delete-index="${i}" title="Удалить проект"
                    aria-label="Удалить проект">
              ${trashSvg}
            </button>
          </div>
        </li>`;
    }).join('');

    listEl.querySelectorAll('[data-open-index]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = Number(btn.dataset.openIndex);
        openProject(projects[idx]);
      });
    });

    listEl.querySelectorAll('[data-download-dataset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = Number(btn.dataset.downloadDataset);
        downloadDataset(projects[idx]);
      });
    });

    listEl.querySelectorAll('[data-download-model]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = Number(btn.dataset.downloadModel);
        downloadModel(projects[idx]);
      });
    });

    listEl.querySelectorAll('[data-delete-index]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = Number(btn.dataset.deleteIndex);
        openDeleteModal(projects[idx]);
      });
    });

    updateStats(projects);
  };

  let redirecting = false;

  const checkProjectsState = async () => {
    if (redirecting) return;

    try {
      const url = `/api/checkprojectsstate?login=${encodeURIComponent(login)}&password=${encodeURIComponent(password)}`;
      const res = await fetch(url, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (!res.ok) return;

      const rawText = await res.text();
      let data = null;
      try {
        data = rawText ? JSON.parse(rawText) : null;
      } catch (e) {
        return;
      }

      if (data === false) {
        redirecting = true;
        window.location.href = 'index.html';
        return;
      }

      if (Array.isArray(data)) {
        renderProjects(data);
      }
    } catch (err) {
      console.error('checkprojectsstate error:', err);
    }
  };

  const modal = document.getElementById('modal-create');
  const btnNewProject = document.getElementById('btn-new-project');
  const formCreate = document.getElementById('form-create');
  const stateCheckbox = document.getElementById('p-state');
  const trainFields = document.getElementById('train-fields');
  const msgEl = document.getElementById('create-msg');
  const btnSubmit = document.getElementById('btn-create-submit');
  const fileInput = document.getElementById('p-coco');
  const fileNameEl = document.getElementById('p-coco-name');
  const fileLabel = fileInput ? fileInput.closest('.file') : null;

  const openModal = () => {
    if (!modal) return;
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
    msgEl.className = 'form__msg';
    msgEl.textContent = '';
    if (btnSubmit) btnSubmit.disabled = false;
    if (fileLabel) fileLabel.classList.remove('file--error');
    const nameInput = document.getElementById('p-name');
    if (nameInput) setTimeout(() => nameInput.focus(), 50);
  };

  const closeModal = () => {
    if (!modal) return;
    modal.hidden = true;
    document.body.style.overflow = '';
    if (formCreate) formCreate.reset();
    if (trainFields) trainFields.hidden = true;
    if (fileNameEl) {
      fileNameEl.textContent = 'Перетащите или выберите coco.txt';
      fileNameEl.classList.remove('file__text--set');
    }
    if (fileLabel) fileLabel.classList.remove('file--error');
    msgEl.className = 'form__msg';
    msgEl.textContent = '';
    if (btnSubmit) btnSubmit.disabled = false;
  };

  const showError = (text) => {
    msgEl.className = 'form__msg form__msg--error';
    msgEl.textContent = text;
  };

  const showSuccess = (text) => {
    msgEl.className = 'form__msg form__msg--success';
    msgEl.textContent = text;
  };

  if (btnNewProject) btnNewProject.addEventListener('click', openModal);

  if (modal) {
    modal.querySelectorAll('[data-modal-close]').forEach((el) => {
      el.addEventListener('click', closeModal);
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && !modal.hidden) closeModal();
  });

  if (stateCheckbox && trainFields) {
    stateCheckbox.addEventListener('change', () => {
      trainFields.hidden = !stateCheckbox.checked;
    });
  }

  if (fileInput && fileNameEl) {
    fileInput.addEventListener('change', () => {
      if (fileLabel) fileLabel.classList.remove('file--error');
      const f = fileInput.files && fileInput.files[0];
      if (f) {
        fileNameEl.textContent = f.name;
        fileNameEl.classList.add('file__text--set');
      } else {
        fileNameEl.textContent = 'Перетащите или выберите coco.txt';
        fileNameEl.classList.remove('file__text--set');
      }
    });
  }

  if (formCreate) {
    formCreate.addEventListener('submit', async (e) => {
      e.preventDefault();

      const name = document.getElementById('p-name').value.trim();
      if (!name) {
        showError('Введите название проекта');
        return;
      }

      const cocoFile = fileInput && fileInput.files ? fileInput.files[0] : null;
      if (!cocoFile) {
        if (fileLabel) fileLabel.classList.add('file--error');
        showError('Загрузите файл coco.txt');
        return;
      }

      if (!/\.txt$/i.test(cocoFile.name)) {
        if (fileLabel) fileLabel.classList.add('file--error');
        showError('Файл должен быть с расширением .txt');
        return;
      }

      const type = document.getElementById('p-type').value;
      const state = stateCheckbox.checked;

      const fd = new FormData();
      fd.append('login', login);
      fd.append('password', password);
      fd.append('name', name);
      fd.append('type', type);
      fd.append('state', state ? 'true' : 'false');
      fd.append('coco', cocoFile, cocoFile.name);

      if (state) {
        fd.append('useGPU', document.getElementById('p-gpu').checked ? 'true' : 'false');
        fd.append('modelSelected', document.getElementById('p-model').value);
        fd.append('imgsz', document.getElementById('p-imgsz').value);
        fd.append('lr', document.getElementById('p-lr').value);
        fd.append('batchSize', document.getElementById('p-batch').value);
        fd.append('worker', document.getElementById('p-workers').value);
      }

      btnSubmit.disabled = true;
      showSuccess('Создаём проект и загружаем файл…');

      try {
        const res = await fetch('/api/createproject', {
          method: 'POST',
          headers: { 'Accept': 'application/json' },
          body: fd
        });

        const rawText = await res.text();
        let data = null;
        try {
          data = rawText ? JSON.parse(rawText) : null;
        } catch (err) {
          data = null;
        }

        if (data === false) {
          redirecting = true;
          window.location.href = 'index.html';
          return;
        }

        if (!res.ok) {
          showError(`Сервер вернул ошибку ${res.status}`);
          btnSubmit.disabled = false;
          return;
        }

        if (data === true || (data && data.success === true)) {
          closeModal();
          await checkProjectsState();
          return;
        }

        if (data && typeof data === 'object' && data.error) {
          showError(String(data.error));
          btnSubmit.disabled = false;
          return;
        }

        showError('Не удалось создать проект');
        btnSubmit.disabled = false;
      } catch (err) {
        console.error('createproject error:', err);
        showError('Не удалось связаться с сервером');
        btnSubmit.disabled = false;
      }
    });
  }

  checkProjectsState();
  setInterval(checkProjectsState, 10000);
});