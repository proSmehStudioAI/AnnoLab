document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('toggle-pass');
  const pass = document.getElementById('login-pass');
  const login = document.getElementById('login-name');
  const btn = document.querySelector('.btn');

  if (toggle && pass) {
    const sync = () => {
      pass.type = toggle.checked ? 'text' : 'password';
      const label = toggle.checked ? 'Скрыть пароль' : 'Показать пароль';
      toggle.setAttribute('aria-label', label);
      const labelEl = document.querySelector('.field__toggle');
      if (labelEl) labelEl.title = label;
    };
    toggle.addEventListener('change', sync);
    sync();
  }

  const clean = (v) => v.replace(/[\s\n\t]/g, '');

  const setCookie = (name, value, days = 30) => {
    const maxAge = days * 24 * 60 * 60;
    const secure = location.protocol === 'https:' ? '; Secure' : '';
    document.cookie =
      `${encodeURIComponent(name)}=${encodeURIComponent(value)}` +
      `; Max-Age=${maxAge}` +
      `; Path=/` +
      `; SameSite=Lax` +
      secure;
  };

  const getCookie = (name) => {
    const key = encodeURIComponent(name) + '=';
    const parts = document.cookie ? document.cookie.split('; ') : [];
    for (const part of parts) {
      if (part.indexOf(key) === 0) {
        return decodeURIComponent(part.slice(key.length));
      }
    }
    return null;
  };

  const deleteCookie = (name) => {
    document.cookie =
      `${encodeURIComponent(name)}=; Max-Age=0; Path=/; SameSite=Lax`;
  };

  let msgEl = null;

  const ensureMsg = () => {
    if (msgEl) return msgEl;
    msgEl = document.createElement('div');
    msgEl.className = 'form__msg';
    msgEl.id = 'login-msg';
    btn.after(msgEl);
    return msgEl;
  };

  const showMessage = (text, type, { shake = true } = {}) => {
    const el = ensureMsg();
    el.textContent = text;
    el.classList.remove('form__msg--error', 'form__msg--wait', 'form__msg--shake');
    el.classList.add(`form__msg--${type}`);
    if (shake) {
      void el.offsetWidth;
      el.classList.add('form__msg--shake');
    }
  };

  const hideMessage = () => {
    if (!msgEl) return;
    msgEl.textContent = '';
    msgEl.className = 'form__msg';
  };

  const setFieldsError = (isError) => {
    document.querySelectorAll('.field__control').forEach((c) => {
      c.classList.toggle('field__control--error', isError);
    });
  };

  const startPause = (seconds) => {
    let remaining = Math.ceil(Number(seconds));
    if (!Number.isFinite(remaining) || remaining <= 0) return;

    btn.disabled = true;

    const tick = (first) => {
      if (remaining > 0) {
        showMessage(
          `Подождите ${remaining} сек. перед следующей попыткой`,
          'wait',
          { shake: first }
        );
        remaining--;
        setTimeout(() => tick(false), 1000);
      } else {
        btn.disabled = false;
        hideMessage();
      }
    };

    tick(true);
  };

  const requestLogin = async (loginValue, passwordValue) => {
    const url = `/api/isLogin?login=${encodeURIComponent(loginValue)}&password=${encodeURIComponent(passwordValue)}`;
    const res = await fetch(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    const rawText = await res.text();
    let data = null;
    try {
      data = rawText ? JSON.parse(rawText) : null;
    } catch (e) {
      console.warn('isLogin: тело ответа не JSON', rawText);
    }
    return { ok: res.ok, status: res.status, data };
  };

  const applyLoginResult = ({ ok, status, data }, { loginValue, passwordValue, isAuto = false }) => {
    if (data === true) {
      setCookie('login', loginValue, 30);
      setCookie('password', passwordValue, 30);
      window.location.href = 'mainMenu.html';
      return false;
    }

    if (data === false) {
      deleteCookie('login');
      deleteCookie('password');
      setFieldsError(true);
      showMessage('Неверный логин или пароль', 'error');
      return false;
    }

    if (data && typeof data === 'object') {
      const pauseNum = Number(data.pause);
      if (Number.isFinite(pauseNum) && pauseNum > 0) {
        if (isAuto) {
          deleteCookie('login');
          deleteCookie('password');
        }
        startPause(pauseNum);
        return true;
      }
    }

    if (!ok) {
      showMessage(`Сервер вернул ошибку ${status}`, 'error');
      return false;
    }

    console.warn('Неизвестный ответ сервера:', data);
    return false;
  };

  const handleSubmit = async () => {
    if (btn.disabled) return;

    const loginValue = clean(login.value);
    const passwordValue = clean(pass.value);

    setFieldsError(false);
    hideMessage();
    btn.disabled = true;

    try {
      const result = await requestLogin(loginValue, passwordValue);
      const paused = applyLoginResult(result, { loginValue, passwordValue });
      if (!paused) btn.disabled = false;
    } catch (err) {
      console.error('isLogin fetch error:', err);
      showMessage('Не удалось связаться с сервером', 'error');
      btn.disabled = false;
    }
  };

  const tryAutoLogin = async () => {
    const savedLogin = getCookie('login');
    const savedPass = getCookie('password');

    if (savedLogin === null || savedPass === null) return;

    if (login) login.value = savedLogin;
    if (pass) pass.value = savedPass;

    btn.disabled = true;

    try {
      const result = await requestLogin(savedLogin, savedPass);
      const paused = applyLoginResult(result, {
        loginValue: savedLogin,
        passwordValue: savedPass,
        isAuto: true
      });
      if (!paused) btn.disabled = false;
    } catch (err) {
      console.warn('Auto-login error:', err);
      btn.disabled = false;
    }
  };

  if (btn) btn.addEventListener('click', handleSubmit);

  tryAutoLogin();
});