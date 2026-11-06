/* App shell: storage, settings, security, first-run setup, home and new case */
'use strict';
const APP_VERSION = '2.0.0';

// ---------- DOM helpers ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
const $ = (s, r = document) => r.querySelector(s);
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
const fmtDate = (t) => t ? new Date(t).toLocaleDateString('ar', { year: 'numeric', month: 'short', day: 'numeric' }) : '';
const fmtDateTime = (t) => t ? new Date(t).toLocaleString('ar', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
const num = (v, d = 1) => v == null || Number.isNaN(+v) ? '—' : (+v).toFixed(d);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function toast(msg, ms = 2600) { const t = h('div', { class: 'toast', role: 'status' }, msg); document.body.append(t); setTimeout(() => t.remove(), ms); }
function modal(title, body, buttons) {
  return new Promise((resolve) => {
    const bg = h('div', { class: 'modalbg' });
    const close = (v) => { bg.remove(); resolve(v); };
    bg.addEventListener('click', (e) => { if (e.target === bg) close(null); });
    const btns = (buttons || [{ label: 'حسناً', value: true, primary: true }]).map((b) => h('button', { class: 'btn ' + (b.primary ? 'primary grow' : b.danger ? 'danger' : ''), onclick: () => close(typeof b.value === 'function' ? b.value() : b.value) }, b.label));
    bg.append(h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' }, h('h2', null, title), body, h('div', { class: 'row', style: { marginTop: '16px' } }, btns)));
    document.body.append(bg);
  });
}
const confirmBox = (title, text, ok = 'تأكيد', danger = false) => modal(title, h('p', null, text), [{ label: 'إلغاء', value: false }, { label: ok, value: true, primary: !danger, danger }]);
async function promptBox(title, label, type = 'text', value = '') {
  const inp = h('input', { type, value, id: 'prompt-input' });
  return modal(title, h('label', { class: 'field' }, h('span', null, label), inp), [{ label: 'إلغاء', value: null }, { label: 'موافق', value: () => inp.value, primary: true }]);
}

// ---------- IndexedDB ----------
const DB = {
  db: null,
  open() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('dsd-clinic-v2', 1);
      r.onupgradeneeded = () => {
        const d = r.result;
        d.createObjectStore('patients', { keyPath: 'id' });
        const cs = d.createObjectStore('cases', { keyPath: 'id' }); cs.createIndex('patientId', 'patientId');
        d.createObjectStore('blobs', { keyPath: 'id' }); d.createObjectStore('kv', { keyPath: 'k' });
      };
      r.onsuccess = () => { DB.db = r.result; res(); }; r.onerror = () => rej(r.error);
    });
  },
  tx(store, mode, fn) {
    return new Promise((res, rej) => {
      const t = DB.db.transaction(store, mode), s = t.objectStore(store); let out; const q = fn(s); if (q) q.onsuccess = () => { out = q.result; };
      t.oncomplete = () => res(out); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error);
    });
  },
  get: (s, k) => DB.tx(s, 'readonly', (st) => st.get(k)), put: (s, v) => DB.tx(s, 'readwrite', (st) => st.put(v)),
  del: (s, k) => DB.tx(s, 'readwrite', (st) => st.delete(k)), all: (s) => DB.tx(s, 'readonly', (st) => st.getAll()),
  async getKV(k, d) { const r = await DB.get('kv', k); return r ? r.v : d; }, setKV: (k, v) => DB.put('kv', { k, v })
};
async function putBlob(blob, id = uid()) { await DB.put('blobs', { id, blob }); return id; }
async function getBlob(id) { if (!id) return null; const r = await DB.get('blobs', id); return r ? r.blob : null; }

// ---------- settings ----------
const DEFAULT_SETTINGS = { clinicName: 'عيادة الأسنان', doctorName: '', theme: 'auto', rules: C.DEFAULT_RULES, ipdMm: 63, pin: null, disclaimerAccepted: false, material: 'إيماكس (ليثيوم ديسيليكات)', serverUrl: '' };
let S = null;
async function loadSettings() {
  S = Object.assign(C.clone(DEFAULT_SETTINGS), await DB.getKV('settings', {}) || {});
  const have = new Set(S.rules.map((r) => r.id)); for (const r of C.DEFAULT_RULES) if (!have.has(r.id)) S.rules.push(C.clone(r));
  applyTheme();
}
const saveSettings = () => DB.setKV('settings', S);
function applyTheme() { if (S.theme === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', S.theme); }

// ---------- crypto ----------
const b64 = {
  enc(buf) { let s = ''; const b = new Uint8Array(buf); for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); },
  dec(str) { const s = atob(str), b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; }
};
async function deriveKey(pass, salt, iter = 250000) {
  const base = await crypto.subtle.importKey('raw', new TextEncoder().encode(pass), 'PBKDF2', false, ['deriveKey']);
  return crypto.subtle.deriveKey({ name: 'PBKDF2', salt, iterations: iter, hash: 'SHA-256' }, base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
}
async function hashPin(pin, salt) {
  const base = await crypto.subtle.importKey('raw', new TextEncoder().encode(pin), 'PBKDF2', false, ['deriveBits']);
  return b64.enc(await crypto.subtle.deriveBits({ name: 'PBKDF2', salt, iterations: 150000, hash: 'SHA-256' }, base, 256));
}

// ---------- router ----------
let cleanup = null;
function go(hash) { if (location.hash === hash) render(); else location.hash = hash; }
window.addEventListener('hashchange', () => render());
async function render() {
  if (cleanup) { try { cleanup(); } catch (e) { /* ignore */ } cleanup = null; }
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  const root = $('#app'); root.innerHTML = '';
  try {
    if (!parts.length) await viewHome(root);
    else if (parts[0] === 'new') await viewNewCase(root, parts[1]);
    else if (parts[0] === 'capture') await viewCapture(root, parts[1]);
    else if (parts[0] === 'case' && parts[2] === 'analyzing') return;     // the analysis screen draws itself
    else if (parts[0] === 'case') await viewResults(root, parts[1]);
    else if (parts[0] === 'present') await viewPresent(root, parts[1]);
    else if (parts[0] === 'settings') await viewSettings(root);
    else await viewHome(root);
  } catch (e) {
    console.error(e);
    root.append(appbar('خطأ', '#/'), h('main', null, h('div', { class: 'card' }, h('p', null, String(e && e.message || e)), h('a', { class: 'btn', href: '#/' }, 'الرئيسية'))));
  }
}
function appbar(title, back, ...extra) {
  return h('header', { class: 'appbar' },
    back ? h('a', { class: 'iconbtn', href: back, 'aria-label': 'رجوع' }, '→') : null,
    h('h1', null, title), ...extra);
}
function actionbar(...kids) { return h('div', { class: 'actionbar' }, h('div', { class: 'inner' }, kids)); }

// ---------- first-run setup: download AI tools with visible progress (never ask the user to pick files) ----------
let setupShown = false;
function ensureReady() {
  const p = Engine.prepare(onSetupStatus);
  return p;
}
const setupState = { model: { state: 'idle', got: 0, total: 0 }, face: { state: 'idle' } };
function onSetupStatus(s) { Object.assign(setupState[s.key] || (setupState[s.key] = {}), s); drawSetup(); }
function drawSetup() {
  const el = $('#setup'); if (!el) return;
  const row = (key, title, sub) => {
    const s = setupState[key] || {}, pct = s.state === 'done' ? 100 : s.total ? Math.round(100 * s.got / s.total) : (s.state === 'loading' ? 8 : 0);
    return h('div', null, h('div', { class: 'row' }, h('b', { class: 'grow' }, title), h('span', { class: 'small muted' }, s.state === 'done' ? 'جاهز' : s.total ? pct + '%' : 'جارٍ…')),
      h('div', { class: 'small muted' }, sub), h('div', { class: 'prog' }, h('i', { style: { width: pct + '%' } })));
  };
  el.querySelector('.rows').replaceChildren(row('model', 'نموذج تحديد الأسنان', 'نحو 16 ميغابايت، من موقع التطبيق نفسه'), row('face', 'كاشف الوجه', 'Google MediaPipe، نحو 4 ميغابايت'));
}
async function withReady(fn) {
  const done = await Promise.race([Engine.prepare(onSetupStatus).then(() => true), new Promise((r) => setTimeout(() => r(false), 150))]);
  if (done) return fn();
  const ov = h('div', { class: 'overlay', id: 'setup' }, h('div', { class: 'box' }, h('h2', null, 'تجهيز النظام لأول مرة'),
    h('p', { class: 'muted' }, 'يُحمَّل الذكاء الاصطناعي مرة واحدة فقط، ثم يعمل التطبيق بلا إنترنت.'), h('div', { class: 'rows' }), h('div', { class: 'err' })));
  document.body.append(ov); drawSetup();
  try { await Engine.prepare(onSetupStatus); ov.remove(); return fn(); }
  catch (e) {
    console.error(e);
    ov.querySelector('.err').replaceChildren(h('div', { class: 'notice fail' }, 'تعذّر التجهيز: ' + e.message + '. تأكد من الاتصال بالإنترنت ثم أعد المحاولة.'),
      h('button', { class: 'btn primary block', onclick: () => { ov.remove(); withReady(fn); } }, 'إعادة المحاولة'),
      h('button', { class: 'btn link block', onclick: () => ov.remove() }, 'إلغاء'));
  }
}

// ---------- home ----------
async function viewHome(root) {
  const cases = (await DB.all('cases')).sort((a, b) => (b.updated || 0) - (a.updated || 0));
  const patients = Object.fromEntries((await DB.all('patients')).map((p) => [p.id, p]));
  const list = h('ul', { class: 'list' });
  const search = h('input', { type: 'search', id: 'search', placeholder: 'بحث بالاسم أو رقم الملف', 'aria-label': 'بحث' });
  const draw = async () => {
    const q = search.value.trim(); list.innerHTML = '';
    const shown = cases.filter((c) => { const p = patients[c.patientId] || {}; return !q || (p.name || '').includes(q) || (p.fileNo || '').includes(q); });
    if (!shown.length) list.append(h('li', { class: 'empty' }, cases.length ? 'لا نتائج' : 'لا توجد حالات بعد.'));
    for (const c of shown) {
      const p = patients[c.patientId] || {};
      const img = h('img', { alt: '' });
      getBlob(c.thumbId).then((b) => { if (b) img.src = URL.createObjectURL(b); else img.replaceWith(h('div', { class: 'ph' })); });
      list.append(h('li', null, h('a', { class: 'caseitem', href: '#/case/' + c.id }, img,
        h('div', { class: 'grow' }, h('b', null, p.name || 'بلا اسم'), h('div', { class: 'small muted' }, [p.fileNo ? 'ملف ' + p.fileNo : null, fmtDate(c.created)].filter(Boolean).join(' · '))),
        h('span', { class: 'badge ' + (c.approval ? 'ok' : c.analysis ? '' : 'warn') }, c.approval ? 'معتمدة' : c.analysis ? 'مسودة' : 'بلا صورة'))));
    }
  };
  search.addEventListener('input', draw); draw();
  root.append(appbar(S.clinicName || 'تصميم الابتسامة', null, h('a', { class: 'iconbtn', href: '#/settings', 'aria-label': 'الإعدادات' }, '⚙')),
    h('main', null,
      h('button', { class: 'newcase', onclick: () => go('#/new') }, h('span', { class: 'plus' }, '+'), h('span', null, h('b', null, 'حالة جديدة'), h('span', { class: 'small' }, 'صوّر ابتسامة المريض واحصل على التحليل والتصميم تلقائياً'))),
      cases.length > 6 ? h('div', { style: { marginBottom: '10px' } }, search) : null,
      h('div', { class: 'card' }, h('h2', null, 'الحالات'), list),
      backupReminder()));
  Engine.prepare(onSetupStatus).catch(() => {});   // warm up in the background
}
function backupReminder() {
  const el = h('div');
  DB.getKV('lastBackup', null).then(async (last) => {
    const n = (await DB.all('cases')).length;
    if (n && (!last || Date.now() - last > 7 * 864e5)) el.append(h('div', { class: 'notice warn' }, h('span', { class: 'grow' }, last ? 'مرّ أسبوع على آخر نسخة احتياطية.' : 'لم تُؤخذ نسخة احتياطية بعد. البيانات على هذا الجهاز فقط.'), h('a', { class: 'btn link', href: '#/settings' }, 'نسخ الآن')));
  });
  return el;
}

// ---------- new case: patient + consent, then camera ----------
async function viewNewCase(root, patientId) {
  const patients = await DB.all('patients');
  let p = patientId ? patients.find((x) => x.id === patientId) : null;
  const name = h('input', { type: 'text', id: 'pname', list: 'plist', autocomplete: 'off', value: p?.name || '' });
  const file = h('input', { type: 'text', id: 'pfile', value: p?.fileNo || '' });
  const consent = h('input', { type: 'checkbox', id: 'consent' });
  const dl = h('datalist', { id: 'plist' }, patients.map((x) => h('option', { value: x.name }, x.fileNo || '')));
  const start = async (gallery) => {
    if (!name.value.trim()) { toast('اكتب اسم المريض'); name.focus(); return; }
    if (!consent.checked) { toast('سجّل موافقة المريض على التصوير'); return; }
    let pt = patients.find((x) => x.name === name.value.trim() && (x.fileNo || '') === file.value.trim());
    if (!pt) { pt = { id: uid(), name: name.value.trim(), fileNo: file.value.trim(), created: Date.now() }; }
    pt.updated = Date.now(); await DB.put('patients', pt);
    const c = { id: uid(), patientId: pt.id, created: Date.now(), updated: Date.now(), consent: { photo: true, at: Date.now() }, design: { ...DEFAULT_DESIGN }, approval: null };
    await DB.put('cases', c);
    if (gallery) pickFromGallery(c.id); else go('#/capture/' + c.id);
  };
  root.append(appbar('حالة جديدة', '#/'), h('main', null, h('div', { class: 'card' },
    h('label', { class: 'field' }, h('span', null, 'اسم المريض'), name, dl),
    h('label', { class: 'field' }, h('span', null, 'رقم الملف (اختياري)'), file),
    h('label', { class: 'check' }, consent, h('span', null, 'وافق المريض على تصويره وحفظ الصورة في ملفه لغرض العلاج.'))),
    h('div', { class: 'card small muted' }, h('b', null, 'للحصول على أفضل نتيجة: '), 'المريض ينظر إلى الكاميرا ويبتسم ابتسامة عريضة، والوجه كله ظاهر، في إضاءة جيدة. الكاميرا ترشدك وتلتقط تلقائياً.')),
  actionbar(h('button', { class: 'btn primary grow', onclick: () => start(false) }, '📷 فتح الكاميرا'), h('button', { class: 'btn', onclick: () => start(true) }, 'من المعرض')));
  setTimeout(() => name.focus(), 50);
}
function pickFromGallery(caseId) {
  const inp = h('input', { type: 'file', accept: 'image/*', class: 'hidden' });
  inp.addEventListener('change', () => { const f = inp.files[0]; inp.remove(); if (f) startAnalysis(caseId, f); else go('#/case/' + caseId); });
  document.body.append(inp); inp.click();
}

// ---------- lock & disclaimer ----------
let lastHidden = 0;
async function showLock() {
  if (!S.pin) return;
  await new Promise((resolve) => {
    const inp = h('input', { type: 'password', inputmode: 'numeric', id: 'pin', autocomplete: 'off', placeholder: 'رمز الدخول', style: { textAlign: 'center', fontSize: '22px' } });
    const msg = h('div', { class: 'small muted', style: { minHeight: '22px' } });
    const tryIt = async () => { if (await hashPin(inp.value, b64.dec(S.pin.salt)) === S.pin.hash) { el.remove(); resolve(); } else { msg.textContent = 'رمز غير صحيح'; inp.value = ''; } };
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') tryIt(); });
    const el = h('div', { class: 'lock' }, h('div', { style: { maxWidth: '320px', width: '100%', textAlign: 'center' } }, h('div', { style: { fontSize: '40px' } }, '🔒'), h('h2', null, S.clinicName), inp, msg, h('button', { class: 'btn primary block', onclick: tryIt }, 'دخول')));
    document.body.append(el); setTimeout(() => inp.focus(), 50);
  });
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) lastHidden = Date.now();
  else if (S?.pin && lastHidden && Date.now() - lastHidden > 5 * 60 * 1000 && !$('.lock')) showLock();
});
async function showDisclaimer() {
  if (S.disclaimerAccepted) return;
  await modal('قبل البدء', h('div', null,
    h('p', null, 'هذا النظام يساعد الطبيب في التحليل والتصميم وعرض النتيجة المتوقعة، وليس جهاز تشخيص.'),
    h('p', null, 'المحاكاة تعديل لصورة المريض نفسها بحسب التصميم، وهي تقريبية. القرار السريري والاعتماد للطبيب.'),
    h('p', null, 'الصور والبيانات تبقى على هذا الجهاز فقط. خذ نسخة احتياطية مشفّرة بانتظام.')), [{ label: 'فهمت', value: true, primary: true }]);
  S.disclaimerAccepted = true; await saveSettings();
}
