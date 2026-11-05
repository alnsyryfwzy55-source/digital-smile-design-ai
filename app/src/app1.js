/* ===== DSD Clinic app — part 1: infrastructure ===== */
'use strict';
const C = window.DSDCore;
const APP_VERSION = '1.0.0';
const ORT_VERSION = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_FILE = 'teeth_unet_v2.onnx';

// ---------- DOM helpers ----------
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
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
const ltr = (t) => h('bdi', { dir: 'ltr' }, t ?? '—');
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function toast(msg, ms = 2600) {
  const t = h('div', { class: 'toast', role: 'status' }, msg);
  document.body.append(t); setTimeout(() => t.remove(), ms);
}
function modal(title, body, buttons) {
  return new Promise((resolve) => {
    const bg = h('div', { class: 'modalbg' });
    const close = (v) => { bg.remove(); resolve(v); };
    const box = h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' }, h('h2', null, title), body,
      h('div', { class: 'row', style: { marginTop: '14px', justifyContent: 'flex-end' } },
        (buttons || [{ label: 'حسناً', value: true, primary: true }]).map((b) => h('button', { class: 'btn' + (b.primary ? ' primary' : '') + (b.danger ? ' danger' : ''), onclick: () => close(typeof b.value === 'function' ? b.value() : b.value) }, b.label))));
    bg.append(box); document.body.append(bg);
  });
}
const confirmBox = (title, text, okLabel = 'تأكيد', danger = false) => modal(title, h('p', null, text), [{ label: 'إلغاء', value: false }, { label: okLabel, value: true, primary: !danger, danger }]);
async function promptBox(title, label, type = 'text', value = '') {
  const inp = h('input', { type, value });
  const r = await modal(title, h('label', { class: 'field' }, h('span', null, label), inp), [{ label: 'إلغاء', value: null }, { label: 'موافق', value: () => inp.value, primary: true }]);
  return r;
}

// ---------- IndexedDB ----------
const DB = {
  db: null,
  open() {
    return new Promise((res, rej) => {
      const r = indexedDB.open('dsd-clinic', 1);
      r.onupgradeneeded = () => {
        const d = r.result;
        d.createObjectStore('patients', { keyPath: 'id' });
        const cs = d.createObjectStore('cases', { keyPath: 'id' }); cs.createIndex('patientId', 'patientId');
        d.createObjectStore('blobs', { keyPath: 'id' });
        d.createObjectStore('kv', { keyPath: 'k' });
      };
      r.onsuccess = () => { DB.db = r.result; res(); };
      r.onerror = () => rej(r.error);
    });
  },
  tx(store, mode, fn) {
    return new Promise((res, rej) => {
      const t = DB.db.transaction(store, mode); const s = t.objectStore(store);
      let out; const q = fn(s); if (q) q.onsuccess = () => { out = q.result; };
      t.oncomplete = () => res(out); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error);
    });
  },
  get: (s, k) => DB.tx(s, 'readonly', (st) => st.get(k)),
  put: (s, v) => DB.tx(s, 'readwrite', (st) => st.put(v)),
  del: (s, k) => DB.tx(s, 'readwrite', (st) => st.delete(k)),
  all: (s) => DB.tx(s, 'readonly', (st) => st.getAll()),
  byIndex: (s, i, v) => DB.tx(s, 'readonly', (st) => st.index(i).getAll(v)),
  async getKV(k, d) { const r = await DB.get('kv', k); return r ? r.v : d; },
  setKV: (k, v) => DB.put('kv', { k, v })
};
async function putBlob(blob, id = uid()) { await DB.put('blobs', { id, blob, type: blob.type }); return id; }
async function getBlob(id) { if (!id) return null; const r = await DB.get('blobs', id); return r ? r.blob : null; }

// ---------- settings ----------
const DEFAULT_SETTINGS = {
  clinicName: 'عيادة الأسنان', doctorName: '', theme: 'auto',
  rules: C.DEFAULT_RULES,
  design: { proportion: 'red70', customP: 0.70, wl: 0.80, spaceMode: 'preserve', symmetric: true, gingivaMode: 'keep', shape: 'square', shade: 'A1', material: 'إيماكس (ليثيوم ديسيليكات)' },
  modelSize: 384, confMin: 0.91, autoCrop: true, ipdMm: 63, useTTA: true, autoStraighten: true,
  pin: null, disclaimerAccepted: false
};
let S = null;
async function loadSettings() {
  const s = await DB.getKV('settings', null);
  S = Object.assign(C.clone(DEFAULT_SETTINGS), s || {});
  S.design = Object.assign(C.clone(DEFAULT_SETTINGS.design), S.design || {});
  // merge newly added default rules by id (keep doctor's edits)
  const have = new Set(S.rules.map((r) => r.id));
  for (const r of C.DEFAULT_RULES) if (!have.has(r.id)) S.rules.push(C.clone(r));
  applyTheme();
}
const saveSettings = () => DB.setKV('settings', S);
function applyTheme() { if (S.theme === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', S.theme); }

// ---------- crypto (backup + PIN) ----------
const b64 = {
  enc(buf) { let s = ''; const b = new Uint8Array(buf); for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); },
  dec(str) { const s = atob(str); const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; }
};
async function deriveKey(pass, salt, iter = 250000) {
  const base = await crypto.subtle.importKey('raw', new TextEncoder().encode(pass), 'PBKDF2', false, ['deriveKey', 'deriveBits']);
  return crypto.subtle.deriveKey({ name: 'PBKDF2', salt, iterations: iter, hash: 'SHA-256' }, base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
}
async function hashPin(pin, salt) {
  const base = await crypto.subtle.importKey('raw', new TextEncoder().encode(pin), 'PBKDF2', false, ['deriveBits']);
  const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', salt, iterations: 150000, hash: 'SHA-256' }, base, 256);
  return b64.enc(bits);
}

// ---------- image helpers ----------
async function blobToBitmap(blob, maxDim = 2400) {
  let bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const m = Math.max(bmp.width, bmp.height);
  if (m > maxDim) {
    const k = maxDim / m;
    const c = canvasOf(Math.round(bmp.width * k), Math.round(bmp.height * k));
    c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
    bmp.close?.(); bmp = await createImageBitmap(c);
  }
  return bmp;
}
function canvasOf(w, h) { const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c; }
function rotatedCanvas(src, totalDeg) {
  const a = totalDeg * Math.PI / 180, W = src.width, H = src.height;
  const cs = Math.abs(Math.cos(a)), sn = Math.abs(Math.sin(a));
  const c = canvasOf(W * cs + H * sn, W * sn + H * cs);
  const x = c.getContext('2d'); x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
  x.imageSmoothingQuality = 'high';
  x.translate(c.width / 2, c.height / 2); x.rotate(a); x.drawImage(src, -W / 2, -H / 2);
  return c;
}
function cropCanvas(src, r, maxDim = 1600) {
  const sx = r.x * src.width, sy = r.y * src.height, sw = Math.max(8, r.w * src.width), sh = Math.max(8, r.h * src.height);
  const k = Math.min(1, maxDim / Math.max(sw, sh));
  const c = canvasOf(sw * k, sh * k);
  const x = c.getContext('2d'); x.imageSmoothingQuality = 'high'; x.drawImage(src, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return c;
}
const canvasToBlob = (c, type = 'image/jpeg', q = 0.92) => new Promise((r) => c.toBlob(r, type, q));
function bitmapToCanvas(bmp) { const c = canvasOf(bmp.width, bmp.height); c.getContext('2d').drawImage(bmp, 0, 0); return c; }
const imageCache = new Map();
async function getCanvasForBlob(id) {
  if (!id) return null;
  if (imageCache.has(id)) return imageCache.get(id);
  const b = await getBlob(id); if (!b) return null;
  const bmp = await blobToBitmap(b, 4000); const c = bitmapToCanvas(bmp); bmp.close?.();
  imageCache.set(id, c); return c;
}

// ---------- router & layout ----------
const app = () => $('#app');
let routeCleanup = null;
function go(hash) { if (location.hash === hash) render(); else location.hash = hash; }
window.addEventListener('hashchange', () => render());
async function render() {
  if (routeCleanup) { try { routeCleanup(); } catch (e) { /* ignore */ } routeCleanup = null; }
  const parts = (location.hash.replace(/^#\/?/, '') || '').split('/').filter(Boolean);
  const root = app(); root.innerHTML = '';
  try {
    if (!parts.length) await viewHome(root);
    else if (parts[0] === 'patient') await viewPatient(root, parts[1]);
    else if (parts[0] === 'case') await viewCase(root, parts[1], parts[2] || 'photos');
    else if (parts[0] === 'settings') await viewSettings(root);
    else if (parts[0] === 'present') await viewPresent(root, parts[1]);
    else await viewHome(root);
  } catch (e) {
    console.error(e);
    root.append(h('main', null, h('div', { class: 'card' }, h('h2', null, 'حدث خطأ'), h('p', { class: 'muted' }, String(e && e.message || e)), h('a', { class: 'btn', href: '#/' }, 'العودة للرئيسية'))));
  }
  window.scrollTo(0, 0);
}
function topbar(title, back, extra) {
  return h('header', { class: 'topbar' },
    back ? h('a', { class: 'iconbtn', href: back, 'aria-label': 'رجوع', style: { textDecoration: 'none', display: 'grid', placeItems: 'center' } }, '→') : h('span', { class: 'brand' }, '◐'),
    h('h1', null, title), extra || null,
    h('a', { class: 'iconbtn', href: '#/settings', 'aria-label': 'الإعدادات', title: 'الإعدادات', style: { textDecoration: 'none', display: 'grid', placeItems: 'center' } }, '⚙'));
}

// ---------- home ----------
async function viewHome(root) {
  const patients = (await DB.all('patients')).sort((a, b) => (b.updated || 0) - (a.updated || 0));
  const cases = await DB.all('cases');
  const count = {}; for (const c of cases) count[c.patientId] = (count[c.patientId] || 0) + 1;
  const list = h('ul', { class: 'list' });
  const search = h('input', { type: 'search', placeholder: 'بحث بالاسم أو رقم الملف', 'aria-label': 'بحث' });
  const draw = () => {
    const q = search.value.trim();
    list.innerHTML = '';
    const shown = patients.filter((p) => !q || (p.name || '').includes(q) || (p.fileNo || '').includes(q));
    if (!shown.length) list.append(h('li', { class: 'empty' }, patients.length ? 'لا نتائج' : 'لا يوجد مرضى بعد. أضف أول مريض.'));
    for (const p of shown) list.append(h('li', null, h('a', { class: 'item', href: '#/patient/' + p.id },
      h('div', { class: 'avatar' }, (p.name || '؟').trim().charAt(0)),
      h('div', { class: 'grow' }, h('div', null, h('b', null, p.name || 'بلا اسم')), h('div', { class: 'muted small' }, [p.fileNo ? 'ملف ' + p.fileNo : null, (count[p.id] || 0) + ' حالة', fmtDate(p.updated)].filter(Boolean).join(' · '))),
      h('span', { class: 'muted' }, '‹'))));
  };
  search.addEventListener('input', draw); draw();
  root.append(topbar(S.clinicName || 'تصميم الابتسامة الرقمي'), h('main', null,
    h('div', { class: 'card' }, h('div', { class: 'row between' }, h('h2', { style: { margin: 0 } }, 'المرضى'), h('button', { class: 'btn primary', onclick: () => editPatient() }, '+ مريض جديد')),
      h('div', { style: { marginTop: '10px' } }, search), list),
    storageCard()));
}
function storageCard() {
  const el = h('div', { class: 'card muted small' }, 'جارٍ فحص التخزين…');
  (async () => {
    let persisted = false, est = null;
    try { persisted = await navigator.storage?.persisted?.(); est = await navigator.storage?.estimate?.(); } catch (e) { /* ignore */ }
    const last = await DB.getKV('lastBackup', null);
    el.innerHTML = '';
    el.append(...[h('div', null, 'البيانات محفوظة على هذا الجهاز فقط. ', persisted ? '(تخزين دائم مفعّل)' : '(التخزين الدائم غير مفعّل — قد يحذفه المتصفح عند امتلاء الذاكرة)'),
      est ? h('div', null, `المستخدم: ${(est.usage / 1048576).toFixed(1)} ميغابايت`) : null,
      h('div', { style: { color: !last || Date.now() - last > 7 * 864e5 ? 'var(--warn)' : '' } }, last ? 'آخر نسخة احتياطية: ' + fmtDateTime(last) : 'لم تُؤخذ نسخة احتياطية بعد — من الإعدادات.')].filter(Boolean));
  })();
  return el;
}

// ---------- patient ----------
async function editPatient(p) {
  const isNew = !p; p = p ? { ...p } : { id: uid(), created: Date.now() };
  const f = (k, label, type = 'text') => { const i = h('input', { type, value: p[k] || '' }); i.addEventListener('input', () => p[k] = i.value); return h('label', { class: 'field' }, h('span', null, label), i); };
  const sex = h('select', null, h('option', { value: '' }, '—'), h('option', { value: 'ذكر', selected: p.sex === 'ذكر' }, 'ذكر'), h('option', { value: 'أنثى', selected: p.sex === 'أنثى' }, 'أنثى'));
  sex.addEventListener('change', () => p.sex = sex.value);
  const notes = h('textarea', null, p.notes || ''); notes.addEventListener('input', () => p.notes = notes.value);
  const body = h('div', null, f('name', 'الاسم'), h('div', { class: 'grid2' }, f('fileNo', 'رقم الملف'), f('age', 'العمر', 'number')), h('div', { class: 'grid2' }, h('label', { class: 'field' }, h('span', null, 'الجنس'), sex), f('phone', 'الجوال', 'tel')), h('label', { class: 'field' }, h('span', null, 'ملاحظات طبية'), notes));
  const ok = await modal(isNew ? 'مريض جديد' : 'تعديل بيانات المريض', body, [{ label: 'إلغاء', value: false }, { label: 'حفظ', value: true, primary: true }]);
  if (!ok) return;
  if (!(p.name || '').trim()) { toast('الاسم مطلوب'); return; }
  p.updated = Date.now(); await DB.put('patients', p);
  go('#/patient/' + p.id);
}
async function viewPatient(root, id) {
  const p = await DB.get('patients', id); if (!p) return go('#/');
  const cases = (await DB.byIndex('cases', 'patientId', id)).sort((a, b) => b.created - a.created);
  const list = h('ul', { class: 'list' });
  if (!cases.length) list.append(h('li', { class: 'empty' }, 'لا توجد حالات. أنشئ حالة لبدء التصوير والتحليل.'));
  for (const c of cases) list.append(h('li', null, h('a', { class: 'item', href: '#/case/' + c.id },
    h('div', { class: 'grow' }, h('b', null, c.title || 'حالة'), h('div', { class: 'muted small' }, fmtDate(c.created))),
    h('span', { class: 'badge ' + (c.approval ? 'ok' : '') }, c.approval ? 'معتمدة' : 'مسودة'))));
  root.append(topbar(p.name, '#/'), h('main', null,
    h('div', { class: 'card' }, h('div', { class: 'row between' }, h('h2', { style: { margin: 0 } }, p.name), h('button', { class: 'btn sm', onclick: () => editPatient(p) }, 'تعديل')),
      h('div', { class: 'kv', style: { marginTop: '8px' } }, h('b', null, 'رقم الملف'), h('span', null, p.fileNo || '—'), h('b', null, 'العمر'), h('span', null, p.age || '—'), h('b', null, 'الجنس'), h('span', null, p.sex || '—'), h('b', null, 'الجوال'), h('span', null, p.phone || '—')),
      p.notes ? h('p', { class: 'muted' }, p.notes) : null),
    h('div', { class: 'card' }, h('div', { class: 'row between' }, h('h2', { style: { margin: 0 } }, 'الحالات'), h('button', { class: 'btn primary', onclick: () => newCase(p) }, '+ حالة جديدة')), list),
    h('div', { class: 'card' }, h('button', { class: 'btn danger', onclick: async () => {
      if (!await confirmBox('حذف المريض', 'سيُحذف المريض وكل حالاته وصوره من هذا الجهاز نهائياً. هل أنت متأكد؟', 'حذف', true)) return;
      for (const c of cases) await deleteCaseData(c);
      await DB.del('patients', id); toast('حُذف المريض'); go('#/');
    } }, 'حذف المريض'))));
}
async function newCase(p) {
  const c = {
    id: uid(), patientId: p.id, created: Date.now(), updated: Date.now(), title: 'تصميم ابتسامة ' + fmtDate(Date.now()),
    consent: { photo: false, training: false, at: null }, images: {}, crop: { q: 0, angle: 0, rect: { x: 0, y: 0, w: 1, h: 1 } }, faceQ: 0,
    intra: { teeth: null, source: null, det: null, maskPng: null }, calib: { method: 'none', ruler: null, tooth: { id: '11', mm: '' } },
    face: { points: {} }, design: Object.assign(C.clone(S.design), { midline: 'current', treat: Object.fromEntries(C.TEETH.map((t) => [t, 'veneer'])), override: {}, notes: '', plan: '', labNotes: '' }),
    approval: null
  };
  await DB.put('cases', c); p.updated = Date.now(); await DB.put('patients', p);
  go('#/case/' + c.id + '/photos');
}
async function deleteCaseData(c) {
  for (const id of Object.values(c.images || {})) if (id) { await DB.del('blobs', id); imageCache.delete(id); }
  await DB.del('cases', c.id);
}

// ---------- lock & disclaimer ----------
let lastHidden = 0;
async function showLock() {
  if (!S.pin) return;
  await new Promise((resolve) => {
    const inp = h('input', { type: 'password', inputmode: 'numeric', autocomplete: 'off', placeholder: 'رمز الدخول', style: { textAlign: 'center', fontSize: '22px' } });
    const msg = h('div', { class: 'muted small', style: { minHeight: '20px' } });
    const tryIt = async () => {
      const hsh = await hashPin(inp.value, b64.dec(S.pin.salt));
      if (hsh === S.pin.hash) { el.remove(); resolve(); } else { msg.textContent = 'رمز غير صحيح'; inp.value = ''; }
    };
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') tryIt(); });
    const el = h('div', { class: 'lock' }, h('div', { class: 'box' }, h('div', { style: { fontSize: '40px' } }, '🔒'), h('h2', null, S.clinicName), inp, msg, h('button', { class: 'btn primary', style: { width: '100%', marginTop: '8px' }, onclick: tryIt }, 'دخول')));
    document.body.append(el); setTimeout(() => inp.focus(), 50);
  });
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) lastHidden = Date.now();
  else if (S && S.pin && lastHidden && Date.now() - lastHidden > 5 * 60 * 1000 && !$('.lock')) showLock();
});
async function showDisclaimer() {
  if (S.disclaimerAccepted) return;
  await modal('قبل البدء', h('div', null,
    h('p', null, 'هذا النظام أداة مساعدة للتحليل والتصميم، وليس جهاز تشخيص. كل قياس واقتراح يراجعه الطبيب ويعدّله ويعتمده قبل إرساله للمعمل أو عرضه على المريض.'),
    h('p', null, 'كشف الأسنان التلقائي قد يخطئ، خصوصاً مع الأسنان المصبوغة أو المتآكلة أو المزدحمة؛ لذلك كل الحدود قابلة للتعديل يدوياً، والقياس بالملم يحتاج معايرة.'),
    h('p', null, 'البيانات والصور تبقى على هذا الجهاز فقط. أنت مسؤول عن موافقة المرضى وعن النسخ الاحتياطي المشفّر بانتظام.')),
  [{ label: 'فهمت وأوافق', value: true, primary: true }]);
  S.disclaimerAccepted = true; await saveSettings();
}
