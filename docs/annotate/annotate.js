/* Tooth annotation tool: the current model pre-labels the six upper front teeth, a person corrects the outlines,
   and the result is exported in the training format (images/<id>.jpg + masks/<id>.png with values 0-6).
   Everything runs in the browser; photos never leave the device. */
'use strict';
const C = window.DSDCore, SM = window.DSDSmile, TEETH = C.TEETH;
const COLORS = { '13': '#ff9f43', '12': '#feca57', '11': '#48dbfb', '21': '#1dd1a1', '22': '#ff6b6b', '23': '#c56cf0' };
const STATUS = [['present', 'موجود'], ['broken', 'مكسور'], ['missing', 'مفقود'], ['hidden', 'غير ظاهر']];
const S = 384, MAXW = 1024;
const $ = (id) => document.getElementById(id);
let session = null, face = null;
const items = [];
let cur = -1, sel = '11', drag = null, drawing = null, zoom = 1, pan = { x: 0, y: 0 }, panning = null, pressTimer = null;

// ------------------------------------------------------------------ models
async function loadModels() {
  if (typeof ort === 'undefined') return 'review';   // stand-alone review file: drafts only, no model needed
  ort.env.wasm.wasmPaths = new URL('../vendor/', location.href).href;
  ort.env.wasm.numThreads = 1;
  const buf = await (await fetch('../teeth_unet_v2.onnx')).arrayBuffer();
  session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
  try {
    if (window.__ANN_TEST_FACE) { face = { detect: window.__ANN_TEST_FACE }; return; }
    const base = new URL('../vendor/mediapipe', location.href).href;
    const vision = await import(base + '/vision_bundle.mjs');
    const fs = await vision.FilesetResolver.forVisionTasks(base + '/wasm');
    const mb = new Uint8Array(await (await fetch('../vendor/face_landmarker.task')).arrayBuffer());
    const mk = (d) => vision.FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetBuffer: mb, delegate: d }, runningMode: 'IMAGE', numFaces: 1 });
    let fl; try { fl = await mk('GPU'); } catch (e) { fl = await mk('CPU'); }
    face = { detect: (c) => fl.detect(c).faceLandmarks || [] };
  } catch (e) { console.warn('face detector unavailable', e); face = null; }
}

const canvasOf = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
function rot(cv, k) {
  const c = k % 2 ? canvasOf(cv.height, cv.width) : canvasOf(cv.width, cv.height), x = c.getContext('2d');
  x.translate(c.width / 2, c.height / 2); x.rotate(k * Math.PI / 2); x.drawImage(cv, -cv.width / 2, -cv.height / 2);
  return c;
}

async function segment(crop) {
  const c = canvasOf(S, S), x = c.getContext('2d', { willReadFrequently: true });
  x.imageSmoothingQuality = 'high'; x.drawImage(crop, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data, n = S * S, t = new Float32Array(3 * n), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) t[k * n + i] = (d[i * 4 + k] / 255 - mean[k]) / std[k];
  const out = (await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, S, S]) }))[session.outputNames[0]].data;
  const labels = new Uint8Array(n), conf = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0, m = out[i];
    for (let c2 = 1; c2 < 7; c2++) { const v = out[c2 * n + i]; if (v > m) { m = v; b = c2; } }
    let s = 0; for (let c2 = 0; c2 < 7; c2++) s += Math.exp(out[c2 * n + i] - m);
    labels[i] = b; conf[i] = 1 / s;
  }
  return { labels, conf };
}

// outline of one class: rays from the centroid to the farthest pixel of the class (teeth are close to star-shaped)
function rayPolygon(lab, c, n = 40) {
  let sx = 0, sy = 0, k = 0;
  for (let i = 0; i < S * S; i++) if (lab[i] === c) { sx += i % S; sy += (i / S) | 0; k++; }
  if (k < 20) return null;
  const cx = sx / k, cy = sy / k, pts = [];
  for (let a = 0; a < n; a++) {
    const th = 2 * Math.PI * a / n, dx = Math.cos(th), dy = Math.sin(th);
    let last = 0;
    for (let r = 0; r < S; r += 0.5) {
      const x = Math.round(cx + dx * r), y = Math.round(cy + dy * r);
      if (x < 0 || y < 0 || x >= S || y >= S) break;
      if (lab[y * S + x] === c) last = r; else if (r - last > 6) break;
    }
    pts.push({ x: cx + 0.5 + dx * (last + 0.5), y: cy + 0.5 + dy * (last + 0.5) });
  }
  return pts;
}

// fewer, evenly useful handles: Ramer-Douglas-Peucker on the closed outline
function simplify(P, eps) {
  if (P.length < 8) return P;
  const rdp = (pts) => {
    if (pts.length < 3) return pts;
    const a = pts[0], b = pts[pts.length - 1], dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    let k = -1, dmax = 0;
    for (let i = 1; i < pts.length - 1; i++) { const d = Math.abs(dy * pts[i].x - dx * pts[i].y + b.x * a.y - b.y * a.x) / L; if (d > dmax) { dmax = d; k = i; } }
    return dmax > eps ? rdp(pts.slice(0, k + 1)).slice(0, -1).concat(rdp(pts.slice(k))) : [a, b];
  };
  const h = Math.floor(P.length / 2), A = rdp(P.slice(0, h + 1)), B = rdp(P.slice(h).concat([P[0]]));
  return A.slice(0, -1).concat(B.slice(0, -1));
}

// ------------------------------------------------------------------ one photo
function storeKey(it) { return 'ann:' + it.file.name + ':' + it.file.size; }
function save(it) {
  try { localStorage.setItem(storeKey(it), JSON.stringify({ polys: it.polys, status: it.status, reviewed: it.reviewed, who: it.who, at: it.at })); } catch (e) { /* storage full or blocked */ }
}

async function processItem(it) {
  if (it.draft) return processDraft(it);
  const bmp = await createImageBitmap(it.file, { imageOrientation: 'from-image' });
  let cv = canvasOf(bmp.width, bmp.height); cv.getContext('2d').drawImage(bmp, 0, 0);
  let P = null;
  if (face) {
    let f = face.detect(cv);
    if (!f.length) for (const k of [1, 3, 2]) { const r = rot(cv, k); f = face.detect(r); if (f.length) { cv = r; break; } }
    if (f.length) {
      P = SM.toPx(f[0], cv.width, cv.height);
      const fix = SM.uprightTurns(P);            // eyes must be above the mouth
      if (fix) { cv = rot(cv, fix); const f2 = face.detect(cv); P = f2.length ? SM.toPx(f2[0], cv.width, cv.height) : null; }
    }
  }
  let crop, lips;
  if (P) {
    const cr = SM.mouthCrop(P, { maxOut: MAXW });
    crop = canvasOf(cr.outW, cr.outH);
    const x = crop.getContext('2d'), Mi = cr.Minv;
    x.imageSmoothingQuality = 'high'; x.setTransform(Mi[0], Mi[3], Mi[1], Mi[4], Mi[2], Mi[5]); x.drawImage(cv, 0, 0);
    const toC = (p) => SM.applyAffine(cr.Minv, p);
    lips = { upper: SM.INNER_UPPER.map((i) => toC(P[i])), lower: SM.INNER_LOWER.map((i) => toC(P[i])), poly: SM.innerLipPolygon(P).map(toC) };
    it.mode = 'face'; it.M = cr.M;
  } else {                                        // close-up / retracted photo: the photo itself is the crop
    const k = Math.min(1, MAXW / cv.width);
    crop = canvasOf(Math.round(cv.width * k), Math.round(cv.height * k));
    crop.getContext('2d').drawImage(cv, 0, 0, crop.width, crop.height);
    const w = crop.width, h = crop.height;
    lips = { upper: [{ x: 0, y: 0 }, { x: w, y: 0 }], lower: [{ x: w, y: h }, { x: 0, y: h }], poly: [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }, { x: 0, y: h }] };
    it.mode = 'close-up'; it.M = null;
  }
  it.crop = crop; it.lips = lips;
  let saved = null; try { saved = JSON.parse(localStorage.getItem(storeKey(it)) || 'null'); } catch (e) { saved = null; }
  if (saved && saved.polys) { Object.assign(it, { polys: saved.polys, status: saved.status, reviewed: !!saved.reviewed, who: saved.who, at: saved.at }); return; }
  const { labels, conf } = await segment(crop);
  const rgba = crop.getContext('2d').getImageData(0, 0, crop.width, crop.height).data;
  const pick = SM.selectUpperTeeth(labels, conf, S, crop.width, crop.height, lips, { rgba });
  it.polys = {}; it.status = {};
  for (const t of TEETH) {
    const poly = rayPolygon(pick.clean, C.CLASS_OF[t]);
    if (poly) { it.polys[t] = simplify(poly.map((p) => ({ x: p.x * crop.width / S, y: p.y * crop.height / S })), Math.max(1.2, crop.width / 500)); it.status[t] = 'present'; }
    else it.status[t] = (t === '13' || t === '23') ? 'hidden' : 'missing';
  }
  it.reviewed = false;
}

// a draft from training/bootstrap: the image is already the mouth crop and the outlines are in crop pixels
async function processDraft(it) {
  const bmp = await createImageBitmap(it.file);
  const crop = canvasOf(bmp.width, bmp.height); crop.getContext('2d').drawImage(bmp, 0, 0);
  const d = it.draft;
  it.crop = crop; it.mode = d.mode || 'face'; it.M = d.crop_to_image || null;
  it.lips = d.lips && d.lips.poly ? d.lips : { poly: [{ x: 0, y: 0 }, { x: crop.width, y: 0 }, { x: crop.width, y: crop.height }, { x: 0, y: crop.height }] };
  let saved = null; try { saved = JSON.parse(localStorage.getItem(storeKey(it)) || 'null'); } catch (e) { saved = null; }
  if (saved && saved.polys) { Object.assign(it, { polys: saved.polys, status: saved.status, reviewed: !!saved.reviewed, who: saved.who, at: saved.at }); return; }
  it.polys = JSON.parse(JSON.stringify(d.teeth || {}));
  it.status = Object.assign(Object.fromEntries(TEETH.map((t) => [t, it.polys[t] ? 'present' : 'missing'])), d.status || {});
  it.reviewed = false;
}

// minimal ZIP reader (stored and deflate entries) for draft packages
async function unzip(file) {
  const buf = new Uint8Array(await file.arrayBuffer()), dv = new DataView(buf.buffer);
  let e = buf.length - 22; while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new Error('ليس ملف ZIP صالحاً');
  const count = dv.getUint16(e + 10, true); let p = dv.getUint32(e + 16, true);
  const dec = new TextDecoder(), out = {};
  for (let k = 0; k < count; k++) {
    const method = dv.getUint16(p + 10, true), csize = dv.getUint32(p + 20, true), nlen = dv.getUint16(p + 28, true),
      xlen = dv.getUint16(p + 30, true), clen = dv.getUint16(p + 32, true), off = dv.getUint32(p + 42, true);
    const name = dec.decode(buf.subarray(p + 46, p + 46 + nlen));
    const lstart = off + 30 + dv.getUint16(off + 26, true) + dv.getUint16(off + 28, true);
    const raw = buf.subarray(lstart, lstart + csize);
    if (!name.endsWith('/')) {
      if (method === 0) out[name] = raw;
      else if (method === 8) out[name] = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer());
    }
    p += 46 + nlen + xlen + clen;
  }
  return out;
}
async function addDraftZip(file) {
  const files = await unzip(file), dec = new TextDecoder();
  let order = Object.keys(files).filter((n) => /^images\/.+\.jpg$/.test(n)).map((n) => n.slice(7, -4));
  if (files['manifest.csv']) {                          // most uncertain drafts first
    const ids = dec.decode(files['manifest.csv']).replace(/^\ufeff/, '').split('\n').slice(1).map((l) => l.split(',')[0]).filter(Boolean);
    order = ids.filter((id) => order.includes(id)).concat(order.filter((id) => !ids.includes(id)));
  }
  const list = [];
  for (const id of order) {
    const m = files[`meta/${id}.json`]; if (!m) continue;
    list.push({ file: new File([files[`images/${id}.jpg`]], `${id}.jpg`, { type: 'image/jpeg' }), crop: null, draft: JSON.parse(dec.decode(m)) });
  }
  await addItems(list);
}

// ------------------------------------------------------------------ list + panel
function renderList() {
  const L = $('list'); L.innerHTML = '';
  items.forEach((it, i) => {
    const d = document.createElement('div');
    d.className = i === cur ? 'cur' : '';
    const tag = it.error ? '<span class="tag err">خطأ</span>' : !it.crop ? '<span class="tag">جارٍ…</span>' : it.reviewed ? '<span class="tag ok">✓</span>' : '<span class="tag">للمراجعة</span>';
    d.innerHTML = `<span>${escapeHtml(it.file.name)}</span>${tag}`;
    d.onclick = () => openItem(i);
    L.appendChild(d);
  });
  const n = items.filter((it) => it.reviewed).length;
  $('export').disabled = !n;
  $('status').textContent = items.length ? `روجعت ${n} من ${items.length}` : 'جاهز — افتح صوراً';
}
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

function renderPanel() {
  const it = items[cur], box = $('teeth'); box.innerHTML = '';
  for (const t of TEETH) {
    const row = document.createElement('div'); row.className = 'tooth' + (t === sel ? ' sel' : '');
    const b = document.createElement('b'); b.textContent = t; b.style.background = COLORS[t]; b.onclick = () => { sel = t; renderPanel(); draw(); };
    const s = document.createElement('select');
    const undecided = it && it.status && !STATUS.some(([v]) => v === it.status[t]);   // e.g. 'unsure' from a draft
    if (undecided) { const o = document.createElement('option'); o.value = it.status[t]; o.textContent = '— حدّد —'; s.appendChild(o); s.classList.add('need'); }
    for (const [v, l] of STATUS) { const o = document.createElement('option'); o.value = v; o.textContent = l; s.appendChild(o); }
    s.value = it && it.status ? it.status[t] : 'present'; s.disabled = !(it && it.status);
    s.onchange = () => { pushUndo(it); it.status[t] = s.value; if (s.value === 'missing' || s.value === 'hidden') delete it.polys[t]; it.reviewed = false; save(it); renderPanel(); draw(); renderList(); };
    row.append(b, s); box.appendChild(row);
  }
}

async function openItem(i) {
  cur = i; zoom = 1; pan = { x: 0, y: 0 }; drawing = null;
  const it = items[i];
  renderList(); renderPanel();
  $('empty').hidden = !!it.crop; $('cv').hidden = !it.crop;
  if (!it.crop) $('empty').textContent = it.error ? 'تعذّرت معالجة الصورة: ' + it.error : 'جارٍ التحليل…';
  draw();
}

// ------------------------------------------------------------------ canvas
function view() {
  const it = items[cur], st = $('stage'), cv = $('cv');
  const W = st.clientWidth, H = st.clientHeight, dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.width = W + 'px'; cv.style.height = H + 'px'; }
  const s = Math.min(W / it.crop.width, H / it.crop.height) * zoom;
  return { s, ox: (W - it.crop.width * s) / 2 + pan.x, oy: (H - it.crop.height * s) / 2 + pan.y, dpr };
}
function toCrop(e) { const r = $('cv').getBoundingClientRect(), v = view(); return { x: (e.clientX - r.left - v.ox) / v.s, y: (e.clientY - r.top - v.oy) / v.s }; }

function draw() {
  const it = items[cur]; if (!it || !it.crop) return;
  const cv = $('cv'), x = cv.getContext('2d'), v = view();
  x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, cv.width, cv.height);
  x.setTransform(v.s * v.dpr, 0, 0, v.s * v.dpr, v.ox * v.dpr, v.oy * v.dpr);
  x.drawImage(it.crop, 0, 0);
  const lw = 1.5 / v.s;
  if (it.mode === 'face') { x.strokeStyle = 'rgba(80,140,255,.7)'; x.lineWidth = lw; x.beginPath(); it.lips.poly.forEach((p, k) => k ? x.lineTo(p.x, p.y) : x.moveTo(p.x, p.y)); x.closePath(); x.stroke(); }
  for (const t of TEETH) {
    const P = it.polys[t]; if (!P) continue;
    x.beginPath(); P.forEach((p, k) => k ? x.lineTo(p.x, p.y) : x.moveTo(p.x, p.y)); x.closePath();
    x.fillStyle = COLORS[t] + (t === sel ? '55' : '33'); x.fill();
    x.strokeStyle = COLORS[t]; x.lineWidth = (t === sel ? 2.5 : 1.5) / v.s; x.stroke();
    const c = centroid(P); x.fillStyle = '#000'; x.font = `${13 / v.s}px sans-serif`; x.textAlign = 'center'; x.fillText(t, c.x, c.y);
    if (t === sel) for (const p of P) { x.fillStyle = '#fff'; x.strokeStyle = COLORS[t]; x.lineWidth = 2 / v.s; x.beginPath(); x.arc(p.x, p.y, 5 / v.s, 0, 7); x.fill(); x.stroke(); }
  }
  if (drawing) {
    x.strokeStyle = COLORS[sel]; x.lineWidth = 2 / v.s; x.beginPath(); drawing.forEach((p, k) => k ? x.lineTo(p.x, p.y) : x.moveTo(p.x, p.y)); x.stroke();
    for (const p of drawing) { x.fillStyle = COLORS[sel]; x.beginPath(); x.arc(p.x, p.y, 4 / v.s, 0, 7); x.fill(); }
  }
}
const centroid = (P) => ({ x: P.reduce((s, p) => s + p.x, 0) / P.length, y: P.reduce((s, p) => s + p.y, 0) / P.length });
function nearestVertex(P, q, tolPx) { const v = view(); let best = -1, bd = tolPx / v.s; (P || []).forEach((p, k) => { const d = Math.hypot(p.x - q.x, p.y - q.y); if (d < bd) { bd = d; best = k; } }); return best; }
function nearestEdge(P, q, tolPx) {
  const v = view(); let best = -1, bd = tolPx / v.s;
  for (let k = 0; k < P.length; k++) {
    const a = P[k], b = P[(k + 1) % P.length], dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy || 1;
    const u = Math.max(0, Math.min(1, ((q.x - a.x) * dx + (q.y - a.y) * dy) / L)), d = Math.hypot(a.x + u * dx - q.x, a.y + u * dy - q.y);
    if (d < bd) { bd = d; best = k; }
  }
  return best;
}

const undoStack = new WeakMap();
function pushUndo(it) { const s = undoStack.get(it) || []; s.push(JSON.stringify({ polys: it.polys, status: it.status })); if (s.length > 60) s.shift(); undoStack.set(it, s); }
function undo() { const it = items[cur]; const s = it && undoStack.get(it); if (!s || !s.length) return; Object.assign(it, JSON.parse(s.pop())); it.reviewed = false; save(it); renderPanel(); renderList(); draw(); }

function onDown(e) {
  const it = items[cur]; if (!it || !it.crop) return;
  const q = toCrop(e);
  if (drawing) {
    if (drawing.length >= 3 && Math.hypot(drawing[0].x - q.x, drawing[0].y - q.y) < 12 / view().s) {
      pushUndo(it); it.polys[sel] = drawing; it.status[sel] = it.status[sel] === 'broken' ? 'broken' : 'present'; drawing = null; it.reviewed = false; save(it); renderPanel(); renderList();
    } else drawing.push(q);
    draw(); return;
  }
  const P = it.polys[sel], k = nearestVertex(P, q, 14);
  if (k >= 0 && e.button === 2) { deleteVertex(it, k); return; }
  if (k >= 0) {
    pushUndo(it); drag = { k }; $('cv').setPointerCapture(e.pointerId);
    if (e.pointerType === 'touch') pressTimer = setTimeout(() => { drag = null; deleteVertex(it, k); }, 650);
    return;
  }
  for (const t of TEETH) if (it.polys[t] && SM.pointInPoly(q.x, q.y, it.polys[t])) { sel = t; renderPanel(); draw(); return; }
  panning = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }; $('cv').setPointerCapture(e.pointerId);
}
function onMove(e) {
  const it = items[cur]; if (!it) return;
  if (drag) { clearTimeout(pressTimer); const q = toCrop(e); it.polys[sel][drag.k] = { x: Math.max(0, Math.min(it.crop.width, q.x)), y: Math.max(0, Math.min(it.crop.height, q.y)) }; draw(); }
  else if (panning) { pan.x = panning.px + e.clientX - panning.x; pan.y = panning.py + e.clientY - panning.y; draw(); }
}
function onUp() { clearTimeout(pressTimer); if (drag) { const it = items[cur]; it.reviewed = false; save(it); renderList(); } drag = null; panning = null; }
function deleteVertex(it, k) { const P = it.polys[sel]; if (!P || P.length <= 3) return; pushUndo(it); P.splice(k, 1); it.reviewed = false; save(it); draw(); }
function onDbl(e) {
  const it = items[cur]; if (!it || drawing) return;
  const P = it.polys[sel]; if (!P) return;
  const q = toCrop(e), k = nearestEdge(P, q, 10);
  if (k >= 0) { pushUndo(it); P.splice(k + 1, 0, q); it.reviewed = false; save(it); draw(); }
}

// ------------------------------------------------------------------ export
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(b) { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
function zip(files) {                    // stored (no compression); UTF-8 names
  const enc = new TextEncoder(), parts = [], central = []; let off = 0;
  const d = new Date(), dt = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate(), tm = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  for (const f of files) {
    const name = enc.encode(f.name), crc = crc32(f.data), n = f.data.length;
    const h = new DataView(new ArrayBuffer(30));
    [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [10, tm, 2], [12, dt, 2], [14, crc, 4], [18, n, 4], [22, n, 4], [26, name.length, 2], [28, 0, 2]].forEach(([o, v, s]) => s === 4 ? h.setUint32(o, v, true) : h.setUint16(o, v, true));
    parts.push(new Uint8Array(h.buffer), name, f.data);
    const c = new DataView(new ArrayBuffer(46));
    [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [10, 0, 2], [12, tm, 2], [14, dt, 2], [16, crc, 4], [20, n, 4], [24, n, 4], [28, name.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2], [36, 0, 2], [38, 0, 4], [42, off, 4]].forEach(([o, v, s]) => s === 4 ? c.setUint32(o, v, true) : c.setUint16(o, v, true));
    central.push(new Uint8Array(c.buffer), name);
    off += 30 + name.length + n;
  }
  const csize = central.reduce((s, a) => s + a.length, 0), e = new DataView(new ArrayBuffer(22));
  [[0, 0x06054b50, 4], [4, 0, 2], [6, 0, 2], [8, files.length, 2], [10, files.length, 2], [12, csize, 4], [16, off, 4], [20, 0, 2]].forEach(([o, v, s]) => s === 4 ? e.setUint32(o, v, true) : e.setUint16(o, v, true));
  return new Blob([...parts, ...central, new Uint8Array(e.buffer)], { type: 'application/zip' });
}
const blobBytes = async (b) => new Uint8Array(await b.arrayBuffer());
const toBlob = (c, type, q) => new Promise((r) => c.toBlob(r, type, q));

function maskOf(it) {
  const w = it.crop.width, h = it.crop.height, m = new Uint8Array(w * h);
  for (const t of TEETH) {
    if (!it.polys[t] || !(it.status[t] === 'present' || it.status[t] === 'broken')) continue;
    const r = SM.rasterPoly(it.polys[t], w, h), c = C.CLASS_OF[t];
    for (let i = 0; i < m.length; i++) if (r[i]) m[i] = c;
  }
  return m;
}
async function exportZip() {
  const files = [], rows = ['id,mode,width,height,teeth,status,reviewer,reviewed_at'];
  const enc = new TextEncoder();
  for (const it of items.filter((x) => x.reviewed)) {
    const id = it.file.name.replace(/\.[^.]+$/, '').replace(/[^\w\-]+/g, '_');
    const w = it.crop.width, h = it.crop.height, m = maskOf(it);
    const mc = canvasOf(w, h), mx = mc.getContext('2d'), img = mx.createImageData(w, h);
    for (let i = 0; i < m.length; i++) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = m[i]; img.data[i * 4 + 3] = 255; }
    mx.putImageData(img, 0, 0);
    files.push({ name: `images/${id}.jpg`, data: await blobBytes(await toBlob(it.crop, 'image/jpeg', 0.95)) });
    files.push({ name: `masks/${id}.png`, data: await blobBytes(await toBlob(mc, 'image/png')) });
    const meta = { id, source_file: it.file.name, mode: it.mode, width: w, height: h, crop_to_image: it.M, lips: it.mode === 'face' ? it.lips : null,
      teeth: it.polys, status: it.status, reviewer: it.who || '', reviewed_at: it.at,
      source: it.draft ? it.draft.source || null : null, regions: it.draft ? it.draft.regions || null : null, from_draft: !!it.draft, classes: { 0: 'background', 1: '13', 2: '12', 3: '11', 4: '21', 5: '22', 6: '23' } };
    files.push({ name: `meta/${id}.json`, data: enc.encode(JSON.stringify(meta)) });
    rows.push([id, it.mode, w, h, TEETH.filter((t) => it.polys[t] && (it.status[t] === 'present' || it.status[t] === 'broken')).length,
      TEETH.map((t) => t + ':' + it.status[t]).join(' '), (it.who || '').replace(/,/g, ' '), it.at].join(','));
  }
  files.push({ name: 'manifest.csv', data: enc.encode('﻿' + rows.join('\n') + '\n') });
  const a = document.createElement('a'); a.href = URL.createObjectURL(zip(files));
  a.download = `annotations_${new Date().toISOString().slice(0, 10)}.zip`; document.body.appendChild(a); a.click(); a.remove();
  window.__lastExport = files;            // for tests
}

// ------------------------------------------------------------------ wiring
async function addFiles(list) { return addItems(list.map((f) => ({ file: f, crop: null }))); }
async function addItems(list) {
  const start = items.length;
  for (const x of list) items.push(x);
  renderList();
  if (cur < 0) openItem(start);
  for (let i = start; i < items.length; i++) {
    try { await processItem(items[i]); } catch (e) { items[i].error = String(e && e.message || e); }
    renderList(); if (i === cur) openItem(i);
  }
}
function next() {
  const it = items[cur]; if (!it || !it.crop) return;
  // a tooth left undecided would teach the model nothing (training drops the photo): ask for it here
  const open = TEETH.filter((t) => !STATUS.some(([v]) => v === it.status[t]));
  if (open.length) { $('status').textContent = `حدّد حالة الأسنان: ${open.join('، ')} — موجود أو مكسور أو مفقود أو غير ظاهر`; $('status').classList.add('warn'); return; }
  $('status').classList.remove('warn');
  it.reviewed = true; it.who = $('who').value.trim(); it.at = new Date().toISOString(); save(it);
  const n = items.findIndex((x, k) => k > cur && !x.reviewed);
  openItem(n >= 0 ? n : cur);
}

$('open').onclick = () => $('files').click();
$('openzip').onclick = () => $('zipfile').click();
$('zipfile').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) addDraftZip(f).catch((err) => { $('status').textContent = 'تعذّر فتح الحزمة: ' + err.message; }); };
$('files').onchange = (e) => { addFiles([...e.target.files]); e.target.value = ''; };
$('export').onclick = exportZip;
$('done').onclick = next;
$('undo').onclick = undo;
$('redraw').onclick = () => { if (items[cur] && items[cur].crop) { drawing = []; draw(); } };
$('del').onclick = () => { const it = items[cur]; if (it && it.polys[sel]) { pushUndo(it); delete it.polys[sel]; it.status[sel] = 'missing'; it.reviewed = false; save(it); renderPanel(); renderList(); draw(); } };
$('zin').onclick = () => { zoom = Math.min(6, zoom * 1.4); draw(); };
$('zout').onclick = () => { zoom = Math.max(1, zoom / 1.4); if (zoom === 1) pan = { x: 0, y: 0 }; draw(); };
const cv = $('cv');
cv.addEventListener('pointerdown', onDown); cv.addEventListener('pointermove', onMove);
cv.addEventListener('pointerup', onUp); cv.addEventListener('pointercancel', onUp);
cv.addEventListener('dblclick', onDbl); cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('wheel', (e) => { e.preventDefault(); zoom = Math.max(1, Math.min(6, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); if (zoom === 1) pan = { x: 0, y: 0 }; draw(); }, { passive: false });
window.addEventListener('resize', draw);
$('who').value = localStorage.getItem('ann:who') || '';
$('who').onchange = () => { try { localStorage.setItem('ann:who', $('who').value); } catch (e) { /* ignore */ } };
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key >= '1' && e.key <= '6') { sel = TEETH[+e.key - 1]; renderPanel(); draw(); }
  else if (e.key === 'Enter') next();
  else if (e.key === 'Escape') { drawing = null; draw(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
});
renderPanel();
window.__annReady = loadModels().then((m) => { $('status').textContent = m === 'review' ? 'وضع المراجعة — اضغط «فتح مسودات (ZIP)»' : 'جاهز — افتح صوراً'; }).catch((e) => { $('status').textContent = 'تعذّر تحميل النموذج: ' + e.message; throw e; });
window.__ann = { items, addFiles, addDraftZip, next, exportZip, maskOf, get cur() { return cur; } };
