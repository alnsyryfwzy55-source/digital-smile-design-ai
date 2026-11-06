/* Engine: loads the AI tools, runs the automatic analysis pipeline and the design, talks to the background worker. */
'use strict';
const C = window.DSDCore, SM = window.DSDSmile, Sim = window.DSDSim;
const VERSIONS = { ort: '1.20.1', mediapipe: '0.10.14' };
const CDN = {
  ort: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSIONS.ort}/dist/`,
  mp: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSIONS.mediapipe}`,
  faceModel: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
};
// the model's fingerprint is part of its URL: the offline cache keeps model files forever, so a new model must be a new URL
const MODEL_URL = 'teeth_unet_v2.onnx?v=7f33b3a87605', MODEL_SIZE = 384, MAX_DIM = 2048;

async function exists(url) { try { const r = await fetch(url, { method: 'HEAD', cache: 'no-store' }); return r.ok && !(r.headers.get('content-type') || '').includes('text/html'); } catch (e) { return false; } }

const Engine = {
  worker: null, seq: 0, pending: new Map(), listeners: new Set(), info: { ep: null, warmMs: null, face: null, vendor: false, errors: [] },
  face: null, faceMode: null, readyP: null,

  start() {
    if (this.worker) return;
    this.worker = new Worker('worker.js');
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { this.listeners.forEach((f) => f(m)); return; }
      const p = this.pending.get(m.id); if (!p) return;
      this.pending.delete(m.id);
      m.type === 'error' ? p.reject(new Error(m.message)) : p.resolve(m);
    };
    this.worker.onerror = (e) => { for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker error')); this.pending.clear(); };
  },
  call(type, payload = {}, transfer = []) {
    this.start();
    const id = ++this.seq;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ type, id, ...payload }, transfer); });
  },
  onProgress(f) { this.listeners.add(f); return () => this.listeners.delete(f); },

  // Download and initialise everything once (cached for offline use afterwards). onStatus({key, state, got, total})
  prepare(onStatus = () => {}) {
    if (this.readyP) return this.readyP;
    this.readyP = (async () => {
      const vendor = await exists('vendor/ort.webgpu.min.js');
      this.info.vendor = vendor;
      const ortBase = vendor ? new URL('vendor/', location.href).href : CDN.ort;
      const off = this.onProgress((m) => onStatus({ key: m.label, state: 'loading', got: m.got, total: m.total }));
      const model = (async () => {
        if (window.__DSD_TEST_SEGMENT) { onStatus({ key: 'model', state: 'done' }); return; }
        onStatus({ key: 'model', state: 'loading' });
        let selfTest = null;
        try {
          const [b, expect] = await Promise.all([fetch('selftest.jpg').then((r) => r.blob()), fetch('selftest.json').then((r) => r.json())]);
          const bmp = await createImageBitmap(b), c = canvasOf(bmp.width, bmp.height); c.getContext('2d').drawImage(bmp, 0, 0);
          selfTest = { input: modelInput(c).buffer, expect };
        } catch (e) { console.warn('self-test asset missing', e); }
        const r = await this.call('init', { selfTest, ortUrl: ortBase + 'ort.webgpu.min.js', ortBase, modelUrl: new URL(MODEL_URL, location.href).href, size: MODEL_SIZE, webgpu: !/[?&]cpu\b/.test(location.search) });
        this.info.ep = r.ep; this.info.warmMs = r.warmMs; this.info.fallback = r.fallback; this.info.selfTest = r.selfTest; onStatus({ key: 'model', state: 'done' });
      })();
      const face = (async () => {
        onStatus({ key: 'face', state: 'loading' });
        await this.loadFace(vendor); onStatus({ key: 'face', state: 'done' });
      })();
      const lib = (async () => { await this.sendLibrary(); })();
      try { await Promise.all([model, face, lib]); } finally { off(); }
      return this.info;
    })();
    this.readyP.catch(() => { this.readyP = null; });
    return this.readyP;
  },

  async loadFace(vendor) {
    if (window.__DSD_TEST_FACE) { this.info.face = 'test'; return; }
    const base = vendor && await exists('vendor/mediapipe/vision_bundle.mjs') ? new URL('vendor/mediapipe', location.href).href : CDN.mp;
    const vision = await import(base + '/vision_bundle.mjs');
    const fileset = await vision.FilesetResolver.forVisionTasks(base + '/wasm');
    const modelUrl = await exists('vendor/face_landmarker.task') ? new URL('vendor/face_landmarker.task', location.href).href : CDN.faceModel;
    const buf = new Uint8Array(await fetchCachedMain(modelUrl));
    const opts = (delegate) => ({ baseOptions: { modelAssetBuffer: buf, delegate }, runningMode: 'IMAGE', numFaces: 2, minFaceDetectionConfidence: 0.5 });
    try { this.face = await vision.FaceLandmarker.createFromOptions(fileset, opts('GPU')); this.info.face = 'GPU'; }
    catch (e) { this.face = await vision.FaceLandmarker.createFromOptions(fileset, opts('CPU')); this.info.face = 'CPU'; }
    this.faceMode = 'IMAGE';
  },
  async faceMode_(mode) { if (this.face && this.faceMode !== mode) { await this.face.setOptions({ runningMode: mode }); this.faceMode = mode; } },
  async detectImage(canvas) {
    if (window.__DSD_TEST_FACE) return window.__DSD_TEST_FACE(canvas);
    await this.faceMode_('IMAGE');
    return this.face.detect(canvas).faceLandmarks || [];
  },
  async detectVideo(video, ts) {
    if (window.__DSD_TEST_FACE) return window.__DSD_TEST_FACE(video);
    await this.faceMode_('VIDEO');
    return this.face.detectForVideo(video, ts).faceLandmarks || [];
  },

  async sendLibrary() {
    const L = window.DSDLibrary; if (!L) return;
    const items = {}, transfer = [];
    for (const [t, url] of Object.entries(L.teeth)) {
      const bmp = await createImageBitmap(await (await fetch(url)).blob());
      const c = canvasOf(bmp.width, bmp.height), x = c.getContext('2d'); x.drawImage(bmp, 0, 0);
      const d = x.getImageData(0, 0, c.width, c.height);
      items[t] = { w: c.width, h: c.height, data: d.data.buffer }; transfer.push(d.data.buffer);
    }
    await this.call('library', { items }, transfer);
  },

  async segment(input) {
    if (window.__DSD_TEST_SEGMENT) return window.__DSD_TEST_SEGMENT(input, MODEL_SIZE);
    // flip-TTA doubles the time for ~+0.004 Dice: use it only when the GPU engine is running
    const r = await this.call('segment', { input, size: MODEL_SIZE, tta: this.info.ep === 'webgpu' }, [input.buffer]);
    return r;
  }
};

async function fetchCachedMain(url) {
  let cache = null;
  try { cache = await caches.open('dsd-models-v2'); const hit = await cache.match(url); if (hit) return await hit.arrayBuffer(); } catch (e) { /* ignore */ }
  const r = await fetch(url); if (!r.ok) throw new Error('تعذّر تحميل كاشف الوجه');
  const b = await r.arrayBuffer();
  try { if (cache) await cache.put(url, new Response(b.slice(0))); } catch (e) { /* quota */ }
  return b;
}

function canvasOf(w, h) { const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c; }
function rotateCanvas(src, turns) {
  if (!turns) return src;
  const c = turns % 2 ? canvasOf(src.height, src.width) : canvasOf(src.width, src.height), x = c.getContext('2d');
  x.translate(c.width / 2, c.height / 2); x.rotate(turns * Math.PI / 2); x.drawImage(src, -src.width / 2, -src.height / 2);
  return c;
}
async function blobToCanvas(blob) {
  let bmp;
  try { bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' }); }
  catch (e) {
    const heic = /hei[cf]/i.test(blob.type || '') || /\.hei[cf]$/i.test(blob.name || '');
    throw new AnalysisError('BAD_IMAGE', heic ? 'هذا المتصفح لا يقرأ صور HEIC. في إعدادات الكاميرا اختر «الأكثر توافقاً» (JPG)، أو التقط الصورة من داخل التطبيق.'
      : 'تعذّرت قراءة الصورة: الملف تالف أو ليس صورة. التقط صورة جديدة أو اختر صورة JPG.');
  }
  const k = Math.min(1, MAX_DIM / Math.max(bmp.width, bmp.height));
  const c = canvasOf(bmp.width * k, bmp.height * k), x = c.getContext('2d');
  x.imageSmoothingQuality = 'high'; x.drawImage(bmp, 0, 0, c.width, c.height); bmp.close?.();
  return c;
}
const toBlob = (c, q = 0.92) => new Promise((r) => c.toBlob(r, 'image/jpeg', q));

// ---------- the automatic pipeline ----------
class AnalysisError extends Error { constructor(code, msg) { super(msg); this.code = code; } }

// The face detector works on a downscaled copy, so a face that fills only a small part of a large photo can be missed.
// Look again in crops of the photo (centre first, then four overlapping corners) and map the points back.
async function detectZoomed(canvas) {
  const W = canvas.width, H = canvas.height, f = 0.6;
  const wins = [[(1 - f) / 2, (1 - f) / 2]];
  for (const oy of [0, 1 - f]) for (const ox of [0, 1 - f]) wins.push([ox, oy]);
  for (const [ox, oy] of wins) {
    const cw = Math.round(W * f), ch = Math.round(H * f), x0 = Math.round(W * ox), y0 = Math.round(H * oy);
    const c = canvasOf(cw, ch); c.getContext('2d').drawImage(canvas, x0, y0, cw, ch, 0, 0, cw, ch);
    const found = await Engine.detectImage(c);
    if (found.length) return found.map((lm) => lm.map((q) => ({ ...q, x: (q.x * cw + x0) / W, y: (q.y * ch + y0) / H })));
  }
  return [];
}
async function findFace(canvas) {
  let faces = await Engine.detectImage(canvas), turns = 0;
  if (!faces.length) {
    for (const k of [1, 3, 2]) { const r = rotateCanvas(canvas, k); const f = await Engine.detectImage(r); if (f.length) { canvas = r; faces = f; turns = k; break; } }
  }
  if (!faces.length) faces = await detectZoomed(canvas);    // a small face in a large photo: look closer
  if (!faces.length) throw new AnalysisError('NO_FACE', 'لم يُعثر على وجه في الصورة');
  const pick = (fs) => fs.map((lm) => SM.toPx(lm, canvas.width, canvas.height)).sort((a, b) => Math.abs(b[473].x - b[468].x) - Math.abs(a[473].x - a[468].x))[0];
  let P = pick(faces);
  const fix = SM.uprightTurns(P);              // eyes must be above the mouth — never trust anything else for orientation
  if (fix) { canvas = rotateCanvas(canvas, fix); turns = (turns + fix) % 4; const f2 = await Engine.detectImage(canvas); if (!f2.length) throw new AnalysisError('NO_FACE', 'لم يُعثر على وجه بعد تصحيح الاتجاه'); P = pick(f2); }
  return { canvas, P, turns, faces: faces.length };
}

function cropCanvasFrom(full, crop) {
  const c = canvasOf(crop.outW, crop.outH), x = c.getContext('2d'), Mi = crop.Minv;
  x.imageSmoothingQuality = 'high';
  x.setTransform(Mi[0], Mi[3], Mi[1], Mi[4], Mi[2], Mi[5]);
  x.drawImage(full, 0, 0);
  return c;
}
function lipsInCrop(P, crop) {
  const toC = (p) => SM.applyAffine(crop.Minv, p);
  return { upper: SM.INNER_UPPER.map((i) => toC(P[i])), lower: SM.INNER_LOWER.map((i) => toC(P[i])), poly: SM.innerLipPolygon(P).map(toC) };
}
function modelInput(cropCanvas) {
  const S = MODEL_SIZE, c = canvasOf(S, S), x = c.getContext('2d', { willReadFrequently: true });
  x.imageSmoothingQuality = 'high'; x.drawImage(cropCanvas, 0, 0, S, S);
  const d = x.getImageData(0, 0, S, S).data, n = S * S, t = new Float32Array(3 * n), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) t[k * n + i] = (d[i * 4 + k] / 255 - mean[k]) / std[k];
  return t;
}
function masksFromLabels(clean, S, w, h) {
  const masks = {};
  for (const t of C.TEETH) {
    const c = C.CLASS_OF[t], m = new Uint8Array(w * h); let any = 0;
    for (let y = 0; y < h; y++) { const ly = Math.min(S - 1, Math.floor((y + 0.5) * S / h)); for (let x = 0; x < w; x++) { const lx = Math.min(S - 1, Math.floor((x + 0.5) * S / w)); if (clean[ly * S + lx] === c) { m[y * w + x] = 1; any = 1; } } }
    if (any) masks[t] = m;
  }
  return masks;
}

// Full automatic analysis of a smile photo. onStep(key) reports progress.
async function analyzePhoto(blob, settings, onStep = () => {}) {
  const T = {}, t0 = performance.now(), mark = (k) => { T[k] = Math.round(performance.now() - t0); };
  onStep('face');
  const { canvas: full, P, turns, faces } = await findFace(await blobToCanvas(blob)); mark('face');
  const fctx = full.getContext('2d', { willReadFrequently: true });
  const fullImg = fctx.getImageData(0, 0, full.width, full.height);
  const stats = SM.imageStats(fullImg.data, full.width, full.height, P);
  const quality = SM.qualityChecks(P, full.width, full.height, stats);
  onStep('mouth');
  const crop = SM.mouthCrop(P), cropC = cropCanvasFrom(full, crop), lips = lipsInCrop(P, crop); mark('mouth');
  onStep('teeth');
  let seg = await Engine.segment(modelInput(cropC)); mark('teeth');
  // second safeguard: the GPU found nothing in an open mouth → check on the CPU engine; if the CPU sees teeth, keep using it
  const toothPx = (l) => { let n = 0; for (let i = 0; i < l.length; i++) if (l[i]) n++; return n; };
  if (!window.__DSD_TEST_SEGMENT && Engine.info.ep === 'webgpu' && mouthOpen(P) && toothPx(seg.labels) < 0.01 * seg.labels.length) {
    await Engine.call('cpu');
    const seg2 = await Engine.segment(modelInput(cropC));
    if (toothPx(seg2.labels) > toothPx(seg.labels)) { seg = seg2; Engine.info.ep = 'wasm'; Engine.info.fallback = 'GPU found no teeth; switched to CPU'; }
  }
  const geom = { P, crop, lips, labels: seg.labels, conf: seg.conf, turns, fullW: full.width, fullH: full.height, faces };
  const st = buildState(full, fullImg, cropC, geom, settings);
  st.quality = quality; st.stats = stats; st.timings = T; st.segMs = seg.ms; st.ep = seg.ep || Engine.info.ep;
  onStep('design'); mark('state');
  await serverCheck(st, settings); mark('server');
  return { st, rotatedBlob: turns ? await toBlob(full) : null };
}

// ---------- optional analysis server (second opinion) ----------
// When the doctor sets a server address, the upright photo and the face points the phone already found are sent to
// it. The server applies the full Python core (no-overlap guarantees, exact FDI numbering, dark-gap and lower-tooth
// checks, validated design, 1:1 mm lab file). Wherever the two disagree on a tooth, that tooth becomes 'unsure' so
// the doctor confirms it. If the server is slow or unreachable the on-device result stands, with a note.
const SERVER_TIMEOUT_MS = 25000;
const CONFIRM_BELOW = 0.95;      // per-tooth confidence under which the doctor confirms (same value as the server; calibrated for model v3)
const serverBase = (u) => (u || '').trim().replace(/\/+$/, '');
async function serverHealth(url) {
  const ctl = new AbortController(), tm = setTimeout(() => ctl.abort(), 8000);
  try { const r = await fetch(serverBase(url) + '/api/health', { signal: ctl.signal }); const j = await r.json(); return { ok: r.ok && !!j.ok, version: j.version }; }
  catch (e) { return { ok: false, error: e.name === 'AbortError' ? 'انتهت المهلة' : 'لا يستجيب أو العنوان خاطئ' }; }
  finally { clearTimeout(tm); }
}
async function serverAnalyze(url, full, P, options = {}) {
  const b = await toBlob(full, 0.9), buf = new Uint8Array(await b.arrayBuffer());
  let bin = ''; for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  const body = JSON.stringify({ image_b64: btoa(bin), landmarks: P.map((p) => [p.x, p.y]), options, overlay: false });
  const ctl = new AbortController(), tm = setTimeout(() => ctl.abort(), SERVER_TIMEOUT_MS), t0 = performance.now();
  try {
    const r = await fetch(serverBase(url) + '/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: ctl.signal });
    let j = null; try { j = await r.json(); } catch (e) { /* not JSON */ }
    const ms = Math.round(performance.now() - t0);
    if (!r.ok) { const d = j && (j.detail || j); return { ok: false, code: d?.error || 'HTTP_' + r.status, message: d?.message || 'رفض الخادم الطلب', ms }; }
    return { ok: true, ms, r: j };
  } catch (e) {
    return { ok: false, code: e.name === 'AbortError' ? 'TIMEOUT' : 'NETWORK', message: e.name === 'AbortError' ? 'انتهت المهلة' : 'تعذّر الاتصال بالخادم' };
  } finally { clearTimeout(tm); }
}
// Merge the server's tooth verdicts into the on-device state (pure; used by tests too)
function mergeServerStatus(local, server) {
  const status = { ...local }, disagree = [];
  const visible = (s) => s === 'present' || s === 'broken';
  for (const t of C.TEETH) {
    const a = local[t], b = server?.[t];
    if (!b || a === b) continue;
    // only a disagreement about whether a real tooth is there counts (missing vs hidden both mean 'nothing to modify')
    if (visible(a) !== visible(b)) { status[t] = 'unsure'; disagree.push(t); }
  }
  return { status, disagree };
}
async function serverCheck(st, settings) {
  if (!settings || !serverBase(settings.serverUrl) || window.__DSD_NO_SERVER) return;
  const res = await serverAnalyze(settings.serverUrl, st.full, st.geom.P, { proportion: 'red70' });
  if (!res.ok) { st.server = { ok: false, code: res.code, message: res.message, ms: res.ms }; return; }
  const r = res.r, m = mergeServerStatus(st.status, r.teeth.status);
  for (const t of m.disagree) delete st.toothMasks[t];
  st.status = m.status;
  st.server = { ok: true, version: r.version, ms: res.ms, status: r.teeth.status, disagree: m.disagree, needs: r.teeth.needs_confirmation,
    scale: r.scale && { source: r.scale.source, rel_err: r.scale.rel_err }, validation: r.validation, design: r.design, svg: r.design_svg };
  if (m.disagree.length) computeMeasures(st);
}

// Everything after segmentation; also used to reopen a saved case without re-running the model.
function buildState(full, fullImg, cropC, geom, settings, saved = {}) {
  const { P, crop, lips } = geom, w = crop.outW, h = crop.outH, S = MODEL_SIZE;
  const cropImg = cropC.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h);
  let sel;
  if (saved.teeth) sel = { teeth: saved.selTeeth, clean: geom.clean, orderOk: saved.orderOk, meanConf: saved.meanConf, presentCount: Object.keys(saved.selTeeth).length, info: saved.info || {} };
  else {
    sel = SM.selectUpperTeeth(geom.labels, geom.conf || null, S, w, h, lips, { rgba: cropImg.data });
    geom.clean = sel.clean;
    // a 'tooth' far darker than its neighbours is a gap the network painted over: take it out before classification
    const look = SM.appearanceCheck(sel.clean, S, cropImg.data, w, h);
    sel.appearance = look;
    for (const t of Object.keys(look.verdict)) {
      delete sel.teeth[t];
      const c = C.CLASS_OF[t]; for (let i = 0; i < sel.clean.length; i++) if (sel.clean[i] === c) sel.clean[i] = 0;
    }
  }
  let status, teeth, edentulous = !!saved.edentulous;
  if (saved.status) { status = saved.status; teeth = saved.teeth; }
  else if (!Object.keys(sel.teeth).length && mouthOpen(P)) {
    // no front teeth visible but the mouth is open: design a full anterior set from the face (trauma / extraction cases)
    edentulous = true;
    teeth = SM.designFromFace(lips, facialMidX(P, crop, lips), SM.irisMmPerPx(P) ? SM.irisMmPerPx(P) * crop.scale : null);
    status = Object.fromEntries(C.TEETH.map((t) => [t, 'missing']));
  } else {
    const absent = SM.classifyAbsent(sel, cropImg.data, w, h, lips);
    const looked = (sel.appearance && sel.appearance.verdict) || {};
    status = Object.fromEntries(C.TEETH.map((t) => [t, sel.teeth[t] ? 'present' : looked[t] || absent[t]]));
    teeth = C.fillMissing(sel.teeth, w, h).teeth;
    const edge = geom && geom.crop ? Math.max(0.75 * geom.crop.outW / MODEL_SIZE, 1 / geom.crop.scale) : 0;
    const br = SM.detectBroken(teeth, status, edge);
    for (const t of br) status[t] = 'broken';
    for (const t of br.unsure) status[t] = 'unsure';            // too close to call at this resolution: the doctor decides
    for (const t of SM.gapCheck(teeth, status)) status[t] = 'unsure';   // a tooth-sized gap or a split tooth: numbering may be shifted
  }
  const toothMasks = masksFromLabels(geom.clean, S, w, h);
  for (const t of C.TEETH) if (status[t] !== 'present' && status[t] !== 'broken') delete toothMasks[t];
  const mouthMask = SM.rasterPoly(lips.poly, w, h);
  const st = { full, fullImg, cropC, cropImg, geom, sel, status, teeth, toothMasks, mouthMask, settings, edentulous };
  computeMeasures(st);
  return st;
}

const mouthOpen = (P) => Math.hypot(P[13].x - P[14].x, P[13].y - P[14].y) / Math.max(1, Math.hypot(P[61].x - P[291].x, P[61].y - P[291].y)) >= 0.06;
function facialMidX(P, crop, lips) {
  const g = SM.applyAffine(crop.Minv, P[SM.LM.glabella]), m = SM.applyAffine(crop.Minv, P[SM.LM.menton]);
  const y = lips.upper.reduce((s, p) => s + p.y, 0) / lips.upper.length;
  return g.x + (m.x - g.x) * (y - g.y) / ((m.y - g.y) || 1);
}
function computeMeasures(st) {
  const { geom, teeth, status } = st, { P, crop } = geom, settings = st.settings;
  const iris = SM.irisMmPerPx(P);
  const cal = st.caliperMm && st.teeth['11'] ? st.caliperMm / (st.teeth['11'].x1 - st.teeth['11'].x0) : null;
  st.mmppCrop = cal || (iris ? iris * crop.scale : null);
  st.mmppImg = st.mmppCrop ? st.mmppCrop / crop.scale : null;
  st.scaleSource = cal ? 'caliper' : iris ? 'iris' : null;
  const mStatus = {}; for (const t of C.TEETH) mStatus[t] = status[t] === 'present' ? 'present' : status[t] === 'broken' ? 'broken' : 'missing';   // 'estimated' boxes are not measured
  st.meas = C.measure(teeth, st.mmppCrop, mStatus);
  const ext = SM.dentitionExtent(st.cropImg.data, crop.outW, crop.outH, st.mouthMask);
  const pts = SM.facePoints(P, Object.fromEntries(C.TEETH.filter((t) => status[t] === 'present').map((t) => [t, teeth[t]])), crop.M, ext);
  const ipdPx = Math.hypot(pts.pupilL.x - pts.pupilR.x, pts.pupilL.y - pts.pupilR.y);
  st.facePts = pts;
  st.face = C.faceAnalysis(pts, { ipdMm: st.mmppImg ? ipdPx * st.mmppImg : settings.ipdMm || 63 });
}

const SHADE_OPTIONS = [
  { id: 'BL2', label: 'أبيض ناصع' }, { id: 'B1', label: 'أبيض طبيعي' }, { id: 'A1', label: 'فاتح' }, { id: 'A2', label: 'طبيعي' }
];
const DEFAULT_DESIGN = { shape: 'square', shade: 'B1', proportion: 'red70', lengthMm: 0, midline: 'current' };

function computeDesign(st, d) {
  const { teeth, status, geom } = st, { crop, lips, P } = geom;
  const rules = st.settings.rules;
  // facial midline at the incisal level, in crop coordinates (crop is aligned to the eye line)
  const g = SM.applyAffine(crop.Minv, P[SM.LM.glabella]), mt = SM.applyAffine(crop.Minv, P[SM.LM.menton]);
  const yInc = (teeth['11'].y1 + teeth['21'].y1) / 2;
  const facialX = g.x + (mt.x - g.x) * (yInc - g.y) / ((mt.y - g.y) || 1);
  const prop = C.proposeDesign(teeth, st.meas, rules, { proportion: d.proportion, wl: 0.8, spaceMode: 'preserve', symmetric: true, gingivaMode: 'keep', midline: d.midline, facialMidlineX: facialX });
  if (!prop.ok) return { ok: false, error: prop.error };
  const lower = (x) => SM.chainY(lips.lower, x);
  let target = SM.applySmileArc(prop.target, lower, st.mmppCrop, { lengthDeltaMm: d.lengthMm });
  target = SM.limitChanges(teeth, target, st.mmppCrop, { shortenMm: 1.5 + Math.max(0, -d.lengthMm), lengthenMm: 2.5 + Math.max(0, d.lengthMm) }, status);
  const treat = {};
  for (const t of C.TEETH) treat[t] = d.treat?.[t] || (status[t] === 'missing' ? 'implant' : status[t] === 'broken' ? 'crown' : status[t] === 'hidden' ? 'none' : 'veneer');
  const mStatus = {}; for (const t of C.TEETH) mStatus[t] = treat[t] === 'none' && status[t] !== 'present' ? 'missing' : 'present';
  const measAfter = C.measure(target, st.mmppCrop, mStatus);
  const pP = C.PROPORTIONS[d.proportion]?.p ?? 0.7;
  // precision of one tooth edge in crop pixels: the coarser of the model grid and the photo's own pixels
  const edgePx = st.geom && st.geom.crop ? Math.max(0.75 * st.geom.crop.outW / MODEL_SIZE, 1 / st.geom.crop.scale) : 0;
  const faceEdgePx = st.geom && st.geom.crop ? edgePx * st.geom.crop.scale : 0;      // the same uncertainty in photo pixels (face points)
  const rulesNow = C.evaluateRules(st.meas, rules, { proportionP: pP, teeth, face: st.face, edgePx, faceEdgePx });
  const rulesAfter = C.evaluateRules(measAfter, rules.filter((r) => r.group !== 'face'), { proportionP: pP, teeth: target });
  const changes = {};
  for (const t of C.TEETH) {
    const a = teeth[t], b = target[t], mm = st.mmppCrop;
    changes[t] = { w: (b.x1 - b.x0) * (mm || 1), h: (b.y1 - b.y0) * (mm || 1), dW: ((b.x1 - b.x0) - (a.x1 - a.x0)) * (mm || 1), dL: ((b.y1 - b.y0) - (a.y1 - a.y0)) * (mm || 1), dIncisal: (b.y1 - a.y1) * (mm || 1), dGingival: (b.y0 - a.y0) * (mm || 1) };
  }
  return { ok: true, target, treat, prop, measAfter, rulesNow, rulesAfter, changes, facialX, mm: !!st.mmppCrop };
}

async function initSimulation(st) {
  const masks = {}, transfer = [];
  for (const [t, m] of Object.entries(st.toothMasks)) { const b = m.slice().buffer; masks[t] = b; transfer.push(b); }
  const crop = st.cropImg.data.slice().buffer, full = st.fullImg.data.slice().buffer, mouth = st.mouthMask.slice().buffer;
  transfer.push(crop, full, mouth);
  await Engine.call('simInit', { crop, cropW: st.cropImg.width, cropH: st.cropImg.height, full, fullW: st.fullImg.width, fullH: st.fullImg.height,
    mouth, toothMasks: masks, current: st.teeth, status: st.status, M: st.geom.crop.M, Minv: st.geom.crop.Minv }, transfer);
}
async function runSimulation(st, design, d) {
  const r = await Engine.call('simulate', { target: design.target, treat: design.treat, status: st.status, shape: d.shape, shadeHex: C.SHADES[d.shade] || C.SHADES.B1, whiten: 1 });
  return { full: new ImageData(new Uint8ClampedArray(r.full), st.fullImg.width, st.fullImg.height), crop: new ImageData(new Uint8ClampedArray(r.cropImg), st.cropImg.width, st.cropImg.height), ms: r.ms };
}

// Plain-language findings for the results screen, most important first
function findings(st, design) {
  const out = [], R = Object.fromEntries((design?.rulesNow || []).map((r) => [r.id, r])), M = st.face?.metrics || {}, mm = !!st.mmppCrop;
  const sev = { fail: 0, warn: 1, uncertain: 2, ok: 3, na: 4 };
  const add = (id, title, text) => { const r = R[id]; if (!r || r.status === 'na') return; out.push({ id, title, text: r.status === 'uncertain' ? `${text} — ضمن هامش خطأ القياس، لا يُحكم من دقة هذه الصورة` : text, status: r.status }); };
  if (M.midline_dev != null) add('midline_dev', 'خط الوسط', M.midline_dev < 0.5 ? 'متطابق مع منتصف الوجه' : `منحرف ${M.midline_dev.toFixed(1)} ملم ${M.midline_dir}`);
  if (M.smile_arc) add('smile_arc', 'قوس الابتسامة', { consonant: 'يوازي الشفة السفلى', flat: 'مسطّح: الحواف القاطعة على خط مستقيم', reverse: 'معكوس: الثنايا أقصر من الجانبية', nonparallel: 'لا يوازي الشفة السفلى' }[M.smile_arc]);
  if (M.gingival_display != null) add('gingival_display', 'ظهور اللثة', M.gingival_display > 0 ? `تظهر ${M.gingival_display.toFixed(1)} ملم من اللثة عند الابتسام` : 'الشفة تغطي حد اللثة');
  if (M.buccal_corridor != null) add('buccal_corridor', 'امتلاء الابتسامة', `الأسنان تملأ ${Math.round(M.buccal_corridor * 100)}% من عرض الابتسامة`);
  add('central_wl', 'شكل الثنايا', `نسبة العرض إلى الطول ${(st.meas.ratios.central_wl ?? 0).toFixed(2)} (المثالي 0.75–0.85)`);
  add('lat_central_w', 'تناسب العرض', `عرض الرباعية ${Math.round((st.meas.ratios.lat_central_w ?? 0) * 100)}% من الثنية`);
  const sym = ['sym_central', 'sym_lateral'].map((k) => R[k]).filter((r) => r && r.status !== 'na').sort((a, b) => sev[a.status] - sev[b.status])[0];
  if (sym) out.push({ id: sym.id, title: 'التماثل', text: sym.status === 'ok' ? 'الجانبان متماثلان' : sym.status === 'uncertain' ? `فرق بين الجانبين ${sym.valueText} — ضمن هامش خطأ القياس، لا يُحكم من دقة هذه الصورة` : `فرق بين الجانبين ${sym.valueText}`, status: sym.status });
  if (M.incisal_cant != null) add('incisal_cant', 'ميل الأسنان', Math.abs(M.incisal_cant) < 1 ? 'المستوى القاطع موازٍ لخط العينين' : `المستوى القاطع مائل ${Math.abs(M.incisal_cant).toFixed(1)}°`);
  out.sort((a, b) => sev[a.status] - sev[b.status]);
  return out;
}

const noTeeth = (st) => !st.edentulous && !C.TEETH.some((t) => st.status[t] === 'present' || st.status[t] === 'broken' || st.status[t] === 'missing');
// Problems the doctor must look at before approving (detection, not design)
// Plausibility of the visible (frontal) widths of the six teeth. Ratios do not depend on the mm scale; the mm range
// of the central incisor allows the ±10% iris-scale error. A tooth wider than a central, or a central far outside the
// human range, almost always means two teeth merged into one box or a shifted number (crowding, gaps, broken teeth).
function implausibleTeeth(st) {
  const ok = (t) => ['present', 'broken'].includes(st.status[t]) && st.teeth[t];
  const w = (t) => st.teeth[t].x1 - st.teeth[t].x0, out = [];
  const cen = ['11', '21'].filter(ok);
  if (!cen.length) return out;
  const cw = cen.reduce((a, t) => a + w(t), 0) / cen.length;
  if (cen.length === 2 && Math.min(w('11'), w('21')) / Math.max(w('11'), w('21')) < 0.72) out.push({ t: '11/21', text: 'الثنيتان مختلفتان كثيراً في العرض' });
  for (const t of ['12', '22']) if (ok(t) && w(t) / cw > 0.95) out.push({ t, text: `الرباعية ${t} بعرض الثنية أو أكبر` });
  for (const t of ['13', '23']) if (ok(t) && w(t) / cw > 1.0) out.push({ t, text: `الناب ${t} أعرض من الثنية` });
  if (st.mmppCrop) for (const t of cen) { const mm = w(t) * st.mmppCrop; if (mm < 6.0 || mm > 11.8) out.push({ t, text: `عرض الثنية ${t} ${mm.toFixed(1)} ملم (خارج المدى الطبيعي)` }); }
  return out;
}

function detectionIssues(st) {
  const out = [];
  if (noTeeth(st)) return [{ level: 'fail', text: 'لا تظهر الأسنان العلوية في الصورة. صوّر المريض وهو يبتسم ابتسامة عريضة تُظهر أسنانه.', retake: true }];
  if (st.edentulous) out.push({ level: 'warn', text: 'لا تظهر أسنان أمامية: صُمّمت أسنان كاملة على مقاييس الوجه (خط الوسط وعرض الفم). راجع مواضعها.', needsReview: true });
  const broken = C.TEETH.filter((t) => st.status[t] === 'broken');
  if (broken.length) out.push({ level: 'warn', text: `أسنان تبدو مكسورة: ${broken.join('، ')} — ستُبنى كاملة. تأكد منها.`, needsReview: true });
  const missing = st.edentulous ? [] : C.TEETH.filter((t) => st.status[t] === 'missing'), unsure = C.TEETH.filter((t) => st.status[t] === 'unsure' && !(st.server?.disagree || []).includes(t)), hidden = C.TEETH.filter((t) => st.status[t] === 'hidden');
  if (missing.length) out.push({ level: 'warn', text: `أسنان تبدو مفقودة: ${missing.join('، ')} — تأكد منها.`, needsReview: true });
  if (unsure.length) out.push({ level: 'fail', text: `لم يتضح تحديد: ${unsure.join('، ')} — راجع التحديد.`, needsReview: true });
  if (hidden.length) out.push({ level: 'info', text: `غير ظاهرة في الابتسامة: ${hidden.join('، ')} (لن تُعدَّل في المحاكاة).` });
  const est = C.TEETH.filter((t) => st.status[t] === 'estimated');
  if (est.length) out.push({ level: 'info', text: `حُدّد تقديرياً: ${est.join('، ')} (ظاهر لكن لم يُقَس). يمكنك تصحيح حدوده في المراجعة.` });
  if (st.sel && st.sel.orderOk === false) out.push({ level: 'fail', text: 'ترتيب الأسنان المكتشفة غير منطقي — راجع التحديد.', needsReview: true });
  // per-tooth confidence: below CONFIRM_BELOW the doctor confirms that tooth (model v3, labelled test photos x 6 hard
  // variants: catches 42 of 47 wrong teeth and asks about 3 of 257 correct teeth in clean photos)
  const lowConf = st.sel && st.sel.info ? C.TEETH.filter((t) => ['present', 'broken'].includes(st.status[t]) && st.sel.info[t] && st.sel.info[t].conf < CONFIRM_BELOW) : [];
  if (lowConf.length) out.push({ level: 'warn', text: `ثقة منخفضة في تحديد: ${lowConf.join('، ')} — تأكد من حدودها.`, needsReview: true });
  if (!lowConf.length && st.sel && st.sel.meanConf && st.sel.meanConf < 0.85) out.push({ level: 'warn', text: `ثقة الكشف منخفضة (${st.sel.meanConf.toFixed(2)}) — راجع التحديد.`, needsReview: true });
  const odd = implausibleTeeth(st);
  if (odd.length) out.push({ level: 'fail', text: `قياسات غير منطقية: ${odd.map((o) => o.text).join('، ')} — غالباً سنّان دُمجا أو حدود خاطئة. صحّح الحدود في المراجعة.`, needsReview: true, implausible: true });
  const hard = C.TEETH.filter((t) => ['missing', 'broken', 'unsure', 'estimated'].includes(st.status[t]));
  if (!st.edentulous && (hard.length >= 3 || (hard.length >= 2 && odd.length)))
    out.push({ level: 'warn', text: 'حالة معقدة (فقد أو كسر أو تزاحم في عدة أسنان): المحاكاة التلقائية تقريبية ولا تمثل النتيجة النهائية. استخدمها للتوضيح فقط، والتصميم النهائي يدوي من الطبيب.', complex: true });
  if (!st.mmppCrop) out.push({ level: 'warn', text: 'تعذّر تقدير المقياس بالملم من العين.' });
  if (st.server && st.server.ok && st.server.disagree.length)
    out.push({ level: 'warn', text: `الخادم والجهاز اختلفا في: ${st.server.disagree.join('، ')} — حدّد حالة كل منها في المراجعة.`, needsReview: true });
  if (st.server && !st.server.ok)
    out.push({ level: 'info', text: `تعذّر التحقق من خادم التحليل (${st.server.message}). النتيجة من التحليل على الجهاز فقط.` });
  return out;
}
