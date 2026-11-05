/* ===== part 4: auto face points, auto-crop, SAM tap-to-segment, tooth library ===== */
const MP_VER = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const FACE_MODEL_FILE = 'face_landmarker.task';
const FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const TF_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const SAM_ID = 'Xenova/slimsam-77-uniform';

// Fetch a model file: bundled next to the app → saved on device → remote (then saved for offline).
async function fetchModelFile(localName, blobKey, remoteUrl) {
  try {
    const r = await fetch(localName, { cache: 'force-cache' });
    if (r.ok && !(r.headers.get('content-type') || '').includes('text/html')) { const b = await r.arrayBuffer(); if (b.byteLength > 100000) return b; }
  } catch (e) { /* not bundled */ }
  const saved = await getBlob(blobKey); if (saved) return saved.arrayBuffer();
  if (!remoteUrl) return null;
  const r = await fetch(remoteUrl); if (!r.ok) throw new Error('DL_FAIL');
  const b = await r.arrayBuffer();
  await DB.put('blobs', { id: blobKey, blob: new Blob([b]), type: 'application/octet-stream' });
  return b;
}

// ---------- MediaPipe face landmarks ----------
let faceLm = null;
async function getFaceLandmarker() {
  if (faceLm) return faceLm;
  let vision;
  try { vision = await import(`${MP_BASE}/vision_bundle.mjs`); } catch (e) { throw new Error('MP_LOAD'); }
  const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const buf = await fetchModelFile(FACE_MODEL_FILE, 'faceModel', FACE_MODEL_URL);
  faceLm = await vision.FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetBuffer: new Uint8Array(buf), delegate: 'CPU' }, runningMode: 'IMAGE', numFaces: 1 });
  return faceLm;
}
async function faceLandmarks(canvas) {
  if (window.__DSD_TEST_FACE) return window.__DSD_TEST_FACE(canvas);
  const f = await getFaceLandmarker();
  const r = f.detect(canvas);
  return r.faceLandmarks && r.faceLandmarks[0] ? r.faceLandmarks[0] : null;
}
const INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
const OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
function interpY(pts, x) {
  const s = [...pts].sort((a, b) => a.x - b.x);
  if (x <= s[0].x) return s[0].y; if (x >= s[s.length - 1].x) return s[s.length - 1].y;
  for (let i = 1; i < s.length; i++) if (x <= s[i].x) { const k = (x - s[i - 1].x) / (s[i].x - s[i - 1].x || 1); return s[i - 1].y + k * (s[i].y - s[i - 1].y); }
  return s[s.length - 1].y;
}
// Returns {points, auto: [ids], manual: [ids]}
async function autoFacePoints(img) {
  const lm = await faceLandmarks(img);
  if (!lm) throw new Error('NO_FACE');
  const P = (i) => ({ x: lm[i].x * img.width, y: lm[i].y * img.height });
  const byX = (a, b) => (a.x <= b.x ? [a, b] : [b, a]);
  const [pR, pL] = byX(P(468), P(473)), [cR, cL] = byX(P(61), P(291));
  const pts = { pupilR: pR, pupilL: pL, glabella: P(9), subnasale: P(2), menton: P(152), commR: cR, commL: cL, upperLip: P(13), lowerLipMid: P(14) };
  const lower = INNER_LOWER.map(P);
  let teethAuto = false;
  // teeth inside the mouth: run the tooth model on a crop of the mouth (best effort)
  try {
    const ring = OUTER_LIP.map(P);
    let x0 = Math.min(...ring.map((p) => p.x)), x1 = Math.max(...ring.map((p) => p.x)), y0 = Math.min(...ring.map((p) => p.y)), y1 = Math.max(...ring.map((p) => p.y));
    const mx = (x1 - x0) * 0.08, my = (y1 - y0) * 0.15; x0 = Math.max(0, x0 - mx); x1 = Math.min(img.width, x1 + mx); y0 = Math.max(0, y0 - my); y1 = Math.min(img.height, y1 + my);
    const crop = cropCanvas(img, { x: x0 / img.width, y: y0 / img.height, w: (x1 - x0) / img.width, h: (y1 - y0) / img.height }, 1600);
    const k = (x1 - x0) / crop.width;
    const r = await detectOn(crop);
    if (r.det.presentCount >= 5 && r.det.orderOk && r.det.teeth['11'] && r.det.teeth['21'] && r.det.teeth['13'] && r.det.teeth['23']) {
      const T = {}; for (const t of C.TEETH) if (r.det.teeth[t]) { const b = r.det.teeth[t]; T[t] = { x0: x0 + b.x0 * k, x1: x0 + b.x1 * k, y0: y0 + b.y0 * k, y1: y0 + b.y1 * k }; }
      const mid = (b) => (b.x0 + b.x1) / 2;
      pts.zenith11 = { x: mid(T['11']), y: T['11'].y0 }; pts.incisal11 = { x: mid(T['11']), y: T['11'].y1 };
      pts.midlineDental = { x: (T['11'].x1 + T['21'].x0) / 2, y: (T['11'].y1 + T['21'].y1) / 2 - (T['11'].y1 - T['11'].y0) * 0.05 };
      pts.cusp13 = { x: mid(T['13']), y: T['13'].y1 }; pts.cusp23 = { x: mid(T['23']), y: T['23'].y1 };
      teethAuto = true;
    }
  } catch (e) { if (String(e.message) !== 'NO_MODEL') console.warn(e); }
  if (pts.cusp13) { pts.lowerLipR = { x: pts.cusp13.x, y: interpY(lower, pts.cusp13.x) }; pts.lowerLipL = { x: pts.cusp23.x, y: interpY(lower, pts.cusp23.x) }; }
  else { const [a, b] = byX(P(178), P(402)); pts.lowerLipR = a; pts.lowerLipL = b; }
  const manual = C.FACE_POINTS.map((p) => p.id).filter((id) => !pts[id]);
  return { points: pts, teethAuto, manual };
}
function faceError(e) {
  console.error(e);
  const m = String(e && e.message || e);
  const txt = m === 'NO_FACE' ? 'لم يُعثر على وجه في الصورة. تأكد أن الوجه كاملاً ظاهر ومستقيم، أو ضع النقاط يدوياً.'
    : m === 'MP_LOAD' || m === 'DL_FAIL' ? 'تعذّر تحميل كاشف الوجه. يلزم إنترنت أول مرة (نحو 4 ميغابايت)، ثم يعمل بلا اتصال.'
      : 'خطأ في الكشف التلقائي للوجه: ' + m;
  return modal('نقاط الوجه التلقائية', h('p', null, txt));
}

// ---------- auto-crop of the intra-oral photo ----------
function rectAroundTeeth(teeth, W, H) {
  const bs = C.TEETH.map((t) => teeth[t]).filter(Boolean); if (!bs.length) return null;
  let x0 = Math.min(...bs.map((b) => b.x0)), x1 = Math.max(...bs.map((b) => b.x1)), y0 = Math.min(...bs.map((b) => b.y0)), y1 = Math.max(...bs.map((b) => b.y1));
  const mx = (x1 - x0) * 0.22, my = (y1 - y0) * 0.6;
  x0 = Math.max(0, x0 - mx); x1 = Math.min(W, x1 + mx); y0 = Math.max(0, y0 - my); y1 = Math.min(H, y1 + my);
  return { x: x0 / W, y: y0 / H, w: (x1 - x0) / W, h: (y1 - y0) / H };
}
// Returns {q, rect, how} with rect relative to rotatedCanvas(src, q*90).
async function autoCropIntra(src) {
  let best = null;
  for (const q of [0, 1, 2, 3]) { const cv = rotatedCanvas(src, q * 90); const r = await detectOn(cv); if (!best || r.det.score > best.r.det.score) best = { q, r, cv }; }
  if (best.r.det.presentCount >= 5 && best.r.det.orderOk) return { q: best.q, rect: rectAroundTeeth(best.r.det.teeth, best.cv.width, best.cv.height), how: 'الأسنان' };
  // full-face photo: find the mouth first, then the teeth inside it
  for (const q of [...new Set([best.q, 0, 1, 2, 3])]) {
    const cv = rotatedCanvas(src, q * 90);
    let lm = null; try { lm = await faceLandmarks(cv); } catch (e) { if (q === best.q) console.warn(e); }
    if (!lm) continue;
    const ring = OUTER_LIP.map((i) => ({ x: lm[i].x * cv.width, y: lm[i].y * cv.height }));
    let x0 = Math.min(...ring.map((p) => p.x)), x1 = Math.max(...ring.map((p) => p.x)), y0 = Math.min(...ring.map((p) => p.y)), y1 = Math.max(...ring.map((p) => p.y));
    const mx = (x1 - x0) * 0.1, my = (y1 - y0) * 0.25; x0 = Math.max(0, x0 - mx); x1 = Math.min(cv.width, x1 + mx); y0 = Math.max(0, y0 - my); y1 = Math.min(cv.height, y1 + my);
    const mouth = { x: x0 / cv.width, y: y0 / cv.height, w: (x1 - x0) / cv.width, h: (y1 - y0) / cv.height };
    const crop = cropCanvas(cv, mouth, 1600); const k = (x1 - x0) / crop.width;
    const r = await detectOn(crop);
    if (r.det.presentCount >= 4 && r.det.orderOk) {
      const T = {}; for (const t of C.TEETH) if (r.det.teeth[t]) { const b = r.det.teeth[t]; T[t] = { x0: x0 + b.x0 * k, x1: x0 + b.x1 * k, y0: y0 + b.y0 * k, y1: y0 + b.y1 * k }; }
      return { q, rect: rectAroundTeeth(T, cv.width, cv.height), how: 'الفم ثم الأسنان' };
    }
    return { q, rect: mouth, how: 'الفم (لم تُكتشف الأسنان بوضوح)' };
  }
  return null;
}

// ---------- SAM: tap inside a tooth to outline it ----------
let samState = null;
async function getSam() {
  if (samState) return samState;
  let T; try { T = await import(TF_URL); } catch (e) { throw new Error('SAM_LOAD'); }
  T.env.allowLocalModels = false;
  const [model, processor] = await Promise.all([T.SamModel.from_pretrained(SAM_ID), T.AutoProcessor.from_pretrained(SAM_ID)]);
  samState = { T, model, processor, key: null };
  return samState;
}
// Returns candidate masks [{score, w, h, has(i)}] in canvas pixel grid.
async function samMasks(canvas, cacheKey, pt) {
  if (window.__DSD_TEST_SAM) return window.__DSD_TEST_SAM(canvas, pt);
  const s = await getSam();
  if (s.key !== cacheKey) {
    const image = await s.T.RawImage.fromBlob(await canvasToBlob(canvas, 'image/png'));
    s.inputs = await s.processor(image); s.emb = await s.model.get_image_embeddings(s.inputs); s.key = cacheKey;
  }
  const [rh, rw] = s.inputs.reshaped_input_sizes[0];
  const input_points = new s.T.Tensor('float32', [pt.x / canvas.width * rw, pt.y / canvas.height * rh], [1, 1, 1, 2]);
  const input_labels = new s.T.Tensor('int64', new BigInt64Array([1n]), [1, 1, 1]);
  const out = await s.model({ ...s.emb, input_points, input_labels });
  const masks = await s.processor.post_process_masks(out.pred_masks, s.inputs.original_sizes, s.inputs.reshaped_input_sizes);
  const m = masks[0][0]; const n = m.dims[0], H = m.dims[1], W = m.dims[2], data = m.data, scores = out.iou_scores.data;
  const res = [];
  for (let j = 0; j < n; j++) { const off = j * H * W; res.push({ score: scores[j], w: W, h: H, has: (i) => !!data[off + i] }); }
  return res;
}
function maskBox(mk, pt) {
  const xs = [], ys = [];
  for (let y = 0; y < mk.h; y += 1) for (let x = 0; x < mk.w; x += 1) if (mk.has(y * mk.w + x)) { xs.push(x); ys.push(y); }
  if (xs.length < 30) return null;
  const pi = Math.round(pt.y) * mk.w + Math.round(pt.x);
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
  const q = (a, p) => a[Math.min(a.length - 1, Math.round(p / 100 * (a.length - 1)))];
  return { box: { x0: q(xs, 1), x1: q(xs, 99) + 1, y0: q(ys, 1), y1: q(ys, 99) + 1 }, area: xs.length / (mk.w * mk.h), hit: mk.has(pi) };
}
// choose the tooth-sized mask containing the tap
function pickToothBox(masks, pt, prev, W) {
  const maxW = prev ? Math.max((prev.x1 - prev.x0) * 1.8, W * 0.06) : W * 0.25;
  const cands = masks.map((m) => ({ m, b: maskBox(m, pt) })).filter((c) => c.b && c.b.hit && c.b.area < 0.25 && (c.b.box.x1 - c.b.box.x0) <= maxW && c.b.area > 0.0015);
  cands.sort((a, b) => b.m.score - a.m.score);
  return cands[0] ? { box: cands[0].b.box, score: cands[0].m.score } : null;
}
function samError(e) {
  console.error(e);
  const m = String(e && e.message || e);
  return modal('التحديد بنقرة', h('p', null, m === 'SAM_LOAD' || /fetch|network|Failed/i.test(m) ? 'تعذّر تحميل نموذج SAM. يلزم إنترنت أول مرة (عشرات الميغابايت)، ثم يُحفظ للعمل بلا اتصال.' : 'خطأ في التحديد بنقرة: ' + m));
}

// ---------- tooth library (natural teeth from approved cases) ----------
let LIB = null; const libImgCache = new Map();
async function loadLibrary() { if (!LIB) LIB = await DB.getKV('toothLibrary', []); return LIB; }
async function saveLibrary() { await DB.setKV('toothLibrary', LIB); }
async function addCaseToLibrary(c) {
  if (!c.intra.teeth || !C.TEETH.every((t) => c.intra.teeth[t])) { toast('حدّد الأسنان الستة أولاً'); return; }
  if (!c.consent.training && !await confirmBox('موافقة المريض', 'لم يُسجَّل في هذه الحالة أن المريض وافق على استخدام صوره. لا تحفظ أسنانه في المكتبة إلا بموافقته. متابعة؟', 'لدي موافقته')) return;
  const name = await promptBox('اسم المجموعة', 'اسم مختصر (مثلاً: طبيعية شابة A1، أو حالة 2025-09)', 'text', 'مجموعة ' + ((await loadLibrary()).length + 1));
  if (!name) return;
  const img = await getCanvasForBlob(c.images.intraWork);
  const teeth = {};
  for (const t of C.TEETH) {
    const b = c.intra.teeth[t], w = b.x1 - b.x0, hh = b.y1 - b.y0, k = 320 / hh;
    const cv = canvasOf(w * k, 320); cv.getContext('2d').drawImage(img, b.x0, b.y0, w, hh, 0, 0, cv.width, cv.height);
    teeth[t] = cv.toDataURL('image/jpeg', 0.9);
  }
  LIB.push({ id: uid(), name, created: Date.now(), teeth });
  await saveLibrary(); toast('حُفظت الأسنان في المكتبة');
}
function libImage(src) {
  if (libImgCache.has(src)) return libImgCache.get(src);
  const im = new Image(); im.src = src; libImgCache.set(src, im); return im;
}
async function preloadLibrarySet(id) {
  const set = (await loadLibrary()).find((s) => s.id === id); if (!set) return null;
  await Promise.all(Object.values(set.teeth).map((u) => libImage(u).decode().catch(() => {})));
  return set;
}
// draw one tooth from a library photo, fitted to the target shape and tinted toward the chosen shade
function drawLibTooth(ctx, set, t, b, d) {
  const mirror = { '11': '21', '21': '11', '12': '22', '22': '12', '13': '23', '23': '13' };
  let src = set.teeth[t], flip = false;
  if (!src && set.teeth[mirror[t]]) { src = set.teeth[mirror[t]]; flip = true; }
  if (!src) return false;
  const im = libImage(src); if (!im.complete || !im.naturalWidth) return false;
  const w = b.x1 - b.x0, hh = b.y1 - b.y0;
  const path = new Path2D(C.toothPath(d.shape, b, t === '13' || t === '23'));
  ctx.save(); ctx.clip(path);
  if (flip) { ctx.translate(b.x0 + w, b.y0); ctx.scale(-1, 1); ctx.drawImage(im, 0, -hh * 0.04, w, hh * 1.08); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clip(path); }
  else ctx.drawImage(im, b.x0, b.y0 - hh * 0.04, w, hh * 1.08);
  const shade = hexRgb(C.SHADES[d.shade] || C.SHADES.A1);
  ctx.globalCompositeOperation = 'color'; ctx.fillStyle = rgba(shade, 0.45); ctx.fillRect(b.x0, b.y0, w, hh);
  ctx.globalCompositeOperation = 'soft-light'; ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.fillRect(b.x0, b.y0, w, hh);
  ctx.restore();
  ctx.save(); ctx.strokeStyle = 'rgba(110,90,70,0.35)'; ctx.lineWidth = Math.max(1, w * 0.015); ctx.stroke(path); ctx.restore();
  return true;
}
function libraryManager() {
  const box = h('div');
  const draw = async () => {
    box.innerHTML = '';
    const L = await loadLibrary();
    if (!L.length) { box.append(h('p', { class: 'muted small' }, 'المكتبة فارغة. من خطوة «الأسنان» في حالة أسنانها جميلة وطبيعية (وبموافقة المريض) اضغط «حفظ أسنان الحالة في المكتبة».')); return; }
    for (const s of L) {
      box.append(h('div', { class: 'row', style: { padding: '8px 0', borderBottom: '1px solid var(--line)' } },
        h('div', { class: 'row', style: { gap: '2px', direction: 'ltr' } }, C.TEETH.map((t) => s.teeth[t] ? h('img', { src: s.teeth[t], alt: t, style: { height: '40px', borderRadius: '4px' } }) : null)),
        h('b', { class: 'grow' }, s.name),
        h('button', { class: 'btn sm', onclick: async () => { const n = await promptBox('إعادة تسمية', 'الاسم', 'text', s.name); if (n) { s.name = n; await saveLibrary(); draw(); } } }, 'تسمية'),
        h('button', { class: 'btn sm danger', onclick: async () => { if (await confirmBox('حذف', `حذف «${s.name}» من المكتبة؟`, 'حذف', true)) { LIB = L.filter((x) => x.id !== s.id); await saveLibrary(); draw(); } } }, 'حذف')));
    }
  };
  draw(); return box;
}
async function warmUp(kind, btn) {
  const old = btn.textContent; btn.disabled = true; btn.textContent = '⏳ جارٍ التحميل…';
  try {
    if (kind === 'face') await getFaceLandmarker(); else if (kind === 'sam') await getSam(); else await getSession();
    toast('جاهز للعمل بلا إنترنت');
  } catch (e) { (kind === 'face' ? faceError : kind === 'sam' ? samError : modelError)(e); }
  finally { btn.disabled = false; btn.textContent = old; }
}
