/* ===== part 3: model, design, outputs, settings ===== */
// ---------- ONNX model ----------
let ortSession = null;
function loadScript(url) {
  return new Promise((res, rej) => { const sc = document.createElement('script'); sc.src = url; sc.onload = res; sc.onerror = () => rej(new Error('ORT_LOAD')); document.head.append(sc); });
}
async function getModelBuffer() {
  try {
    const r = await fetch(MODEL_FILE, { cache: 'force-cache' });
    if (r.ok && !(r.headers.get('content-type') || '').includes('text/html')) { const b = await r.arrayBuffer(); if (b.byteLength > 100000) return { buf: b, src: 'مع التطبيق' }; }
  } catch (e) { /* not bundled */ }
  const blob = await getBlob('model');
  if (blob) return { buf: await blob.arrayBuffer(), src: 'محفوظ على الجهاز' };
  return null;
}
async function getSession() {
  if (ortSession) return ortSession;
  if (!window.ort) await loadScript(ORT_BASE + 'ort.min.js');
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const m = await getModelBuffer(); if (!m) throw new Error('NO_MODEL');
  ortSession = await ort.InferenceSession.create(m.buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  return ortSession;
}
async function inferProb(canvas) {
  const Sz = +S.modelSize || 384, n = Sz * Sz;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const run = async (flip) => {
    const cv = canvasOf(Sz, Sz), x = cv.getContext('2d', { willReadFrequently: true });
    if (flip) { x.translate(Sz, 0); x.scale(-1, 1); }
    x.imageSmoothingQuality = 'high'; x.drawImage(canvas, 0, 0, Sz, Sz);
    const d = x.getImageData(0, 0, Sz, Sz).data, t = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) t[k * n + i] = (d[i * 4 + k] / 255 - mean[k]) / std[k];
    if (window.__DSD_TEST_INFER) return window.__DSD_TEST_INFER(t, Sz, flip);
    const sess = await getSession();
    const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', t, [1, 3, Sz, Sz]) });
    const o = out[sess.outputNames[0]];
    if (o.dims.length !== 4 || o.dims[1] !== 7 || o.dims[2] !== Sz || o.dims[3] !== Sz) throw new Error('شكل مخرج النموذج ' + o.dims.join('×') + ' لا يطابق حجم المدخل ' + Sz + ' في الإعدادات');
    return o.data;
  };
  let prob = C.softmaxNCHW(await run(false), 7, Sz);
  if (S.useTTA) prob = C.mergeFlipTTA(prob, C.softmaxNCHW(await run(true), 7, Sz), 7, Sz);
  return { prob, Sz };
}
async function detectOn(canvas) {
  const { prob, Sz } = await inferProb(canvas);
  const { labels, conf } = C.argmaxNCHW(prob, 7, Sz);
  const clean = C.largestComponents(labels, Sz, 7);
  return { det: C.labelsToTeeth(clean, conf, Sz, canvas.width, canvas.height), labels: clean, Sz };
}
function modelError(e) {
  console.error(e);
  const m = String(e && e.message || e);
  if (m === 'NO_MODEL') return modal('النموذج غير محمّل', h('div', null, h('p', null, 'لم يُعثر على ملف النموذج ' + MODEL_FILE + '. حمّله مرة واحدة من الإعدادات ← «نموذج كشف الأسنان»، أو ضعه بجانب ملفات التطبيق عند النشر.'), h('p', { class: 'muted small' }, 'يمكنك المتابعة بالتحديد اليدوي الآن.')), [{ label: 'لاحقاً', value: false }, { label: 'فتح الإعدادات', value: true, primary: true }]).then((v) => { if (v) go('#/settings'); });
  if (m === 'ORT_LOAD') return modal('تعذّر تحميل محرك التشغيل', h('p', null, 'يلزم اتصال بالإنترنت أول مرة لتحميل ONNX Runtime، ثم يعمل بلا إنترنت. تحقق من الاتصال وحاول مجدداً، أو استخدم التحديد اليدوي.'));
  return modal('خطأ في الكشف التلقائي', h('p', null, m));
}
function rotateSame(src, degs) { // rotate around centre keeping size
  const cv = canvasOf(src.width, src.height), x = cv.getContext('2d');
  x.fillStyle = '#000'; x.fillRect(0, 0, cv.width, cv.height); x.imageSmoothingQuality = 'high';
  x.translate(cv.width / 2, cv.height / 2); x.rotate(degs * Math.PI / 180); x.drawImage(src, -src.width / 2, -src.height / 2);
  return cv;
}
async function replaceWork(c, canvas) {
  if (c.images.intraWork) { await DB.del('blobs', c.images.intraWork); imageCache.delete(c.images.intraWork); }
  c.images.intraWork = await putBlob(await canvasToBlob(canvas));
  imageCache.set(c.images.intraWork, canvas);
}
async function runDetection(c, btn) {
  const old = btn.textContent; btn.disabled = true; btn.textContent = '⏳ جارٍ التحليل…';
  try {
    let img = await getCanvasForBlob(c.images.intraWork);
    let r = await detectOn(img), rotated = 0, straightened = 0;
    if (r.det.presentCount < 5 || !r.det.orderOk) {
      let best = { r, q: 0 };
      for (const q of [2, 1, 3]) { const cv = rotatedCanvas(img, q * 90); const rr = await detectOn(cv); if (rr.det.score > best.r.det.score + 0.5) best = { r: rr, q, cv }; }
      if (best.q) { img = best.cv; r = best.r; rotated = best.q * 90; await replaceWork(c, img); }
    }
    if (S.autoStraighten && r.det.presentCount >= 4 && r.det.orderOk && Math.abs(r.det.tilt) > 1.5 && Math.abs(r.det.tilt) < 25) {
      straightened = -r.det.tilt; img = rotateSame(img, straightened); await replaceWork(c, img); r = await detectOn(img);
    }
    const det = r.det, q = C.qualityReport(det, +S.confMin || 0.91);
    const fm = C.fillMissing(det.teeth, img.width, img.height);
    c.intra.teeth = fm.teeth; c.intra.reviewed = false;
    c.intra.status = {}; for (const t of fm.filled) c.intra.status[t] = 'missing';
    for (const t of fm.filled) if (['veneer', 'composite'].includes(c.design.treat[t])) c.design.treat[t] = 'implant';
    c.intra.source = 'auto';
    c.intra.det = { meanConf: det.meanConf, presentCount: det.presentCount, orderOk: det.orderOk, tilt: det.tilt, issues: q.issues, level: q.level, tta: !!S.useTTA, straightened, rotated, at: Date.now() };
    if (fm.filled.length) { c.intra.det.issues = c.intra.det.issues.filter((i) => !i.text.startsWith('أسنان لم تُكتشف')); if (c.intra.det.level === 'fail' && !c.intra.det.issues.some((i) => i.level === 'fail')) c.intra.det.level = 'warn'; }
    if (fm.filled.length) c.intra.det.issues.unshift({ level: 'warn', text: `لم يُكتشف: ${fm.filled.join('، ')}. وُضع مستطيله في الفراغ وعُلّم «مفقود» واقتُرحت زرعة. إن كان السن موجوداً فغيّر حالته وعدّل حدوده (أو استخدم التحديد بنقرة).` });
    if (rotated) c.intra.det.issues.unshift({ level: 'warn', text: `دُوّرت الصورة ${rotated}° تلقائياً لأن الأسنان بدت مقلوبة.` });
    c.intra.maskPng = labelsToPng(r.labels, r.Sz);
    c.design.override = {};
    await saveCase(c, { design: true });
    toast(q.level === 'ok' ? 'تم الكشف — راجع الحدود' : 'تم الكشف مع تنبيهات — راجع الحدود بعناية');
    return true;
  } catch (e) { await modelError(e); return false; } finally { btn.disabled = false; btn.textContent = old; }
}
function labelsToPng(labels, Sz) {
  const cv = canvasOf(Sz, Sz), x = cv.getContext('2d'), im = x.createImageData(Sz, Sz);
  const rgb = C.TEETH.map((t) => hexRgb(TOOTH_COLORS[t]));
  for (let i = 0; i < labels.length; i++) { const l = labels[i]; if (!l) continue; const [r, g, b] = rgb[l - 1]; im.data.set([r, g, b, 200], i * 4); }
  x.putImageData(im, 0, 0); return cv.toDataURL('image/png');
}
async function autoOrient(src) {
  try {
    toast('جارٍ فحص الاتجاهات الأربعة…');
    let best = null;
    for (const q of [0, 1, 2, 3]) { const r = await detectOn(rotatedCanvas(src, q * 90)); if (!best || r.det.score > best.score) best = { q, score: r.det.score, n: r.det.presentCount }; }
    toast(best.q ? `الاتجاه المقترح: تدوير ${best.q * 90}°` : 'الاتجاه الحالي صحيح');
    return best.q;
  } catch (e) { await modelError(e); return null; }
}

// ---------- colour & rendering ----------
function hexRgb(hx) { const v = parseInt(hx.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function mix(rgb, to, k) { return rgb.map((v, i) => Math.round(v + (to[i] - v) * k)); }
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
function drawSim(ctx, img, target, d, teeth) {
  ctx.drawImage(img, 0, 0);
  const base = hexRgb(C.SHADES[d.shade] || C.SHADES.A1);
  const set = d.libraryId && LIB ? LIB.find((x) => x.id === d.libraryId) : null;
  for (const t of C.TEETH) {
    if (d.treat[t] === 'none' || !target[t]) continue;
    if (set && drawLibTooth(ctx, set, t, target[t], d)) continue;
    const b = target[t], w = b.x1 - b.x0, hgt = b.y1 - b.y0;
    const path = new Path2D(C.toothPath(d.shape, b, t === '13' || t === '23'));
    const g = ctx.createLinearGradient(0, b.y0, 0, b.y1);
    g.addColorStop(0, rgba(mix(base, [190, 150, 110], 0.22), 0.97));
    g.addColorStop(0.35, rgba(base, 0.97));
    g.addColorStop(0.8, rgba(mix(base, [255, 255, 255], 0.25), 0.96));
    g.addColorStop(1, rgba(mix(base, [200, 210, 225], 0.45), 0.9));
    ctx.save(); ctx.fillStyle = g; ctx.fill(path);
    ctx.clip(path);
    const hl = ctx.createLinearGradient(b.x0, 0, b.x1, 0);
    hl.addColorStop(0, 'rgba(90,70,50,0.18)'); hl.addColorStop(0.3, 'rgba(255,255,255,0.10)'); hl.addColorStop(0.5, 'rgba(255,255,255,0.18)'); hl.addColorStop(0.7, 'rgba(255,255,255,0.08)'); hl.addColorStop(1, 'rgba(90,70,50,0.2)');
    ctx.fillStyle = hl; ctx.fillRect(b.x0, b.y0 - hgt * 0.1, w, hgt * 1.2);
    const inc = ctx.createLinearGradient(0, b.y1 - hgt * 0.22, 0, b.y1);
    inc.addColorStop(0, 'rgba(170,185,205,0)'); inc.addColorStop(1, 'rgba(150,170,195,0.35)');
    ctx.fillStyle = inc; ctx.fillRect(b.x0, b.y1 - hgt * 0.22, w, hgt * 0.22);
    if (t !== '13' && t !== '23') for (const f of [0.33, 0.66]) { const lg = ctx.createLinearGradient(b.x0 + w * (f - 0.08), 0, b.x0 + w * (f + 0.08), 0); lg.addColorStop(0, 'rgba(255,255,255,0)'); lg.addColorStop(0.5, 'rgba(120,100,80,0.07)'); lg.addColorStop(1, 'rgba(255,255,255,0)'); ctx.fillStyle = lg; ctx.fillRect(b.x0 + w * (f - 0.08), b.y0 + hgt * 0.15, w * 0.16, hgt * 0.7); }
    ctx.restore();
    ctx.save(); ctx.strokeStyle = 'rgba(110,90,70,0.45)'; ctx.lineWidth = Math.max(1, w * 0.02); ctx.stroke(path); ctx.restore();
  }
}
function drawMeasure(ctx, teeth, mm, target) {
  const W = ctx.canvas.width, lw = Math.max(2, W / 400), fs = Math.max(12, W / 55);
  ctx.font = `700 ${fs}px sans-serif`; ctx.textAlign = 'center'; ctx.lineJoin = 'round';
  for (const t of C.TEETH) {
    const b = teeth[t]; if (!b) continue;
    ctx.strokeStyle = TOOTH_COLORS[t]; ctx.lineWidth = lw; ctx.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
    const txt = mm ? `${t}: ${num((b.x1 - b.x0) * mm, 1)}×${num((b.y1 - b.y0) * mm, 1)}` : t;
    ctx.lineWidth = fs / 4; ctx.strokeStyle = '#000'; ctx.strokeText(txt, (b.x0 + b.x1) / 2, b.y0 - fs * 0.4); ctx.fillStyle = TOOTH_COLORS[t]; ctx.fillText(txt, (b.x0 + b.x1) / 2, b.y0 - fs * 0.4);
  }
  if (target) { ctx.setLineDash([lw * 3, lw * 2]); ctx.strokeStyle = '#fff'; ctx.lineWidth = lw; for (const t of C.TEETH) if (target[t]) ctx.stroke(new Path2D(C.toothPath('square', target[t], t === '13' || t === '23'))); ctx.setLineDash([]); }
}
function simCanvas(img, c, A) { if (c.design.libraryId) loadLibrary(); const cv = canvasOf(img.width, img.height); drawSim(cv.getContext('2d'), img, A.design.target, c.design, c.intra.teeth); return cv; }
function measureCanvas(img, c, A) { const cv = canvasOf(img.width, img.height), x = cv.getContext('2d'); x.drawImage(img, 0, 0); drawMeasure(x, c.intra.teeth, A.mm); return cv; }

// ----- step 5: design -----
async function stepDesign(body, c) {
  if (!c.intra.teeth) { body.append(h('div', { class: 'card empty' }, 'حدّد الأسنان أولاً.'), nextBtn(c, 'teeth', '→ الأسنان')); return; }
  const img = await getCanvasForBlob(c.images.intraWork);
  const d = c.design; let view = 'outline', sel = '11';
  const lib = await loadLibrary(); if (d.libraryId && !lib.find((x) => x.id === d.libraryId)) d.libraryId = null;
  if (d.libraryId) await preloadLibrarySet(d.libraryId);
  let A = computeAll(c);
  if (A.design?.error) { body.append(h('div', { class: 'card issue fail' }, A.design.error)); return; }
  const opt = h('div', { class: 'card' }, h('h2', null, 'خيارات التصميم'));
  const sel_ = (key, options, label, onAfter) => {
    const el = h('select', null, Object.entries(options).map(([v, l]) => h('option', { value: v, selected: String(d[key]) === v }, l)));
    el.addEventListener('change', () => { d[key] = el.value === 'true' ? true : el.value === 'false' ? false : el.value; onAfter?.(); changed(); });
    return h('label', { class: 'field' }, h('span', null, label), el);
  };
  const numIn = (key, label, step, ph) => {
    const el = h('input', { type: 'number', step, value: d[key] ?? '', placeholder: ph || '' });
    el.addEventListener('change', () => { d[key] = el.value === '' ? null : +el.value; changed(); });
    return h('label', { class: 'field' }, h('span', null, label), el);
  };
  const faceAvail = A.face?.metrics?.midline_dev_px != null;
  const propOpts = Object.fromEntries([...Object.entries(C.PROPORTIONS).map(([k, v]) => [k, v.label]), ['custom', 'نسبة مخصصة']]);
  opt.append(h('div', { class: 'grid2' },
    sel_('proportion', propOpts, 'تناسب العرض الظاهري', () => render()), d.proportion === 'custom' ? numIn('customP', 'النسبة المخصصة', '0.01') : h('div'),
    numIn('wl', 'نسبة عرض الثنية إلى طولها المستهدفة', '0.01'),
    sel_('spaceMode', { preserve: 'الحفاظ على المسافة الحالية بين النابين', central: 'البدء من عرض الثنية' }, 'توزيع العرض'),
    sel_('symmetric', { true: 'متماثل الجانبين', false: 'كل جانب حسب مسافته' }, 'التماثل'),
    sel_('gingivaMode', { keep: 'إبقاء حد اللثة الحالي', ideal: 'خط لثة مثالي (قد يحتاج قطع لثة)' }, 'اللثة'),
    sel_('midline', faceAvail ? { current: 'خط الوسط السني الحالي', facial: 'تصحيح إلى خط وسط الوجه' } : { current: 'خط الوسط السني الحالي' }, 'خط الوسط'),
    A.rest && A.rest.display != null && !A.rest.hidden && A.mm && A.status['11'] !== 'missing' ? sel_('lengthMode', { ratio: 'من نسبة العرض إلى الطول', rest: 'من ظهورها في الراحة' }, 'طول الثنية') : null,
    A.mm ? numIn('centralWidthMm', 'عرض الثنية المطلوب (ملم، اختياري)', '0.1', 'تلقائي') : null,
    A.mm ? numIn('centralLenMm', 'طول الثنية المطلوب (ملم، اختياري)', '0.1', 'تلقائي') : null));
  if (d.spaceMode === 'preserve' && (d.centralWidthMm)) opt.append(h('div', { class: 'muted small' }, 'ملاحظة: عرض الثنية المطلوب يُطبّق فقط في وضع «البدء من عرض الثنية».'));
  const shapeChips = h('div', { class: 'chips' }, Object.entries(SHAPES).map(([k, l]) => h('button', { class: 'chip' + (d.shape === k ? ' on' : ''), onclick: () => { d.shape = k; changed(true); } }, l)));
  const shade = h('select', null, Object.keys(C.SHADES).map((k) => h('option', { value: k, selected: d.shade === k }, k)));
  shade.addEventListener('change', () => { d.shade = shade.value; changed(); });
  const mat = h('select', null, MATERIALS.map((k) => h('option', { value: k, selected: d.material === k }, k)));
  mat.addEventListener('change', () => { d.material = mat.value; changed(); });
  const libSel = h('select', null, h('option', { value: '' }, 'رسم مولَّد (بلا مكتبة)'), lib.map((s) => h('option', { value: s.id, selected: d.libraryId === s.id }, 'مكتبة: ' + s.name)));
  libSel.addEventListener('change', async () => { d.libraryId = libSel.value || null; if (d.libraryId) await preloadLibrarySet(d.libraryId); changed(); });
  opt.append(h('h3', null, 'شكل الأسنان'), shapeChips, h('label', { class: 'field', style: { marginTop: '10px' } }, h('span', null, 'مصدر الأسنان في المحاكاة'), libSel),
    !lib.length ? h('div', { class: 'muted small' }, 'لا توجد مكتبة أسنان بعد: احفظ أسنان حالة جميلة من خطوة «الأسنان» لتظهر هنا.') : null, h('div', { class: 'grid2', style: { marginTop: '10px' } }, h('label', { class: 'field' }, h('span', null, 'اللون (VITA، تقريبي للعرض)'), shade), h('label', { class: 'field' }, h('span', null, 'المادة'), mat)));
  const treatGrid = h('div', { class: 'grid2' }, C.TEETH.map((t) => {
    const el = h('select', null, Object.entries(TREAT).map(([k, l]) => h('option', { value: k, selected: d.treat[t] === k }, l)));
    el.addEventListener('change', () => { d.treat[t] = el.value; changed(); });
    const ts = A.status[t]; return h('label', { class: 'field' }, h('span', null, t + ' ' + C.TOOTH_NAME[t] + (ts && ts !== 'present' ? ' — ' + STATUS[ts] : '')), el);
  }));
  opt.append(h('h3', null, 'العلاج لكل سن'), treatGrid);
  body.append(opt);

  const stCard = h('div', { class: 'card' }, h('h2', null, 'التصميم فوق الصورة'));
  const st = makeStage(img, {
    onDown(pt, target) {
      if (view !== 'outline') return null;
      const hk = target.dataset?.h, tk = target.dataset?.t;
      if (tk && !hk) { sel = tk; draw(); return null; }
      if (!hk) return null;
      const t = sel; if (d.treat[t] === 'none') return null;
      const b = d.override[t] ? d.override[t] : (d.override[t] = { ...A.design.target[t] });
      return { move(q) {
        if (hk === 'x0') b.x0 = Math.min(q.x, b.x1 - 6); if (hk === 'x1') b.x1 = Math.max(q.x, b.x0 + 6);
        if (hk === 'y0') b.y0 = Math.min(q.y, b.y1 - 6); if (hk === 'y1') b.y1 = Math.max(q.y, b.y0 + 6);
        A = computeAll(c); draw();
      }, up() { changed(); } };
    }
  });
  const ctx = st.base.getContext('2d');
  const draw = () => {
    if (view === 'sim') drawSim(ctx, img, A.design.target, d, c.intra.teeth); else ctx.drawImage(img, 0, 0);
    st.svg.innerHTML = '';
    if (view !== 'outline') return;
    const T = c.intra.teeth, sw = st.r(1.5);
    for (const t of C.TEETH) {
      const a = T[t], b = A.design.target[t];
      st.svg.append(s('rect', { x: a.x0, y: a.y0, width: a.x1 - a.x0, height: a.y1 - a.y0, fill: 'none', stroke: TOOTH_COLORS[t], 'stroke-width': sw, 'stroke-dasharray': st.r(5) + ' ' + st.r(4), opacity: 0.8 }));
      if (d.treat[t] === 'none') continue;
      st.svg.append(s('path', { d: C.toothPath(d.shape, b, t === '13' || t === '23'), fill: t === sel ? 'rgba(255,255,255,.22)' : 'rgba(255,255,255,.08)', stroke: '#fff', 'stroke-width': st.r(t === sel ? 3 : 2), 'data-t': t }));
      st.svg.append(s('text', { x: (b.x0 + b.x1) / 2, y: b.y1 + st.r(16), fill: '#fff', 'font-size': st.r(13), 'text-anchor': 'middle', 'font-weight': 700, 'paint-order': 'stroke', stroke: '#000', 'stroke-width': st.r(3), 'pointer-events': 'none' }, t + (d.override[t] ? '*' : '')));
    }
    const mid = A.design.prop.mid, am = A.design.prop.actualMid;
    st.svg.append(s('line', { x1: mid, x2: mid, y1: 0, y2: st.H, stroke: '#ff2d95', 'stroke-width': st.r(1.5) }));
    if (Math.abs(mid - am) > 1) st.svg.append(s('line', { x1: am, x2: am, y1: 0, y2: st.H, stroke: '#ff2d95', 'stroke-width': st.r(1), 'stroke-dasharray': st.r(4) + ' ' + st.r(4), opacity: 0.6 }));
    const b = A.design.target[sel];
    if (b && d.treat[sel] !== 'none') {
      const mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
      for (const [k, x, y] of [['x0', b.x0, my], ['x1', b.x1, my], ['y0', mx, b.y0], ['y1', mx, b.y1]]) st.svg.append(s('circle', { cx: x, cy: y, r: st.r(12), fill: '#fff', stroke: '#ff2d95', 'stroke-width': st.r(3), 'data-h': k }));
    }
  };
  st.draw = draw;
  const viewChips = h('div', { class: 'chips' });
  const drawViewChips = () => { viewChips.innerHTML = ''; for (const [k, l] of [['outline', 'المخطط (قابل للتعديل)'], ['sim', 'محاكاة بعد'], ['before', 'قبل']]) viewChips.append(h('button', { class: 'chip' + (view === k ? ' on' : ''), onclick: () => { view = k; drawViewChips(); draw(); } }, l)); };
  drawViewChips();
  const nOv = Object.keys(d.override || {}).length;
  focusTeeth(st, c.intra.teeth);
  stCard.append(viewChips, h('div', { class: 'toolbar' }, ...zoomBar(st), nOv ? h('button', { class: 'btn sm', onclick: () => { d.override = {}; changed(true); } }, `إلغاء التعديلات اليدوية (${nOv})`) : null,
    h('span', { class: 'muted small' }, 'المستطيلات المتقطعة = الوضع الحالي · الأبيض = المقترح · الوردي = خط الوسط')), st.wrap);
  body.append(stCard);
  draw();

  const chg = h('div', { class: 'card' }); body.append(chg);
  const drawChanges = () => {
    chg.innerHTML = '';
    const mm = A.mm, D = A.design;
    const blocks = A.safety.filter((x) => x.level === 'block' && x.text.includes('مفقود'));
    chg.append(h('h2', null, 'التغييرات المقترحة لكل سن'), h('div', null, blocks.map((b) => h('div', { class: 'issue fail' }, b.text))), h('div', null, !mm ? h('div', { class: 'issue warn' }, 'بدون معايرة: التغييرات بالبكسل وتصلح للمقارنة فقط. عاير الصورة لإرسال مقاسات للمعمل.') : null),
      h('div', { class: 'tablewrap' }, h('table', { class: 't' }, h('thead', null, h('tr', null, h('th', null, 'السن'), h('th', null, 'العلاج'), h('th', null, 'المقترح'), h('th', null, 'التغيير'))),
        h('tbody', null, C.TEETH.map((t) => { const ch = D.changes[t]; return h('tr', null, h('td', null, h('b', null, t)), h('td', null, TREAT[d.treat[t]]),
          h('td', null, mm ? `ع ${num(ch.wmm, 1)} · ط ${num(ch.hmm, 1)} ملم` : `ع ${num(ch.w, 0)} · ط ${num(ch.h, 0)} px`),
          h('td', null, d.treat[t] === 'none' ? '—' : A.status[t] === 'missing' ? 'سن جديد مكان الفراغ' : C.describeChange(ch, !!mm))); })))),
      D.restLen ? h('div', { class: 'issue info' }, `طول الثنية من الراحة: يظهر حالياً ${num(D.restLen.display, 1)} ملم، والهدف ${D.restLen.goal} ملم ← طول مستهدف ${num(D.restLen.target, 1)} ملم (${D.restLen.delta >= 0 ? 'إطالة' : 'تقصير'} ${num(Math.abs(D.restLen.delta), 1)} ملم).`) : null,
      h('div', { class: 'muted small' }, `النسبة المطبقة: ${num(D.prop.p, 3)} · عرض/طول الثنية: ${num(D.measAfter.ratios.central_wl, 2)}${D.facialX != null ? ' · خط الوسط مصحح إلى الوجه' : ''}`),
      rulesTable(A.rulesNow.filter((r) => r.group !== 'face'), 'مطابقة القواعد قبل وبعد', D.rulesAfter));
  };
  drawChanges();
  const notes = h('div', { class: 'card' }, h('h2', null, 'خطة العلاج وملاحظات المعمل'));
  for (const [k, l] of [['plan', 'خطة العلاج (للطبيب والمريض)'], ['labNotes', 'ملاحظات للمعمل'], ['notes', 'ملاحظات داخلية']]) {
    const ta = h('textarea', null, d[k] || ''); ta.addEventListener('input', () => { d[k] = ta.value; saveSoon(c, { design: true }); });
    notes.append(h('label', { class: 'field' }, h('span', null, l), ta));
  }
  body.append(notes, nextBtn(c, 'output', 'التالي: الاعتماد والمخرجات ←'));
  function changed(rerender) { saveSoon(c, { design: true }); if (rerender) return setTimeout(render, 450); A = computeAll(c); draw(); drawChanges(); }
}

// ----- step 6: approval & outputs -----
async function stepOutput(body, c, p) {
  if (!c.intra.teeth) { body.append(h('div', { class: 'card empty' }, 'أكمل تحديد الأسنان والتصميم أولاً.')); return; }
  const A = computeAll(c);
  const blocking = A.safety.filter((x) => x.level === 'block');
  const ap = h('div', { class: 'card' }, h('h2', null, 'فحص السلامة والاعتماد'));
  if (!A.safety.length) ap.append(h('div', { class: 'issue ok' }, 'لا ملاحظات من فحص السلامة.'));
  for (const w of A.safety) ap.append(h('div', { class: 'issue ' + (w.level === 'block' ? 'fail' : 'warn') }, (w.level === 'block' ? '⛔ ' : '⚠ ') + w.text));
  if (c.approval) {
    ap.append(h('div', { class: 'issue ok' }, `معتمد من ${c.approval.by} في ${fmtDateTime(c.approval.at)}`), h('button', { class: 'btn', onclick: async () => { c.approval = null; await saveCase(c); render(); } }, 'إلغاء الاعتماد'));
  } else {
    const chk = h('input', { type: 'checkbox' });
    ap.append(h('label', { class: 'check' }, chk, h('span', null, 'راجعتُ حدود الأسنان والقياسات والتصميم بنفسي، وأعتمدها كقرار سريري مني.')),
      blocking.length ? h('p', { class: 'muted small' }, 'الاعتماد متوقف حتى تُحل البنود ⛔.') : null,
      h('button', { class: 'btn primary', disabled: !!blocking.length, onclick: async () => {
        if (!chk.checked) { toast('أكّد المراجعة أولاً'); return; }
        let by = S.doctorName; if (!by) { by = await promptBox('اسم الطبيب', 'اسم الطبيب المعتمد'); if (!by) return; S.doctorName = by; await saveSettings(); }
        c.approval = { by, at: Date.now() }; await saveCase(c); toast('اعتُمد التصميم'); render();
      } }, 'اعتماد التصميم'));
  }
  body.append(ap);
  const out = h('div', { class: 'card' }, h('h2', null, 'المخرجات'),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', onclick: () => printSheet(c, p) }, '🖨 Case Sheet (طباعة / PDF)'),
      h('a', { class: 'btn', href: '#/present/' + c.id }, '👁 عرض المريض (قبل/بعد)'),
      h('button', { class: 'btn', onclick: () => shareLab(c, p) }, '📤 مشاركة مع المعمل'),
      h('button', { class: 'btn', onclick: () => downloadImages(c, p) }, '⬇ تنزيل الصور')),
    !c.approval ? h('p', { class: 'muted small' }, 'المخرجات قبل الاعتماد تحمل علامة «مسودة غير معتمدة».') : null,
    h('p', { class: 'muted small' }, 'على أندرويد: من نافذة الطباعة اختر «حفظ بتنسيق PDF».'));
  body.append(out);
  body.append(h('div', { class: 'card' }, h('button', { class: 'btn danger', onclick: async () => {
    if (!await confirmBox('حذف الحالة', 'ستُحذف الحالة وصورها نهائياً من هذا الجهاز.', 'حذف', true)) return;
    await deleteCaseData(c); go('#/patient/' + c.patientId);
  } }, 'حذف الحالة')));
}
function teethRect(img, ...sets) {
  const bs = sets.flatMap((T) => T ? C.TEETH.map((t) => T[t]).filter(Boolean) : []);
  if (!bs.length) return null;
  let x0 = Math.min(...bs.map((b) => b.x0)), x1 = Math.max(...bs.map((b) => b.x1)), y0 = Math.min(...bs.map((b) => b.y0)), y1 = Math.max(...bs.map((b) => b.y1));
  const mx = (x1 - x0) * 0.18, my = (y1 - y0) * 0.45;
  x0 = Math.max(0, x0 - mx); x1 = Math.min(img.width, x1 + mx); y0 = Math.max(0, y0 - my); y1 = Math.min(img.height, y1 + my);
  return { x: x0 / img.width, y: y0 / img.height, w: (x1 - x0) / img.width, h: (y1 - y0) / img.height };
}
async function caseImages(c) {
  const img = await getCanvasForBlob(c.images.intraWork); const A = computeAll(c);
  if (c.design.libraryId) await preloadLibrarySet(c.design.libraryId);
  const r = teethRect(img, c.intra.teeth, A.design?.target);
  const cut = (cv) => cv && r ? cropCanvas(cv, r, 1600) : cv;
  const before = measureCanvas(img, c, A), after = A.design?.target ? simCanvas(img, c, A) : null;
  return { A, img: cut(img), before: cut(before), after: cut(after) };
}
async function printSheet(c, p) {
  const { A, before, after } = await caseImages(c);
  const mm = A.mm, d = c.design, D = A.design;
  const statusTxt = (st) => `<span class="st-${st}">${{ ok: 'مطابق', warn: 'قريب', fail: 'خارج', na: '—' }[st]}</span>`;
  const aMap = D ? Object.fromEntries(D.rulesAfter.map((r) => [r.id, r])) : {};
  const sz = (b) => mm ? `العرض ${num((b.x1 - b.x0) * mm, 1)} · الطول ${num((b.y1 - b.y0) * mm, 1)} ملم` : `نسبة ع/ط ${num((b.x1 - b.x0) / (b.y1 - b.y0), 2)}`;
  const L = (t) => `<bdi dir="ltr">${esc(t ?? '—')}</bdi>`;
  const faceRows = A.rulesNow.filter((r) => r.group === 'face' && r.status !== 'na');
  const html = `<div class="sheet" dir="rtl">
  <div class="hdr"><div><h1>${esc(S.clinicName)}</h1><div>Case Sheet — تصميم ابتسامة رقمي</div></div><div style="text-align:left">${fmtDate(Date.now())}<br>رقم الحالة: ${esc(c.id.slice(0, 8))}</div></div>
  ${!c.approval ? '<div class="warn"><b>مسودة غير معتمدة</b> — لا تُنفّذ قبل اعتماد الطبيب.</div>' : ''}
  ${!mm ? '<div class="warn">الصورة غير معايرة: المقاسات بالنسب فقط. يلزم قياس أو نموذج دراسي لتحديد الأبعاد الفعلية.</div>' : ''}
  <h2>المريض</h2><table><tr><th>الاسم</th><td>${esc(p?.name)}</td><th>رقم الملف</th><td>${esc(p?.fileNo || '—')}</td></tr><tr><th>العمر</th><td>${esc(p?.age || '—')}</td><th>الجنس</th><td>${esc(p?.sex || '—')}</td></tr></table>
  <h2>الصور</h2><div class="imgs"><div><b>الوضع الحالي والقياسات</b><img src="${before.toDataURL('image/jpeg', 0.85)}"></div>${after ? `<div><b>التصميم المقترح (محاكاة)</b><img src="${after.toDataURL('image/jpeg', 0.85)}"></div>` : ''}</div>
  <h2>مواصفات التلبيسات</h2>
  <table><tr><th>السن</th><th>الحالة</th><th>العلاج</th><th>الحالي</th><th>المقترح</th><th>التغيير</th></tr>
  ${C.TEETH.map((t) => `<tr><td><b>${t}</b> ${C.TOOTH_NAME[t]}</td><td>${STATUS[A.status[t] || 'present']}</td><td>${TREAT[d.treat[t]]}</td><td>${A.status[t] === 'missing' ? 'فراغ: ' + sz(c.intra.teeth[t]) : sz(c.intra.teeth[t])}</td><td>${D?.target ? sz(D.target[t]) : '—'}</td><td>${D?.changes && d.treat[t] !== 'none' ? C.describeChange(D.changes[t], !!mm) : '—'}</td></tr>`).join('')}
  </table>
  <table style="margin-top:6px"><tr><th>الشكل</th><td>${SHAPES[d.shape]}</td><th>اللون</th><td>${esc(d.shade)}</td><th>المادة</th><td>${esc(d.material)}</td></tr>
  <tr><th>التناسب</th><td>${D ? num(D.prop.p, 3) : '—'}</td><th>ع/ط الثنية</th><td>${D ? num(D.measAfter.ratios.central_wl, 2) : '—'}</td><th>اللثة</th><td>${d.gingivaMode === 'keep' ? 'بلا تعديل' : 'خط لثة مثالي'}</td></tr></table>
  <h2>مطابقة القواعد</h2><table><tr><th>القاعدة</th><th>الحالي</th><th>بعد التصميم</th><th>المرجع</th></tr>
  ${A.rulesNow.filter((r) => r.group !== 'face').map((r) => `<tr><td>${esc(r.label)}</td><td>${L(r.valueText)} ${statusTxt(r.status)}</td><td>${aMap[r.id] ? L(aMap[r.id].valueText) + ' ' + statusTxt(aMap[r.id].status) : '—'}</td><td>${L(r.target || '')}</td></tr>`).join('')}</table>
  ${faceRows.length ? `<h2>تحليل الوجه والابتسامة</h2><table><tr><th>البند</th><th>القيمة</th><th>المرجع</th></tr>${faceRows.map((r) => `<tr><td>${esc(r.label)}${r.msg ? ' — ' + esc(r.msg) : ''}</td><td>${L(r.valueText)} ${statusTxt(r.status)}</td><td>${L(r.target || '')}</td></tr>`).join('')}</table><div>المقياس: ${esc(A.face?.scaleSource || '—')}</div>` : ''}
  ${A.rest && A.rest.display != null ? `<h2>الشفاه في الراحة</h2><div>ظهور الثنية: ${A.rest.hidden ? 'مغطاة بالشفة' : L(num(A.rest.display, 1) + ' mm')}${A.rest.approx ? ' (مقياس تقريبي)' : ''}${D?.restLen ? ` — طول الثنية المستهدف ${L(num(D.restLen.target, 1) + ' mm')}` : ''}</div>` : ''}
  ${d.plan ? `<h2>خطة العلاج</h2><div style="white-space:pre-wrap">${esc(d.plan)}</div>` : ''}
  ${d.labNotes ? `<h2>ملاحظات للمعمل</h2><div style="white-space:pre-wrap">${esc(d.labNotes)}</div>` : ''}
  <div class="sig"><div>الطبيب: ${esc(c.approval?.by || S.doctorName || '................')}<br>${c.approval ? 'اعتُمد: ' + fmtDateTime(c.approval.at) : 'غير معتمد'}</div><div>التوقيع: ....................</div></div>
  <p style="color:#666;font-size:10px;margin-top:14px">المحاكاة توضيحية وليست النتيجة النهائية. القياسات من صورة ثنائية الأبعاد: العرض المقاس للأنياب والرباعيات عرض ظاهري يتأثر بانحناء القوس. القرار السريري للطبيب.</p></div>`;
  let pr = $('#print-root'); if (!pr) { pr = h('div', { id: 'print-root' }); document.body.append(pr); }
  pr.innerHTML = html;
  await Promise.all([...pr.querySelectorAll('img')].map((im) => im.decode().catch(() => {})));
  window.__lastSheet = html;
  if (!window.__DSD_NO_PRINT) window.print();
}
async function shareLab(c, p) {
  const { A, before, after } = await caseImages(c);
  const mm = A.mm, d = c.design;
  const lines = [`${S.clinicName} — ${p?.name || ''} (${p?.fileNo || ''})`, c.approval ? `معتمد: ${c.approval.by}` : 'مسودة غير معتمدة', `الشكل: ${SHAPES[d.shape]} · اللون: ${d.shade} · المادة: ${d.material}`];
  for (const t of C.TEETH) { if (d.treat[t] === 'none') continue; const ch = A.design.changes[t]; lines.push(`${t}${A.status[t] === 'missing' ? ' (مفقود)' : ''} ${TREAT[d.treat[t]]}: ${mm ? num(ch.wmm, 1) + '×' + num(ch.hmm, 1) + ' ملم' : 'نسب فقط'}`); }
  if (d.labNotes) lines.push('ملاحظات: ' + d.labNotes);
  const text = lines.join('\n');
  const files = [];
  for (const [n, cv] of [['before', before], ['design', after]]) if (cv) files.push(new File([await canvasToBlob(cv, 'image/jpeg', 0.9)], `${n}-${(p?.fileNo || c.id.slice(0, 6))}.jpg`, { type: 'image/jpeg' }));
  try {
    if (navigator.canShare && navigator.canShare({ files })) { await navigator.share({ files, text, title: 'Case ' + (p?.name || '') }); return; }
  } catch (e) { if (e.name === 'AbortError') return; }
  try { await navigator.clipboard.writeText(text); toast('نُسخ النص — والصور ستُنزَّل الآن'); } catch (e) { /* ignore */ }
  for (const f of files) downloadBlob(f, f.name);
}
async function downloadImages(c, p) {
  const { before, after } = await caseImages(c);
  downloadBlob(await canvasToBlob(before), `before-${p?.fileNo || c.id.slice(0, 6)}.jpg`);
  if (after) downloadBlob(await canvasToBlob(after), `design-${p?.fileNo || c.id.slice(0, 6)}.jpg`);
}
function downloadBlob(blob, name) { const u = URL.createObjectURL(blob); const a = h('a', { href: u, download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 4000); }

// ---------- presentation ----------
async function viewPresent(root, id) {
  const c = await DB.get('cases', id); if (!c || !c.intra.teeth) return go('#/');
  const { after, img } = await caseImages(c);
  if (!after) return go(`#/case/${id}/design`);
  const before = canvasOf(img.width, img.height); before.getContext('2d').drawImage(img, 0, 0);
  const afterWrap = h('div', { class: 'after' }, after);
  const handle = h('div', { class: 'handle' });
  const ba = h('div', { class: 'ba' }, before, afterWrap, handle, h('span', { class: 'lbl', style: { left: '10px' } }, 'بعد'), h('span', { class: 'lbl', style: { right: '10px' } }, 'قبل'));
  const view = h('div', { class: 'view' }, ba);
  const set = (k) => { k = clamp(k, 0, 1); afterWrap.style.width = (k * 100) + '%'; handle.style.left = (k * 100) + '%'; };
  const fit = () => { const r = before.getBoundingClientRect(); after.style.width = r.width + 'px'; after.style.height = r.height + 'px'; };
  let dragging = false;
  const pos = (e) => { const r = before.getBoundingClientRect(); set((e.clientX - r.left) / r.width); };
  view.addEventListener('pointerdown', (e) => { dragging = true; view.setPointerCapture(e.pointerId); pos(e); });
  view.addEventListener('pointermove', (e) => { if (dragging) pos(e); });
  view.addEventListener('pointerup', () => { dragging = false; });
  view.style.touchAction = 'none';
  const el = h('div', { class: 'present' }, h('div', { class: 'bar' }, h('a', { class: 'iconbtn', href: `#/case/${id}/output`, 'aria-label': 'إغلاق', style: { textDecoration: 'none' } }, '✕'), h('b', null, 'تصميم ابتسامتك المقترح')), view,
    h('div', { class: 'note' }, 'تصميم مقترح للتوضيح وليس النتيجة النهائية. النتيجة الفعلية تعتمد على حالة الأسنان واللثة وخطة العلاج.'));
  root.append(el);
  requestAnimationFrame(() => { fit(); set(0.5); });
  window.addEventListener('resize', fit); routeCleanup = () => window.removeEventListener('resize', fit);
}

// ---------- settings ----------
async function viewSettings(root) {
  const main = h('main'); root.append(topbar('الإعدادات', '#/'), main);
  const txt = (k, label) => { const i = h('input', { type: 'text', value: S[k] || '' }); i.addEventListener('change', async () => { S[k] = i.value; await saveSettings(); }); return h('label', { class: 'field' }, h('span', null, label), i); };
  const theme = h('select', null, [['auto', 'حسب الجهاز'], ['light', 'فاتح'], ['dark', 'داكن']].map(([v, l]) => h('option', { value: v, selected: S.theme === v }, l)));
  theme.addEventListener('change', async () => { S.theme = theme.value; applyTheme(); await saveSettings(); });
  main.append(h('div', { class: 'card' }, h('h2', null, 'العيادة'), txt('clinicName', 'اسم العيادة'), txt('doctorName', 'اسم الطبيب'), h('label', { class: 'field' }, h('span', null, 'المظهر'), theme)));

  // model
  const mStatus = h('div', { class: 'muted small' }, 'جارٍ الفحص…');
  (async () => { const m = await getModelBuffer().catch(() => null); mStatus.textContent = m ? `النموذج متوفر (${m.src}، ${(m.buf.byteLength / 1048576).toFixed(1)} ميغابايت)` : 'النموذج غير محمّل — الكشف التلقائي غير متاح، والتحديد اليدوي يعمل.'; })();
  const file = h('input', { type: 'file', accept: '.onnx', class: 'hidden' });
  file.addEventListener('change', async () => {
    const f = file.files[0]; if (!f) return;
    if (f.size < 100000) { toast('الملف صغير جداً — ليس نموذج ONNX'); return; }
    await DB.put('blobs', { id: 'model', blob: f, type: 'application/octet-stream' }); ortSession = null; toast('حُفظ النموذج على الجهاز'); render();
  });
  const numSet = (k, label, step) => { const i = h('input', { type: 'number', step, value: S[k] }); i.addEventListener('change', async () => { S[k] = +i.value; ortSession = null; await saveSettings(); }); return h('label', { class: 'field' }, h('span', null, label), i); };
  const boolSet = (k, label) => { const i = h('input', { type: 'checkbox', checked: !!S[k] }); i.addEventListener('change', async () => { S[k] = i.checked; await saveSettings(); }); return h('label', { class: 'check' }, i, h('span', null, label)); };
  main.append(h('div', { class: 'card' }, h('h2', null, 'نموذج كشف الأسنان (ONNX)'), mStatus,
    h('p', { class: 'muted small' }, `حمّل ملف ${MODEL_FILE} مرة واحدة (U-Net + MobileNetV2، 7 أصناف). يلزم إنترنت أول مرة لتحميل محرك ONNX Runtime ثم يعمل بلا اتصال.`),
    h('div', { class: 'row' }, file, h('button', { class: 'btn primary', onclick: () => file.click() }, 'تحميل ملف النموذج'),
      h('button', { class: 'btn danger', onclick: async () => { await DB.del('blobs', 'model'); ortSession = null; toast('حُذف'); render(); } }, 'حذف النموذج من الجهاز')),
    h('div', { class: 'grid2', style: { marginTop: '10px' } }, numSet('modelSize', 'حجم مدخل النموذج', '32'), numSet('confMin', 'حد الثقة للتنبيه', '0.01')),
    boolSet('useTTA', 'استخدام TTA (أدق وأبطأ مرتين)'), boolSet('autoStraighten', 'تصحيح ميل الصورة تلقائياً بعد الكشف')));

  const wf = h('button', { class: 'btn sm', onclick: () => warmUp('face', wf) }, 'تحميل كاشف الوجه');
  const ws = h('button', { class: 'btn sm', onclick: () => warmUp('sam', ws) }, 'تحميل نموذج التحديد بنقرة');
  const wo = h('button', { class: 'btn sm', onclick: () => warmUp('onnx', wo) }, 'تجهيز نموذج الأسنان');
  main.append(h('div', { class: 'card' }, h('h2', null, 'أدوات مساعدة إضافية'),
    h('p', { class: 'muted small' }, 'كاشف نقاط الوجه (MediaPipe Face Landmarker، نحو 4 ميغابايت) ونموذج التحديد بنقرة (SlimSAM، عشرات الميغابايت) يُحمَّلان من الإنترنت أول مرة ثم يعملان بلا اتصال. حمّلهما الآن على شبكة الواي فاي قبل استخدامهما في العيادة.'),
    h('div', { class: 'row' }, wo, wf, ws), boolSet('autoCrop', 'قص تلقائي على الأسنان بعد رفع الصورة')));
  main.append(h('div', { class: 'card' }, h('h2', null, 'مكتبة الأسنان الطبيعية'), h('p', { class: 'muted small' }, 'صور أسنان حقيقية من حالات جميلة تُستخدم في المحاكاة بدل الرسم المولَّد. احفظ فقط بموافقة المريض.'), libraryManager()));
  // design defaults
  const dSel = (k, opts, label) => { const i = h('select', null, Object.entries(opts).map(([v, l]) => h('option', { value: v, selected: S.design[k] === v }, l))); i.addEventListener('change', async () => { S.design[k] = i.value; await saveSettings(); }); return h('label', { class: 'field' }, h('span', null, label), i); };
  main.append(h('div', { class: 'card' }, h('h2', null, 'القيم الافتراضية للحالات الجديدة'), h('div', { class: 'grid2' },
    dSel('proportion', Object.fromEntries(Object.entries(C.PROPORTIONS).map(([k, v]) => [k, v.label])), 'التناسب'), dSel('shape', SHAPES, 'الشكل'),
    dSel('shade', Object.fromEntries(Object.keys(C.SHADES).map((k) => [k, k])), 'اللون'), dSel('material', Object.fromEntries(MATERIALS.map((m) => [m, m])), 'المادة'),
    numSet('ipdMm', 'مسافة العينين الافتراضية (ملم) عند غياب معايرة', '0.5'))));

  // rules
  const rc = h('div', { class: 'card' }, h('h2', null, 'قواعد التصميم'), h('p', { class: 'muted small' }, 'قيم افتراضية من الأدبيات. عدّلها حسب معاييرك؛ تُطبّق على كل الحالات فوراً.'));
  const tb = h('tbody');
  const cell = (r, k, step) => { if (r[k] === undefined) return h('td', { class: 'muted' }, '—'); const i = h('input', { type: 'number', step, value: r[k] ?? '', style: { minWidth: '70px' } }); i.addEventListener('change', async () => { r[k] = i.value === '' ? null : +i.value; await saveSettings(); }); return h('td', null, i); };
  for (const r of S.rules) {
    const en = h('input', { type: 'checkbox', checked: r.enabled, 'aria-label': 'تفعيل' }); en.addEventListener('change', async () => { r.enabled = en.checked; await saveSettings(); });
    tb.append(h('tr', null, h('td', null, en), h('td', null, r.label, r.note ? h('div', { class: 'muted small' }, r.note) : null, h('div', { class: 'muted small' }, { ratio: 'نسبة', pct: 'نسبة فرق (0.05 = 5%)', mm: 'ملم', deg: 'درجة', bool: 'نعم/لا', class: 'تصنيف' }[r.unit] || '')), cell(r, 'min', '0.01'), cell(r, 'max', '0.01'), cell(r, 'target', '0.01'), cell(r, 'tol', '0.01')));
  }
  rc.append(h('div', { class: 'tablewrap' }, h('table', { class: 't' }, h('thead', null, h('tr', null, h('th', null, ''), h('th', null, 'القاعدة'), h('th', null, 'أدنى'), h('th', null, 'أعلى'), h('th', null, 'الهدف'), h('th', null, 'تسامح'))), tb)),
    h('div', { class: 'row', style: { marginTop: '10px' } },
      h('button', { class: 'btn', onclick: async () => { if (await confirmBox('استعادة القواعد', 'إرجاع كل القواعد إلى القيم الافتراضية؟')) { S.rules = C.clone(C.DEFAULT_RULES); await saveSettings(); render(); } } }, 'استعادة الافتراضي'),
      h('button', { class: 'btn', onclick: async () => {
        const ta = h('textarea', { style: { minHeight: '260px', direction: 'ltr', fontFamily: 'monospace', fontSize: '12px' } }, JSON.stringify(S.rules, null, 1));
        const v = await modal('القواعد (JSON)', ta, [{ label: 'إلغاء', value: null }, { label: 'حفظ', value: () => ta.value, primary: true }]);
        if (v == null) return;
        try { const arr = JSON.parse(v); if (!Array.isArray(arr) || !arr.every((r) => r.id && r.label)) throw 0; S.rules = arr; await saveSettings(); render(); } catch (e) { toast('JSON غير صالح'); }
      } }, 'تحرير/نسخ JSON')));
  main.append(rc);

  // security & backup
  const sec = h('div', { class: 'card' }, h('h2', null, 'الأمان والنسخ الاحتياطي'),
    h('p', { class: 'muted small' }, 'رمز الدخول يمنع فتح التطبيق، لكنه لا يشفّر البيانات داخل المتصفح. احمِ الجهاز نفسه بقفل شاشة. النسخة الاحتياطية مشفّرة بكلمة مرور (AES-256).'),
    h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: async () => {
        const a = await promptBox('رمز الدخول', 'رمز جديد (4 أرقام على الأقل)، اتركه فارغاً للإلغاء', 'password'); if (a == null) return;
        if (!a) { S.pin = null; await saveSettings(); toast('أُلغي رمز الدخول'); return; }
        if (a.length < 4) { toast('الرمز قصير'); return; }
        const b = await promptBox('تأكيد الرمز', 'أعد كتابة الرمز', 'password'); if (a !== b) { toast('الرمزان غير متطابقين'); return; }
        const salt = crypto.getRandomValues(new Uint8Array(16)); S.pin = { salt: b64.enc(salt), hash: await hashPin(a, salt) }; await saveSettings(); toast('فُعّل رمز الدخول');
      } }, S.pin ? 'تغيير/إلغاء رمز الدخول' : 'تفعيل رمز الدخول'),
      h('button', { class: 'btn primary', onclick: exportBackup }, '⬇ تصدير نسخة احتياطية مشفّرة'),
      h('button', { class: 'btn', onclick: importBackup }, '⬆ استيراد نسخة'),
      h('button', { class: 'btn', onclick: async () => { const ok = await navigator.storage?.persist?.(); toast(ok ? 'فُعّل التخزين الدائم' : 'رفض المتصفح التخزين الدائم — ثبّت التطبيق على الشاشة الرئيسية وحاول مجدداً'); } }, 'طلب تخزين دائم')),
    storageCard());
  main.append(sec);
  main.append(h('div', { class: 'card muted small' }, h('h2', null, 'حول النظام'),
    h('p', null, `الإصدار ${APP_VERSION}. أداة مساعدة لتحليل وتصميم الابتسامة، وليست جهاز تشخيص. كل اقتراح يراجعه الطبيب ويعتمده.`),
    h('p', null, 'نموذج الكشف: U-Net بمشفّر MobileNetV2، دُرّب على مجموعات Roboflow مفتوحة (CC BY 4.0): Teeth-detection-FDI-System (rizka-milandga-2k04u)، tooth_frontal (gradproject-g265b)، teeth (tooth-yqbyw). Dice 0.956 على 20 صورة اختبار من نفس المصادر؛ الدقة على صور العيادة لم تُقَس بعد مقابل قياسات الفرجار.'),
    h('p', null, 'القيم الافتراضية للقواعد من أدبيات تصميم الابتسامة (نسبة عرض/طول الثنية 75–85%، النسبة الذهبية وRED، خط الوسط، ممر الشدق، قوس الابتسامة) ويُفترض أن يراجعها الطبيب.'),
    h('button', { class: 'btn', onclick: async () => { S.disclaimerAccepted = false; await showDisclaimer(); } }, 'عرض التنبيه')));
}
async function exportBackup() {
  const pw = await promptBox('كلمة مرور النسخة', 'كلمة مرور قوية (8 أحرف على الأقل). بدونها لا يمكن فتح النسخة.', 'password'); if (!pw) return;
  if (pw.length < 8) { toast('كلمة المرور قصيرة'); return; }
  const pw2 = await promptBox('تأكيد', 'أعد كتابة كلمة المرور', 'password'); if (pw !== pw2) { toast('غير متطابقة'); return; }
  toast('جارٍ التجهيز…');
  const blobs = [];
  for (const r of await DB.all('blobs')) { if (r.id === 'model') continue; blobs.push({ id: r.id, type: r.type, b64: b64.enc(await r.blob.arrayBuffer()) }); }
  const settings = { ...S, pin: null };
  const plain = new TextEncoder().encode(JSON.stringify({ patients: await DB.all('patients'), cases: await DB.all('cases'), blobs, settings, at: Date.now(), v: APP_VERSION }));
  const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(pw, salt);
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, plain);
  const file = JSON.stringify({ format: 'dsd-backup', v: 1, kdf: 'PBKDF2-SHA256', iter: 250000, salt: b64.enc(salt), iv: b64.enc(iv), data: b64.enc(ct) });
  downloadBlob(new Blob([file], { type: 'application/octet-stream' }), `dsd-backup-${new Date().toISOString().slice(0, 10)}.dsdb`);
  await DB.setKV('lastBackup', Date.now()); toast('حُفظت النسخة — انقلها إلى مكان آمن خارج الجهاز');
}
async function importBackup() {
  const inp = h('input', { type: 'file', accept: '.dsdb,application/octet-stream,application/json' });
  inp.addEventListener('change', async () => {
    const f = inp.files[0]; if (!f) return;
    let j; try { j = JSON.parse(await f.text()); if (j.format !== 'dsd-backup') throw 0; } catch (e) { toast('ملف غير صالح'); return; }
    const pw = await promptBox('كلمة مرور النسخة', 'كلمة المرور', 'password'); if (!pw) return;
    let data;
    try { const key = await deriveKey(pw, b64.dec(j.salt), j.iter); data = JSON.parse(new TextDecoder().decode(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64.dec(j.iv) }, key, b64.dec(j.data)))); }
    catch (e) { toast('كلمة مرور خاطئة أو ملف تالف'); return; }
    if (!await confirmBox('استيراد', `ستُضاف/تُحدّث ${data.patients.length} مريض و${data.cases.length} حالة. السجلات بنفس المعرّف تُستبدل. متابعة؟`, 'استيراد')) return;
    for (const p of data.patients) await DB.put('patients', p);
    for (const c of data.cases) await DB.put('cases', c);
    for (const b of data.blobs) await DB.put('blobs', { id: b.id, type: b.type, blob: new Blob([b64.dec(b.b64)], { type: b.type }) });
    if (data.settings && await confirmBox('الإعدادات', 'استيراد الإعدادات والقواعد من النسخة أيضاً؟', 'نعم')) { const pin = S.pin; S = Object.assign(S, data.settings, { pin }); await saveSettings(); }
    imageCache.clear(); toast('تم الاستيراد'); go('#/');
  });
  inp.click();
}

// ---------- init ----------
(async function init() {
  try {
    await DB.open(); await loadSettings();
  } catch (e) { document.body.innerHTML = '<p style="padding:20px">تعذّر فتح قاعدة البيانات في هذا المتصفح. جرّب Chrome حديثاً خارج وضع التصفح الخفي.</p>'; return; }
  if ('serviceWorker' in navigator && location.protocol === 'https:' && !window.__NO_SW) navigator.serviceWorker.register('sw.js').catch(() => {});
  try { if (navigator.storage?.persist && !(await navigator.storage.persisted())) await navigator.storage.persist(); } catch (e) { /* ignore */ }
  await showLock();
  await render();
  showDisclaimer();
})();
