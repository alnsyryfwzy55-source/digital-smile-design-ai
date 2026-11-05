/* ===== part 2: case workspace ===== */
const STEPS = [
  { id: 'photos', label: '1 الصور والموافقة' }, { id: 'crop', label: '2 القص والاستقامة' },
  { id: 'teeth', label: '3 الأسنان والقياس' }, { id: 'face', label: '4 الوجه والابتسامة' },
  { id: 'rest', label: '5 الشفاه في الراحة' }, { id: 'design', label: '6 التصميم' }, { id: 'output', label: '7 الاعتماد والمخرجات' }
];
const TOOTH_COLORS = { '13': '#e6194b', '12': '#f58231', '11': '#ffe119', '21': '#3cb44b', '22': '#42d4f4', '23': '#4363d8' };
const TREAT = { veneer: 'فينير', crown: 'تلبيسة (تاج)', composite: 'كومبوزيت تجميلي', implant: 'زرعة + تاج', pontic: 'جسر: سن معلّق', abutment: 'جسر: تاج داعم', temp: 'مؤقت / طقم جزئي', none: 'بلا علاج' };
const STATUS = { present: 'موجود', missing: 'مفقود', broken: 'مكسور / متآكل' };
const statusOf = (c) => c.intra.status || {};
const MATERIALS = ['إيماكس (ليثيوم ديسيليكات)', 'زيركون', 'زيركون متعدد الطبقات', 'بورسلين على معدن', 'كومبوزيت', 'أخرى'];
const SHAPES = { square: 'مربع', oval: 'بيضاوي', triangular: 'مثلث' };

async function saveCase(c, { design = false } = {}) {
  c.updated = Date.now();
  if (design && c.approval) { c.approval = null; toast('عُدّل التصميم بعد اعتماده — يلزم اعتماد جديد'); }
  await DB.put('cases', c);
}
let saveTimer = null;
function saveSoon(c, opt) { clearTimeout(saveTimer); saveTimer = setTimeout(() => saveCase(c, opt), 400); }

function mmPerPx(c) {
  const cal = c.calib || {};
  if (cal.method === 'ruler' && cal.ruler && +cal.ruler.mm > 0) {
    const d = Math.hypot(cal.ruler.b.x - cal.ruler.a.x, cal.ruler.b.y - cal.ruler.a.y);
    return d > 3 ? +cal.ruler.mm / d : null;
  }
  if (cal.method === 'tooth' && cal.tooth && +cal.tooth.mm > 0 && c.intra.teeth && c.intra.teeth[cal.tooth.id]) {
    const b = c.intra.teeth[cal.tooth.id]; const w = b.x1 - b.x0;
    return w > 3 ? +cal.tooth.mm / w : null;
  }
  return null;
}
function faceResult(c, meas) {
  const pts = c.face?.points || {};
  if (!pts.pupilR || !pts.pupilL) return null;
  return C.faceAnalysis(pts, { centralLenMm: meas?.teeth?.['11']?.hmm || null, ipdMm: S.ipdMm });
}
function computeAll(c) {
  const teeth = c.intra.teeth;
  const mm = mmPerPx(c);
  const status = statusOf(c);
  const meas = teeth ? C.measure(teeth, mm, status) : null;
  const face = meas ? faceResult(c, meas) : (c.face?.points?.pupilR ? C.faceAnalysis(c.face.points, { ipdMm: S.ipdMm }) : null);
  const d = c.design;
  const pP = d.proportion === 'custom' ? +d.customP : (C.PROPORTIONS[d.proportion]?.p ?? 0.7);
  const rest = restResult(c, face);
  const rulesNow = meas ? C.evaluateRules(meas, S.rules, { proportionP: pP, teeth, face, rest }) : [];
  let restLen = null;
  if (d.lengthMode === 'rest' && rest && rest.display != null && !rest.hidden && mm && teeth?.['11'] && status['11'] !== 'missing') {
    const tgt = S.rules.find((r) => r.id === 'rest_display')?.target ?? 2;
    const cur = (teeth['11'].y1 - teeth['11'].y0) * mm;
    restLen = { current: cur, delta: tgt - rest.display, target: cur + (tgt - rest.display), display: rest.display, goal: tgt };
  }
  let design = null;
  if (teeth && C.TEETH.every((t) => teeth[t])) {
    let facialX = null;
    if (d.midline === 'facial' && face?.metrics?.midline_dev_px != null) {
      const fp = c.face.points;
      const facePx = fp.zenith11 && fp.incisal11 ? Math.hypot(fp.incisal11.x - fp.zenith11.x, fp.incisal11.y - fp.zenith11.y) : null;
      const ratio = facePx ? meas.teeth['11'].h / facePx : (face.scale && mm ? face.scale / mm : null);
      if (ratio) facialX = ((teeth['11'].x1 + teeth['21'].x0) / 2) - face.metrics.midline_dev_px * ratio;
    }
    const prop = C.proposeDesign(teeth, meas, S.rules, { ...d, facialMidlineX: facialX, centralWidthMm: +d.centralWidthMm || null, centralLenMm: +d.centralLenMm || (restLen ? restLen.target : null) });
    if (prop.ok) {
      const target = {}, changes = {};
      for (const t of C.TEETH) {
        if (d.treat[t] === 'none') target[t] = { ...teeth[t] };
        else target[t] = d.override?.[t] ? { ...d.override[t] } : { ...prop.target[t] };
        const a = teeth[t], b = target[t];
        const ch = { dW: (b.x1 - b.x0) - (a.x1 - a.x0), dL: (b.y1 - b.y0) - (a.y1 - a.y0), dIncisal: b.y1 - a.y1, dGingival: b.y0 - a.y0, w: b.x1 - b.x0, h: b.y1 - b.y0 };
        if (mm) for (const k of Object.keys(ch)) ch[k + 'mm'] = ch[k] * mm;
        changes[t] = ch;
      }
      const stAfter = {}; for (const t of C.TEETH) stAfter[t] = d.treat[t] === 'none' ? (status[t] || 'present') : 'present';
      const measAfter = C.measure(target, mm, stAfter);
      const rulesAfter = C.evaluateRules(measAfter, S.rules.filter((r) => r.group !== 'face'), { proportionP: pP, teeth: target });
      design = { prop, target, changes, measAfter, rulesAfter, facialX, restLen };
    } else design = { error: prop.error };
  }
  const out = { teeth, mm, meas, face, rest, rulesNow, design, pP, status };
  out.safety = C.safetyChecks({ teeth, status, treat: d.treat, reviewed: !!c.intra.reviewed, meas, detLevel: c.intra.det?.level, source: c.intra.source, rest, design, consent: !!c.consent.photo });
  return out;
}
function restResult(c, face) {
  const P = c.rest?.points; if (!P || !P.pupilR || !P.pupilL || !P.lipRest) return null;
  let ipdMm = S.ipdMm, measured = false;
  const fp = c.face?.points;
  if (face && face.scale && face.scaleSource && !face.scaleSource.includes('مفترضة') && fp?.pupilR && fp?.pupilL) { ipdMm = Math.hypot(fp.pupilL.x - fp.pupilR.x, fp.pupilL.y - fp.pupilR.y) * face.scale; measured = true; }
  return C.restAnalysis(P, { ipdMm, ipdMeasured: measured, hidden: !!c.rest.hidden });
}

// ---------- interactive stage ----------
function makeStage(srcCanvas, opts = {}) {
  const W = srcCanvas.width, H = srcCanvas.height;
  const wrap = h('div', { class: 'stagewrap' });
  const stage = h('div', { class: 'stage' });
  const base = canvasOf(W, H); base.getContext('2d').drawImage(srcCanvas, 0, 0);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('preserveAspectRatio', 'none');
  const extra = h('div', { style: { position: 'absolute', inset: 0, pointerEvents: 'none' } });
  stage.append(base, extra, svg); wrap.append(stage);
  let zoom = 1;
  const api = {
    W, H, wrap, stage, svg, base, extra,
    scale: () => (stage.clientWidth || 1) / W,
    r: (px) => px / api.scale(),
    toImg(e) { const b = svg.getBoundingClientRect(); return { x: (e.clientX - b.left) / b.width * W, y: (e.clientY - b.top) / b.height * H }; },
    setZoom(z) {
      const cx = (wrap.scrollLeft + wrap.clientWidth / 2) / (stage.clientWidth || 1), cy = (wrap.scrollTop + wrap.clientHeight / 2) / (stage.clientHeight || 1);
      zoom = Math.min(5, Math.max(1, z)); stage.style.width = (zoom * 100) + '%';
      wrap.scrollLeft = cx * stage.clientWidth - wrap.clientWidth / 2; wrap.scrollTop = cy * stage.clientHeight - wrap.clientHeight / 2;
      opts.onZoom?.(); api.draw?.();
    },
    zoom: () => zoom
  };
  let drag = null;
  svg.addEventListener('pointerdown', (e) => {
    if (e.button && e.button !== 0) return;
    const pt = api.toImg(e);
    const handler = opts.onDown ? opts.onDown(pt, e.target, e) : null;
    drag = handler ? { handler, id: e.pointerId } : { pan: true, id: e.pointerId, sx: e.clientX, sy: e.clientY, sl: wrap.scrollLeft, st: wrap.scrollTop, moved: 0 };
    svg.setPointerCapture(e.pointerId); e.preventDefault();
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag || drag.id !== e.pointerId) return;
    if (drag.handler) { drag.handler.move?.(api.toImg(e)); return; }
    const dx = e.clientX - drag.sx, dy = e.clientY - drag.sy; drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
    wrap.scrollLeft = drag.sl - dx; wrap.scrollTop = drag.st - dy;
  });
  const end = (e) => {
    if (!drag || drag.id !== e.pointerId) return;
    if (drag.handler) drag.handler.up?.(api.toImg(e));
    else if (drag.moved < 8) opts.onTap?.(api.toImg(e));
    drag = null;
  };
  svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
  return api;
}
function focusTeeth(st, teeth) {
  if (!teeth) return;
  const bs = C.TEETH.map((t) => teeth[t]).filter(Boolean); if (!bs.length) return;
  const x0 = Math.min(...bs.map((b) => b.x0)), x1 = Math.max(...bs.map((b) => b.x1)), y0 = Math.min(...bs.map((b) => b.y0)), y1 = Math.max(...bs.map((b) => b.y1));
  const z = clamp(0.8 * st.W / (x1 - x0), 1, 4);
  requestAnimationFrame(() => { st.stage.style.width = (z * 100) + '%'; st.setZoom(z);
    const sc = st.scale(); st.wrap.scrollLeft = ((x0 + x1) / 2) * sc - st.wrap.clientWidth / 2; st.wrap.scrollTop = ((y0 + y1) / 2) * sc - st.wrap.clientHeight / 2; });
}
function zoomBar(st) {
  return [h('button', { class: 'btn sm', onclick: () => st.setZoom(st.zoom() * 1.5), 'aria-label': 'تكبير' }, '＋'), h('button', { class: 'btn sm', onclick: () => st.setZoom(st.zoom() / 1.5), 'aria-label': 'تصغير' }, '－')];
}
const svgNS = 'http://www.w3.org/2000/svg';
function s(tag, attrs, text) { const e = document.createElementNS(svgNS, tag); for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v); if (text != null) e.textContent = text; return e; }

// ---------- case view ----------
async function viewCase(root, id, step) {
  const c = await DB.get('cases', id); if (!c) return go('#/');
  const p = await DB.get('patients', c.patientId);
  const done = { photos: !!c.images.intraOrig && c.consent.photo, crop: !!c.images.intraWork, teeth: !!c.intra.teeth, face: Object.keys(c.face.points || {}).length >= 5, rest: !!(c.rest?.points?.lipRest), design: !!c.intra.teeth, output: !!c.approval };
  const nav = h('nav', { class: 'steps', 'aria-label': 'خطوات الحالة' }, STEPS.map((st) => h('a', { href: `#/case/${id}/${st.id}`, class: (st.id === step ? 'on ' : '') + (done[st.id] ? 'done' : '') }, st.label)));
  const main = h('main');
  root.append(topbar((p?.name || '') + ' — ' + (c.title || 'حالة'), '#/patient/' + c.patientId), main);
  main.append(nav);
  const body = h('div'); main.append(body);
  const fn = { photos: stepPhotos, crop: stepCrop, teeth: stepTeeth, face: stepFace, rest: stepRest, design: stepDesign, output: stepOutput }[step] || stepPhotos;
  await fn(body, c, p);
  setTimeout(() => nav.querySelector('.on')?.scrollIntoView({ inline: 'center', block: 'nearest' }), 30);
}
function nextBtn(c, step, label = 'التالي ←', disabled = false) {
  return h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '8px' } }, h('button', { class: 'btn primary', disabled, onclick: () => go(`#/case/${c.id}/${step}`) }, label));
}

// ----- step 1: photos & consent -----
async function stepPhotos(body, c) {
  const photo = h('input', { type: 'checkbox', checked: c.consent.photo });
  const train = h('input', { type: 'checkbox', checked: c.consent.training });
  const upd = async () => { c.consent.photo = photo.checked; c.consent.training = train.checked; c.consent.at = Date.now(); await saveCase(c); render(); };
  photo.addEventListener('change', upd); train.addEventListener('change', upd);
  body.append(h('div', { class: 'card' }, h('h2', null, 'موافقة المريض'),
    h('label', { class: 'check' }, photo, h('span', null, 'حصلتُ على موافقة المريض على التصوير وحفظ الصور في ملفه لغرض العلاج.')),
    h('label', { class: 'check' }, train, h('span', null, 'وافق المريض (اختياري) على استخدام صوره مجهولة الهوية لتحسين النظام.')),
    c.consent.at ? h('div', { class: 'muted small' }, 'سُجّلت: ' + fmtDateTime(c.consent.at)) : null));
  const title = h('input', { type: 'text', value: c.title || '' });
  title.addEventListener('change', async () => { c.title = title.value; await saveCase(c); });
  body.append(h('div', { class: 'card' }, h('label', { class: 'field' }, h('span', null, 'عنوان الحالة'), title)));
  const imgCard = async (key, workKey, label, hint, required) => {
    const card = h('div', { class: 'card' }, h('h2', null, label), h('p', { class: 'muted small' }, hint));
    const url = c.images[workKey] || c.images[key];
    if (url) { const b = await getBlob(url); if (b) { const u = URL.createObjectURL(b); card.append(h('img', { class: 'thumb', src: u, alt: label })); } }
    const inp = h('input', { type: 'file', accept: 'image/*', class: 'hidden' });
    const cam = h('input', { type: 'file', accept: 'image/*', capture: 'environment', class: 'hidden' });
    const onFile = async (f) => {
      if (!f) return;
      if (!c.consent.photo) { toast('سجّل موافقة المريض أولاً'); return; }
      if (key === 'intraOrig' && c.intra.teeth && !await confirmBox('استبدال الصورة', 'ستُحذف حدود الأسنان والتصميم الحالي لهذه الصورة. متابعة؟', 'استبدال', true)) return;
      try { const bmp = await blobToBitmap(f, 2400); bmp.close?.(); } catch (e) { toast('تعذّر فتح الصورة'); return; }
      for (const k of [key, workKey]) if (c.images[k]) { await DB.del('blobs', c.images[k]); imageCache.delete(c.images[k]); c.images[k] = null; }
      c.images[key] = await putBlob(f);
      if (key === 'intraOrig') { c.crop = { q: 0, angle: 0, rect: { x: 0, y: 0, w: 1, h: 1 }, autoPending: true }; c.intra = { teeth: null, source: null, det: null, maskPng: null, status: {}, reviewed: false }; c.design.override = {}; }
      else if (key === 'restOrig') { c.restQ = 0; c.rest = { points: {}, hidden: false }; await makeFaceWork(c, 'rest'); }
      else { c.faceQ = 0; c.face = { points: {} }; await makeFaceWork(c); }
      await saveCase(c, { design: true });
      if (key === 'intraOrig') go(`#/case/${c.id}/crop`); else render();
    };
    inp.addEventListener('change', () => onFile(inp.files[0])); cam.addEventListener('change', () => onFile(cam.files[0]));
    card.append(h('div', { class: 'row', style: { marginTop: '10px' } }, inp, cam,
      h('button', { class: 'btn primary', disabled: !c.consent.photo, onclick: () => cam.click() }, '📷 تصوير'),
      h('button', { class: 'btn', disabled: !c.consent.photo, onclick: () => inp.click() }, '🖼 من المعرض'),
      !required && c.images[key] ? h('button', { class: 'btn danger', onclick: async () => { for (const k of [key, workKey]) if (c.images[k]) { await DB.del('blobs', c.images[k]); c.images[k] = null; } if (key === 'restOrig') c.rest = { points: {}, hidden: false }; else c.face = { points: {} }; await saveCase(c, { design: true }); render(); } }, 'حذف') : null));
    return card;
  };
  body.append(await imgCard('intraOrig', 'intraWork', 'صورة الأسنان الأمامية بمباعد الشفاه (أساسية)', 'الجوال أفقي، المباعد موضوع، الأسنان العلوية الأمامية كلها ظاهرة في وسط الصورة، والميل أقل من 10°. ضع مسطرة صغيرة أو معرفة عرض سن بالفرجار للقياس بالملم.', true));
  body.append(await imgCard('faceOrig', 'faceWork', 'صورة الوجه مع ابتسامة (اختيارية)', 'المريض ينظر للكاميرا مباشرة ويبتسم ابتسامة كاملة، والوجه كله ظاهر. تُستخدم لخط الوسط وخط العينين وقوس الابتسامة وظهور اللثة.', false));
  body.append(await imgCard('restOrig', 'restWork', 'صورة الوجه والشفاه في الراحة (اختيارية، مهمة لطول الثنية)', 'المريض ينظر للأمام، الشفاه مرتاحة ومفتوحة قليلاً بلا ابتسامة (كأنه قال «إيمّا» ثم ارتخى). تحدد كم يظهر من الثنية، وهو أهم مرجع لطولها.', false));
  body.append(nextBtn(c, 'crop', 'التالي: القص ←', !c.images.intraOrig));
}
async function makeFaceWork(c, kind = 'face') {
  const orig = kind === 'rest' ? 'restOrig' : 'faceOrig', work = kind === 'rest' ? 'restWork' : 'faceWork', q = kind === 'rest' ? (c.restQ || 0) : (c.faceQ || 0);
  const b = await getBlob(c.images[orig]); if (!b) return;
  const bmp = await blobToBitmap(b, 2400);
  const rc = rotatedCanvas(bmp, q * 90); bmp.close?.();
  const w = cropCanvas(rc, { x: 0, y: 0, w: 1, h: 1 }, 1600);
  if (c.images[work]) { await DB.del('blobs', c.images[work]); imageCache.delete(c.images[work]); }
  c.images[work] = await putBlob(await canvasToBlob(w));
}

// ----- step 2: crop & straighten -----
async function stepCrop(body, c) {
  if (!c.images.intraOrig) { body.append(h('div', { class: 'card empty' }, 'أضف صورة الأسنان أولاً.'), nextBtn(c, 'photos', '→ الصور')); return; }
  const bmp = await blobToBitmap(await getBlob(c.images.intraOrig), 2400);
  const src = bitmapToCanvas(bmp); bmp.close?.();
  const card = h('div', { class: 'card' }, h('h2', null, 'قص الصورة على الأسنان وتعديل الاستقامة'),
    h('p', { class: 'muted small' }, 'اسحب زوايا المربع ليحيط بالأسنان العلوية الأمامية مع هامش بسيط، ثم عدّل الميل حتى يكون خط الأسنان أفقياً.'));
  body.append(card);
  const holder = h('div'); card.append(holder);
  const angle = h('input', { type: 'range', min: -25, max: 25, step: 0.5, value: c.crop.angle, style: { width: '100%' }, 'aria-label': 'الميل' });
  const angleLbl = h('b', null, num(c.crop.angle, 1) + '°');
  let st, rc;
  const rect = { ...c.crop.rect };
  const build = () => {
    rc = rotatedCanvas(src, c.crop.q * 90 + c.crop.angle);
    holder.innerHTML = '';
    st = makeStage(rc, {
      onDown(pt, target) {
        const k = target.dataset?.h; if (!k) {
          const inside = pt.x > rect.x * rc.width && pt.x < (rect.x + rect.w) * rc.width && pt.y > rect.y * rc.height && pt.y < (rect.y + rect.h) * rc.height;
          if (!inside) return null;
          const o = { ...rect }, p0 = pt;
          return { move(q) { rect.x = clamp(o.x + (q.x - p0.x) / rc.width, 0, 1 - rect.w); rect.y = clamp(o.y + (q.y - p0.y) / rc.height, 0, 1 - rect.h); draw(); } };
        }
        return { move(q) {
          const x = clamp(q.x / rc.width, 0, 1), y = clamp(q.y / rc.height, 0, 1);
          let x0 = rect.x, y0 = rect.y, x1 = rect.x + rect.w, y1 = rect.y + rect.h;
          if (k.includes('l')) x0 = Math.min(x, x1 - 0.05); if (k.includes('r')) x1 = Math.max(x, x0 + 0.05);
          if (k.includes('t')) y0 = Math.min(y, y1 - 0.05); if (k.includes('b')) y1 = Math.max(y, y0 + 0.05);
          Object.assign(rect, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }); draw();
        } };
      }
    });
    holder.append(st.wrap);
    const draw = () => {
      const W = rc.width, H = rc.height, x0 = rect.x * W, y0 = rect.y * H, x1 = (rect.x + rect.w) * W, y1 = (rect.y + rect.h) * H, r = st.r(16);
      st.svg.innerHTML = '';
      st.svg.append(s('path', { d: `M0 0H${W}V${H}H0Z M${x0} ${y0}V${y1}H${x1}V${y0}Z`, fill: 'rgba(0,0,0,.55)', 'fill-rule': 'evenodd' }));
      st.svg.append(s('rect', { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: 'none', stroke: '#fff', 'stroke-width': st.r(2) }));
      for (let i = 1; i < 3; i++) {
        st.svg.append(s('line', { x1: x0, x2: x1, y1: y0 + (y1 - y0) * i / 3, y2: y0 + (y1 - y0) * i / 3, stroke: 'rgba(255,255,255,.5)', 'stroke-width': st.r(1) }));
        st.svg.append(s('line', { y1: y0, y2: y1, x1: x0 + (x1 - x0) * i / 3, x2: x0 + (x1 - x0) * i / 3, stroke: 'rgba(255,255,255,.5)', 'stroke-width': st.r(1) }));
      }
      for (const [k, x, y] of [['tl', x0, y0], ['tr', x1, y0], ['bl', x0, y1], ['br', x1, y1]]) st.svg.append(s('circle', { cx: x, cy: y, r, fill: '#fff', stroke: '#0f7b6c', 'stroke-width': st.r(3), 'data-h': k }));
    };
    st.draw = draw; requestAnimationFrame(draw);
  };
  build();
  angle.addEventListener('input', () => { angleLbl.textContent = num(+angle.value, 1) + '°'; });
  angle.addEventListener('change', () => { c.crop.angle = +angle.value; build(); });
  const runAuto = async (silent) => {
    const old = autoBtn.textContent; autoBtn.disabled = true; autoBtn.textContent = '⏳ جارٍ البحث عن الأسنان…';
    try {
      const r = await autoCropIntra(src);
      if (!r) { if (!silent) toast('لم يُعثر على الأسنان أو الفم — قصّ يدوياً'); return; }
      c.crop.q = r.q; c.crop.angle = 0; angle.value = 0; angleLbl.textContent = '0.0°'; Object.assign(rect, r.rect); build();
      toast('قص تلقائي حسب ' + r.how + ' — راجعه ثم اضغط تطبيق');
    } catch (e) { if (!silent || String(e.message) !== 'NO_MODEL') await modelError(e); }
    finally { autoBtn.disabled = false; autoBtn.textContent = old; }
  };
  const autoBtn = h('button', { class: 'btn primary sm', onclick: () => runAuto(false) }, '✂ قص تلقائي');
  if (c.crop.autoPending) { c.crop.autoPending = false; saveCase(c); if (S.autoCrop && (await getModelBuffer().catch(() => null) || window.__DSD_TEST_INFER)) setTimeout(() => runAuto(true), 50); }
  const setQ = (d) => { c.crop.q = (c.crop.q + d + 4) % 4; Object.assign(rect, { x: 0, y: 0, w: 1, h: 1 }); build(); };
  card.append(h('div', { class: 'toolbar' }, h('button', { class: 'btn sm', onclick: () => setQ(-1) }, '↺ 90°'), h('button', { class: 'btn sm', onclick: () => setQ(1) }, '↻ 90°'),
    h('button', { class: 'btn sm', onclick: () => { Object.assign(rect, { x: 0, y: 0, w: 1, h: 1 }); st.draw(); } }, 'كامل الصورة'),
    autoBtn,
    h('button', { class: 'btn sm', onclick: async () => { const q = await autoOrient(src); if (q != null) { c.crop.q = q; Object.assign(rect, { x: 0, y: 0, w: 1, h: 1 }); build(); } } }, '🔍 اتجاه تلقائي'),
    ...zoomBar({ setZoom: (z) => st.setZoom(z), zoom: () => st.zoom() })));
  card.append(h('div', null, h('div', { class: 'row between' }, h('span', { class: 'muted small' }, 'الميل'), angleLbl), angle));
  card.append(h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '10px' } }, h('button', { class: 'btn primary', onclick: async () => {
    if (c.intra.teeth && !await confirmBox('تطبيق القص', 'تغيير القص يعيد تحديد الأسنان والتصميم من جديد. متابعة؟', 'متابعة')) return;
    c.crop.rect = { ...rect };
    await applyCrop(c, src);
    c.intra = { teeth: null, source: null, det: null, maskPng: null, status: c.intra.status || {}, reviewed: false }; c.design.override = {};
    if (c.calib.method === 'ruler') c.calib.ruler = null;
    await saveCase(c, { design: true });
    go(`#/case/${c.id}/teeth`);
  } }, 'تطبيق والمتابعة ←')));
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
async function applyCrop(c, src) {
  if (!src) { const bmp = await blobToBitmap(await getBlob(c.images.intraOrig), 2400); src = bitmapToCanvas(bmp); bmp.close?.(); }
  const rc = rotatedCanvas(src, c.crop.q * 90 + c.crop.angle);
  const w = cropCanvas(rc, c.crop.rect, 1600);
  if (c.images.intraWork) { await DB.del('blobs', c.images.intraWork); imageCache.delete(c.images.intraWork); }
  c.images.intraWork = await putBlob(await canvasToBlob(w));
}

// ----- step 3: teeth & measurement -----
async function stepTeeth(body, c) {
  if (!c.images.intraWork) { body.append(h('div', { class: 'card empty' }, 'طبّق القص أولاً.'), nextBtn(c, 'crop', '→ القص')); return; }
  const img = await getCanvasForBlob(c.images.intraWork);
  const W = img.width, H = img.height;
  let sel = '11', mode = 'teeth', showMask = true, link = true;
  const card = h('div', { class: 'card' }, h('h2', null, 'حدود الأسنان الستة'),
    h('p', { class: 'muted small' }, 'لكل سن: الحد الأيمن والأيسر (نقاط التماس)، والحد العلوي (قمة اللثة zenith)، والحد السفلي (الحافة القاطعة). اختر السن ثم اسحب نقاطه.'));
  body.append(card);
  const issuesBox = h('div');
  const st = makeStage(img, {
    onTap(pt) { if (mode === 'sam') samTap(pt); },
    onDown(pt, target) {
      const hk = target.dataset?.h, tk = target.dataset?.t;
      if (mode === 'sam') return null;
      if (hk === 'ra' || hk === 'rb') { const P = c.calib.ruler[hk === 'ra' ? 'a' : 'b']; return { move(q) { P.x = clamp(q.x, 0, W); P.y = clamp(q.y, 0, H); draw(); refresh(); }, up() { saveSoon(c, { design: true }); } }; }
      if (mode !== 'teeth' || !c.intra.teeth) return null;
      if (tk && !hk) { sel = tk; drawChips(); }
      const T = c.intra.teeth;
      if (hk && T[sel]) {
        const b = T[sel], i = C.TEETH.indexOf(sel), left = C.TEETH[i - 1], right = C.TEETH[i + 1];
        return { move(q) {
          const minS = 6;
          if (hk === 'x0') { b.x0 = clamp(q.x, 0, b.x1 - minS); if (link && left && T[left]) T[left].x1 = Math.max(b.x0, T[left].x0 + minS); }
          if (hk === 'x1') { b.x1 = clamp(q.x, b.x0 + minS, W); if (link && right && T[right]) T[right].x0 = Math.min(b.x1, T[right].x1 - minS); }
          if (hk === 'y0') b.y0 = clamp(q.y, 0, b.y1 - minS);
          if (hk === 'y1') b.y1 = clamp(q.y, b.y0 + minS, H);
          c.intra.source = c.intra.source === 'auto' ? 'auto+edited' : (c.intra.source || 'manual'); draw(); refresh();
        }, up() { saveSoon(c, { design: true }); } };
      }
      if (tk && T[tk]) {
        const b = T[tk], o = { ...b }, p0 = pt; let started = false;
        return { move(q) { const dx = q.x - p0.x, dy = q.y - p0.y; if (!started && Math.hypot(dx, dy) < st.r(8)) return; started = true; b.x0 = o.x0 + dx; b.x1 = o.x1 + dx; b.y0 = o.y0 + dy; b.y1 = o.y1 + dy; draw(); refresh(); }, up() { saveSoon(c, { design: true }); } };
      }
      return null;
    }
  });
  let maskImg = null;
  const placeMask = () => {
    st.extra.innerHTML = '';
    if (c.intra.maskPng && showMask) { maskImg = h('img', { class: 'maskimg', src: c.intra.maskPng, alt: '' }); st.extra.append(maskImg); }
  };
  const draw = () => {
    st.svg.innerHTML = '';
    const T = c.intra.teeth, sw = st.r(2);
    if (T) for (const t of C.TEETH) {
      const b = T[t]; if (!b) continue;
      const on = t === sel && mode === 'teeth', ts = statusOf(c)[t];
      st.svg.append(s('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: ts === 'missing' ? 'rgba(0,0,0,.35)' : on ? 'rgba(255,255,255,.12)' : 'rgba(0,0,0,0)', stroke: TOOTH_COLORS[t], 'stroke-width': on ? sw * 1.6 : sw, 'stroke-dasharray': ts === 'missing' ? st.r(6) + ' ' + st.r(4) : null, 'data-t': t }));
      st.svg.append(s('text', { x: (b.x0 + b.x1) / 2, y: b.y0 - st.r(6), fill: TOOTH_COLORS[t], 'font-size': st.r(14), 'text-anchor': 'middle', 'font-weight': '700', 'paint-order': 'stroke', stroke: '#000', 'stroke-width': st.r(3) }, t + (ts === 'missing' ? ' ✕' : ts === 'broken' ? ' ⚡' : '')));
    }
    if (T && T[sel] && mode === 'teeth') {
      const b = T[sel], r = Math.min(st.r(13), (b.x1 - b.x0) * 0.22), mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
      for (const [k, x, y] of [['x0', b.x0, my], ['x1', b.x1, my], ['y0', mx, b.y0], ['y1', mx, b.y1]])
        st.svg.append(s('circle', { cx: x, cy: y, r, fill: '#fff', stroke: TOOTH_COLORS[sel], 'stroke-width': st.r(4), 'data-h': k }));
    }
    const R = c.calib.method === 'ruler' && c.calib.ruler;
    if (R) {
      st.svg.append(s('line', { x1: R.a.x, y1: R.a.y, x2: R.b.x, y2: R.b.y, stroke: '#ff2d95', 'stroke-width': st.r(3), 'stroke-dasharray': st.r(8) + ' ' + st.r(5) }));
      for (const [k, P] of [['ra', R.a], ['rb', R.b]]) st.svg.append(s('circle', { cx: P.x, cy: P.y, r: st.r(13), fill: 'rgba(255,45,149,.35)', stroke: '#ff2d95', 'stroke-width': st.r(3), 'data-h': k }));
    }
  };
  st.draw = draw;
  const chips = h('div', { class: 'chips', role: 'group', 'aria-label': 'اختيار السن' });
  const statusSel = h('select', { 'aria-label': 'حالة السن' }, Object.entries(STATUS).map(([k, l]) => h('option', { value: k }, l)));
  const statusLbl = h('span');
  statusSel.addEventListener('change', async () => {
    c.intra.status = { ...statusOf(c), [sel]: statusSel.value };
    if (statusSel.value === 'missing' && ['veneer', 'composite'].includes(c.design.treat[sel])) { c.design.treat[sel] = 'implant'; toast(`السن ${sel} مفقود: غُيّر العلاج المقترح إلى زرعة + تاج (عدّله في التصميم)`); }
    if (statusSel.value === 'missing' && c.intra.teeth) toast('ضع مستطيل السن المفقود على الفراغ الذي سيشغله');
    await saveCase(c, { design: true }); drawChips(); refresh();
  });
  const drawChips = () => {
    chips.innerHTML = '';
    for (const t of C.TEETH) { const ts = statusOf(c)[t]; chips.append(h('button', { class: 'chip' + (t === sel ? ' on' : ''), style: { borderInlineStart: `5px solid ${TOOTH_COLORS[t]}` }, onclick: () => { sel = t; if (mode === 'sam') samBtn.textContent = '👆 انقر داخل السن ' + sel; drawChips(); draw(); } }, t + (ts === 'missing' ? ' ✕' : ts === 'broken' ? ' ⚡' : ''))); }
    statusSel.value = statusOf(c)[sel] || 'present'; statusLbl.textContent = 'حالة السن ' + sel;
    draw();
  };
  const maskBtn = h('button', { class: 'btn sm', onclick: () => { showMask = !showMask; maskBtn.textContent = showMask ? 'إخفاء القناع' : 'إظهار القناع'; placeMask(); } }, 'إخفاء القناع');
  const linkBtn = h('button', { class: 'btn sm', onclick: () => { link = !link; linkBtn.textContent = link ? '🔗 نقاط التماس مربوطة' : 'نقاط التماس منفصلة'; } }, '🔗 نقاط التماس مربوطة');
  const samBtn = h('button', { class: 'btn sm', onclick: () => {
    if (!c.intra.teeth) { c.intra.teeth = C.defaultTeeth(W, H); c.intra.source = 'manual'; }
    mode = mode === 'sam' ? 'teeth' : 'sam'; samBtn.classList.toggle('primary', mode === 'sam');
    samBtn.textContent = mode === 'sam' ? '👆 انقر داخل السن ' + sel : '👆 تحديد بنقرة'; draw();
  } }, '👆 تحديد بنقرة');
  const samTap = async (pt) => {
    samBtn.disabled = true; const old = samBtn.textContent; samBtn.textContent = '⏳ جارٍ التحديد…';
    try {
      const masks = await samMasks(img, c.images.intraWork, pt);
      const pick = pickToothBox(masks, pt, c.intra.teeth?.[sel], W);
      if (!pick) { toast('لم يُحدَّد سن واضح — انقر في منتصف السن أو عدّل يدوياً'); return; }
      c.intra.teeth[sel] = pick.box; c.intra.source = c.intra.source === 'auto' ? 'auto+edited' : (c.intra.source || 'manual');
      await saveCase(c, { design: true });
      const i = C.TEETH.indexOf(sel); if (i < 5) { sel = C.TEETH[i + 1]; drawChips(); }
      refresh(); toast('حُدّد السن — راجع الحدود');
    } catch (e) { await samError(e); mode = 'teeth'; samBtn.classList.remove('primary'); }
    finally { samBtn.disabled = false; samBtn.textContent = mode === 'sam' ? '👆 انقر داخل السن ' + sel : old; draw(); }
  };
  const detectBtn = h('button', { class: 'btn primary sm', onclick: async () => {
    if (c.intra.teeth && !await confirmBox('كشف تلقائي', 'سيستبدل الكشف التلقائي الحدود الحالية. متابعة؟', 'متابعة')) return;
    const ok = await runDetection(c, detectBtn);
    if (ok) render();
  } }, '🤖 كشف تلقائي');
  card.append(h('div', { class: 'toolbar' }, detectBtn,
    h('button', { class: 'btn sm', onclick: async () => {
      if (c.intra.teeth && !await confirmBox('تحديد يدوي', 'إعادة الحدود إلى الوضع الافتراضي؟', 'متابعة')) return;
      c.intra.teeth = C.defaultTeeth(W, H); c.intra.source = 'manual'; c.intra.det = null; c.intra.maskPng = null; c.intra.reviewed = false; await saveCase(c, { design: true }); render();
    } }, '✋ تحديد يدوي'),
    samBtn, c.intra.maskPng ? maskBtn : null, linkBtn, ...zoomBar(st),
    c.intra.teeth ? h('button', { class: 'btn sm', onclick: () => addCaseToLibrary(c) }, '📚 حفظ الأسنان في المكتبة') : null));
  card.append(chips, h('label', { class: 'field', style: { marginTop: '8px', maxWidth: '320px' } }, h('span', null, statusLbl), statusSel), st.wrap, issuesBox);
  placeMask(); drawChips(); focusTeeth(st, c.intra.teeth);

  // calibration
  const calCard = h('div', { class: 'card' }, h('h2', null, 'المعايرة بالملم'));
  const method = h('select', { 'aria-label': 'طريقة المعايرة' }, h('option', { value: 'none' }, 'بلا معايرة (نسب فقط)'), h('option', { value: 'ruler' }, 'مسطرة في الصورة'), h('option', { value: 'tooth' }, 'عرض سن مقاس بالفرجار'));
  method.value = c.calib.method || 'none';
  const calBody = h('div');
  const drawCal = () => {
    calBody.innerHTML = '';
    if (method.value === 'ruler') {
      if (!c.calib.ruler) c.calib.ruler = { a: { x: W * 0.3, y: H * 0.85 }, b: { x: W * 0.5, y: H * 0.85 }, mm: 10 };
      const mm = h('input', { type: 'number', step: '0.1', min: '0.5', value: c.calib.ruler.mm });
      mm.addEventListener('input', () => { c.calib.ruler.mm = +mm.value; refresh(); saveSoon(c, { design: true }); });
      calBody.append(h('p', { class: 'muted small' }, 'اسحب طرفي الخط الوردي على علامتين في المسطرة، ثم اكتب المسافة بينهما. يجب أن تكون المسطرة في مستوى الثنايا.'), h('label', { class: 'field' }, h('span', null, 'المسافة بين الطرفين (ملم)'), mm));
    } else if (method.value === 'tooth') {
      const t = h('select', null, C.TEETH.map((k) => h('option', { value: k, selected: c.calib.tooth?.id === k }, k + ' ' + C.TOOTH_NAME[k])));
      const mm = h('input', { type: 'number', step: '0.1', min: '1', value: c.calib.tooth?.mm || '' });
      const upd = () => { c.calib.tooth = { id: t.value, mm: +mm.value }; refresh(); saveSoon(c, { design: true }); };
      t.addEventListener('change', upd); mm.addEventListener('input', upd);
      calBody.append(h('p', { class: 'muted small' }, 'قِس العرض الظاهري للسن (من الأمام) بالفرجار على المريض، ويُفضّل الثنية. يعتمد القياس على دقة حدود هذا السن.'), h('div', { class: 'grid2' }, h('label', { class: 'field' }, h('span', null, 'السن'), t), h('label', { class: 'field' }, h('span', null, 'العرض (ملم)'), mm)));
    } else calBody.append(h('p', { class: 'muted small' }, 'بدون معايرة تظهر النسب فقط، ولا تُحسب الأطوال بالملم ولا قواعد اللثة.'));
    draw();
  };
  method.addEventListener('change', () => { c.calib.method = method.value; drawCal(); refresh(); saveSoon(c, { design: true }); });
  calCard.append(method, calBody); drawCal();
  body.append(calCard);

  const measCard = h('div', { class: 'card' }); body.append(measCard);
  const refresh = () => {
    issuesBox.innerHTML = '';
    if (c.intra.det) {
      const d = c.intra.det;
      issuesBox.append(h('div', { class: 'issue info' }, `كشف تلقائي: ثقة ${num(d.meanConf, 2)} · ${d.presentCount}/6 أسنان · ${d.tta ? 'مع TTA' : 'بدون TTA'}${d.straightened ? ` · صُحّح الميل ${num(d.straightened, 1)}°` : ''}${c.intra.source === 'auto+edited' ? ' · عُدّلت يدوياً' : ''}`));
      for (const i of d.issues || []) issuesBox.append(h('div', { class: 'issue ' + i.level }, i.text));
      if (!(d.issues || []).length) issuesBox.append(h('div', { class: 'issue ok' }, 'لا مشاكل ظاهرة، لكن راجع الحدود بالعين قبل الاعتماد.'));
    } else if (!c.intra.teeth) issuesBox.append(h('div', { class: 'issue info' }, 'اضغط «كشف تلقائي» أو «تحديد يدوي» للبدء.'));
    measCard.innerHTML = '';
    if (!c.intra.teeth) return;
    const A = computeAll(c), m = A.meas, u = m.calibrated;
    measCard.append(h('h2', null, 'القياسات'), u ? h('div', { class: 'muted small' }, `المعايرة: ${num(A.mm * 100, 3)} ملم لكل 100 بكسل`) : h('div', { class: 'issue warn' }, 'غير معاير: الأرقام بالبكسل، والنسب صحيحة.'));
    const tb = h('table', { class: 't' }, h('thead', null, h('tr', null, h('th', null, 'السن'), h('th', null, 'العرض'), h('th', null, 'الطول'), h('th', null, 'عرض/طول'))), h('tbody', null,
      C.TEETH.filter((t) => m.teeth[t] || m.missing.includes(t)).map((t) => !m.teeth[t] ? h('tr', null, h('td', null, h('b', null, t), ' ', h('span', { class: 'muted small' }, C.TOOTH_NAME[t])), h('td', { colspan: 3, class: 'muted' }, 'مفقود — مستطيله يمثل الفراغ')) : h('tr', null, h('td', null, h('b', { style: { color: 'var(--ink)' } }, t), ' ', h('span', { class: 'muted small' }, C.TOOTH_NAME[t])),
        h('td', null, ltr(u ? num(m.teeth[t].wmm, 2) + ' mm' : num(m.teeth[t].w, 0) + ' px')), h('td', null, ltr(u ? num(m.teeth[t].hmm, 2) + ' mm' : num(m.teeth[t].h, 0) + ' px')), h('td', null, m.teeth[t].broken ? h('span', { class: 'muted small' }, 'مكسور — الطول لا يُحتسب') : num(m.teeth[t].wl, 2))))));
    measCard.append(h('div', { class: 'tablewrap' }, tb), rulesTable(A.rulesNow.filter((r) => r.group !== 'face'), 'مطابقة القواعد (الوضع الحالي)'));
  };
  refresh();
  if (c.intra.teeth) {
    const rv = h('input', { type: 'checkbox', checked: !!c.intra.reviewed });
    rv.addEventListener('change', async () => { c.intra.reviewed = rv.checked; c.intra.reviewedAt = Date.now(); await saveCase(c, { design: true }); });
    body.append(h('div', { class: 'card' }, h('label', { class: 'check' }, rv, h('span', null, h('b', null, 'راجعتُ بنفسي حدود الأسنان الستة وحالتها (موجود/مفقود/مكسور) والمعايرة.'), h('div', { class: 'muted small' }, 'شرط للاعتماد. يُلغى تلقائياً عند إعادة الكشف أو تغيير الصورة.')))));
  }
  body.append(nextBtn(c, 'face', 'التالي: الوجه ←', !c.intra.teeth));
}
function statusBadge(st) { return h('span', { class: 'badge ' + st }, { ok: 'مطابق', warn: 'قريب', fail: 'خارج النطاق', na: 'غير متاح' }[st] || st); }
function rulesTable(rows, title, after) {
  const box = h('div'); if (title) box.append(h('h3', null, title));
  if (!rows.length) { box.append(h('div', { class: 'muted small' }, 'لا قواعد مفعّلة.')); return box; }
  const aMap = after ? Object.fromEntries(after.map((r) => [r.id, r])) : null;
  box.append(h('div', { class: 'tablewrap' }, h('table', { class: 't' },
    h('thead', null, h('tr', null, h('th', null, 'القاعدة'), h('th', null, 'الحالي'), aMap ? h('th', null, 'بعد التصميم') : null, h('th', null, 'المرجع'))),
    h('tbody', null, rows.map((r) => h('tr', null,
      h('td', null, r.label, r.msg ? h('div', { class: 'muted small' }, r.msg) : null),
      h('td', null, ltr(r.valueText), h('br'), statusBadge(r.status)),
      aMap ? h('td', null, aMap[r.id] ? [ltr(aMap[r.id].valueText), h('br'), statusBadge(aMap[r.id].status)] : '—') : null,
      h('td', { class: 'muted small' }, ltr(r.target || ''))))))));
  return box;
}

// ----- step 4: face -----
async function stepFace(body, c) {
  if (!c.images.faceWork) {
    body.append(h('div', { class: 'card' }, h('h2', null, 'تحليل الوجه والابتسامة'), h('p', { class: 'muted' }, 'لم تُضف صورة وجه. هذه الخطوة اختيارية: تحسب انحراف خط الوسط، وميل زوايا الفم والمستوى القاطع، وظهور اللثة، وممر الشدق، وقوس الابتسامة.'),
      h('div', { class: 'row' }, h('a', { class: 'btn', href: `#/case/${c.id}/photos` }, 'إضافة صورة وجه'), h('a', { class: 'btn primary', href: `#/case/${c.id}/design` }, 'تخطي إلى التصميم ←'))));
    return;
  }
  const img = await getCanvasForBlob(c.images.faceWork);
  const P = c.face.points; const PTS = C.FACE_POINTS;
  let cur = PTS.findIndex((p) => !P[p.id]); if (cur < 0) cur = 0;
  const card = h('div', { class: 'card' }, h('h2', null, 'نقاط الوجه'), h('p', { class: 'muted small' }, 'اضغط على الصورة لوضع النقطة المطلوبة، ثم ينتقل النظام للنقطة التالية. يمكنك سحب أي نقطة لتعديلها. كبّر الصورة للدقة.'));
  body.append(card);
  const curLbl = h('div', { class: 'issue info' });
  const st = makeStage(img, {
    onDown(pt, target) {
      const k = target.dataset?.p; if (!k) return null;
      const p0 = pt, orig = { ...P[k] }; let moved = false;
      return {
        move(q) { if (!moved && Math.hypot(q.x - p0.x, q.y - p0.y) < st.r(6)) return; if (!moved) { moved = true; cur = PTS.findIndex((p) => p.id === k); drawList(); } P[k] = { x: clamp(orig.x + q.x - p0.x, 0, img.width), y: clamp(orig.y + q.y - p0.y, 0, img.height) }; draw(); },
        up() { if (!moved) { st.tapHere(p0); return; } saveSoon(c, { design: true }); refresh(); }
      };
    },
    onTap(pt) { st.tapHere(pt); }
  });
  st.tapHere = (pt) => {
      P[PTS[cur].id] = pt;
      const nxt = PTS.findIndex((p, i) => i > cur && !P[p.id]); cur = nxt >= 0 ? nxt : Math.min(cur + 1, PTS.length - 1);
      draw(); drawList(); saveSoon(c, { design: true }); refresh();
  };
  const draw = () => {
    st.svg.innerHTML = '';
    const L = (a, b, col, dash) => { if (P[a] && P[b]) st.svg.append(s('line', { x1: P[a].x, y1: P[a].y, x2: P[b].x, y2: P[b].y, stroke: col, 'stroke-width': st.r(2), 'stroke-dasharray': dash ? st.r(6) + ' ' + st.r(4) : null })); };
    L('pupilR', 'pupilL', '#42d4f4'); L('commR', 'commL', '#ffe119'); L('cusp13', 'cusp23', '#f58231', true);
    if (P.glabella && P.menton) {
      const dx = P.menton.x - P.glabella.x, dy = P.menton.y - P.glabella.y;
      st.svg.append(s('line', { x1: P.glabella.x - dx * 0.1, y1: P.glabella.y - dy * 0.1, x2: P.menton.x + dx * 0.05, y2: P.menton.y + dy * 0.05, stroke: '#3cb44b', 'stroke-width': st.r(2) }));
    }
    if (P.midlineDental) st.svg.append(s('line', { x1: P.midlineDental.x, y1: P.midlineDental.y - st.r(40), x2: P.midlineDental.x, y2: P.midlineDental.y + st.r(25), stroke: '#ff2d95', 'stroke-width': st.r(2) }));
    PTS.forEach((p, i) => {
      if (!P[p.id]) return;
      const on = i === cur;
      st.svg.append(s('circle', { cx: P[p.id].x, cy: P[p.id].y, r: st.r(on ? 11 : 8), fill: on ? 'rgba(255,45,149,.5)' : 'rgba(255,255,255,.45)', stroke: on ? '#ff2d95' : '#0f7b6c', 'stroke-width': st.r(2.5), 'data-p': p.id }));
      st.svg.append(s('text', { x: P[p.id].x + st.r(12), y: P[p.id].y - st.r(8), fill: '#fff', 'font-size': st.r(11), 'paint-order': 'stroke', stroke: '#000', 'stroke-width': st.r(3), 'pointer-events': 'none' }, String(i + 1)));
    });
  };
  st.draw = draw;
  const list = h('div', { class: 'pointlist' });
  const drawList = () => {
    list.innerHTML = '';
    PTS.forEach((p, i) => list.append(h('button', { class: (i === cur ? 'cur ' : '') + (P[p.id] ? 'set' : ''), onclick: () => { cur = i; drawList(); draw(); } }, h('span', { class: 'dot' }), h('span', null, `${i + 1}. ${p.label}`))));
    curLbl.textContent = 'النقطة الحالية: ' + (cur + 1) + '. ' + PTS[cur].label;
    list.children[cur]?.scrollIntoView({ block: 'nearest' });
  };
  const autoBtn = h('button', { class: 'btn primary sm', onclick: async () => {
    if (Object.keys(P).length && !await confirmBox('نقاط تلقائية', 'ستُستبدل النقاط الحالية بنقاط مكتشفة تلقائياً. متابعة؟', 'متابعة')) return;
    const old = autoBtn.textContent; autoBtn.disabled = true; autoBtn.textContent = '⏳ جارٍ الكشف…';
    try {
      const r = await autoFacePoints(img);
      for (const k of Object.keys(P)) delete P[k]; Object.assign(P, r.points);
      cur = Math.max(0, PTS.findIndex((p) => !P[p.id])); draw(); drawList(); await saveCase(c, { design: true }); refresh();
      const names = r.manual.map((id) => PTS.findIndex((p) => p.id === id) + 1).join('، ');
      await modal('تم وضع النقاط', h('div', null, h('p', null, r.teethAuto ? 'وُضعت نقاط الوجه والشفاه والأسنان تلقائياً.' : 'وُضعت نقاط الوجه والشفاه تلقائياً. نقاط الأسنان لم تُكتشف بوضوح في الابتسامة.'),
        r.manual.length ? h('p', null, 'ضع يدوياً النقاط: ' + names + '.') : null, h('p', { class: 'muted small' }, 'راجع كل نقطة بتكبير الصورة، خصوصاً نقاط الأسنان والشفة واللثة.')));
    } catch (e) { await faceError(e); } finally { autoBtn.disabled = false; autoBtn.textContent = old; }
  } }, '🤖 نقاط تلقائية');
  card.append(curLbl, h('div', { class: 'toolbar' }, autoBtn, ...zoomBar(st),
    h('button', { class: 'btn sm', onclick: () => { delete P[PTS[cur].id]; draw(); drawList(); saveSoon(c, { design: true }); refresh(); } }, 'حذف النقطة الحالية'),
    h('button', { class: 'btn sm', onclick: async () => {
      if (Object.keys(P).length && !await confirmBox('تدوير صورة الوجه', 'سيمسح التدوير النقاط الحالية. متابعة؟', 'تدوير')) return;
      c.faceQ = ((c.faceQ || 0) + 1) % 4; c.face.points = {}; await makeFaceWork(c); await saveCase(c, { design: true }); render();
    } }, '↻ 90°')), st.wrap, h('div', { style: { height: '10px' } }), list);
  draw(); drawList();
  const res = h('div', { class: 'card' }); body.append(res);
  const refresh = () => {
    res.innerHTML = '';
    const A = computeAll(c);
    if (!A.face) { res.append(h('h2', null, 'النتائج'), h('p', { class: 'muted small' }, 'ضع نقطتي العينين على الأقل لبدء التحليل.')); return; }
    const M = A.face.metrics;
    res.append(h('h2', null, 'النتائج'), h('div', { class: 'muted small' }, 'المقياس: ' + (A.face.scaleSource || 'غير متوفر — الأرقام بالملم لن تظهر')));
    const rows = A.rulesNow.filter((r) => r.group === 'face');
    res.append(rulesTable(rows.length ? rows : C.evaluateRules({ teeth: {}, ratios: {}, symmetry: {}, calibrated: false }, S.rules.filter((r) => r.group === 'face'), { face: A.face })));
    if (M.facial_midline_angle != null) res.append(h('div', { class: 'muted small' }, `ميل خط الوسط الوجهي عن العمودي على خط العينين: ${num(M.facial_midline_angle, 1)}°`));
    if (M.incisal_to_lip != null) res.append(h('div', { class: 'muted small' }, `المسافة بين الحافة القاطعة والشفة السفلى: ${num(M.incisal_to_lip, 1)} ملم`));
  };
  refresh();
  body.append(nextBtn(c, 'rest', 'التالي: الشفاه في الراحة ←'));
}

// ----- step 5: lips at rest -----
const REST_POINTS = [
  { id: 'pupilR', label: 'بؤبؤ العين اليمنى للمريض' }, { id: 'pupilL', label: 'بؤبؤ العين اليسرى للمريض' },
  { id: 'lipRest', label: 'الحد السفلي للشفة العليا عند المنتصف' }, { id: 'incisal11', label: 'الحافة القاطعة للثنية 11 (إن ظهرت)' }
];
async function stepRest(body, c) {
  c.rest = c.rest || { points: {}, hidden: false };
  if (!c.images.restWork) {
    body.append(h('div', { class: 'card' }, h('h2', null, 'الشفاه في الراحة'), h('p', { class: 'muted' }, 'لم تُضف صورة الراحة. هي اختيارية لكنها أهم مرجع لطول الثنية: كم يظهر منها والشفاه مرتاحة (عادةً 1–3 ملم، وأكثر عند الشباب والنساء).'),
      h('div', { class: 'row' }, h('a', { class: 'btn', href: `#/case/${c.id}/photos` }, 'إضافة صورة الراحة'), h('a', { class: 'btn primary', href: `#/case/${c.id}/design` }, 'تخطي إلى التصميم ←'))));
    return;
  }
  const img = await getCanvasForBlob(c.images.restWork);
  const P = c.rest.points; let cur = REST_POINTS.findIndex((p) => !P[p.id]); if (cur < 0) cur = 0;
  const card = h('div', { class: 'card' }, h('h2', null, 'نقاط صورة الراحة'), h('p', { class: 'muted small' }, 'اضغط لوضع النقطة الحالية، واسحب أي نقطة لتعديلها. كبّر الصورة عند الشفاه للدقة.'));
  body.append(card);
  const curLbl = h('div', { class: 'issue info' });
  const st = makeStage(img, {
    onDown(pt, target) {
      const k = target.dataset?.p; if (!k) return null;
      const p0 = pt, orig = { ...P[k] }; let moved = false;
      return { move(q) { if (!moved && Math.hypot(q.x - p0.x, q.y - p0.y) < st.r(6)) return; moved = true; P[k] = { x: clamp(orig.x + q.x - p0.x, 0, img.width), y: clamp(orig.y + q.y - p0.y, 0, img.height) }; draw(); },
        up() { if (!moved) { place(p0); return; } saveSoon(c, { design: true }); refresh(); } };
    },
    onTap(pt) { place(pt); }
  });
  const place = (pt) => { P[REST_POINTS[cur].id] = pt; const n = REST_POINTS.findIndex((p, i) => i > cur && !P[p.id]); cur = n >= 0 ? n : cur; draw(); drawList(); saveSoon(c, { design: true }); refresh(); };
  const draw = () => {
    st.svg.innerHTML = '';
    if (P.pupilR && P.pupilL) st.svg.append(s('line', { x1: P.pupilR.x, y1: P.pupilR.y, x2: P.pupilL.x, y2: P.pupilL.y, stroke: '#42d4f4', 'stroke-width': st.r(2) }));
    if (P.lipRest && P.incisal11) st.svg.append(s('line', { x1: P.lipRest.x, y1: P.lipRest.y, x2: P.incisal11.x, y2: P.incisal11.y, stroke: '#ff2d95', 'stroke-width': st.r(2) }));
    REST_POINTS.forEach((p, i) => { if (!P[p.id]) return; st.svg.append(s('circle', { cx: P[p.id].x, cy: P[p.id].y, r: st.r(i === cur ? 10 : 7), fill: 'rgba(255,255,255,.45)', stroke: i === cur ? '#ff2d95' : '#0f7b6c', 'stroke-width': st.r(2.5), 'data-p': p.id })); });
  };
  st.draw = draw;
  const list = h('div', { class: 'pointlist' });
  const drawList = () => { list.innerHTML = ''; REST_POINTS.forEach((p, i) => list.append(h('button', { class: (i === cur ? 'cur ' : '') + (P[p.id] ? 'set' : ''), onclick: () => { cur = i; drawList(); draw(); } }, h('span', { class: 'dot' }), h('span', null, `${i + 1}. ${p.label}`)))); curLbl.textContent = 'النقطة الحالية: ' + REST_POINTS[cur].label; };
  const hid = h('input', { type: 'checkbox', checked: !!c.rest.hidden });
  hid.addEventListener('change', async () => { c.rest.hidden = hid.checked; if (hid.checked) delete P.incisal11; await saveCase(c, { design: true }); draw(); drawList(); refresh(); });
  const autoBtn = h('button', { class: 'btn primary sm', onclick: async () => {
    const old = autoBtn.textContent; autoBtn.disabled = true; autoBtn.textContent = '⏳ جارٍ الكشف…';
    try {
      const lm = await faceLandmarks(img); if (!lm) throw new Error('NO_FACE');
      const Q = (i) => ({ x: lm[i].x * img.width, y: lm[i].y * img.height });
      const [a, b] = Q(468).x <= Q(473).x ? [Q(468), Q(473)] : [Q(473), Q(468)];
      P.pupilR = a; P.pupilL = b; P.lipRest = Q(13); cur = 3;
      await saveCase(c, { design: true }); draw(); drawList(); refresh(); toast('وُضعت العينان والشفة — ضع الحافة القاطعة يدوياً أو علّم أنها مغطاة');
    } catch (e) { await faceError(e); } finally { autoBtn.disabled = false; autoBtn.textContent = old; }
  } }, '🤖 نقاط تلقائية');
  card.append(curLbl, h('div', { class: 'toolbar' }, autoBtn, ...zoomBar(st),
    h('button', { class: 'btn sm', onclick: async () => { if (Object.keys(P).length && !await confirmBox('تدوير', 'سيمسح التدوير النقاط. متابعة؟', 'تدوير')) return; c.restQ = ((c.restQ || 0) + 1) % 4; c.rest.points = {}; await makeFaceWork(c, 'rest'); await saveCase(c, { design: true }); render(); } }, '↻ 90°')),
    st.wrap, h('label', { class: 'check' }, hid, h('span', null, 'الحافة القاطعة مغطاة بالشفة تماماً في الراحة')), list);
  draw(); drawList();
  const res = h('div', { class: 'card' }); body.append(res);
  const refresh = () => {
    res.innerHTML = '';
    const A = computeAll(c); const R = A.rest;
    res.append(h('h2', null, 'النتيجة'));
    if (!R) { res.append(h('p', { class: 'muted small' }, 'ضع العينين والشفة العليا أولاً.')); return; }
    res.append(h('div', { class: 'muted small' }, R.approx ? `المقياس: مسافة عينين مفترضة ${R.ipdMm} ملم (تقريبي). للدقة أضف صورة الابتسامة مع نقاط الثنية وعاير صورة الأسنان.` : `المقياس: مسافة العينين المقاسة من صورة الابتسامة (${num(R.ipdMm, 1)} ملم).`));
    res.append(rulesTable(A.rulesNow.filter((r) => r.id === 'rest_display')));
    if (R.hidden) res.append(h('div', { class: 'issue warn' }, 'الحافة مغطاة: لا يمكن حساب الإطالة المطلوبة تلقائياً. حدّد طول الثنية يدوياً في التصميم.'));
    else if (R.display != null) {
      const tgt = S.rules.find((r) => r.id === 'rest_display')?.target ?? 2, dlt = tgt - R.display;
      res.append(h('div', { class: 'issue info' }, `للوصول إلى ظهور ${tgt} ملم: ${Math.abs(dlt) < 0.2 ? 'الطول الحالي مناسب' : (dlt > 0 ? 'إطالة' : 'تقصير') + ' الحافة القاطعة نحو ' + num(Math.abs(dlt), 1) + ' ملم'}. اختر «طول الثنية: من ظهورها في الراحة» في التصميم لتطبيقه.`));
    }
  };
  refresh();
  body.append(nextBtn(c, 'design', 'التالي: التصميم ←'));
}
