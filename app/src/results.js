/* Results: before/after on the patient's own photo, findings, simple design choices, review, approval, outputs */
'use strict';
const Results = { pending: null, live: null };
const TREAT_LABEL = { veneer: 'فينير', crown: 'تاج', composite: 'كومبوزيت', implant: 'زرعة + تاج', pontic: 'جسر: سن معلّق', abutment: 'جسر: تاج داعم', denture: 'طقم', none: 'بلا علاج' };
const STATUS_LABEL = { present: 'موجود', estimated: 'تقديري', broken: 'مكسور', missing: 'مفقود', hidden: 'غير ظاهر', unsure: 'غير واضح' };
const SHAPE_LABEL = { square: 'مربع', oval: 'بيضاوي', triangular: 'مثلث' };

// view rectangles on the full photo: whole face, or the smile
function viewRects(st) {
  const P = st.geom.P, W = st.fullImg.width, H = st.fullImg.height;
  const outer = SM.OUTER_LIP.map((i) => P[i]);
  const mx0 = Math.min(...outer.map((p) => p.x)), mx1 = Math.max(...outer.map((p) => p.x)), my0 = Math.min(...outer.map((p) => p.y)), my1 = Math.max(...outer.map((p) => p.y));
  const cx = (mx0 + mx1) / 2, cy = (my0 + my1) / 2, sw = (mx1 - mx0) * 1.7, sh = sw / 1.6;
  const fx0 = Math.min(P[234].x, P[454].x), fx1 = Math.max(P[234].x, P[454].x), fy0 = P[10].y, fy1 = P[152].y;
  const fw = (fx1 - fx0) * 1.35, fh = (fy1 - fy0) * 1.3, fcx = (fx0 + fx1) / 2, fcy = (fy0 + fy1) / 2;
  const fit = (x, y, w, h) => { w = Math.min(w, W); h = Math.min(h, H); return { x: Math.max(0, Math.min(W - w, x - w / 2)), y: Math.max(0, Math.min(H - h, y - h / 2)), w, h }; };
  return { smile: fit(cx, cy, sw, sh), face: fit(fcx, fcy, Math.max(fw, fh * 0.8), Math.max(fh, fw / 0.8)) };
}

function beforeAfter(beforeSrc, afterSrc, rect, opts = {}) {
  const k = Math.min(1, (opts.maxW || 1400) / rect.w), w = Math.round(rect.w * k), hgt = Math.round(rect.h * k);
  const b = canvasOf(w, hgt), a = canvasOf(w, hgt);
  const draw = (cv, src) => { const x = cv.getContext('2d'); x.imageSmoothingQuality = 'high'; x.drawImage(src, rect.x, rect.y, rect.w, rect.h, 0, 0, w, hgt); };
  draw(b, beforeSrc); if (afterSrc) draw(a, afterSrc);
  const afterWrap = h('div', { class: 'after' }, a), handle = h('div', { class: 'handle' }), busy = h('div', { class: 'busy hidden' }, 'جارٍ التحديث…');
  const el = h('div', { class: 'ba', style: { '--ar': (w / hgt).toFixed(3) } }, b, afterWrap, handle, h('span', { class: 'tag', style: { left: '10px' } }, 'قبل'), h('span', { class: 'tag', style: { right: '10px' } }, 'بعد'), busy);
  let pos = 0.5;
  const set = (f) => { pos = Math.max(0, Math.min(1, f)); afterWrap.style.clipPath = `inset(0 0 0 ${pos * 100}%)`; handle.style.left = pos * 100 + '%'; };
  let drag = false;
  const at = (e) => { const r = el.getBoundingClientRect(); set((e.clientX - r.left) / r.width); };
  el.addEventListener('pointerdown', (e) => { drag = true; el.setPointerCapture(e.pointerId); at(e); });
  el.addEventListener('pointermove', (e) => { if (drag) at(e); });
  el.addEventListener('pointerup', () => { drag = false; }); el.addEventListener('pointercancel', () => { drag = false; });
  set(0.5);
  return { el, setAfter(src) { draw(a, src); }, setBusy(v) { busy.classList.toggle('hidden', !v); } };
}

async function viewResults(root, caseId) {
  const c = await DB.get('cases', caseId); if (!c) return go('#/');
  const p = await DB.get('patients', c.patientId);
  if (!c.analysis) {
    root.append(appbar(p?.name || 'حالة', '#/'), h('main', null, h('div', { class: 'card empty' }, 'لا توجد صورة لهذه الحالة بعد.')),
      actionbar(h('button', { class: 'btn primary grow', onclick: () => go('#/capture/' + c.id) }, '📷 تصوير'), h('button', { class: 'btn', onclick: () => pickFromGallery(c.id) }, 'من المعرض')));
    return;
  }
  const main = h('main', null, h('div', { class: 'card empty' }, 'جارٍ التحضير…'));
  root.append(appbar(p?.name || 'حالة', '#/', h('button', { class: 'iconbtn', 'aria-label': 'المزيد', onclick: () => moreMenu(c) }, '⋮')), main);
  await withReady(async () => {
    let st = Results.pending && Results.pending.geom ? Results.pending : await restoreState(c);
    Results.pending = null;
    st.caliperMm = c.caliperMm || null; st.broken = c.broken || {};
    Results.live = { st, c };
    await initSimulation(st);
    const d = Object.assign({ ...DEFAULT_DESIGN }, c.design || {});
    let design = computeDesign(st, d);
    const rects = viewRects(st);
    let view = 'smile';
    const afterCanvas = canvasOf(st.fullImg.width, st.fullImg.height);
    let ba = beforeAfter(st.full, null, rects[view]);
    const baHolder = h('div', { class: 'mediawrap' }, ba.el);
    const viewSeg = h('div', { class: 'seg', role: 'tablist' });
    const drawSeg = () => viewSeg.replaceChildren(...[['smile', 'الابتسامة'], ['face', 'الوجه كاملاً']].map(([k, l]) => h('button', { class: view === k ? 'on' : '', role: 'tab', onclick: () => { view = k; drawSeg(); ba = beforeAfter(st.full, afterCanvas, rects[view]); baHolder.replaceChildren(ba.el); } }, l)));
    drawSeg();

    let simSeq = 0, timer = null;
    const refresh = async (immediate) => {
      clearTimeout(timer);
      const run = async () => {
        const my = ++simSeq; ba.setBusy(true);
        design = computeDesign(st, d);
        if (!design.ok) { ba.setBusy(false); return; }
        const r = await runSimulation(st, design, d);
        if (my !== simSeq) return;
        afterCanvas.getContext('2d').putImageData(r.full, 0, 0); ba.setAfter(afterCanvas); ba.setBusy(false);
        drawFindings(); drawIssues();
        st.lastSimMs = r.ms;
      };
      if (immediate) return run(); timer = setTimeout(run, 120);
    };
    const save = async () => { c.design = { ...d }; c.updated = Date.now(); if (c.approval) { c.approval = null; toast('عُدّل التصميم: يلزم اعتماد جديد'); drawAction(); } await DB.put('cases', c); };
    const change = (k, v) => { d[k] = v; drawControls(); refresh(); save(); };

    const issuesBox = h('div');
    const drawIssues = () => {
      const iss = detectionIssues(st);
      issuesBox.replaceChildren(...iss.map((i) => h('div', { class: 'notice ' + i.level }, h('span', { class: 'grow' }, i.text), i.needsReview ? h('button', { class: 'btn link', onclick: () => openReview() }, 'مراجعة') : null)));
    };
    const findingsBox = h('div', { class: 'card' });
    const drawFindings = () => {
      const f = findings(st, design);
      findingsBox.replaceChildren(h('h2', null, 'التحليل'), ...(f.length ? f.slice(0, 6).map((x) => h('div', { class: 'finding' }, h('span', { class: 'dot ' + x.status }), h('div', null, h('b', null, x.title), h('div', { class: 'small muted' }, x.text)))) : [h('div', { class: 'muted small' }, 'لا تتوفر قياسات كافية.')]),
        h('div', { class: 'small muted', style: { marginTop: '8px' } }, st.scaleSource === 'caliper' ? 'المقاسات معايرة بعرض الثنية المقاس بالفرجار.' : 'المقاسات بالملم تقديرية من حجم قزحية العين (±10%). لدقة أعلى أدخل عرض الثنية بالفرجار عند ورقة المعمل.'));
    };
    const controls = h('div', { class: 'card' });
    const chipRow = (opts, cur, onPick) => h('div', { class: 'chips' }, opts.map(([v, l]) => h('button', { class: 'chip' + (cur === v ? ' on' : ''), onclick: () => onPick(v) }, l)));
    const drawControls = () => {
      const out = h('output', null, (d.lengthMm > 0 ? '+' : '') + d.lengthMm.toFixed(1) + ' ملم');
      controls.replaceChildren(h('h2', null, 'التصميم'),
        h('div', { class: 'small muted' }, 'الشكل'), chipRow(Object.entries(SHAPE_LABEL), d.shape, (v) => change('shape', v)),
        h('div', { class: 'small muted', style: { marginTop: '12px' } }, 'اللون'), chipRow(SHADE_OPTIONS.map((s) => [s.id, s.label]), d.shade, (v) => change('shade', v)),
        h('div', { class: 'row', style: { marginTop: '12px' } }, h('div', { class: 'grow' }, h('div', { class: 'small muted' }, 'طول الأسنان'),
          h('div', { class: 'stepper' }, h('button', { 'aria-label': 'أقصر', onclick: () => change('lengthMm', Math.max(-2, +(d.lengthMm - 0.5).toFixed(1))) }, '−'), out, h('button', { 'aria-label': 'أطول', onclick: () => change('lengthMm', Math.min(3, +(d.lengthMm + 0.5).toFixed(1))) }, '+')))),
        h('details', { class: 'adv' }, h('summary', null, 'خيارات متقدمة'),
          h('div', { class: 'small muted' }, 'تناسب العرض'), chipRow([['red70', 'طبيعي (70%)'], ['golden', 'النسبة الذهبية'], ['red80', 'عريض (80%)']], d.proportion, (v) => change('proportion', v)),
          h('div', { class: 'small muted', style: { marginTop: '10px' } }, 'خط الوسط'), chipRow([['current', 'الحالي'], ['facial', 'منتصف الوجه']], d.midline, (v) => change('midline', v)),
          h('div', { class: 'small muted', style: { marginTop: '10px' } }, 'العلاج لكل سن'),
          h('div', { class: 'tablewrap' }, h('table', { class: 't' }, h('tbody', null, C.TEETH.map((t) => {
            const sel = h('select', { id: 'treat-' + t }, Object.entries(TREAT_LABEL).map(([k, l]) => h('option', { value: k, selected: design.treat?.[t] === k }, l)));
            sel.addEventListener('change', () => { d.treat = { ...(d.treat || {}), [t]: sel.value }; refresh(); save(); });
            return h('tr', null, h('td', null, h('b', null, t), ' ', h('span', { class: 'small muted' }, STATUS_LABEL[st.status[t]] || '')), h('td', null, sel));
          }))))));
    };
    const action = h('div');
    const drawAction = () => action.replaceChildren(actionbar(
      c.approval ? h('button', { class: 'btn primary grow', onclick: () => outputs(c, p, st, design, d, afterCanvas) }, 'المخرجات والمشاركة') : h('button', { class: 'btn primary grow', onclick: () => approve() }, 'اعتماد التصميم'),
      h('a', { class: 'btn', href: '#/present/' + c.id, onclick: () => { Results.present = { st, afterCanvas, rects }; } }, 'عرض للمريض')));
    const openReview = () => reviewTeeth(st, c, async () => { await initSimulation(st); refresh(true); drawIssues(); });
    const approve = async () => {
      const needs = detectionIssues(st).filter((i) => i.needsReview);
      if (needs.length && !c.reviewed) { if (await confirmBox('راجع تحديد الأسنان أولاً', needs.map((n) => n.text).join(' '), 'مراجعة الآن')) openReview(); return; }
      const chk = h('input', { type: 'checkbox', id: 'approve-check' }), name = h('input', { type: 'text', id: 'approve-name', value: S.doctorName || '' });
      const ok = await modal('اعتماد التصميم', h('div', null, h('label', { class: 'check' }, chk, h('span', null, 'راجعتُ التحليل والتصميم والمحاكاة، وأعتمدها قراراً سريرياً مني.')), h('label', { class: 'field' }, h('span', null, 'اسم الطبيب'), name)),
        [{ label: 'إلغاء', value: false }, { label: 'اعتماد', value: () => chk.checked && name.value.trim() ? true : 'invalid', primary: true }]);
      if (ok === 'invalid') { toast('أكّد المراجعة واكتب اسم الطبيب'); return; }
      if (!ok) return;
      S.doctorName = name.value.trim(); await saveSettings();
      c.approval = { by: S.doctorName, at: Date.now() }; c.updated = Date.now(); await DB.put('cases', c);
      await saveThumb(c, st, afterCanvas, rects.smile); drawAction(); toast('اعتُمد التصميم'); outputs(c, p, st, design, d, afterCanvas);
    };
    if (noTeeth(st)) {
      const still = canvasOf(rects.face.w, rects.face.h); still.getContext('2d').drawImage(st.full, rects.face.x, rects.face.y, rects.face.w, rects.face.h, 0, 0, still.width, still.height);
      still.style.cssText = 'width:100%;border-radius:14px;display:block';
      main.replaceChildren(h('div', { class: 'mediawrap' }, still), h('div', { class: 'card' }, h('h2', null, 'لا يمكن تصميم هذه الصورة'),
        h('p', { class: 'muted' }, 'لا تظهر الأسنان العلوية. التحليل والتصميم يحتاجان صورة ابتسامة عريضة تظهر فيها الأسنان.'),
        h('details', { class: 'adv' }, h('summary', null, 'تفاصيل للمطوّر'), h('pre', { class: 'small', style: { whiteSpace: 'pre-wrap', direction: 'ltr', textAlign: 'left' } }, JSON.stringify({
          engine: Engine.info.ep, fallback: Engine.info.fallback, selfTest: Engine.info.selfTest && { ep: Engine.info.selfTest.ep, ok: Engine.info.selfTest.ok }, segMs: st.segMs,
          toothPixels: st.geom.labels ? Array.from({ length: 6 }, (_, i) => st.geom.labels.reduce((s, v) => s + (v === i + 1), 0)) : 'n/a',
          rejected: st.sel.rejected, open: +(st.quality?.open || 0).toFixed(3), mouthW: Math.round(st.quality?.mouthW || 0), crop: [st.geom.crop.outW, st.geom.crop.outH], failed: (st.quality?.items || []).filter((i) => !i.ok).map((i) => i.id)
        }, null, 1)))),
        actionbar(h('button', { class: 'btn primary grow', onclick: () => go('#/capture/' + c.id) }, '📷 التقاط صورة جديدة'), h('button', { class: 'btn', onclick: () => pickFromGallery(c.id) }, 'من المعرض')));
      return;
    }
    main.replaceChildren(baHolder, h('div', { style: { margin: '0 0 12px' } }, viewSeg), issuesBox, findingsBox, controls, action);
    drawControls(); drawIssues(); drawFindings(); drawAction();
    await refresh(true);
    if (!c.thumbId) saveThumb(c, st, afterCanvas, rects.smile);
  });
}

async function saveThumb(c, st, afterCanvas, rect) {
  const cv = canvasOf(240, 150); cv.getContext('2d').drawImage(afterCanvas, rect.x, rect.y, rect.w, rect.h, 0, 0, 240, 150);
  if (c.thumbId) await DB.del('blobs', c.thumbId);
  c.thumbId = await putBlob(await toBlob(cv, 0.8)); await DB.put('cases', c);
}

async function moreMenu(c) {
  const v = await modal('خيارات الحالة', h('div'), [{ label: 'إلغاء', value: null }, { label: 'تصوير جديد', value: 'retake' }, { label: 'حذف الحالة', value: 'del', danger: true }]);
  if (v === 'retake') go('#/capture/' + c.id);
  if (v === 'del' && await confirmBox('حذف الحالة', 'ستُحذف الحالة وصورتها من هذا الجهاز نهائياً.', 'حذف', true)) {
    for (const k of ['photoId', 'thumbId']) if (c[k]) await DB.del('blobs', c[k]);
    await DB.del('cases', c.id); go('#/');
  }
}

// ---------- tooth review: only when needed; tap a tooth, drag its edges, set its status ----------
function reviewTeeth(st, c, onDone) {
  const w = st.cropImg.width, hh = st.cropImg.height, teeth = C.clone(st.teeth), status = { ...st.status };
  let sel = C.TEETH.find((t) => status[t] === 'unsure') || '11';
  const cv = canvasOf(w, hh); cv.getContext('2d').putImageData(st.cropImg, 0, 0);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', `0 0 ${w} ${hh}`); svg.setAttribute('preserveAspectRatio', 'none');
  const ed = h('div', { class: 'editor' }, cv, svg);
  const COLORS = { '13': '#e6194b', '12': '#f58231', '11': '#ffe119', '21': '#3cb44b', '22': '#42d4f4', '23': '#4363d8' };
  const sz = () => w / (ed.clientWidth || w);
  const S_ = (tag, a) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); return e; };
  const draw = () => {
    svg.innerHTML = ''; const r = sz();
    for (const t of C.TEETH) {
      const b = teeth[t], on = t === sel, miss = status[t] !== 'present';
      svg.append(S_('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: on ? 'rgba(255,255,255,.15)' : 'transparent', stroke: COLORS[t], 'stroke-width': (on ? 3 : 1.6) * r, 'stroke-dasharray': miss ? `${5 * r} ${4 * r}` : 'none', 'data-t': t }));
      const tx = S_('text', { x: (b.x0 + b.x1) / 2, y: b.y0 - 5 * r, fill: COLORS[t], 'font-size': 14 * r, 'text-anchor': 'middle', 'font-weight': 700, stroke: '#000', 'stroke-width': 3 * r, 'paint-order': 'stroke' }); tx.textContent = t; svg.append(tx);
    }
    const b = teeth[sel], mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
    for (const [k, x, y] of [['x0', b.x0, my], ['x1', b.x1, my], ['y0', mx, b.y0], ['y1', mx, b.y1]]) svg.append(S_('circle', { cx: x, cy: y, r: Math.min(13 * r, (b.x1 - b.x0) / 3), fill: '#fff', stroke: COLORS[sel], 'stroke-width': 3 * r, 'data-h': k }));
  };
  let drag = null;
  const pt = (e) => { const r = svg.getBoundingClientRect(); return { x: (e.clientX - r.left) / r.width * w, y: (e.clientY - r.top) / r.height * hh }; };
  svg.addEventListener('pointerdown', (e) => {
    const hk = e.target.dataset?.h, tk = e.target.dataset?.t;
    if (tk && !hk) { sel = tk; drawStatus(); draw(); return; }
    if (hk) { drag = { k: hk, id: e.pointerId }; svg.setPointerCapture(e.pointerId); e.preventDefault(); }
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return; const q = pt(e), b = teeth[sel], i = C.TEETH.indexOf(sel), L = C.TEETH[i - 1], R = C.TEETH[i + 1];
    if (drag.k === 'x0') { b.x0 = Math.min(q.x, b.x1 - 6); if (L) teeth[L].x1 = Math.max(b.x0, teeth[L].x0 + 6); }
    if (drag.k === 'x1') { b.x1 = Math.max(q.x, b.x0 + 6); if (R) teeth[R].x0 = Math.min(b.x1, teeth[R].x1 - 6); }
    if (drag.k === 'y0') b.y0 = Math.min(q.y, b.y1 - 6); if (drag.k === 'y1') b.y1 = Math.max(q.y, b.y0 + 6);
    draw();
  });
  svg.addEventListener('pointerup', () => { drag = null; });
  const statusBox = h('div');
  const drawStatus = () => statusBox.replaceChildren(h('div', { class: 'small muted', style: { margin: '10px 0 6px' } }, `السن ${sel}: ${C.TOOTH_NAME[sel]}`),
    h('div', { class: 'chips' }, [['present', 'موجود'], ['broken', 'مكسور'], ['missing', 'مفقود'], ['hidden', 'غير ظاهر']].map(([k, l]) => h('button', { class: 'chip' + (status[sel] === k ? ' on' : ''), onclick: () => { status[sel] = k; drawStatus(); draw(); } }, l))),
    h('div', { class: 'chips', style: { marginTop: '8px' } }, C.TEETH.map((t) => h('button', { class: 'chip' + (t === sel ? ' on' : ''), onclick: () => { sel = t; drawStatus(); draw(); } }, t))));
  const bg = h('div', { class: 'modalbg' });
  const box = h('div', { class: 'modal' }, h('h2', null, 'مراجعة تحديد الأسنان'), h('p', { class: 'small muted' }, 'اضغط على السن، واسحب النقاط البيضاء لتصحيح حدوده: الجانبان عند نقاط التماس، والأعلى عند خط اللثة، والأسفل عند الحافة القاطعة.'),
    ed, statusBox, h('div', { class: 'row', style: { marginTop: '14px' } }, h('button', { class: 'btn', onclick: () => bg.remove() }, 'إلغاء'), h('button', { class: 'btn primary grow', onclick: async () => {
      const edited = c.toothMasksEdited || {};
      for (const t of C.TEETH) {
        const a = st.teeth[t], b = teeth[t], moved = Math.abs(a.x0 - b.x0) + Math.abs(a.x1 - b.x1) + Math.abs(a.y0 - b.y0) + Math.abs(a.y1 - b.y1) > 2;
        if (status[t] !== 'present' && status[t] !== 'broken') { delete st.toothMasks[t]; delete edited[t]; }
        else if (moved || !st.toothMasks[t]) { st.toothMasks[t] = maskForBox(st, b); edited[t] = b; }
      }
      st.teeth = teeth; st.status = status; computeMeasures(st);
      c.analysis.teeth = teeth; c.analysis.status = status; c.toothMasksEdited = edited; c.reviewed = true; c.updated = Date.now();
      if (c.approval) c.approval = null;
      await DB.put('cases', c); bg.remove(); onDone();
    } }, 'تم')));
  bg.append(box); document.body.append(bg);
  drawStatus(); requestAnimationFrame(draw);
}

// ---------- outputs ----------
async function outputs(c, p, st, design, d, afterCanvas) {
  const v = await modal('المخرجات', h('p', { class: 'small muted' }, c.approval ? `معتمد من ${c.approval.by} · ${fmtDateTime(c.approval.at)}` : 'مسودة غير معتمدة'),
    [{ label: 'إغلاق', value: null }, { label: 'صورة قبل/بعد', value: 'share' }, ...(st.server?.svg ? [{ label: 'ملف المعمل 1:1 ملم', value: 'svg' }] : []), { label: 'ورقة المعمل', value: 'lab', primary: true }]);
  if (v === 'svg') labFiles(c, p, st);
  if (v === 'share') shareBeforeAfter(c, p, st, afterCanvas);
  if (v === 'lab') labSheet(c, p, st, d, afterCanvas);
}
// The server's validated design: an SVG drawn at 1:1 mm (prints at true size, 10 mm ruler to check) and its JSON
async function labFiles(c, p, st) {
  const base = `smile-lab-${(p?.fileNo || c.id.slice(0, 6))}`;
  const files = [new File([st.server.svg], base + '.svg', { type: 'image/svg+xml' }),
    new File([JSON.stringify({ ...st.server.design, approval: c.approval || null }, null, 1)], base + '.json', { type: 'application/json' })];
  try { if (navigator.canShare && navigator.canShare({ files })) { await navigator.share({ files, title: 'ملف المعمل' }); return; } } catch (e) { if (e.name === 'AbortError') return; }
  for (const f of files) { const u = URL.createObjectURL(f), a = h('a', { href: u, download: f.name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 4000); }
}
async function composite(st, afterCanvas, rect, title) {
  const W = 1600, ph = Math.round((W / 2) * rect.h / rect.w), H = ph + 90, cv = canvasOf(W, H), x = cv.getContext('2d');
  x.fillStyle = '#ffffff'; x.fillRect(0, 0, W, H);
  x.drawImage(st.full, rect.x, rect.y, rect.w, rect.h, 0, 0, W / 2 - 4, ph);
  x.drawImage(afterCanvas, rect.x, rect.y, rect.w, rect.h, W / 2 + 4, 0, W / 2 - 4, ph);
  x.fillStyle = '#16201b'; x.font = '600 30px sans-serif'; x.textAlign = 'center';
  x.fillText('قبل', W / 4, ph + 40); x.fillText('بعد (تصميم مقترح)', 3 * W / 4, ph + 40);
  x.font = '22px sans-serif'; x.fillStyle = '#56635c'; x.fillText(title, W / 2, ph + 76);
  return cv;
}
async function shareBeforeAfter(c, p, st, afterCanvas) {
  const rects = viewRects(st);
  const files = [];
  for (const [k, r] of Object.entries(rects)) files.push(new File([await toBlob(await composite(st, afterCanvas, r, `${S.clinicName} — تصميم مقترح وليس النتيجة النهائية`), 0.9)], `smile-${k}-${(p?.fileNo || c.id.slice(0, 6))}.jpg`, { type: 'image/jpeg' }));
  try { if (navigator.canShare && navigator.canShare({ files })) { await navigator.share({ files, title: 'تصميم الابتسامة' }); return; } } catch (e) { if (e.name === 'AbortError') return; }
  for (const f of files) { const u = URL.createObjectURL(f), a = h('a', { href: u, download: f.name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 4000); }
}
async function labSheet(c, p, st, d, afterCanvas) {
  const cal = h('input', { type: 'number', step: '0.1', min: '5', max: '12', id: 'caliper', value: c.caliperMm || '', placeholder: 'مثلاً 8.6' });
  const mat = h('input', { type: 'text', id: 'material', value: c.material || S.material });
  const notes = h('textarea', { id: 'labnotes' }, c.labNotes || '');
  const go_ = await modal('ورقة المعمل', h('div', null,
    h('label', { class: 'field' }, h('span', null, 'عرض الثنية 11 بالفرجار (ملم، اختياري، يجعل المقاسات دقيقة)'), cal),
    h('label', { class: 'field' }, h('span', null, 'المادة'), mat), h('label', { class: 'field' }, h('span', null, 'ملاحظات للمعمل'), notes)),
  [{ label: 'إلغاء', value: false }, { label: 'طباعة / PDF', value: true, primary: true }]);
  if (!go_) return;
  c.caliperMm = +cal.value || null; c.material = mat.value; c.labNotes = notes.value; await DB.put('cases', c);
  st.caliperMm = c.caliperMm; computeMeasures(st);
  const design = computeDesign(st, d); const r = await runSimulation(st, design, d); afterCanvas.getContext('2d').putImageData(r.full, 0, 0);
  const rects = viewRects(st), smile = await composite(st, afterCanvas, rects.smile, ''), face = await composite(st, afterCanvas, rects.face, '');
  const mm = !!st.mmppCrop, f1 = (v) => mm ? num(v, 1) : num(v, 0);
  const unit = mm ? 'ملم' : 'بكسل';
  const fnd = findings(st, design);
  const html = `<div class="sheet" dir="rtl">
  <div class="hdr"><div><h1>${esc(S.clinicName)}</h1><div>ورقة المعمل — تصميم ابتسامة رقمي</div></div><div style="text-align:left">${fmtDate(Date.now())}<br>رقم الحالة ${esc(c.id.slice(0, 8))}</div></div>
  ${!c.approval ? '<div class="warn"><b>مسودة غير معتمدة</b> — لا تُنفّذ قبل اعتماد الطبيب.</div>' : ''}
  ${st.scaleSource !== 'caliper' ? '<div class="warn">المقاسات بالملم تقديرية من صورة الوجه (±10%). للتنفيذ اعتمد على القياس الفعلي أو النموذج الدراسي.</div>' : ''}
  <h2>المريض</h2><table><tr><th>الاسم</th><td>${esc(p?.name)}</td><th>رقم الملف</th><td>${esc(p?.fileNo || '—')}</td></tr></table>
  <h2>قبل / بعد</h2><img src="${smile.toDataURL('image/jpeg', 0.85)}" style="width:100%;border:1px solid #ccc"><img src="${face.toDataURL('image/jpeg', 0.8)}" style="width:100%;border:1px solid #ccc;margin-top:6px">
  <h2>المواصفات لكل سن (${unit})</h2>
  <table><tr><th>السن</th><th>الحالة</th><th>العلاج</th><th>الحالي ع × ط</th><th>المقترح ع × ط</th><th>التغيير في الطول</th></tr>
  ${C.TEETH.map((t) => { const a = st.teeth[t], ch = design.changes[t], k = st.mmppCrop || 1; return `<tr><td><b>${t}</b> ${C.TOOTH_NAME[t]}</td><td>${STATUS_LABEL[st.status[t]] || ''}</td><td>${TREAT_LABEL[design.treat[t]]}</td><td>${st.status[t] === 'present' ? f1((a.x1 - a.x0) * k) + ' × ' + f1((a.y1 - a.y0) * k) : '—'}</td><td>${design.treat[t] === 'none' ? '—' : f1(ch.w) + ' × ' + f1(ch.h)}</td><td>${design.treat[t] === 'none' || st.status[t] !== 'present' ? '—' : (ch.dIncisal >= 0 ? '+' : '') + f1(ch.dIncisal)}</td></tr>`; }).join('')}
  </table>
  <table style="margin-top:6px"><tr><th>الشكل</th><td>${SHAPE_LABEL[d.shape]}</td><th>اللون</th><td>${esc(d.shade)}</td><th>المادة</th><td>${esc(c.material || '')}</td></tr>
  <tr><th>التناسب</th><td>${num(design.prop.p, 2)}</td><th>ع/ط الثنية</th><td>${num(design.measAfter.ratios.central_wl, 2)}</td><th>خط الوسط</th><td>${d.midline === 'facial' ? 'منتصف الوجه' : 'الحالي'}</td></tr></table>
  <h2>التحليل</h2><table>${fnd.map((x) => `<tr><td>${esc(x.title)}</td><td>${esc(x.text)}</td></tr>`).join('')}</table>
  ${c.labNotes ? `<h2>ملاحظات</h2><div style="white-space:pre-wrap">${esc(c.labNotes)}</div>` : ''}
  <div style="display:flex;justify-content:space-between;margin-top:22px"><div>الطبيب: ${esc(c.approval?.by || S.doctorName || '........')}<br>${c.approval ? 'اعتُمد ' + fmtDateTime(c.approval.at) : 'غير معتمد'}</div><div>التوقيع: ..............</div></div>
  <p style="color:#666;font-size:10px;margin-top:12px">المحاكاة تعديل تقريبي لصورة المريض وليست النتيجة النهائية. القياسات من صورة ثنائية الأبعاد؛ عرض الرباعية والناب ظاهري. القرار السريري للطبيب.</p></div>`;
  let pr = $('#print-root'); if (!pr) { pr = h('div', { id: 'print-root' }); document.body.append(pr); }
  pr.innerHTML = html; await Promise.all([...pr.querySelectorAll('img')].map((im) => im.decode().catch(() => {})));
  window.__lastSheet = html;
  if (!window.__DSD_NO_PRINT) window.print();
}

// ---------- presentation for the patient ----------
async function viewPresent(root, caseId) {
  let ctx = Results.present;
  if (!ctx) {
    const c = await DB.get('cases', caseId); if (!c?.analysis) return go('#/');
    await withReady(async () => {
      const st = await restoreState(c); st.caliperMm = c.caliperMm || null; await initSimulation(st);
      const d = Object.assign({ ...DEFAULT_DESIGN }, c.design || {}), design = computeDesign(st, d), r = await runSimulation(st, design, d);
      const afterCanvas = canvasOf(st.fullImg.width, st.fullImg.height); afterCanvas.getContext('2d').putImageData(r.full, 0, 0);
      ctx = { st, afterCanvas, rects: viewRects(st) };
    });
    if (!ctx) return;
  }
  Results.present = null;
  let view = 'smile';
  const stage = h('div', { class: 'stage' });
  const seg = h('div', { class: 'seg', style: { background: 'rgba(255,255,255,.12)' } });
  const draw = () => { stage.replaceChildren(beforeAfter(ctx.st.full, ctx.afterCanvas, ctx.rects[view], { maxW: 2000 }).el); seg.replaceChildren(...[['smile', 'الابتسامة'], ['face', 'الوجه']].map(([k, l]) => h('button', { class: view === k ? 'on' : '', onclick: () => { view = k; draw(); } }, l))); };
  root.append(h('div', { class: 'present' }, h('div', { class: 'bar' }, h('a', { class: 'iconbtn', href: '#/case/' + caseId, 'aria-label': 'إغلاق' }, '✕'), h('b', { class: 'grow' }, 'ابتسامتك بعد العلاج'), seg), stage,
    h('div', { class: 'note' }, 'تصميم مقترح على صورتك للتوضيح، وليس النتيجة النهائية. اسحب الخط لترى الفرق.')));
  draw();
}
