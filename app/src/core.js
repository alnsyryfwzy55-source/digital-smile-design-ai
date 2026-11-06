/* DSD core: pure functions (no DOM). Works in browser (window.DSDCore) and Node (module.exports). */
(function (root) {
  'use strict';

  // Order left→right as seen in a frontal photo (patient's right appears on viewer's left).
  const TEETH = ['13', '12', '11', '21', '22', '23'];
  const CLASS_OF = { '13': 1, '12': 2, '11': 3, '21': 4, '22': 5, '23': 6 };
  const FLIP_CHANNELS = [0, 6, 5, 4, 3, 2, 1];
  const TOOTH_NAME = {
    '13': 'ناب أيمن', '12': 'رباعية يمنى', '11': 'ثنية يمنى',
    '21': 'ثنية يسرى', '22': 'رباعية يسرى', '23': 'ناب أيسر'
  };

  const PROPORTIONS = {
    golden: { label: 'النسبة الذهبية (62%)', p: 0.618 },
    red70: { label: 'RED ثابتة 70%', p: 0.70 },
    red75: { label: 'RED ثابتة 75%', p: 0.75 },
    red80: { label: 'RED ثابتة 80%', p: 0.80 }
  };

  // Default rules: literature-based starting values; the doctor edits them in Settings.
  const DEFAULT_RULES = [
    { id: 'central_wl', group: 'teeth', label: 'نسبة عرض الثنية إلى طولها', min: 0.75, max: 0.85, target: 0.80, unit: 'ratio', enabled: true },
    { id: 'lat_central_w', group: 'teeth', label: 'عرض الرباعية إلى الثنية (ظاهري)', tol: 0.06, unit: 'ratio', enabled: true, note: 'الهدف من النسبة المختارة في التصميم' },
    { id: 'can_lat_w', group: 'teeth', label: 'عرض الناب إلى الرباعية (ظاهري)', tol: 0.08, unit: 'ratio', enabled: true, note: 'الهدف من النسبة المختارة في التصميم' },
    { id: 'lat_central_len', group: 'teeth', label: 'طول الرباعية إلى طول الثنية', min: 0.80, max: 0.90, target: 0.85, unit: 'ratio', enabled: true },
    { id: 'can_central_len', group: 'teeth', label: 'طول الناب إلى طول الثنية', min: 0.88, max: 1.00, target: 0.95, unit: 'ratio', enabled: true },
    { id: 'sym_central', group: 'teeth', label: 'تماثل الثنيتين (فرق العرض والطول)', max: 0.05, unit: 'pct', enabled: true },
    { id: 'sym_lateral', group: 'teeth', label: 'تماثل الرباعيتين', max: 0.08, unit: 'pct', enabled: true },
    { id: 'sym_canine', group: 'teeth', label: 'تماثل النابين', max: 0.10, unit: 'pct', enabled: true },
    { id: 'central_dominance', group: 'teeth', label: 'سيادة الثنيتين (أعرض الأسنان ظاهرياً)', unit: 'bool', enabled: true },
    { id: 'central_len_mm', group: 'teeth', label: 'طول الثنية بالملم', min: 10.0, max: 11.5, target: 10.8, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'central_w_mm', group: 'teeth', label: 'عرض الثنية بالملم', min: 8.0, max: 9.5, target: 8.6, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'zenith_lateral', group: 'gingiva', label: 'لثة الرباعية أنزل من لثة الثنية', min: 0.0, max: 1.5, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'zenith_canine', group: 'gingiva', label: 'لثة الناب بمستوى لثة الثنية', min: -0.5, max: 0.5, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'zenith_sym', group: 'gingiva', label: 'تماثل مستوى لثة الثنيتين', max: 0.5, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'incisal_step', group: 'teeth', label: 'الحافة القاطعة للرباعية أقصر من الثنية', min: 0.5, max: 1.5, unit: 'mm', needs: 'calib', enabled: true },
    { id: 'midline_dev', group: 'face', label: 'انحراف خط الوسط السني عن خط الوجه', max: 2.0, unit: 'mm', needs: 'face', enabled: true },
    { id: 'commissure_cant', group: 'face', label: 'ميل خط زوايا الفم عن خط العينين', max: 2.0, unit: 'deg', needs: 'face', enabled: true },
    { id: 'incisal_cant', group: 'face', label: 'ميل مستوى الأنياب (المستوى القاطع) عن خط العينين', max: 2.0, unit: 'deg', needs: 'face', enabled: true },
    { id: 'gingival_display', group: 'face', label: 'ظهور اللثة عند الابتسامة', min: 0.0, max: 2.0, unit: 'mm', needs: 'face', enabled: true },
    { id: 'buccal_corridor', group: 'face', label: 'عرض الأسنان الظاهرة إلى عرض الابتسامة', min: 0.82, max: 0.92, unit: 'ratio', needs: 'face', enabled: true },
    { id: 'smile_arc', group: 'face', label: 'قوس الابتسامة يوازي الشفة السفلى', unit: 'class', needs: 'face', enabled: true },
    { id: 'rest_display', group: 'face', label: 'ظهور الثنية والشفاه في الراحة', min: 1.0, max: 3.0, target: 2.0, unit: 'mm', needs: 'rest', enabled: true, note: 'يقلّ مع العمر وعند الرجال عادةً؛ عدّله حسب المريض' },
    { id: 'lower_third', group: 'face', label: 'الثلث السفلي إلى الأوسط من الوجه', min: 0.90, max: 1.10, unit: 'ratio', needs: 'face', enabled: true }
  ];

  const FACE_POINTS = [
    { id: 'pupilR', label: 'بؤبؤ العين اليمنى للمريض (يسار الصورة)' },
    { id: 'pupilL', label: 'بؤبؤ العين اليسرى للمريض (يمين الصورة)' },
    { id: 'glabella', label: 'منتصف ما بين الحاجبين' },
    { id: 'subnasale', label: 'أسفل الأنف (نقطة التقاء الأنف بالشفة)' },
    { id: 'menton', label: 'أسفل الذقن' },
    { id: 'commR', label: 'زاوية الفم اليمنى للمريض' },
    { id: 'commL', label: 'زاوية الفم اليسرى للمريض' },
    { id: 'upperLip', label: 'الحد السفلي للشفة العليا عند المنتصف' },
    { id: 'zenith11', label: 'أعلى لثة الثنية 11' },
    { id: 'incisal11', label: 'الحافة القاطعة للثنية 11' },
    { id: 'midlineDental', label: 'نقطة التماس بين الثنيتين (خط الوسط السني)' },
    { id: 'cusp13', label: 'رأس الناب 13' },
    { id: 'cusp23', label: 'رأس الناب 23' },
    { id: 'lastR', label: 'آخر سن ظاهر يمين المريض' },
    { id: 'lastL', label: 'آخر سن ظاهر يسار المريض' },
    { id: 'lowerLipMid', label: 'الحد العلوي للشفة السفلى عند المنتصف' },
    { id: 'lowerLipR', label: 'الشفة السفلى تحت الناب 13' },
    { id: 'lowerLipL', label: 'الشفة السفلى تحت الناب 23' }
  ];

  const SHADES = {
    BL1: '#F7F4EE', BL2: '#F3EFE6', BL3: '#EFEADF', BL4: '#EBE5D8',
    A1: '#EEE6D4', A2: '#E9DDC4', A3: '#E2D2B5', 'A3.5': '#DCC8A7', A4: '#D2BB97',
    B1: '#F0E9D8', B2: '#EAE0C8', B3: '#E1D2B1', C1: '#E5DDCC', C2: '#DCD1BC', D2: '#E6DCCB'
  };

  // ---------- geometry helpers ----------
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const deg = (r) => r * 180 / Math.PI;
  function percentile(sorted, p) {
    if (!sorted.length) return NaN;
    const i = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))));
    return sorted[i];
  }
  function lineAngleDeg(a, b) { return deg(Math.atan2(b.y - a.y, b.x - a.x)); }
  function normAngle(a) { while (a > 90) a -= 180; while (a < -90) a += 180; return a; }

  // Least squares tilt (degrees) of the tooth row from centers [{x,y}].
  function fitTilt(centers) {
    const pts = centers.filter(Boolean);
    if (pts.length < 2) return 0;
    const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    let sxx = 0, sxy = 0;
    for (const p of pts) { sxx += (p.x - mx) ** 2; sxy += (p.x - mx) * (p.y - my); }
    if (sxx < 1e-9) return 0;
    return deg(Math.atan(sxy / sxx));
  }

  // ---------- model postprocessing ----------
  // logits: Float32Array [C*S*S] (NCHW, N=1). Returns probabilities Float32Array same layout.
  function softmaxNCHW(logits, C, S) {
    const n = S * S, out = new Float32Array(C * n);
    for (let i = 0; i < n; i++) {
      let m = -Infinity;
      for (let c = 0; c < C; c++) m = Math.max(m, logits[c * n + i]);
      let s = 0;
      for (let c = 0; c < C; c++) { const e = Math.exp(logits[c * n + i] - m); out[c * n + i] = e; s += e; }
      for (let c = 0; c < C; c++) out[c * n + i] /= s;
    }
    return out;
  }
  // Average probabilities of an image and its horizontally flipped prediction (TTA).
  function mergeFlipTTA(prob, probFlipped, C, S) {
    const n = S * S, out = new Float32Array(C * n);
    for (let c = 0; c < C; c++) {
      const fc = FLIP_CHANNELS[c];
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        const i = y * S + x, j = y * S + (S - 1 - x);
        out[c * n + i] = 0.5 * (prob[c * n + i] + probFlipped[fc * n + j]);
      }
    }
    return out;
  }
  function argmaxNCHW(prob, C, S) {
    const n = S * S, labels = new Uint8Array(n), conf = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let b = 0, bv = prob[i];
      for (let c = 1; c < C; c++) { const v = prob[c * n + i]; if (v > bv) { bv = v; b = c; } }
      labels[i] = b; conf[i] = bv;
    }
    return { labels, conf };
  }
  // Keep the largest 4-connected component of each class; other pixels of that class become background.
  function largestComponents(labels, S, C) {
    const out = new Uint8Array(labels.length);
    const seen = new Uint8Array(labels.length);
    const stack = new Int32Array(labels.length);
    const best = new Array(C).fill(null);
    for (let i = 0; i < labels.length; i++) {
      const c = labels[i];
      if (c === 0 || seen[i]) continue;
      let sp = 0; stack[sp++] = i; seen[i] = 1; const comp = [];
      while (sp) {
        const k = stack[--sp]; comp.push(k);
        const x = k % S, y = (k / S) | 0;
        const nb = [x > 0 ? k - 1 : -1, x < S - 1 ? k + 1 : -1, y > 0 ? k - S : -1, y < S - 1 ? k + S : -1];
        for (const q of nb) if (q >= 0 && !seen[q] && labels[q] === c) { seen[q] = 1; stack[sp++] = q; }
      }
      if (!best[c] || comp.length > best[c].length) best[c] = comp;
    }
    for (let c = 1; c < C; c++) if (best[c]) for (const k of best[c]) out[k] = c;
    return out;
  }

  // Convert a label grid (S×S, squashed from image W×H) to tooth boxes in image pixels.
  // Coordinates are measured in a frame rotated by -tiltDeg around the image center if tiltDeg given.
  function labelsToTeeth(labels, conf, S, W, H, opts = {}) {
    const minFrac = opts.minAreaFrac ?? (300 / 102400);
    const sx = W / S, sy = H / S;
    const per = {};
    for (let c = 1; c <= 6; c++) per[c] = { xs: [], ys: [], conf: 0, n: 0 };
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = y * S + x, c = labels[i];
      if (!c) continue;
      const p = per[c]; p.xs.push((x + 0.5) * sx); p.ys.push((y + 0.5) * sy); p.conf += conf ? conf[i] : 1; p.n++;
    }
    const centers = {}, teeth = {}, info = {};
    let confSum = 0, confN = 0;
    for (const t of TEETH) {
      const p = per[CLASS_OF[t]];
      const frac = p.n / (S * S);
      info[t] = { area: p.n, frac, conf: p.n ? p.conf / p.n : 0, present: frac >= minFrac };
      if (p.n) { confSum += p.conf; confN += p.n; }
      if (info[t].present) centers[t] = { x: avg(p.xs), y: avg(p.ys) };
    }
    const tilt = fitTilt(TEETH.map((t) => centers[t]));
    const rot = (opts.tiltDeg ?? 0) * Math.PI / 180;
    const cx = W / 2, cy = H / 2, cs = Math.cos(-rot), sn = Math.sin(-rot);
    for (const t of TEETH) {
      if (!info[t].present) continue;
      const p = per[CLASS_OF[t]];
      const xs = [], ys = [];
      for (let k = 0; k < p.xs.length; k++) {
        const dx = p.xs[k] - cx, dy = p.ys[k] - cy;
        xs.push(cx + dx * cs - dy * sn); ys.push(cy + dx * sn + dy * cs);
      }
      xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
      teeth[t] = { x0: percentile(xs, 1), x1: percentile(xs, 99), y0: percentile(ys, 1), y1: percentile(ys, 99) };
    }
    const order = TEETH.filter((t) => centers[t]).map((t) => centers[t].x);
    let orderOk = order.length >= 2;
    for (let k = 1; k < order.length; k++) if (order[k] <= order[k - 1]) orderOk = false;
    const presentCount = TEETH.filter((t) => info[t].present).length;
    const meanConf = confN ? confSum / confN : 0;
    return { teeth, info, centers, tilt, orderOk, presentCount, meanConf, score: presentCount + meanConf + (orderOk ? 2 : 0) };
  }
  function avg(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

  function qualityReport(det, confMin = 0.91) {
    const issues = [];
    const missing = TEETH.filter((t) => !det.info[t].present);
    if (missing.length) issues.push({ level: 'fail', text: 'أسنان لم تُكتشف: ' + missing.join('، ') + ' — أضف حدودها يدوياً.' });
    if (!det.orderOk) issues.push({ level: 'fail', text: 'ترتيب الأسنان غير سليم (قد تكون الصورة مقلوبة أو جانبية).' });
    if (det.meanConf < confMin) issues.push({ level: 'warn', text: `ثقة النموذج ${det.meanConf.toFixed(2)} أقل من ${confMin} — راجع الحدود يدوياً بعناية.` });
    for (const t of TEETH) if (det.info[t].present && det.info[t].conf < confMin - 0.05) issues.push({ level: 'warn', text: `ثقة منخفضة للسن ${t} (${det.info[t].conf.toFixed(2)}).` });
    if (Math.abs(det.tilt) > 20) issues.push({ level: 'warn', text: `ميل الصورة ${det.tilt.toFixed(1)}° كبير.` });
    const level = issues.some((i) => i.level === 'fail') ? 'fail' : issues.length ? 'warn' : 'ok';
    return { level, issues };
  }

  // ---------- measurement ----------
  // status: {t: 'present'|'missing'|'broken'}; missing teeth are left out, broken teeth keep their width but not their length
  function measure(teeth, mmPerPx, status) {
    const st = status || {};
    const m = { calibrated: !!(mmPerPx && mmPerPx > 0), mmPerPx: mmPerPx || null, teeth: {}, missing: [], broken: [] };
    for (const t of TEETH) {
      const b = teeth && teeth[t];
      if (!b) continue;
      if (st[t] === 'missing') { m.missing.push(t); continue; }
      const broken = st[t] === 'broken'; if (broken) m.broken.push(t);
      const w = b.x1 - b.x0, h = b.y1 - b.y0;
      m.teeth[t] = { w, h, broken, wmm: m.calibrated ? w * mmPerPx : null, hmm: m.calibrated ? h * mmPerPx : null, wl: h > 0 && !broken ? w / h : null };
    }
    const g = (t) => m.teeth[t];
    const gh = (t) => (m.teeth[t] && !m.teeth[t].broken ? m.teeth[t].h : null);
    const mean = (a, b) => (a != null && b != null) ? (a + b) / 2 : (a ?? b ?? null);
    const cW = mean(g('11')?.w, g('21')?.w), cH = mean(gh('11'), gh('21'));
    const lW = mean(g('12')?.w, g('22')?.w), lH = mean(gh('12'), gh('22'));
    const kW = mean(g('13')?.w, g('23')?.w), kH = mean(gh('13'), gh('23'));
    m.ratios = {
      central_wl: cW && cH ? cW / cH : null,
      lat_central_w: lW && cW ? lW / cW : null,
      can_lat_w: kW && lW ? kW / lW : null,
      lat_central_len: lH && cH ? lH / cH : null,
      can_central_len: kH && cH ? kH / cH : null
    };
    const side = (a, b) => (g(a) && g(b)) ? {
      a, b, dw: Math.abs(g(a).w - g(b).w) / Math.max(g(a).w, g(b).w),
      dh: (gh(a) != null && gh(b) != null) ? Math.abs(gh(a) - gh(b)) / Math.max(gh(a), gh(b)) : 0
    } : null;
    m.symmetry = { central: side('11', '21'), lateral: side('12', '22'), canine: side('13', '23') };
    if (teeth && teeth['11'] && teeth['21']) m.midlineX = (teeth['11'].x1 + teeth['21'].x0) / 2;
    m.status = st;
    return m;
  }

  // ---------- face analysis ----------
  // points: {id:{x,y}}; scaleMmPerPx: number or null; returns metrics in a frame aligned to the interpupillary line.
  function faceAnalysis(points, scaleInfo = {}) {
    const P = points || {};
    const has = (...ids) => ids.every((i) => P[i]);
    const r = { has: {}, metrics: {}, scale: null, scaleSource: null };
    if (!has('pupilR', 'pupilL')) return r;
    const ipl = lineAngleDeg(P.pupilR, P.pupilL);
    const a = -ipl * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a);
    const o = P.pupilR;
    const T = (p) => p && ({ x: o.x + (p.x - o.x) * cs - (p.y - o.y) * sn, y: o.y + (p.x - o.x) * sn + (p.y - o.y) * cs });
    const Q = {}; for (const k in P) Q[k] = T(P[k]);
    // scale
    let scale = null, src = null;
    if (scaleInfo.centralLenMm && has('zenith11', 'incisal11')) {
      const px = Math.hypot(P.incisal11.x - P.zenith11.x, P.incisal11.y - P.zenith11.y);
      if (px > 2) { scale = scaleInfo.centralLenMm / px; src = 'طول الثنية المقاس في صورة المباعد'; }
    }
    if (!scale && scaleInfo.ipdMm) {
      const px = Math.hypot(P.pupilL.x - P.pupilR.x, P.pupilL.y - P.pupilR.y);
      scale = scaleInfo.ipdMm / px; src = `مسافة بين العينين مفترضة ${scaleInfo.ipdMm} ملم (تقريبي)`;
    }
    r.scale = scale; r.scaleSource = src;
    const mm = (px) => scale ? px * scale : null;
    const M = r.metrics;
    M.ipl_angle = ipl;
    if (has('commR', 'commL')) { M.commissure_cant = normAngle(lineAngleDeg(P.commR, P.commL) - ipl); M.commissure_span_px = Math.hypot(P.commL.x - P.commR.x, P.commL.y - P.commR.y); }
    if (has('cusp13', 'cusp23')) { M.incisal_cant = normAngle(lineAngleDeg(P.cusp13, P.cusp23) - ipl); M.incisal_span_px = Math.hypot(P.cusp23.x - P.cusp13.x, P.cusp23.y - P.cusp13.y); }
    // facial midline: line glabella→menton (through subnasale if present: least squares x = a + b*y)
    const mids = ['glabella', 'subnasale', 'menton'].filter((k) => Q[k]).map((k) => Q[k]);
    if (mids.length >= 2 && Q.midlineDental) {
      const my = avg(mids.map((p) => p.y)), mx = avg(mids.map((p) => p.x));
      let syy = 0, sxy = 0; for (const p of mids) { syy += (p.y - my) ** 2; sxy += (p.y - my) * (p.x - mx); }
      const b = syy > 1e-9 ? sxy / syy : 0;
      const xAt = mx + b * (Q.midlineDental.y - my);
      const dpx = Q.midlineDental.x - xAt; // + : dental midline toward image right (patient's left)
      M.midline_dev_px = dpx; M.midline_dev = mm(Math.abs(dpx));
      M.midline_dir = Math.abs(dpx) < 1e-6 ? '' : (dpx > 0 ? 'إلى يسار المريض' : 'إلى يمين المريض');
      M.facial_midline_angle = deg(Math.atan(b));
      M.facial_midline_xAtDental = xAt;
    }
    if (has('upperLip', 'zenith11')) { const d = Q.zenith11.y - Q.upperLip.y; M.gingival_display_px = -d; M.gingival_display = mm(-d); }
    // positive gingival_display = gum visible above zenith (lip above the zenith)
    if (has('lastR', 'lastL', 'commR', 'commL')) {
      M.dentition_span_px = Math.abs(Q.lastL.x - Q.lastR.x); M.smile_width_px = Math.abs(Q.commL.x - Q.commR.x);
      M.buccal_corridor = M.dentition_span_px / M.smile_width_px;
    }
    if (has('incisal11', 'cusp13', 'cusp23', 'lowerLipMid', 'lowerLipR', 'lowerLipL')) {
      const inc = Q.incisal11.y - (Q.cusp13.y + Q.cusp23.y) / 2; // +: centrals lower than canines (convex)
      const lip = Q.lowerLipMid.y - (Q.lowerLipR.y + Q.lowerLipL.y) / 2;
      const ref = Math.abs(Q.commL.x - Q.commR.x) || Math.abs(Q.cusp23.x - Q.cusp13.x);
      const incN = inc / ref, lipN = lip / ref;
      let cls;
      if (incN < -0.01) cls = 'reverse'; else if (incN < 0.015) cls = 'flat';
      else if (lipN > 0.005 && incN / lipN >= 0.5 && incN / lipN <= 2) cls = 'consonant'; else cls = 'nonparallel';
      M.smile_arc = cls; M.smile_arc_inc = incN; M.smile_arc_lip = lipN; M.smile_arc_ref_px = ref;
      M.incisal_to_lip = mm(Q.lowerLipMid.y - Q.incisal11.y);
    }
    if (has('glabella', 'subnasale', 'menton')) M.lower_third = (Q.menton.y - Q.subnasale.y) / (Q.subnasale.y - Q.glabella.y);
    for (const k of Object.keys(M)) r.has[k] = M[k] != null;
    return r;
  }
  const SMILE_ARC_LABEL = { consonant: 'متوافق مع الشفة', flat: 'مسطّح', reverse: 'معكوس', nonparallel: 'غير موازٍ للشفة' };

  // ---------- rules engine ----------
  function evaluateRules(meas, rules, ctx = {}) {
    const res = [];
    const p = ctx.proportionP ?? 0.70;
    const face = ctx.face?.metrics || {};
    const t = meas.teeth, mmpp = meas.mmPerPx;
    const fmt = (v, d = 2) => v == null || Number.isNaN(v) ? '—' : (+v).toFixed(d);
    // Measurement precision. ctx.edgePx = uncertainty (one standard deviation) of one tooth edge in the same pixels
    // as the boxes. A value outside the range by less than two standard deviations cannot be told apart from a
    // normal one at this photo's resolution: status 'uncertain' instead of a false warning. No edgePx: unchanged.
    const e = ctx.edgePx || 0, sz = Math.SQRT2 * e;
    const avg = (a, b, k) => { const v = [a, b].map((x) => t[x] && (k === 'w' || !t[x].broken) ? t[x][k] : null).filter((x) => x != null); return v.length ? v.reduce((p2, q) => p2 + q, 0) / v.length : null; };
    const sigRatio = (r, A, B) => (e && r != null && A && B) ? r * Math.hypot(sz / A, sz / B) : 0;
    const SIZES = { cW: ['11', '21', 'w'], cH: ['11', '21', 'h'], lW: ['12', '22', 'w'], lH: ['12', '22', 'h'], kW: ['13', '23', 'w'], kH: ['13', '23', 'h'] };
    const S2 = Object.fromEntries(Object.entries(SIZES).map(([k, [a, b, q]]) => [k, avg(a, b, q)]));
    // face measurements are in photo pixels: ctx.faceEdgePx is the same edge uncertainty there; face scale = mm per photo pixel
    const fe = ctx.faceEdgePx || 0, fsc = ctx.face?.scale || 0;
    const sigMm = fe && fsc ? Math.SQRT2 * fe * fsc : 0;
    const sigDeg = (span) => fe && span ? Math.SQRT2 * fe / span * 180 / Math.PI : 0;
    const BAND = { pct: (v) => (v * 100).toFixed(1) + '%', deg: (v) => v.toFixed(1) + '°', mm: (v) => v.toFixed(1) + ' ملم' };
    for (let rule of rules) {
      if (!rule.enabled) continue;
      const out = { id: rule.id, group: rule.group, label: rule.label, value: null, target: '', status: 'na', msg: '' };
      const range = (v, unit = '', sig = 0) => {
        out.value = v;
        if (v == null || Number.isNaN(v)) { out.status = 'na'; return; }
        const lo = rule.min ?? -Infinity, hi = rule.max ?? Infinity;
        if (sig > 0) out.sigma = sig;
        out.target = rule.min != null && rule.max != null ? `${rule.min}–${rule.max}${unit}` : rule.max != null ? `≤ ${rule.max}${unit}` : `≥ ${rule.min}${unit}`;
        const span = (isFinite(hi) && isFinite(lo)) ? (hi - lo) : Math.abs(isFinite(hi) ? hi : lo) || 1;
        if (v >= lo && v <= hi) out.status = 'ok';
        else out.status = (v < lo - span * 0.5 || v > hi + span * 0.5) ? 'fail' : 'warn';
        if (sig > 0 && out.status !== 'ok' && (v < lo ? lo - v : v - hi) < 2 * sig) out.status = 'uncertain';
      };
      const needCalib = rule.needs === 'calib' && !meas.calibrated;
      const needFace = rule.needs === 'face' && !ctx.face;
      if (rule.needs === 'rest' && !(ctx.rest && ctx.rest.display != null)) { out.msg = 'يحتاج صورة الشفاه في الراحة'; res.push(out); continue; }
      if (needCalib) { out.msg = 'يحتاج معايرة بالملم'; res.push(out); continue; }
      if (needFace) { out.msg = 'يحتاج تحليل صورة الوجه'; res.push(out); continue; }
      switch (rule.id) {
        case 'central_wl': {
          const v11 = t['11']?.wl, v21 = t['21']?.wl, f = ctx.face ? face : null;
          // the lip hides the gum line: only part of the crown is visible, so its length (and W/L) cannot be measured
          if (f && (f.gingival_display != null ? f.gingival_display < -0.5 : f.gingival_display_px != null && f.gingival_display_px < -3)) {
            out.msg = 'الشفة تغطي حد اللثة، فالطول الظاهر أقصر من طول السن الحقيقي ولا تُقاس النسبة من هذه الصورة'; break;
          }
          range(meas.ratios.central_wl, '', sigRatio(meas.ratios.central_wl, S2.cW, S2.cH));
          out.msg = `11: ${fmt(v11)} · 21: ${fmt(v21)}`; break;
        }
        case 'lat_central_w': {
          const tol = rule.tol ?? 0.06; rule = { ...rule, min: +(p - tol).toFixed(3), max: +(p + tol).toFixed(3) };
          range(meas.ratios.lat_central_w, '', sigRatio(meas.ratios.lat_central_w, S2.lW, S2.cW)); out.msg = `الهدف ${p}`; break;
        }
        case 'can_lat_w': {
          const tol = rule.tol ?? 0.08; rule = { ...rule, min: +(p - tol).toFixed(3), max: +(p + tol).toFixed(3) };
          range(meas.ratios.can_lat_w, '', sigRatio(meas.ratios.can_lat_w, S2.kW, S2.lW)); out.msg = `الهدف ${p}`; break;
        }
        case 'lat_central_len': range(meas.ratios.lat_central_len, '', sigRatio(meas.ratios.lat_central_len, S2.lH, S2.cH)); break;
        case 'can_central_len': range(meas.ratios.can_central_len, '', sigRatio(meas.ratios.can_central_len, S2.kH, S2.cH)); break;
        case 'sym_central': case 'sym_lateral': case 'sym_canine': {
          const s = meas.symmetry[{ sym_central: 'central', sym_lateral: 'lateral', sym_canine: 'canine' }[rule.id]];
          if (!s) break;
          const v = Math.max(s.dw, s.dh), A = t[s.a], B = t[s.b];
          const den = s.dw >= s.dh ? Math.max(A.w, B.w) : Math.max(A.h, B.h);
          range(v, '', e && den ? Math.SQRT2 * sz / den : 0); out.target = `≤ ${(rule.max * 100).toFixed(0)}%`;
          out.msg = `فرق العرض ${(s.dw * 100).toFixed(1)}% · فرق الطول ${(s.dh * 100).toFixed(1)}%`;
          out.valueText = `${(v * 100).toFixed(1)}%`; break;
        }
        case 'central_dominance': {
          const cs = ['11', '21'].filter((k) => t[k]).map((k) => t[k].w), others = ['12', '22', '13', '23'].filter((k) => t[k]).map((k) => t[k].w);
          if (!cs.length || !others.length) break;
          const ok = Math.min(...cs) >= Math.max(...others);
          out.status = ok ? 'ok' : 'warn'; out.valueText = ok ? 'نعم' : 'لا'; out.target = 'نعم'; break;
        }
        case 'central_len_mm': { const v = avgDef(t['11'] && !t['11'].broken ? t['11'].hmm : null, t['21'] && !t['21'].broken ? t['21'].hmm : null); range(v, ' ملم'); break; }
        case 'central_w_mm': { const v = avgDef(t['11']?.wmm, t['21']?.wmm); range(v, ' ملم'); break; }
        case 'zenith_lateral': case 'zenith_canine': case 'zenith_sym': case 'incisal_step': {
          const st = meas.status || {}; if (!ctx.teeth) break;
          const teeth = {}; for (const k of TEETH) if (ctx.teeth[k] && st[k] !== 'missing' && !(rule.id === 'incisal_step' && st[k] === 'broken')) teeth[k] = ctx.teeth[k];
          const zc = avgDef(teeth['11']?.y0, teeth['21']?.y0);
          if (rule.id === 'zenith_sym') { if (teeth['11'] && teeth['21']) range(Math.abs(teeth['11'].y0 - teeth['21'].y0) * mmpp, ' ملم'); break; }
          if (rule.id === 'incisal_step') {
            const ic = avgDef(teeth['11']?.y1, teeth['21']?.y1), il = avgDef(teeth['12']?.y1, teeth['22']?.y1);
            if (ic != null && il != null) { range((ic - il) * mmpp, ' ملم'); out.msg = 'موجب = الرباعية أقصر'; } break;
          }
          const ids = rule.id === 'zenith_lateral' ? ['12', '22'] : ['13', '23'];
          const z = avgDef(teeth[ids[0]]?.y0, teeth[ids[1]]?.y0);
          if (zc != null && z != null) { range((z - zc) * mmpp, ' ملم'); out.msg = 'موجب = أنزل (أقرب للحافة القاطعة)'; } break;
        }
        case 'midline_dev': range(face.midline_dev, ' ملم', sigMm); if (face.midline_dir) out.msg = face.midline_dir; break;
        case 'commissure_cant': if (face.commissure_cant != null) { range(Math.abs(face.commissure_cant), '°', sigDeg(face.commissure_span_px)); } break;
        case 'incisal_cant': if (face.incisal_cant != null) { range(Math.abs(face.incisal_cant), '°', sigDeg(face.incisal_span_px)); } break;
        case 'gingival_display': range(face.gingival_display, ' ملم', sigMm); out.msg = 'سالب = الشفة تغطي جزءاً من السن'; break;
        case 'buccal_corridor': range(face.buccal_corridor, '', fe && face.dentition_span_px && face.smile_width_px ? face.buccal_corridor * Math.hypot(Math.SQRT2 * fe / face.dentition_span_px, Math.SQRT2 * fe / face.smile_width_px) : 0); break;
        case 'smile_arc': if (face.smile_arc) {
          out.status = face.smile_arc === 'consonant' ? 'ok' : face.smile_arc === 'reverse' ? 'fail' : 'warn'; out.valueText = SMILE_ARC_LABEL[face.smile_arc]; out.target = SMILE_ARC_LABEL.consonant;
          // a class decided by a hair: the measured curve is closer to a class boundary than twice its error
          if (out.status !== 'ok' && fe && face.smile_arc_ref_px) {
            const si = Math.sqrt(1.5) * fe / face.smile_arc_ref_px, inc = face.smile_arc_inc, lip = face.smile_arc_lip, near = (v, b, sg) => Math.abs(v - b) < 2 * sg;
            let fragile = near(inc, -0.01, si) || near(inc, 0.015, si);
            if (!fragile && inc >= 0.015) {
              fragile = near(lip, 0.005, si);
              if (!fragile && lip > 0.005) { const r = inc / lip, sr = r * Math.hypot(si / inc, si / lip); fragile = near(r, 0.5, sr) || near(r, 2, sr); }
            }
            if (fragile) { out.status = 'uncertain'; out.sigma = si; }
          }
        } break;
        case 'lower_third': range(face.lower_third); break;
        case 'rest_display': range(ctx.rest.display, ' ملم'); if (ctx.rest.approx) out.msg = 'المقياس تقريبي'; break;
      }
      if (out.valueText == null && out.value != null) out.valueText = fmt(out.value, rule.unit === 'deg' || rule.unit === 'mm' ? 1 : 2);
      if (out.status === 'uncertain') out.msg = (out.msg ? out.msg + ' · ' : '') + (rule.unit === 'class' ? 'قريب من الحد بين تصنيفين ضمن هامش خطأ القياس — صورة أقرب تحسم' : `ضمن هامش خطأ القياس (±${(BAND[rule.unit] || ((v) => v.toFixed(2)))(2 * out.sigma)}) — صورة أقرب تحسم`);
      res.push(out);
    }
    return res;
  }
  function avgDef(a, b) { return (a != null && b != null) ? (a + b) / 2 : (a ?? b ?? null); }

  // ---------- design proposal ----------
  // Returns target boxes and per-tooth changes. All in working-image pixels.
  function proposeDesign(teeth, meas, rules, opt = {}) {
    const o = Object.assign({ proportion: 'red70', customP: 0.7, wl: null, spaceMode: 'preserve', symmetric: true, gingivaMode: 'keep', midline: 'current', facialMidlineX: null, centralWidthMm: null, centralLenMm: null }, opt);
    const rule = (id) => rules.find((r) => r.id === id) || {};
    const p = o.proportion === 'custom' ? +o.customP : (PROPORTIONS[o.proportion]?.p ?? 0.7);
    const wl = o.wl ?? rule('central_wl').target ?? 0.8;
    const latLen = rule('lat_central_len').target ?? 0.85;
    const canLen = rule('can_central_len').target ?? 0.95;
    const mmpp = meas.mmPerPx;
    const T = teeth;
    if (!['13', '12', '11', '21', '22', '23'].every((k) => T[k])) return { ok: false, error: 'يجب تحديد الأسنان الستة قبل التصميم' };
    const actualMid = (T['11'].x1 + T['21'].x0) / 2;
    const mid = (o.midline === 'facial' && o.facialMidlineX != null) ? o.facialMidlineX : actualMid;
    let cR, cL;
    if (o.spaceMode === 'preserve') {
      const SR = mid - T['13'].x0, SL = T['23'].x1 - mid;
      const k = 1 + p + p * p;
      cR = SR / k; cL = SL / k;
      if (o.symmetric) { const c = (cR + cL) / 2; cR = cL = c; }
    } else {
      const cw = ['11', '21'].map((k) => meas.teeth[k]?.w).filter((v) => v != null);
      let c = cw.length ? cw.reduce((a, b) => a + b, 0) / cw.length : (T['11'].x1 - T['11'].x0 + T['21'].x1 - T['21'].x0) / 2;
      if (o.centralWidthMm && mmpp) c = o.centralWidthMm / mmpp;
      cR = cL = c;
    }
    let Lc = ((cR + cL) / 2) / wl;
    if (o.centralLenMm && mmpp) Lc = o.centralLenMm / mmpp;
    else if (mmpp) {
      const r = rule('central_len_mm');
      if (r.enabled && r.min != null && r.max != null) Lc = Math.min(Math.max(Lc, r.min / mmpp), r.max / mmpp);
    }
    const len = { '11': Lc, '21': Lc, '12': Lc * latLen, '22': Lc * latLen, '13': Lc * canLen, '23': Lc * canLen };
    const wR = { '11': cR, '12': cR * p, '13': cR * p * p }, wL = { '21': cL, '22': cL * p, '23': cL * p * p };
    const X = {};
    X['11'] = [mid - wR['11'], mid]; X['12'] = [X['11'][0] - wR['12'], X['11'][0]]; X['13'] = [X['12'][0] - wR['13'], X['12'][0]];
    X['21'] = [mid, mid + wL['21']]; X['22'] = [X['21'][1], X['21'][1] + wL['22']]; X['23'] = [X['22'][1], X['22'][1] + wL['23']];
    // gingival line
    const zc = (T['11'].y0 + T['21'].y0) / 2;
    const latDrop = mmpp ? 0.75 / mmpp : Lc * 0.07;
    const Y0 = {};
    for (const k of TEETH) {
      if (o.gingivaMode === 'keep') Y0[k] = T[k].y0;
      else Y0[k] = (k === '12' || k === '22') ? zc + latDrop : zc;
    }
    const target = {}, changes = {};
    for (const k of TEETH) {
      target[k] = { x0: X[k][0], x1: X[k][1], y0: Y0[k], y1: Y0[k] + len[k] };
      const a = T[k], b = target[k];
      const c = {
        dW: (b.x1 - b.x0) - (a.x1 - a.x0), dL: (b.y1 - b.y0) - (a.y1 - a.y0),
        dIncisal: b.y1 - a.y1, dGingival: b.y0 - a.y0, dMesialShift: 0,
        w: b.x1 - b.x0, h: b.y1 - b.y0
      };
      if (mmpp) for (const f of ['dW', 'dL', 'dIncisal', 'dGingival', 'w', 'h']) c[f + 'mm'] = c[f] * mmpp;
      changes[k] = c;
    }
    return { ok: true, p, wl, mid, actualMid, target, changes, options: o };
  }

  function describeChange(c, mm) {
    const v = (x) => mm ? `${Math.abs(x).toFixed(1)} ملم` : `${Math.abs(x).toFixed(0)} بكسل`;
    const out = [];
    const dW = mm ? c.dWmm : c.dW, dI = mm ? c.dIncisalmm : c.dIncisal, dG = mm ? c.dGingivalmm : c.dGingival;
    const th = mm ? 0.2 : 2;
    if (dW > th) out.push('توسيع ' + v(dW)); else if (dW < -th) out.push('تضييق ' + v(dW));
    if (dI > th) out.push('إطالة الحافة ' + v(dI)); else if (dI < -th) out.push('تقصير الحافة ' + v(dI));
    if (dG < -th) out.push('رفع اللثة (قطع) ' + v(dG)); else if (dG > th) out.push('إنزال حد التاج ' + v(dG));
    return out.length ? out.join('، ') : 'بلا تغيير يُذكر';
  }

  // ---------- tooth shape path (SVG) ----------
  function toothPath(shape, b, canine) {
    const x0 = b.x0, x1 = b.x1, y0 = b.y0, y1 = b.y1, w = x1 - x0, h = y1 - y0;
    if (w <= 0 || h <= 0) return '';
    const narrow = shape === 'triangular' ? 0.20 : shape === 'oval' ? 0.14 : 0.08;
    const cx0 = x0 + w * narrow, cx1 = x1 - w * narrow;
    const cy = y0 + h * 0.16;
    const ctrlY = 2 * y0 - cy; // quadratic apex at y0
    const zx = x0 + w * 0.5;
    let p = `M ${f(cx0)} ${f(cy)} Q ${f(zx)} ${f(ctrlY)} ${f(cx1)} ${f(cy)} `;
    const bulge = shape === 'oval' ? 0.02 : 0;
    const r = shape === 'oval' ? 0.32 * w : shape === 'triangular' ? 0.10 * w : 0.12 * w;
    if (canine) {
      const tip = { x: x0 + w * 0.5, y: y1 };
      p += `C ${f(x1 + w * bulge)} ${f(y0 + h * 0.4)} ${f(x1)} ${f(y1 - h * 0.35)} ${f(x1 - w * 0.06)} ${f(y1 - h * 0.18)} `;
      p += `Q ${f(x1 - w * 0.25)} ${f(y1 - h * 0.02)} ${f(tip.x)} ${f(tip.y)} `;
      p += `Q ${f(x0 + w * 0.25)} ${f(y1 - h * 0.02)} ${f(x0 + w * 0.06)} ${f(y1 - h * 0.18)} `;
      p += `C ${f(x0)} ${f(y1 - h * 0.35)} ${f(x0 - w * bulge)} ${f(y0 + h * 0.4)} ${f(cx0)} ${f(cy)} Z`;
      return p;
    }
    p += `C ${f(x1 + w * bulge)} ${f(y0 + h * 0.35)} ${f(x1)} ${f(y0 + h * 0.6)} ${f(x1)} ${f(y1 - r)} `;
    p += `Q ${f(x1)} ${f(y1)} ${f(x1 - r)} ${f(y1)} L ${f(x0 + r)} ${f(y1)} Q ${f(x0)} ${f(y1)} ${f(x0)} ${f(y1 - r)} `;
    p += `C ${f(x0)} ${f(y0 + h * 0.6)} ${f(x0 - w * bulge)} ${f(y0 + h * 0.35)} ${f(cx0)} ${f(cy)} Z`;
    return p;
  }
  function f(v) { return Math.round(v * 10) / 10; }

  function defaultTeeth(W, H) {
    // six boxes centred in the image with typical apparent proportions
    const total = W * 0.8, k = 1 + 0.7 + 0.49, c = total / 2 / k;
    const mid = W / 2, y0 = H * 0.25, L = c / 0.8;
    const b = {};
    const wR = [c, c * 0.7, c * 0.49];
    let x = mid; ['11', '12', '13'].forEach((t, i) => { b[t] = { x0: x - wR[i], x1: x, y0: y0 + (i === 1 ? L * 0.05 : 0), y1: y0 + L * [1, 0.85, 0.95][i] }; x -= wR[i]; });
    x = mid; ['21', '22', '23'].forEach((t, i) => { b[t] = { x0: x, x1: x + wR[i], y0: y0 + (i === 1 ? L * 0.05 : 0), y1: y0 + L * [1, 0.85, 0.95][i] }; x += wR[i]; });
    return b;
  }

  // Place boxes for teeth the model did not find, inside the gap between found neighbours.
  const IDEAL_W = { '13': 0.49, '12': 0.7, '11': 1, '21': 1, '22': 0.7, '23': 0.49 };
  const IDEAL_L = { '13': 0.95, '12': 0.85, '11': 1, '21': 1, '22': 0.85, '23': 0.95 };
  function fillMissing(found, W, H) {
    const out = {}; for (const t of TEETH) if (found[t]) out[t] = { ...found[t] };
    const have = TEETH.filter((t) => out[t]);
    if (!have.length) return { teeth: defaultTeeth(W, H), filled: TEETH.slice() };
    // unit width (= one central) and unit length from what was found
    const unitW = avg(have.map((t) => (out[t].x1 - out[t].x0) / IDEAL_W[t]));
    const unitL = avg(have.map((t) => (out[t].y1 - out[t].y0) / IDEAL_L[t]));
    const top = avg(have.map((t) => out[t].y0));
    const filled = [];
    let i = 0;
    while (i < 6) {
      if (out[TEETH[i]]) { i++; continue; }
      let j = i; while (j < 6 && !out[TEETH[j]]) j++;
      const run = TEETH.slice(i, j), L = TEETH[i - 1], R = TEETH[j];
      const weights = run.map((t) => IDEAL_W[t]), sw = weights.reduce((a, b) => a + b, 0);
      let x0, x1;
      if (L && R) { x0 = out[L].x1; x1 = out[R].x0; if (x1 - x0 < unitW * sw * 0.3) { const c = (x0 + x1) / 2; x0 = c - unitW * sw * 0.5; x1 = c + unitW * sw * 0.5; } }
      else if (L) { x0 = out[L].x1; x1 = x0 + unitW * sw; }
      else { x1 = out[R].x0; x0 = x1 - unitW * sw; }
      let x = x0;
      run.forEach((t, k) => { const w = (x1 - x0) * weights[k] / sw; out[t] = { x0: x, x1: x + w, y0: top, y1: top + unitL * IDEAL_L[t] }; x += w; filled.push(t); });
      i = j;
    }
    for (const t of TEETH) { const b = out[t]; b.x0 = Math.max(0, b.x0); b.x1 = Math.min(W, Math.max(b.x1, b.x0 + 4)); b.y1 = Math.min(H, b.y1); }
    return { teeth: out, filled };
  }

  // Lips-at-rest photo: how much of the central incisor shows below the upper lip (mm).
  // points: pupilR, pupilL, lipRest (lower border of upper lip at midline), incisal11; hidden: edge covered by the lip
  function restAnalysis(points, opts = {}) {
    const P = points || {};
    if (!P.pupilR || !P.pupilL || !P.lipRest) return null;
    const ipl = lineAngleDeg(P.pupilR, P.pupilL), a = -ipl * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a), o = P.pupilR;
    const T = (p) => ({ x: o.x + (p.x - o.x) * cs - (p.y - o.y) * sn, y: o.y + (p.x - o.x) * sn + (p.y - o.y) * cs });
    const ipdPx = Math.hypot(P.pupilL.x - P.pupilR.x, P.pupilL.y - P.pupilR.y);
    const ipdMm = opts.ipdMm || 63, scale = ipdMm / ipdPx;
    const r = { scale, approx: !opts.ipdMeasured, ipdMm, display: null, hidden: !!opts.hidden };
    if (opts.hidden) { r.display = 0; r.hiddenNote = 'الحافة مغطاة بالشفة: الظهور 0 ملم أو أقل'; return r; }
    if (!P.incisal11) return r;
    r.display = (T(P.incisal11).y - T(P.lipRest).y) * scale;
    return r;
  }

  // Safety checks shown before approval. Returns [{level:'block'|'warn', text}]
  function safetyChecks(x) {
    const out = [];
    const st = x.status || {};
    if (!x.teeth || !TEETH.every((t) => x.teeth[t])) out.push({ level: 'block', text: 'حدود الأسنان الستة غير مكتملة.' });
    if (!x.reviewed) out.push({ level: 'block', text: 'لم تُؤكَّد مراجعة حدود الأسنان في خطوة «الأسنان».' });
    for (const t of TEETH) {
      const tr = x.treat?.[t];
      if (st[t] === 'missing' && ['veneer', 'composite'].includes(tr)) out.push({ level: 'block', text: `السن ${t} مفقود ولا يصلح له ${tr === 'veneer' ? 'فينير' : 'كومبوزيت'} — اختر زرعة أو جسراً أو بلا علاج.` });
      if (st[t] === 'broken' && tr === 'none') out.push({ level: 'warn', text: `السن ${t} مكسور ولم يُحدَّد له علاج.` });
      if (st[t] !== 'missing' && tr === 'pontic') out.push({ level: 'warn', text: `السن ${t} موجود لكنه محدد كسن معلّق في جسر — هل سيُخلع؟` });
    }
    if (x.meas && x.meas.calibrated) {
      const w = avgDef(x.meas.teeth['11']?.wmm, x.meas.teeth['21']?.wmm);
      if (w != null && (w < 6.5 || w > 11)) out.push({ level: 'warn', text: `عرض الثنية المحسوب ${w.toFixed(1)} ملم غير معتاد — راجع المعايرة.` });
    } else out.push({ level: 'warn', text: 'الصورة غير معايرة: لا مقاسات بالملم في ورقة المعمل.' });
    if (x.detLevel && x.detLevel !== 'ok' && x.source === 'auto') out.push({ level: 'warn', text: 'الكشف التلقائي أعطى تنبيهات ولم تُعدَّل الحدود.' });
    if (x.rest && x.rest.display != null && (x.rest.display < -3 || x.rest.display > 8)) out.push({ level: 'warn', text: `ظهور الثنية في الراحة ${x.rest.display.toFixed(1)} ملم قيمة غير معتادة — راجع نقاط صورة الراحة.` });
    if (x.design && x.design.changes) for (const t of TEETH) {
      const c = x.design.changes[t]; if (!c || c.dWmm == null) continue;
      if (Math.abs(c.dWmm) > 2 && st[t] !== 'missing') out.push({ level: 'warn', text: `تغيير عرض السن ${t} بمقدار ${c.dWmm.toFixed(1)} ملم كبير — قد يحتاج تقويماً أو تحضيراً كبيراً.` });
      if (c.dIncisalmm < -2 && st[t] !== 'missing') out.push({ level: 'warn', text: `تقصير السن ${t} بمقدار ${(-c.dIncisalmm).toFixed(1)} ملم كبير.` });
    }
    if (x.rest && x.rest.ipdMm && !x.rest.approx && (x.rest.ipdMm < 50 || x.rest.ipdMm > 75)) out.push({ level: 'warn', text: `مسافة العينين المحسوبة ${x.rest.ipdMm.toFixed(0)} ملم غير معتادة — راجع معايرة صورة الأسنان ونقاط الثنية في صورة الابتسامة.` });
    if (!x.consent) out.push({ level: 'block', text: 'لا توجد موافقة مسجلة على التصوير.' });
    return out;
  }

  const api = {
    TEETH, CLASS_OF, FLIP_CHANNELS, TOOTH_NAME, PROPORTIONS, DEFAULT_RULES, FACE_POINTS, SHADES, SMILE_ARC_LABEL,
    clone, fitTilt, softmaxNCHW, mergeFlipTTA, argmaxNCHW, largestComponents, labelsToTeeth, qualityReport,
    measure, faceAnalysis, evaluateRules, proposeDesign, fillMissing, restAnalysis, safetyChecks, describeChange, toothPath, defaultTeeth, lineAngleDeg
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DSDCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
