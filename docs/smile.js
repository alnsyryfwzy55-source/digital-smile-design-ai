/* Smile-photo analysis: pure functions on landmarks, label maps and pixel arrays (no DOM). Browser: window.DSDSmile, Node: module.exports */
(function (root) {
  'use strict';
  const C = root.DSDCore || (typeof require !== 'undefined' ? require('./core.js') : null);
  const TEETH = C.TEETH;

  // MediaPipe Face Landmarker (478 points) indices
  const LM = {
    irisA: 468, irisB: 473, ringA: [469, 470, 471, 472], ringB: [474, 475, 476, 477],
    eyeA_out: 33, eyeA_in: 133, eyeB_in: 362, eyeB_out: 263,
    forehead: 10, glabella: 9, nasion: 168, noseTip: 1, subnasale: 2, menton: 152, edgeA: 234, edgeB: 454,
    commA: 61, commB: 291, upperInner: 13, lowerInner: 14
  };
  const INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];   // corner → corner along upper lip
  const INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];   // corner → corner along lower lip
  const OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
  const IRIS_MM = 11.7;            // mean horizontal visible iris diameter in adults

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const deg = (r) => r * 180 / Math.PI;

  function toPx(lm, W, H) { return lm.map((p) => ({ x: p.x * W, y: p.y * H })); }

  // Upright test: eyes must be above the mouth. Returns 0 if upright, else the quarter turns (k*90° clockwise) that fix it.
  function uprightTurns(P) {
    const eyes = mid(P[LM.irisA] || P[LM.eyeA_out], P[LM.irisB] || P[LM.eyeB_out]);
    const mouth = mid(P[LM.upperInner], P[LM.lowerInner]);
    const dx = mouth.x - eyes.x, dy = mouth.y - eyes.y;      // vector eyes → mouth should point down (+y)
    const a = deg(Math.atan2(dx, dy));                       // 0 = down
    if (Math.abs(a) <= 45) return 0;
    if (a > 45 && a <= 135) return 1;      // mouth to the right of eyes → rotate 90° clockwise
    if (a < -45 && a >= -135) return 3;
    return 2;
  }

  function eyes(P) {
    let a = P[LM.irisA], b = P[LM.irisB];
    if (!a || !b) { a = mid(P[LM.eyeA_out], P[LM.eyeA_in]); b = mid(P[LM.eyeB_in], P[LM.eyeB_out]); }
    return a.x <= b.x ? [a, b] : [b, a];         // [image-left (patient right), image-right]
  }

  function pose(P) {
    const [eR, eL] = eyes(P);
    const roll = deg(Math.atan2(eL.y - eR.y, eL.x - eR.x));
    const r = -roll * Math.PI / 180, cs = Math.cos(r), sn = Math.sin(r), o = eR;
    const T = (p) => ({ x: o.x + (p.x - o.x) * cs - (p.y - o.y) * sn, y: o.y + (p.x - o.x) * sn + (p.y - o.y) * cs });
    const nose = T(P[LM.noseTip]), a = T(P[LM.edgeA]), b = T(P[LM.edgeB]);
    const left = a.x < b.x ? a : b, right = a.x < b.x ? b : a;
    const dl = Math.abs(nose.x - left.x), dr = Math.abs(right.x - nose.x);
    const yaw = (dl - dr) / Math.max(1, dl + dr);                 // 0 = frontal; ± = turned
    const em = T(mid(eR, eL)), chin = T(P[LM.menton]);
    const pitch = (nose.y - em.y) / Math.max(1, chin.y - em.y);   // ~0.35–0.5 when frontal
    return { roll, yaw, pitch };
  }

  function innerLipPolygon(P) { return INNER_UPPER.map((i) => P[i]).concat(INNER_LOWER.slice(1, -1).map((i) => P[i])); }

  // Pixel statistics needed by the capture quality checks. rgba: Uint8ClampedArray, W×H.
  function imageStats(rgba, W, H, P, opts = {}) {
    const inner = innerLipPolygon(P);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of inner) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0)); x1 = Math.min(W - 1, Math.ceil(x1)); y1 = Math.min(H - 1, Math.ceil(y1));
    const step = Math.max(1, Math.round((x1 - x0) / 220));
    let n = 0, bright = 0, clip = 0; const Ls = [];
    for (let y = y0; y <= y1; y += step) for (let x = x0; x <= x1; x += step) {
      if (!pointInPoly(x, y, inner)) continue;
      const i = (y * W + x) * 4, r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const L = 0.299 * r + 0.587 * g + 0.114 * b; Ls.push(L);
      n++; if (L > 140 && (mx - mn) < 75) bright++; if (mx >= 250) clip++;
    }
    // face brightness: box between the eyes and the mouth (or supplied by the caller)
    let fs = 0, fn = 0;
    if (opts.faceL != null) { fs = opts.faceL; fn = 1; } else {
      const [eR, eL] = eyes(P), m = mid(P[LM.upperInner], P[LM.lowerInner]);
      const fx0 = Math.round(Math.min(eR.x, eL.x)), fx1 = Math.round(Math.max(eR.x, eL.x)), fy0 = Math.round(Math.min(eR.y, eL.y)), fy1 = Math.round(m.y);
      const fstep = Math.max(1, Math.round((fx1 - fx0) / 60));
      for (let y = Math.max(0, fy0); y <= Math.min(H - 1, fy1); y += fstep) for (let x = Math.max(0, fx0); x <= Math.min(W - 1, fx1); x += fstep) {
        const i = (y * W + x) * 4; fs += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]; fn++;
      }
    }
    // sharpness: variance of the Laplacian over the mouth box, on a ~256px-wide grid
    const sw = Math.max(1, (x1 - x0) / 256);
    const gw = Math.max(3, Math.floor((x1 - x0) / sw)), gh = Math.max(3, Math.floor((y1 - y0) / sw));
    const G = new Float32Array(gw * gh);
    for (let j = 0; j < gh; j++) for (let k = 0; k < gw; k++) {
      const x = Math.min(W - 1, Math.round(x0 + k * sw)), y = Math.min(H - 1, Math.round(y0 + j * sw)), i = (y * W + x) * 4;
      G[j * gw + k] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    }
    let ls = 0, ls2 = 0, ln = 0;
    for (let j = 1; j < gh - 1; j++) for (let k = 1; k < gw - 1; k++) {
      const v = 4 * G[j * gw + k] - G[j * gw + k - 1] - G[j * gw + k + 1] - G[(j - 1) * gw + k] - G[(j + 1) * gw + k];
      ls += v; ls2 += v * v; ln++;
    }
    const lapVar = ln ? ls2 / ln - (ls / ln) ** 2 : 0;
    return { teethFrac: n ? bright / n : 0, clipFrac: n ? clip / n : 0, faceL: fn ? fs / fn : 0, sharpness: lapVar, mouthPx: { w: x1 - x0, h: y1 - y0 } };
  }

  const QUALITY_DEFAULTS = { maxRoll: 4, maxYaw: 0.10, pitchMin: 0.22, pitchMax: 0.60, minMouthW: 200, faceMin: 0.30, faceMax: 0.92,
    minOpen: 0.10, minTeeth: 0.20, minL: 70, maxL: 225, maxClip: 0.25, minSharp: 40, maxCenterOff: 0.22 };

  // P: landmarks in px; stats: imageStats(); returns { ok, items: [{id, ok, label, hint}] }
  function qualityChecks(P, W, H, stats, th = {}) {
    const t = Object.assign({}, QUALITY_DEFAULTS, th);
    const ps = pose(P), items = [];
    const faceH = dist(P[LM.forehead], P[LM.menton]) * 1.12;
    const fc = mid(P[LM.forehead], P[LM.menton]);
    const mouthW = dist(P[LM.commA], P[LM.commB]);
    const open = dist(P[LM.upperInner], P[LM.lowerInner]) / Math.max(1, mouthW);
    const add = (id, ok, label, hint) => items.push({ id, ok, label, hint: ok ? '' : hint });
    const rel = faceH / H;
    add('distance', rel >= t.faceMin && rel <= t.faceMax && mouthW >= t.minMouthW, 'المسافة',
      rel < t.faceMin ? 'قرّب الكاميرا من المريض' : rel > t.faceMax ? 'ابتعد قليلاً: الوجه كله يجب أن يظهر' : 'دقة الصورة منخفضة: استخدم الكاميرا الخلفية واقترب قليلاً');
    const off = Math.max(Math.abs(fc.x / W - 0.5), Math.abs(fc.y / H - 0.5));
    add('center', off <= t.maxCenterOff, 'التوسيط', 'اجعل الوجه في منتصف الصورة');
    add('roll', Math.abs(ps.roll) <= t.maxRoll, 'استقامة الرأس', ps.roll > 0 ? 'الرأس مائل: اطلب من المريض إمالته قليلاً إلى اليسار' : 'الرأس مائل: اطلب من المريض إمالته قليلاً إلى اليمين');
    add('frontal', Math.abs(ps.yaw) <= t.maxYaw && ps.pitch >= t.pitchMin && ps.pitch <= t.pitchMax, 'النظر للأمام',
      Math.abs(ps.yaw) > t.maxYaw ? 'المريض ملتفت: يجب أن ينظر مباشرة إلى الكاميرا' : (ps.pitch < t.pitchMin ? 'الرأس مرفوع: اطلب خفض الذقن قليلاً' : 'الرأس منخفض: اطلب رفع الذقن قليلاً'));
    add('smile', open >= t.minOpen && stats.teethFrac >= t.minTeeth, 'ابتسامة تُظهر الأسنان', 'اطلب ابتسامة عريضة تظهر فيها الأسنان العلوية');
    add('light', stats.faceL >= t.minL && stats.faceL <= t.maxL && stats.clipFrac <= t.maxClip, 'الإضاءة',
      stats.faceL < t.minL ? 'الإضاءة ضعيفة: صوّر قرب نافذة أو شغّل الضوء' : 'إضاءة قوية جداً أو فلاش مباشر على الأسنان');
    add('sharp', stats.sharpness >= t.minSharp, 'الوضوح', 'الصورة مهزوزة أو غير مركّزة: ثبّت الجوال');
    return { ok: items.every((i) => i.ok), items, pose: ps, open, mouthW };
  }

  // Affine crop around the mouth, aligned to the eye line. Aspect 1.6:1 so squashing to the model square matches training distortion.
  function mouthCrop(P, opts = {}) {
    const aspect = opts.aspect || 1.6, widen = opts.widen || 1.18;
    const ps = pose(P);
    const inner = innerLipPolygon(P);
    const c = { x: inner.reduce((s, p) => s + p.x, 0) / inner.length, y: inner.reduce((s, p) => s + p.y, 0) / inner.length };
    const Wc = dist(P[LM.commA], P[LM.commB]) * widen, Hc = Wc / aspect;
    const outW = Math.round(clamp(Wc, opts.minOut || 384, opts.maxOut || 1024)), outH = Math.round(outW / aspect);
    const s = Wc / outW, a = ps.roll * Math.PI / 180, cs = Math.cos(a) * s, sn = Math.sin(a) * s;
    // crop (u,v) → image (x,y) = c + R(roll)·s·(u − outW/2, v − outH/2)
    const M = [cs, -sn, c.x - cs * outW / 2 + sn * outH / 2, sn, cs, c.y - sn * outW / 2 - cs * outH / 2];
    return { M, Minv: invertAffine(M), outW, outH, scale: s, roll: ps.roll, center: c, Hc };
  }
  function invertAffine(M) {
    const [a, b, c, d, e, f] = M, det = a * e - b * d;
    return [e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det];
  }
  const applyAffine = (M, p) => ({ x: M[0] * p.x + M[1] * p.y + M[2], y: M[3] * p.x + M[4] * p.y + M[5] });

  function pointInPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i], b = poly[j];
      if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  }
  function rasterPoly(poly, w, h) {
    const m = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const xs = [], yy = y + 0.5;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const a = poly[i], b = poly[j];
        if ((a.y > yy) !== (b.y > yy)) xs.push((b.x - a.x) * (yy - a.y) / (b.y - a.y) + a.x);
      }
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(0, Math.ceil(xs[k] - 0.5)), xb = Math.min(w - 1, Math.floor(xs[k + 1] - 0.5));
        for (let x = xa; x <= xb; x++) m[y * w + x] = 1;
      }
    }
    return m;
  }
  function dilate(m, w, h, r) {
    let cur = m;
    for (let k = 0; k < r; k++) {
      const out = new Uint8Array(cur);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (cur[i]) continue;
        if ((x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w])) out[i] = 1;
      }
      cur = out;
    }
    return cur;
  }
  // y of a lip chain (array of points sorted or not) at x, linear interpolation
  function chainY(chain, x) {
    const s = [...chain].sort((a, b) => a.x - b.x);
    if (x <= s[0].x) return s[0].y; if (x >= s[s.length - 1].x) return s[s.length - 1].y;
    for (let i = 1; i < s.length; i++) if (x <= s[i].x) { const k = (x - s[i - 1].x) / ((s[i].x - s[i - 1].x) || 1); return s[i - 1].y + k * (s[i].y - s[i - 1].y); }
    return s[s.length - 1].y;
  }

  // Pick the upper six teeth from a model label map of the mouth crop.
  // labels/conf: S×S (crop squashed to square). lips: {upper:[pts], lower:[pts], poly:[pts]} in CROP pixels.
  function selectUpperTeeth(labels, conf, S, cropW, cropH, lips, opts = {}) {
    const sx = cropW / S, sy = cropH / S;
    const toL = (p) => ({ x: p.x / sx, y: p.y / sy });
    const polyL = lips.poly.map(toL), upL = lips.upper.map(toL), loL = lips.lower.map(toL);
    const inMouth = dilate(rasterPoly(polyL, S, S), S, S, opts.lipMargin ?? 3);
    const minArea = (opts.minFrac ?? 0.0025) * S * S;
    const seen = new Uint8Array(S * S), stack = new Int32Array(S * S);
    const chosen = {}, rejected = [];
    for (let c = 1; c <= 6; c++) {
      let best = null; const cands = [];
      for (let i0 = 0; i0 < S * S; i0++) {
        if (labels[i0] !== c || seen[i0] || !inMouth[i0]) continue;
        let sp = 0; stack[sp++] = i0; seen[i0] = 1;
        const comp = []; let cx = 0, cs = 0, ymin = S;
        while (sp) {
          const k = stack[--sp]; comp.push(k);
          const x = k % S, y = (k / S) | 0; cx += x; cs += conf ? conf[k] : 1; if (y < ymin) ymin = y;
          if (x > 0 && labels[k - 1] === c && !seen[k - 1] && inMouth[k - 1]) { seen[k - 1] = 1; stack[sp++] = k - 1; }
          if (x < S - 1 && labels[k + 1] === c && !seen[k + 1] && inMouth[k + 1]) { seen[k + 1] = 1; stack[sp++] = k + 1; }
          if (y > 0 && labels[k - S] === c && !seen[k - S] && inMouth[k - S]) { seen[k - S] = 1; stack[sp++] = k - S; }
          if (y < S - 1 && labels[k + S] === c && !seen[k + S] && inMouth[k + S]) { seen[k + S] = 1; stack[sp++] = k + S; }
        }
        if (comp.length < minArea) continue;
        const mx = cx / comp.length, uy = chainY(upL, mx), ly = chainY(loL, mx);
        const rel = (ymin - uy) / Math.max(1, ly - uy);           // 0 = at the upper lip, 1 = at the lower lip
        cands.push({ comp, conf: cs / comp.length, mx, rel, c });
      }
      // an upper tooth starts near the upper lip / gum line. Prefer components that start high; a component that starts
      // near the lower lip is a lower tooth. With a gummy smile upper teeth start lower, so the limit is generous.
      const high = cands.filter((k) => k.rel <= (opts.lowerStart ?? 0.45)), mid = cands.filter((k) => k.rel > (opts.lowerStart ?? 0.45) && k.rel <= 0.62);
      const pool = high.length ? high : mid;
      for (const k of cands) if (!pool.includes(k)) rejected.push({ c, area: k.comp.length, reason: 'lower' });
      for (const k of pool) if (!best || k.comp.length > best.comp.length) best = k;
      if (best) chosen[c] = best;
    }
    const clean = new Uint8Array(S * S);
    const teeth = {}, info = {}, centers = {};
    let confSum = 0, confN = 0;
    for (const t of TEETH) {
      const c = C.CLASS_OF[t], b = chosen[c];
      if (!b) { info[t] = { present: false }; continue; }
      const xs = [], ys = [];
      for (const k of b.comp) { clean[k] = c; xs.push((k % S + 0.5) * sx); ys.push((((k / S) | 0) + 0.5) * sy); }
      xs.sort((p, q) => p - q); ys.sort((p, q) => p - q);
      const q = (a, p) => a[Math.min(a.length - 1, Math.round(p / 100 * (a.length - 1)))];
      teeth[t] = { x0: q(xs, 1), x1: q(xs, 99), y0: q(ys, 1), y1: q(ys, 99) };
      // an upper tooth merged with the lower tooth below it is far too tall: cut at the dark occlusal line between them
      if (opts.rgba) {
        const cut = occlusalCut(opts.rgba, cropW, cropH, teeth[t], t);
        if (cut != null) {
          teeth[t].y1 = cut; teeth[t].cut = true;
          for (const k of b.comp) if ((((k / S) | 0) + 0.5) * sy > cut) clean[k] = 0;
        }
      }
      centers[t] = b.mx * sx;
      info[t] = { present: true, area: b.comp.length / (S * S), conf: b.conf };
      confSum += b.conf * b.comp.length; confN += b.comp.length;
    }
    const order = TEETH.filter((t) => teeth[t]).map((t) => centers[t]);
    let orderOk = order.length >= 2;
    for (let k = 1; k < order.length; k++) if (order[k] <= order[k - 1]) orderOk = false;
    return { teeth, info, clean, orderOk, meanConf: confN ? confSum / confN : 0, presentCount: Object.keys(teeth).length, rejected };
  }

  const EXPECT_WL = { '13': 0.80, '12': 0.75, '11': 0.80, '21': 0.80, '22': 0.75, '23': 0.80 };
  function occlusalCut(rgba, w, h, box, t) {
    const bw = box.x1 - box.x0, bh = box.y1 - box.y0, expH = bw / EXPECT_WL[t];
    if (bh <= 1.05 * expH) return null;
    const xa = Math.round(box.x0 + 0.2 * bw), xb = Math.round(box.x1 - 0.2 * bw);
    const ya = Math.round(box.y0 + 0.55 * expH), yb = Math.round(box.y1);
    const rows = [];
    for (let y = Math.max(0, Math.round(box.y0)); y <= Math.min(h - 1, yb); y++) {
      let s = 0, n = 0;
      for (let x = Math.max(0, xa); x <= Math.min(w - 1, xb); x++) { const i = (y * w + x) * 4; s += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]; n++; }
      rows.push({ y, L: n ? s / n : 0 });
    }
    const sm = rows.map((r, i) => ({ y: r.y, L: (rows[Math.max(0, i - 1)].L + r.L + rows[Math.min(rows.length - 1, i + 1)].L) / 3 }));
    const cand = sm.filter((r) => r.y >= ya && r.y <= yb - 2);
    if (!cand.length) return null;
    const v = cand.reduce((a, b) => (b.L < a.L ? b : a));
    const above = sm.filter((r) => r.y < v.y && r.y >= box.y0 + 0.2 * expH).map((r) => r.L).sort((a, b) => a - b);
    const ref = above.length ? above[Math.floor(above.length / 2)] : 0;
    if (ref - v.L < 25) return null;
    // a real occlusal line has tooth-bright rows again below it (a translucent incisal edge does not)
    const brightBelow = sm.filter((r) => r.y > v.y + 1 && r.L > ref - 30).length;
    return brightBelow >= 3 ? v.y : null;
  }

  // Decide why a tooth was not found: 'missing' (a real gap between neighbours) or 'hidden' (behind the lips / out of view)
  function classifyAbsent(sel, rgba, w, h, lips) {
    const status = {};
    const present = TEETH.filter((t) => sel.teeth[t]);
    if (!present.length) return status;
    const IDEAL = { '13': 0.49, '12': 0.7, '11': 1, '21': 1, '22': 0.7, '23': 0.49 };
    const unit = present.reduce((s, t) => s + (sel.teeth[t].x1 - sel.teeth[t].x0) / IDEAL[t], 0) / present.length;
    for (let i = 0; i < 6; i++) {
      const t = TEETH[i]; if (sel.teeth[t]) continue;
      let L = null, R = null;
      for (let k = i - 1; k >= 0; k--) if (sel.teeth[TEETH[k]]) { L = TEETH[k]; break; }
      for (let k = i + 1; k < 6; k++) if (sel.teeth[TEETH[k]]) { R = TEETH[k]; break; }
      if (L && R) {
        const gap = sel.teeth[R].x0 - sel.teeth[L].x1;
        // a real gap between neighbours (dark space, or pink gum where a tooth was lost) → missing; otherwise the model missed it
        status[t] = gap >= 0.45 * unit * IDEAL[t] && nonToothFraction(rgba, w, h, sel.teeth[L].x1, sel.teeth[R].x0, sel.teeth[L], sel.teeth[R]) > 0.55 ? 'missing' : 'unsure';
      } else {
        // end tooth (canine) not labelled: look where it should be, next to the lateral, inside the mouth opening
        const N = L || R, nb = sel.teeth[N], cw = 0.8 * (nb.x1 - nb.x0);
        const xa = L ? nb.x1 : nb.x0 - cw, xb = L ? nb.x1 + cw : nb.x0;
        const f = stripFractions(rgba, w, h, xa, xb, nb.y0, nb.y1, lips);
        // a dark corner is usually the buccal corridor, not a lost canine: stay conservative (the doctor can mark it missing)
        status[t] = !f || f.inMouth < 0.3 ? 'hidden' : f.tooth >= 0.45 ? 'estimated' : f.tooth >= 0.2 ? 'unsure' : 'hidden';
      }
    }
    return status;
  }
  const toothLike = (r, g, b) => (0.299 * r + 0.587 * g + 0.114 * b) > 115 && (Math.max(r, g, b) - Math.min(r, g, b)) < 70;
  function nonToothFraction(rgba, w, h, xa, xb, A, B) {
    const y0 = Math.round(Math.max(A.y0, B.y0)), y1 = Math.round(Math.min(A.y1, B.y1));
    let n = 0, d = 0;
    for (let y = Math.max(0, y0); y <= Math.min(h - 1, y1); y += 2) for (let x = Math.max(0, Math.round(xa)); x <= Math.min(w - 1, Math.round(xb)); x += 2) {
      const i = (y * w + x) * 4; n++; if (!toothLike(rgba[i], rgba[i + 1], rgba[i + 2])) d++;
    }
    return n ? d / n : 0;
  }

  function stripFractions(rgba, w, h, xa, xb, ya, yb, lips) {
    const poly = lips && lips.poly; let n = 0, inM = 0, tooth = 0, dark = 0;
    for (let y = Math.max(0, Math.round(ya)); y <= Math.min(h - 1, Math.round(yb)); y += 2) for (let x = Math.max(0, Math.round(xa)); x <= Math.min(w - 1, Math.round(xb)); x += 2) {
      n++; if (poly && !pointInPoly(x, y, poly)) continue; inM++;
      const i = (y * w + x) * 4, r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      if (toothLike(r, g, b)) tooth++; else if (0.299 * r + 0.587 * g + 0.114 * b < 70) dark++;
    }
    return inM ? { tooth: tooth / inM, dark: dark / inM, inMouth: inM / Math.max(1, n) } : null;
  }

  // A fractured incisor is much shorter than its counterpart (or, for both centrals, far too wide for its length)
  // A fractured incisor is much shorter than its counterpart, or far too wide for its length. edgePx (uncertainty of one
  // tooth edge, crop px): a verdict closer to the threshold than twice its error is not decided from this photo; those
  // teeth come back in .unsure for the doctor to confirm. Without edgePx: the plain rule.
  function detectBroken(teeth, status, edgePx = 0) {
    const out = [], unsure = [], ok = (t) => teeth[t] && status[t] === 'present', H = (t) => teeth[t].y1 - teeth[t].y0, W = (t) => teeth[t].x1 - teeth[t].x0;
    const s2 = Math.SQRT2 * edgePx, sig = (r, A, B) => edgePx ? r * Math.hypot(s2 / A, s2 / B) : 0;
    for (const [a, b] of [['11', '21'], ['12', '22']]) {
      if (!(ok(a) && ok(b))) continue;
      const [lo, hi] = H(a) <= H(b) ? [a, b] : [b, a], r = H(lo) / Math.max(1, H(hi)), sg = sig(r, H(lo), H(hi));
      if (r < 0.78 - 2 * sg) out.push(lo); else if (r < 0.78) unsure.push(lo);
    }
    for (const t of ['11', '21']) {
      if (!ok(t) || out.includes(t) || unsure.includes(t)) continue;
      const r = W(t) / Math.max(1, H(t)), sg = sig(r, W(t), H(t));
      if (r > 1.35 + 2 * sg) out.push(t); else if (r > 1.35) unsure.push(t);
    }
    out.unsure = unsure;
    return out;
  }

  // No front teeth at all (trauma, extraction): place an ideal anterior set from the face itself.
  // lips: crop-space chains; midX: facial midline at the mouth (crop px); mmpp: mm per crop px (or null).
  function designFromFace(lips, midX, mmpp, opts = {}) {
    const up = lips.upper, lo = lips.lower;
    const xs = up.map((p) => p.x), mouthW = Math.max(...xs) - Math.min(...xs);
    const cW = mmpp ? (opts.centralWidthMm || 8.6) / mmpp : mouthW * 0.155;
    const cL = mmpp ? (opts.centralLenMm || 10.5) / mmpp : cW / 0.8;
    const p = opts.p || 0.7, ws = [p * p, p, 1, 1, p, p * p].map((k) => k * cW), lens = [0.95, 0.85, 1, 1, 0.85, 0.95];
    const T = {}; let x = midX - cW - p * cW - p * p * cW;
    TEETH.forEach((t, i) => {
      const cx = x + ws[i] / 2, top = chainY(up, cx) - (mmpp ? 1 / mmpp : cL * 0.1);   // the neck sits just under the upper lip
      T[t] = { x0: x, x1: x + ws[i], y0: top, y1: top + cL * lens[i] }; x += ws[i];
    });
    return T;
  }

  function irisMmPerPx(P) {
    const d = [];
    if (P[469] && P[471]) d.push(dist(P[469], P[471]));
    if (P[474] && P[476]) d.push(dist(P[474], P[476]));
    if (!d.length) return null;
    const px = d.reduce((a, b) => a + b, 0) / d.length;
    return px > 4 ? IRIS_MM / px : null;
  }

  // Shape incisal edges into an arc parallel to the lower lip. target: boxes (crop px); lowerY(x): lower-lip y in crop px.
  function applySmileArc(target, lowerY, mmpp, opts = {}) {
    const out = {}; for (const t of TEETH) if (target[t]) out[t] = { ...target[t] };
    if (!out['11'] || !out['21']) return out;
    const midX = (out['11'].x1 + out['21'].x0) / 2;
    const dL = (opts.lengthDeltaMm || 0) / (mmpp || 1) * (mmpp ? 1 : 0);
    const yc = (out['11'].y1 + out['21'].y1) / 2 + dL;
    const step = { '12': 0.75, '22': 0.75, '13': 0.25, '23': 0.25 };
    const lipMid = lowerY(midX), follow = opts.follow ?? 0.85;
    const L11 = yc - (out['11'].y0 + out['21'].y0) / 2;
    for (const t of TEETH) {
      const b = out[t]; if (!b) continue;
      const cx = (b.x0 + b.x1) / 2;
      let y1 = yc + Math.min(0, (lowerY(cx) - lipMid)) * follow;          // lateral teeth rise with the lip, never drop below the centrals
      if (step[t]) y1 -= mmpp ? step[t] / mmpp : L11 * 0.07 * step[t] / 0.75;
      if (t === '11' || t === '21') y1 = yc;
      y1 = Math.min(y1, yc);
      out[t].y1 = Math.max(b.y0 + 0.55 * L11, y1);
    }
    return out;
  }

  // Keep automatic changes within what is usually achievable with veneers/crowns; the doctor can go further on purpose.
  function limitChanges(current, target, mmpp, lim = {}, status = {}) {
    const L = Object.assign({ shortenMm: 1.5, lengthenMm: 2.5, widthMm: 1.5 }, lim), out = {};
    const px = (mm) => mmpp ? mm / mmpp : null;
    for (const t of TEETH) {
      const a = current[t], b = target[t]; if (!b) continue; out[t] = { ...b };
      if (!a || !mmpp || (status[t] && status[t] !== 'present')) continue;   // a missing or broken tooth is rebuilt in full
      out[t].y1 = clamp(b.y1, a.y1 - px(L.shortenMm), a.y1 + px(L.lengthenMm));
    }
    return out;
  }

  // Points for DSDCore.faceAnalysis from landmarks (image px) and tooth boxes (crop px → image via M)
  function facePoints(P, teethCrop, M, extents) {
    const [pR, pL] = eyes(P);
    const cA = P[LM.commA], cB = P[LM.commB], [cR, cL] = cA.x <= cB.x ? [cA, cB] : [cB, cA];
    const pts = { pupilR: pR, pupilL: pL, glabella: P[LM.glabella], subnasale: P[LM.subnasale], menton: P[LM.menton], commR: cR, commL: cL,
      upperLip: P[LM.upperInner], lowerLipMid: P[LM.lowerInner] };
    const lower = INNER_LOWER.map((i) => P[i]);
    const T = teethCrop || {};
    const toImg = (x, y) => applyAffine(M, { x, y });
    const cx = (b) => (b.x0 + b.x1) / 2;
    if (T['11']) { pts.zenith11 = toImg(cx(T['11']), T['11'].y0); pts.incisal11 = toImg(cx(T['11']), T['11'].y1); }
    if (T['11'] && T['21']) pts.midlineDental = toImg((T['11'].x1 + T['21'].x0) / 2, (T['11'].y1 + T['21'].y1) / 2 - (T['11'].y1 - T['11'].y0) * 0.05);
    if (T['13']) pts.cusp13 = toImg(cx(T['13']), T['13'].y1);
    if (T['23']) pts.cusp23 = toImg(cx(T['23']), T['23'].y1);
    if (pts.cusp13) pts.lowerLipR = { x: pts.cusp13.x, y: chainY(lower, pts.cusp13.x) };
    if (pts.cusp23) pts.lowerLipL = { x: pts.cusp23.x, y: chainY(lower, pts.cusp23.x) };
    if (extents) { pts.lastR = toImg(extents.x0, extents.y); pts.lastL = toImg(extents.x1, extents.y); }
    return pts;
  }

  // Horizontal extent of the visible dentition (bright, low-chroma pixels) inside the mouth crop, for the buccal corridor.
  function dentitionExtent(rgba, w, h, mouthMask) {
    const col = new Float32Array(w), cnt = new Float32Array(w);
    let ys = 0, yn = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = y * w + x; if (!mouthMask[k]) continue;
      const i = k * 4, r = rgba[i], g = rgba[i + 1], b = rgba[i + 2], L = 0.299 * r + 0.587 * g + 0.114 * b;
      cnt[x]++;
      if (L > 120 && Math.max(r, g, b) - Math.min(r, g, b) < 80) { col[x]++; ys += y; yn++; }
    }
    let x0 = -1, x1 = -1;
    for (let x = 0; x < w; x++) if (cnt[x] > 2 && col[x] / cnt[x] > 0.25) { if (x0 < 0) x0 = x; x1 = x; }
    return x0 < 0 ? null : { x0, x1, y: yn ? ys / yn : h / 2 };
  }

  // A 'tooth' much darker than the other teeth in the SAME photo is a gap the network filled in from context (a lost
  // tooth): segmentation networks do this. Relative brightness, so dim photos and yellow teeth are not flagged.
  // clean: S×S labels (crop squashed to square), class = 1 + index in C.TEETH order; rgba: crop pixels (w×h).
  // Same rule and thresholds as appearance_check in the Python core (validated: 0 false alarms on 258 real teeth).
  const APPEAR_TEETH = ['13', '12', '11', '21', '22', '23'];
  function appearanceCheck(clean, S, rgba, w, h, relBright = 0.55, darkMissing = 0.5) {
    const vals = APPEAR_TEETH.map(() => []);
    for (let y = 0; y < h; y++) {
      const ly = Math.min(S - 1, Math.floor((y + 0.5) * S / h));
      for (let x = 0; x < w; x++) {
        const c = clean[ly * S + Math.min(S - 1, Math.floor((x + 0.5) * S / w))];
        if (c >= 1 && c <= 6) { const i = (y * w + x) * 4; vals[c - 1].push(Math.max(rgba[i], rgba[i + 1], rgba[i + 2])); }
      }
    }
    const med = {}, stats = {};
    APPEAR_TEETH.forEach((t, k) => {
      const v = vals[k]; if (!v.length) return;
      v.sort((a, b) => a - b);
      const n = v.length, m = n % 2 ? v[n >> 1] : (v[n / 2 - 1] + v[n / 2]) / 2;
      let dark = 0; for (const q of v) if (q < 70) dark++;
      med[t] = m; stats[t] = { brightness: m, dark: dark / n };
    });
    const verdict = {}, ts = Object.keys(med);
    if (ts.length < 2) return { verdict, stats };
    for (const t of ts) {
      const o = ts.filter((k) => k !== t).map((k) => med[k]).sort((a, b) => a - b), n = o.length;
      const ref = n % 2 ? o[n >> 1] : (o[n / 2 - 1] + o[n / 2]) / 2;
      stats[t].relative = med[t] / Math.max(1, ref);
      if (med[t] < relBright * ref) verdict[t] = stats[t].dark >= darkMissing ? 'missing' : 'unsure';
    }
    return { verdict, stats };
  }

  // Numbering safety net (same rule as gap_check in the Python core): neighbours in the sequence cannot have a
  // tooth-sized empty space between them (a tooth is missing there and the outer one may carry the wrong number), and
  // cannot lie one within the other (the same tooth split in two). Returns the teeth to confirm.
  const NEIGHBOURS = [['13', '12'], ['12', '11'], ['21', '22'], ['22', '23']];
  function gapCheck(teeth, status, frac = 0.6, overlap = 0.5) {
    const ext = {};
    for (const t of APPEAR_TEETH) if (status[t] === 'present' && teeth[t]) ext[t] = [teeth[t].x0, teeth[t].x1];
    const ws = Object.values(ext).map(([a, b]) => b - a).sort((a, b) => a - b);
    if (ws.length < 2) return [];
    const w = ws.length % 2 ? ws[ws.length >> 1] : (ws[ws.length / 2 - 1] + ws[ws.length / 2]) / 2, out = [];
    for (const [o, i] of NEIGHBOURS) {
      if (!ext[o] || !ext[i]) continue;
      const [a0, a1] = ext[o], [b0, b1] = ext[i], gap = (o === '13' || o === '12') ? b0 - a1 : a0 - b1;
      if (gap > frac * w) out.push(o);
      if (Math.min(a1, b1) - Math.max(a0, b0) > overlap * Math.min(a1 - a0, b1 - b0)) for (const t of [o, i]) if (!out.includes(t)) out.push(t);
    }
    return out;
  }

  const api = { LM, appearanceCheck, gapCheck, INNER_UPPER, INNER_LOWER, OUTER_LIP, IRIS_MM, QUALITY_DEFAULTS, toPx, uprightTurns, eyes, pose, innerLipPolygon, imageStats,
    qualityChecks, mouthCrop, invertAffine, applyAffine, pointInPoly, rasterPoly, dilate, chainY, selectUpperTeeth, classifyAbsent, detectBroken, designFromFace,
    irisMmPerPx, applySmileArc, limitChanges, facePoints, dentitionExtent };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DSDSmile = api;
})(typeof window !== 'undefined' ? window : globalThis);
