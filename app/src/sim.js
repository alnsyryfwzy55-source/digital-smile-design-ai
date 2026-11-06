/* Smile morphing on the patient's own photo. Pure pixel math on {w,h,data:Uint8ClampedArray RGBA}. Browser: window.DSDSim, Node: module.exports */
(function (root) {
  'use strict';
  const TEETH = ['13', '12', '11', '21', '22', '23'];
  const MIRROR = { '13': '23', '12': '22', '11': '21', '21': '11', '22': '12', '23': '13' };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  const makeImg = (w, h) => ({ w, h, data: new Uint8ClampedArray(w * h * 4) });
  function copyImg(a) { return { w: a.w, h: a.h, data: new Uint8ClampedArray(a.data) }; }

  function sample(img, x, y, out) {        // bilinear, clamped; out = [r,g,b]
    const w = img.w, h = img.h, d = img.data;
    x = clamp(x, 0, w - 1.001); y = clamp(y, 0, h - 1.001);
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = (y0 * w + x0) * 4, j = i + w * 4;
    for (let c = 0; c < 3; c++) out[c] = (d[i + c] * (1 - fx) + d[i + 4 + c] * fx) * (1 - fy) + (d[j + c] * (1 - fx) + d[j + 4 + c] * fx) * fy;
    return out;
  }

  // dst(u,v) = src(M·(u,v)), M = [a,b,c,d,e,f] (dst → src)
  function warpAffine(src, M, w, h) {
    const out = makeImg(w, h), px = [0, 0, 0];
    for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
      const x = M[0] * u + M[1] * v + M[2], y = M[3] * u + M[4] * v + M[5], o = (v * w + u) * 4;
      if (x < -1 || y < -1 || x > src.w || y > src.h) { out.data[o + 3] = 255; continue; }
      sample(src, x, y, px); out.data[o] = px[0]; out.data[o + 1] = px[1]; out.data[o + 2] = px[2]; out.data[o + 3] = 255;
    }
    return out;
  }

  // Paste a crop result back into the full image. Minv: image → crop. alpha: Float32 (crop size) blend weight.
  function pasteBack(dst, crop, alpha, Minv, M) {
    // bounding box of the crop in image coordinates
    const cs = [[0, 0], [crop.w, 0], [0, crop.h], [crop.w, crop.h]].map(([u, v]) => [M[0] * u + M[1] * v + M[2], M[3] * u + M[4] * v + M[5]]);
    const x0 = Math.max(0, Math.floor(Math.min(...cs.map((c) => c[0])))), x1 = Math.min(dst.w - 1, Math.ceil(Math.max(...cs.map((c) => c[0]))));
    const y0 = Math.max(0, Math.floor(Math.min(...cs.map((c) => c[1])))), y1 = Math.min(dst.h - 1, Math.ceil(Math.max(...cs.map((c) => c[1]))));
    const px = [0, 0, 0];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const u = Minv[0] * x + Minv[1] * y + Minv[2], v = Minv[3] * x + Minv[4] * y + Minv[5];
      if (u < 0 || v < 0 || u > crop.w - 1 || v > crop.h - 1) continue;
      const a = sampleAlpha(alpha, crop.w, crop.h, u, v); if (a <= 0.002) continue;
      sample(crop, u, v, px);
      const o = (y * dst.w + x) * 4;
      for (let c = 0; c < 3; c++) dst.data[o + c] = dst.data[o + c] * (1 - a) + px[c] * a;
    }
    return dst;
  }
  function sampleAlpha(A, w, h, x, y) {
    const x0 = Math.min(w - 2, x | 0), y0 = Math.min(h - 2, y | 0), fx = x - x0, fy = y - y0, i = y0 * w + x0;
    return (A[i] * (1 - fx) + A[i + 1] * fx) * (1 - fy) + (A[i + w] * (1 - fx) + A[i + w + 1] * fx) * fy;
  }

  function blurAlpha(A, w, h, r) {           // separable box blur ×2 ≈ gaussian
    if (r <= 0) return Float32Array.from(A);
    let a = Float32Array.from(A), b = new Float32Array(w * h);
    for (let pass = 0; pass < 2; pass++) {
      for (let y = 0; y < h; y++) { let s = 0; for (let x = -r; x <= r; x++) s += a[y * w + clamp(x, 0, w - 1)];
        for (let x = 0; x < w; x++) { b[y * w + x] = s / (2 * r + 1); s += a[y * w + clamp(x + r + 1, 0, w - 1)] - a[y * w + clamp(x - r, 0, w - 1)]; } }
      for (let x = 0; x < w; x++) { let s = 0; for (let y = -r; y <= r; y++) s += b[clamp(y, 0, h - 1) * w + x];
        for (let y = 0; y < h; y++) { a[y * w + x] = s / (2 * r + 1); s += b[clamp(y + r + 1, 0, h - 1) * w + x] - b[clamp(y - r, 0, h - 1) * w + x]; } }
    }
    return a;
  }

  // Fill masked pixels (mask=1) smoothly from their surroundings (push–pull pyramid)
  function inpaint(img, mask) {
    const w0 = img.w, h0 = img.h, levels = [];
    let w = w0, h = h0, C = new Float32Array(w * h * 3), Wt = new Float32Array(w * h);
    for (let k = 0; k < w * h; k++) { if (mask[k]) continue; Wt[k] = 1; C[k * 3] = img.data[k * 4]; C[k * 3 + 1] = img.data[k * 4 + 1]; C[k * 3 + 2] = img.data[k * 4 + 2]; }
    levels.push({ w, h, C, Wt });
    while (w > 2 && h > 2) {
      const nw = Math.ceil(w / 2), nh = Math.ceil(h / 2), nC = new Float32Array(nw * nh * 3), nW = new Float32Array(nw * nh);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const k = y * w + x, q = (y >> 1) * nw + (x >> 1);
        nW[q] += Wt[k]; nC[q * 3] += C[k * 3]; nC[q * 3 + 1] += C[k * 3 + 1]; nC[q * 3 + 2] += C[k * 3 + 2];
      }
      for (let q = 0; q < nw * nh; q++) { const s = nW[q]; if (s > 0) { nC[q * 3] /= s; nC[q * 3 + 1] /= s; nC[q * 3 + 2] /= s; nW[q] = Math.min(1, s); } }
      w = nw; h = nh; C = nC; Wt = nW; levels.push({ w, h, C, Wt });
    }
    for (let l = levels.length - 2; l >= 0; l--) {
      const L = levels[l], U = levels[l + 1];
      for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
        const k = y * L.w + x; if (L.Wt[k] >= 1) continue;
        const q = Math.min(U.h - 1, y >> 1) * U.w + Math.min(U.w - 1, x >> 1), a = L.Wt[k];
        for (let c = 0; c < 3; c++) L.C[k * 3 + c] = L.C[k * 3 + c] * a + U.C[q * 3 + c] * (1 - a);
        L.Wt[k] = 1;
      }
    }
    const out = copyImg(img), F = levels[0].C;
    for (let k = 0; k < w0 * h0; k++) if (mask[k]) { out.data[k * 4] = F[k * 3]; out.data[k * 4 + 1] = F[k * 3 + 1]; out.data[k * 4 + 2] = F[k * 3 + 2]; }
    return out;
  }

  // sRGB ↔ CIELAB (D65)
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const gam = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  const LUT = new Float32Array(256); for (let i = 0; i < 256; i++) LUT[i] = lin(i);
  const fL = (t) => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  function rgb2lab(r, g, b, out) {
    const R = LUT[r | 0], G = LUT[g | 0], B = LUT[b | 0];
    const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
    const fx = fL(X), fy = fL(Y), fz = fL(Z);
    out[0] = 116 * fy - 16; out[1] = 500 * (fx - fy); out[2] = 200 * (fy - fz); return out;
  }
  function lab2rgb(L, a, b, out) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const inv = (t) => t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787;
    const X = inv(fx) * 0.95047, Y = inv(fy), Z = inv(fz) * 1.08883;
    out[0] = clamp(gam(3.2406 * X - 1.5372 * Y - 0.4986 * Z), 0, 255);
    out[1] = clamp(gam(-0.9689 * X + 1.8758 * Y + 0.0415 * Z), 0, 255);
    out[2] = clamp(gam(0.0557 * X - 0.2040 * Y + 1.0570 * Z), 0, 255);
    return out;
  }
  function hexLab(hex) { const v = parseInt(hex.slice(1), 16); return rgb2lab((v >> 16) & 255, (v >> 8) & 255, v & 255, [0, 0, 0]); }

  function labStats(img, mask) {
    const s = [0, 0, 0], s2 = [0, 0, 0], lab = [0, 0, 0]; let n = 0;
    for (let k = 0; k < img.w * img.h; k++) {
      if (!mask[k]) continue;
      rgb2lab(img.data[k * 4], img.data[k * 4 + 1], img.data[k * 4 + 2], lab);
      for (let c = 0; c < 3; c++) { s[c] += lab[c]; s2[c] += lab[c] * lab[c]; } n++;
    }
    if (!n) return null;
    return { mean: s.map((v) => v / n), std: s2.map((v, c) => Math.sqrt(Math.max(0, v / n - (s[c] / n) ** 2))), n };
  }

  function erode(m, w, h, r) {
    let cur = m;
    for (let k = 0; k < r; k++) {
      const out = new Uint8Array(cur);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (!cur[i]) continue;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !cur[i - 1] || !cur[i + 1] || !cur[i - w] || !cur[i + w]) out[i] = 0;
      }
      cur = out;
    }
    return cur;
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

  // Tooth outline polygon inside box b (x0..x1, y0 = gingival zenith, y1 = incisal edge)
  function toothPolygon(shape, b, canine, n = 72) {
    const w = b.x1 - b.x0, h = b.y1 - b.y0, cx = (b.x0 + b.x1) / 2;
    const P = { square: { top: 0.84, ex: 5.0, corner: 0.10 }, oval: { top: 0.76, ex: 2.6, corner: 0.30 }, triangular: { top: 0.60, ex: 3.6, corner: 0.14 } }[shape] || { top: 0.84, ex: 5, corner: 0.1 };
    const pts = [];
    for (let i = 0; i < n; i++) {
      const th = (i / n) * Math.PI * 2;                              // 0 → right middle, π/2 → bottom
      const c = Math.cos(th), s = Math.sin(th);
      let x = Math.sign(c) * Math.abs(c) ** (2 / P.ex), y = Math.sign(s) * Math.abs(s) ** (2 / P.ex);
      const v = (y + 1) / 2;                                          // 0 top … 1 incisal
      x *= P.top + (1 - P.top) * Math.min(1, v * 1.6) ** 0.8;        // narrower at the neck
      if (v < 0.5) { const zen = 1 - (0.5 - v) * 0.12; y = (v * 2 - 1) * zen; }   // softer cervical arc
      if (canine && v > 0.62) { const k = ((v - 0.62) / 0.38); x *= 1 - 0.62 * k ** 1.4; y += 0.06 * (1 - Math.abs(x)); }
      pts.push({ x: cx + x * w / 2, y: b.y0 + (y + 1) / 2 * h });
    }
    return pts;
  }

  function rasterPoly(poly, w, h) {
    const m = new Uint8Array(w * h);
    let ya = Infinity, yb = -Infinity; for (const p of poly) { ya = Math.min(ya, p.y); yb = Math.max(yb, p.y); }
    for (let y = Math.max(0, Math.floor(ya)); y <= Math.min(h - 1, Math.ceil(yb)); y++) {
      const xs = [], yy = y + 0.5;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const a = poly[i], b = poly[j];
        if ((a.y > yy) !== (b.y > yy)) xs.push((b.x - a.x) * (yy - a.y) / (b.y - a.y) + a.x);
      }
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(w - 1, Math.floor(xs[k + 1] - 0.5)); x++) m[y * w + x] = 1;
    }
    return m;
  }

  // Source texture for a tooth: its own pixels (or the mirror tooth's, or a library tooth), with the non-enamel part of the box filled from the enamel.
  function toothPatch(img, mask, box) {
    const x0 = Math.max(0, Math.floor(box.x0)), y0 = Math.max(0, Math.floor(box.y0)), x1 = Math.min(img.w - 1, Math.ceil(box.x1)), y1 = Math.min(img.h - 1, Math.ceil(box.y1));
    const w = Math.max(2, x1 - x0 + 1), h = Math.max(2, y1 - y0 + 1), p = makeImg(w, h), hole = new Uint8Array(w * h), inner = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = (y0 + y) * img.w + (x0 + x), o = (y * w + x) * 4;
      for (let c = 0; c < 4; c++) p.data[o + c] = img.data[k * 4 + c];
      if (mask[k]) inner[y * w + x] = 1; else hole[y * w + x] = 1;
    }
    const core = erode(inner, w, h, 1);                    // avoid edge pixels mixed with gum / gaps
    const fillMask = new Uint8Array(w * h); let any = 0;
    for (let k = 0; k < w * h; k++) { fillMask[k] = core[k] ? 0 : 1; any += core[k]; }
    const filled = any > 10 ? inpaint(p, fillMask) : p;
    return { img: filled, mask: core, stats: labStats(filled, core) };
  }

  /*
    Morph the patient's own photo: every tooth is stretched/narrowed/lengthened to its designed box (texture, highlights,
    gum and lower teeth are kept), gaps close, then a texture-preserving whitening. Only missing/broken teeth are painted
    (from the counterpart tooth or the library).
    opts: { crop, mouthMask, toothMasks:{t:Uint8}, current:{t:box} (all six), target:{t:box}, treat, status, shape, shadeHex, whiten, library }
    returns { img, alpha }
  */
  function simulate(o) {
    const { crop, mouthMask } = o, w = crop.w, h = crop.h;
    const treat = (t) => (o.treat[t] || 'veneer') !== 'none';
    const cur = o.current, tgt = {};
    for (const t of TEETH) tgt[t] = treat(t) && o.target[t] ? o.target[t] : cur[t];
    // ---- 1. horizontal map: tooth edges (current) → tooth edges (target); gaps between teeth collapse to the contact point
    const sk = [0], dk = [0];
    let prev = -Infinity;
    for (let i = 0; i < 6; i++) {
      const t = TEETH[i], a = cur[t], b = tgt[t];
      let x0 = a.x0, x1 = a.x1;
      if (x0 < prev) x0 = prev;                     // overlapping boxes: share the edge
      if (x1 < x0 + 1) x1 = x0 + 1;
      sk.push(x0, x1); dk.push(b.x0, b.x1); prev = x1;
    }
    sk.push(w - 1); dk.push(w - 1);
    for (let i = 1; i < dk.length; i++) if (dk[i] < dk[i - 1]) dk[i] = dk[i - 1];   // keep monotonic
    const invX = (x) => {                         // dst x → src x
      if (x <= dk[0]) return sk[0];
      for (let i = 1; i < dk.length; i++) if (x <= dk[i]) { const span = dk[i] - dk[i - 1]; return span < 1e-6 ? sk[i] : sk[i - 1] + (x - dk[i - 1]) * (sk[i] - sk[i - 1]) / span; }
      return sk[sk.length - 1];
    };
    // ---- 2. vertical map per column: gum line fixed, crown stretched to the new incisal edge, the area below follows down to the lower lip
    const centers = TEETH.map((t) => (tgt[t].x0 + tgt[t].x1) / 2);
    const P = (t) => ({ y0c: cur[t].y0, y1c: cur[t].y1, y0t: tgt[t].y0, y1t: tgt[t].y1 });
    const params = TEETH.map(P);
    const colPar = new Array(w);
    for (let x = 0; x < w; x++) {
      let k = 0; while (k < 5 && x > centers[k + 1]) k++;
      let a = params[k], b = params[Math.min(5, k + 1)], f = 0;
      if (x <= centers[0]) { b = a; f = 0; } else if (x >= centers[5]) { a = b = params[5]; } else f = (x - centers[k]) / Math.max(1, centers[k + 1] - centers[k]);
      const L = (u, v) => u + (v - u) * f;
      // fade the vertical change to nothing towards the mouth corners
      const fade = x < tgt['13'].x0 ? clamp(1 - (tgt['13'].x0 - x) / Math.max(1, tgt['13'].x1 - tgt['13'].x0), 0, 1) : x > tgt['23'].x1 ? clamp(1 - (x - tgt['23'].x1) / Math.max(1, tgt['23'].x1 - tgt['23'].x0), 0, 1) : 1;
      const y0c = L(a.y0c, b.y0c), y1c = L(a.y1c, b.y1c), y0t = L(a.y0t, b.y0t), y1t = L(a.y1t, b.y1t);
      colPar[x] = { y0c, y1c, y0t: y0c + (y0t - y0c) * fade, y1t: y1c + (y1t - y1c) * fade };
    }
    const bottom = new Float32Array(w).fill(h - 1);
    for (let x = 0; x < w; x++) { for (let y = h - 1; y >= 0; y--) if (mouthMask[y * w + x]) { bottom[x] = y; break; } }
    const invY = (x, y) => {
      const c = colPar[Math.max(0, Math.min(w - 1, Math.round(x)))], yb = Math.max(bottom[Math.max(0, Math.min(w - 1, Math.round(x)))], c.y1t + 2, c.y1c + 2);
      if (y <= c.y0t) return y - c.y0t + c.y0c;
      if (y <= c.y1t) return c.y0c + (y - c.y0t) * (c.y1c - c.y0c) / Math.max(1, c.y1t - c.y0t);
      if (y <= yb) return c.y1c + (y - c.y1t) * (yb - c.y1c) / Math.max(1, yb - c.y1t);
      return y;
    };
    // ---- 3. warp the mouth interior; warp the tooth masks the same way
    const out = copyImg(crop), px = [0, 0, 0];
    const oldTeeth = new Uint8Array(w * h);
    for (const t of TEETH) { const m = o.toothMasks[t]; if (m && treat(t) && o.status[t] !== 'missing') for (let k = 0; k < w * h; k++) if (m[k]) oldTeeth[k] = 1; }
    const toothDst = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = y * w + x; if (!mouthMask[k]) continue;
      const sx = invX(x), sy = invY(x, y);
      sample(crop, sx, sy, px);
      out.data[k * 4] = px[0]; out.data[k * 4 + 1] = px[1]; out.data[k * 4 + 2] = px[2];
      const mi = clamp(Math.round(sy), 0, h - 1) * w + clamp(Math.round(sx), 0, w - 1);
      if (oldTeeth[mi]) toothDst[k] = 1;
    }
    // ---- 4. reference enamel colour (after warping) and target colour
    const ref = labStats(out, erode(toothDst, w, h, 2));
    const lib = o.library || {};
    const libRef = Object.values(lib).find((l) => l && l.stats)?.stats;
    const base = ref || libRef || { mean: [72, 2, 16], std: [7, 2, 4] };
    const shade = hexLab(o.shadeHex || '#F0E9D8'), s = clamp(o.whiten ?? 1, 0, 1);
    const tL = Math.max(base.mean[0], base.mean[0] + (shade[0] - base.mean[0]) * s), tA = base.mean[1] + (shade[1] - base.mean[1]) * s, tB = base.mean[2] + (shade[2] - base.mean[2]) * s;
    const lab = [0, 0, 0], rgb = [0, 0, 0];
    // ---- 5. paint missing / broken teeth from the counterpart tooth (or library), matched to the patient's enamel
    const painted = new Uint8Array(w * h);
    const usable = (k) => o.status[k] === 'present' && o.toothMasks[k] && cur[k];
    for (const t of ['13', '23', '12', '22', '11', '21']) {
      if (!treat(t) || (o.status[t] !== 'missing' && o.status[t] !== 'broken' && o.status[t] !== 'unsure')) continue;
      let patch = null, flip = false;
      if (usable(MIRROR[t])) { patch = toothPatch(crop, o.toothMasks[MIRROR[t]], cur[MIRROR[t]]); flip = true; }
      else if (lib[t]) patch = lib[t]; else if (lib[MIRROR[t]]) { patch = lib[MIRROR[t]]; flip = true; }
      if (!patch || !patch.stats) continue;
      const b = tgt[t], st = patch.stats, bw = b.x1 - b.x0, bh = b.y1 - b.y0, Pi = patch.img;
      // a. the gap is filled from its surroundings (gum above, teeth beside) before the new tooth is painted, so no
      //    black corners stay around it. The gap is followed by its shape: the dark pixels connected to the dark pixels
      //    inside the new tooth's box (the estimated box of a missing tooth can be narrower than the real gap, and the
      //    design can make the new tooth narrower or shorter). Bounded to a window around the slot, so the dark mouth
      //    elsewhere is left alone; only dark pixels are touched, so the fragment of a broken tooth stays.
      const ti = TEETH.indexOf(t), xl = ti > 0 ? tgt[TEETH[ti - 1]].x1 : b.x0 - 0.3 * bw, xr = ti < 5 ? tgt[TEETH[ti + 1]].x0 : b.x1 + 0.3 * bw;
      const rx0 = Math.max(0, Math.floor(Math.min(b.x0, xl) - 0.3 * bw)), rx1 = Math.min(w - 1, Math.ceil(Math.max(b.x1, xr) + 0.3 * bw));
      const ry0 = Math.max(0, Math.floor(b.y0 - 0.35 * bh)), ry1 = Math.min(h - 1, Math.ceil(b.y1 + 0.2 * bh));
      {
        const rw = rx1 - rx0 + 1, rh = ry1 - ry0 + 1, sub = makeImg(rw, rh), dark = new Uint8Array(rw * rh), hole = new Uint8Array(rw * rh), q = [];
        for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
          const k = (ry0 + y) * w + rx0 + x, j = y * rw + x;
          for (let c = 0; c < 4; c++) sub.data[j * 4 + c] = out.data[k * 4 + c];
          rgb2lab(out.data[k * 4], out.data[k * 4 + 1], out.data[k * 4 + 2], lab);
          if (mouthMask[k] && lab[0] < 40) {               // dark is never enamel, whatever the warped tooth label says
            dark[j] = 1;
            const X = rx0 + x, Y = ry0 + y;
            if (X >= b.x0 && X <= b.x1 && Y >= b.y0 && Y <= b.y1) { hole[j] = 1; q.push(j); }
          }
        }
        while (q.length) {                                 // grow through connected dark pixels
          const j = q.pop(), x = j % rw, y = (j / rw) | 0;
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const X = x + dx, Y = y + dy; if (X < 0 || Y < 0 || X >= rw || Y >= rh) continue;
            const n = Y * rw + X; if (dark[n] && !hole[n]) { hole[n] = 1; q.push(n); }
          }
        }
        let nh = 0; for (let j = 0; j < rw * rh; j++) nh += hole[j];
        if (nh > 0 && nh < rw * rh * 0.97) {
          const f = inpaint(sub, dilate(hole, rw, rh, 1));
          for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) { const j = y * rw + x; if (!hole[j]) continue; const k = (ry0 + y) * w + rx0 + x; for (let c = 0; c < 3; c++) out.data[k * 4 + c] = f.data[j * 4 + c]; toothDst[k] = 0; }
        }
      }
      // b. outline: the counterpart tooth's own shape (mirrored), not a template, so the new tooth has a natural contour
      let pw = 0, ph = 0, PA = null, PB = null;
      if (patch.mask) {
        pw = Pi.w; ph = Pi.h;
        const r1 = Math.max(1, Math.round(0.012 * pw)), r2 = Math.max(2, Math.round(0.06 * pw));
        PA = blurAlpha(Float32Array.from(patch.mask), pw, ph, r1);          // soft edge
        PB = blurAlpha(Float32Array.from(patch.mask), pw, ph, r2);          // wide falloff: the curved surface turns away at the sides
      }
      const poly = PA ? null : toothPolygon(o.shape || 'square', b, t === '13' || t === '23'), al = PA ? null : blurAlpha(Float32Array.from(rasterPoly(poly, w, h)), w, h, 1);
      for (let y = Math.max(0, Math.floor(b.y0) - 2); y <= Math.min(h - 1, Math.ceil(b.y1) + 2); y++) for (let x = Math.max(0, Math.floor(b.x0) - 2); x <= Math.min(w - 1, Math.ceil(b.x1) + 2); x++) {
        const k = y * w + x; if (!mouthMask[k]) continue;
        const u = clamp((x - b.x0) / bw, 0, 1), v = clamp((y - b.y0) / bh, 0, 1), su = (flip ? 1 - u : u) * (pw - 1), sv = v * (ph - 1);
        let a, shade = 0;
        if (PA) {
          a = Math.min(1, sampleAlpha(PA, pw, ph, su, sv) * 1.25);
          shade = 6 * clamp(1 - sampleAlpha(PB, pw, ph, su, sv) * 1.6, 0, 1);    // up to 6 L* darker towards the outline
        } else a = Math.min(1, al[k] * 1.15);
        if (a <= 0.01) continue;
        sample(Pi, (flip ? 1 - u : u) * (Pi.w - 1), v * (Pi.h - 1), px); rgb2lab(px[0], px[1], px[2], lab);
        lab2rgb(base.mean[0] + (lab[0] - st.mean[0]) - shade, base.mean[1] + (lab[1] - st.mean[1]) * 0.7, base.mean[2] + (lab[2] - st.mean[2]) * 0.7, rgb);
        for (let c = 0; c < 3; c++) out.data[k * 4 + c] = out.data[k * 4 + c] * (1 - a) + rgb[c] * a;
        if (a > 0.5) { painted[k] = 1; toothDst[k] = 1; }
      }
    }
    // ---- 6. tooth form: round the incisal corners (what shows there is what is behind: lower teeth / mouth)
    for (const t of TEETH) {
      if (!treat(t)) continue;
      const b = tgt[t], poly = toothPolygon(o.shape || 'square', b, t === '13' || t === '23'), pm = rasterPoly(poly, w, h), bh = b.y1 - b.y0;
      const yStart = Math.round(b.y0 + 0.55 * bh), i = TEETH.indexOf(t);
      // only where no neighbouring tooth shares the box: overlapping boxes would otherwise cut into the neighbour
      const xa = Math.max(b.x0, i > 0 ? tgt[TEETH[i - 1]].x1 : -Infinity), xb = Math.min(b.x1, i < 5 ? tgt[TEETH[i + 1]].x0 : Infinity);
      for (let y = Math.max(0, yStart); y <= Math.min(h - 1, Math.ceil(b.y1)); y++) for (let x = Math.max(0, Math.ceil(xa)); x <= Math.min(w - 1, Math.floor(xb)); x++) {
        const k = y * w + x; if (pm[k] || !toothDst[k] || !mouthMask[k]) continue;
        const sy = Math.min(h - 1, Math.round(b.y1 + 0.06 * bh)), j = sy * w + x;      // just below the incisal edge
        for (let c = 0; c < 3; c++) out.data[k * 4 + c] = out.data[j * 4 + c];
        toothDst[k] = 0;
      }
    }
    // ---- 7. whitening that keeps the enamel texture and highlights.
    // The region is every enamel-coloured pixel inside the designed teeth (the model mask alone can miss parts of a tooth).
    const refL = base.mean[0], refStd = Math.max(4, base.std[0]), refC = Math.hypot(base.mean[1], base.mean[2]);
    const wm = new Uint8Array(w * h);
    for (const t of TEETH) {
      if (!treat(t)) continue;
      const b = tgt[t], ext = Math.max(3, Math.round(0.08 * (b.x1 - b.x0)));
      const shapeM = dilate(rasterPoly(toothPolygon(o.shape || 'square', b, t === '13' || t === '23'), w, h), w, h, ext), yCut = b.y0 + 0.88 * (b.y1 - b.y0);
      // the strip between two boxes is enamel too: leaving it out shows as a vertical line between whitened teeth
      for (let y = Math.max(0, Math.floor(b.y0) - 1); y <= Math.min(h - 1, Math.ceil(b.y1)); y++) for (let x = Math.max(0, Math.floor(b.x0) - ext); x <= Math.min(w - 1, Math.ceil(b.x1) + ext); x++) {
        const k = y * w + x; if (!mouthMask[k]) continue;
        if (toothDst[k]) { wm[k] = 1; continue; }          // the tooth itself, whole (a template would cut a canine diagonally)
        if (!shapeM[k] || y > yCut) continue;                // near the incisal edge only the tooth itself (not the lower teeth behind it)
        rgb2lab(out.data[k * 4], out.data[k * 4 + 1], out.data[k * 4 + 2], lab);
        if (lab[0] >= refL - 2.2 * refStd && Math.hypot(lab[1], lab[2]) <= refC + 14 && lab[1] < 14) wm[k] = 1;   // enamel, not gum (gum is red: high a*)
      }
    }
    const alpha = blurAlpha(Float32Array.from(wm), w, h, 2);
    const dL = clamp(tL - refL, 0, 16);                      // natural whitening: at most ~16 L* units
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = y * w + x, a = alpha[k]; if (a <= 0.01 || !mouthMask[k]) continue;
      rgb2lab(out.data[k * 4], out.data[k * 4 + 1], out.data[k * 4 + 2], lab);
      const hi = clamp((lab[0] - refL) / Math.max(1, 100 - refL), 0, 1);       // highlights brighten less, so they do not burn out
      // the colour shift fades out in highlights like the brightness does: highlights are already neutral white, and
      // pushing them further from yellow turns them blue-green
      let L = lab[0] + dL * (1 - 0.8 * hi), A = lab[1] + (tA - base.mean[1]) * 0.8 * (1 - hi), B = lab[2] + (tB - base.mean[2]) * (1 - hi);
      // soft shoulder: above L* 92 the brightness approaches white gradually instead of clipping, so highlight
      // detail survives in photos whose teeth are already bright (a burnt-out white patch reads as plastic)
      if (L > 92 && L > lab[0]) L = Math.max(lab[0], 92 + 8 * (1 - Math.exp(-(L - 92) / 8)));
      lab2rgb(L, A, B, rgb);
      for (let c = 0; c < 3; c++) out.data[k * 4 + c] = out.data[k * 4 + c] * (1 - a) + rgb[c] * a;
    }
    // ---- 8. lips stay on top: blend back the original outside the mouth opening
    const ma = blurAlpha(Float32Array.from(mouthMask), w, h, 1);
    for (let k = 0; k < w * h; k++) { const a = ma[k]; if (a >= 1) continue; for (let c = 0; c < 3; c++) out.data[k * 4 + c] = crop.data[k * 4 + c] * (1 - a) + out.data[k * 4 + c] * a; }
    return { img: out, alpha: ma };
  }

  // typical colour of the dark mouth interior: the darkest fifth of the mouth pixels that are not teeth
  function darkColour(img, mouth, teeth) {
    const px = [];
    for (let k = 0; k < img.w * img.h; k += 2) if (mouth[k] && !teeth[k]) { const i = k * 4; px.push([0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2], i]); }
    if (px.length < 20) return [35, 18, 20];
    px.sort((a, b) => a[0] - b[0]);
    const n = Math.max(5, Math.floor(px.length * 0.2)), s = [0, 0, 0];
    for (let j = 0; j < n; j++) for (let c = 0; c < 3; c++) s[c] += img.data[px[j][1] + c];
    return s.map((v) => v / n);
  }

  const api = { TEETH, makeImg, copyImg, sample, warpAffine, pasteBack, blurAlpha, inpaint, rgb2lab, lab2rgb, hexLab, labStats, erode, dilate, toothPolygon, rasterPoly, toothPatch, simulate };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DSDSim = api;
})(typeof window !== 'undefined' ? window : globalThis);
