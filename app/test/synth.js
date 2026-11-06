// Synthetic smiling face: landmarks (478, normalized) + RGBA image with teeth. Used by tests.
const S = require('../src/smile.js');
function face(W = 1200, H = 1600, opts = {}) {
  const roll = (opts.roll || 0) * Math.PI / 180, cx = W / 2, cy = H * 0.5, sc = opts.scale || 1;
  const P = Array.from({ length: 478 }, () => ({ x: cx, y: cy }));
  const set = (i, x, y) => { P[i] = { x, y }; };
  const fw = 520 * sc;                       // face width
  set(10, cx, cy - 520 * sc); set(152, cx, cy + 520 * sc);
  set(234, cx - fw / 2, cy - 60 * sc); set(454, cx + fw / 2, cy - 60 * sc);
  set(468, cx - 110 * sc, cy - 160 * sc); set(473, cx + 110 * sc, cy - 160 * sc);
  const ir = 11.7 / 0.30 / 2 * sc / 1;        // iris radius px so that 1px ≈ 0.30mm at scale 1 → iris diameter 39px
  const irisD = 11.7 / 0.30 * sc;
  for (const [c, ring] of [[468, [469, 470, 471, 472]], [473, [474, 475, 476, 477]]]) {
    const o = P[c]; set(ring[0], o.x + irisD / 2, o.y); set(ring[1], o.x, o.y - irisD / 2); set(ring[2], o.x - irisD / 2, o.y); set(ring[3], o.x, o.y + irisD / 2);
  }
  set(33, cx - 160 * sc, cy - 160 * sc); set(133, cx - 60 * sc, cy - 160 * sc); set(362, cx + 60 * sc, cy - 160 * sc); set(263, cx + 160 * sc, cy - 160 * sc);
  set(9, cx, cy - 210 * sc); set(168, cx, cy - 160 * sc); set(1, cx + (opts.yaw || 0) * 400 * sc, cy + 45 * sc); set(2, cx, cy + 75 * sc);
  const mw = 270 * sc, my = cy + 210 * sc, open = (opts.open ?? 70) * sc;
  set(61, cx - mw / 2, my); set(291, cx + mw / 2, my);
  const arcUp = (t) => my - open / 2 * Math.sin(Math.PI * t) ** 0.6, arcLo = (t) => my + open / 2 * Math.sin(Math.PI * t) ** 0.6 - (opts.lipCurve ?? 10) * sc * (1 - Math.sin(Math.PI * t));
  S.INNER_UPPER.forEach((i, k) => { const t = k / (S.INNER_UPPER.length - 1); set(i, cx - mw / 2 * 0.92 + mw * 0.92 * t, arcUp(t)); });
  S.INNER_LOWER.forEach((i, k) => { const t = 1 - k / (S.INNER_LOWER.length - 1); if (i === 308 || i === 78) return; set(i, cx - mw / 2 * 0.92 + mw * 0.92 * t, arcLo(t)); });
  S.OUTER_LIP.forEach((i, k) => { const a = k / S.OUTER_LIP.length * 2 * Math.PI; if (i === 61 || i === 291) return; set(i, cx - Math.cos(a) * mw / 2, my - Math.sin(a) * (open / 2 + 22 * sc)); });
  // rotate everything by roll around face centre
  const R = (p) => ({ x: cx + (p.x - cx) * Math.cos(roll) - (p.y - cy) * Math.sin(roll), y: cy + (p.x - cx) * Math.sin(roll) + (p.y - cy) * Math.cos(roll) });
  const Pr = P.map(R);
  return { W, H, P: Pr, lm: Pr.map((p) => ({ x: p.x / W, y: p.y / H, z: 0 })), mw, my, open, sc };
}
// Draw the face into an RGBA buffer: skin, lips, dark mouth, gums, yellowish upper teeth (12 narrow + gap option), lower teeth.
function render(f, opts = {}) {
  const { W, H, P } = f, d = new Uint8ClampedArray(W * H * 4);
  const inner = S.innerLipPolygon(P), outer = S.OUTER_LIP.map((i) => P[i]);
  const inPoly = (x, y, poly) => S.pointInPoly(x, y, poly);
  const teeth = opts.teeth || defaultTeeth(f);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let c = [205, 160, 135];                         // skin
    if (inPoly(x, y, outer)) c = [170, 85, 90];       // lips
    if (inPoly(x, y, inner)) {
      c = [45, 20, 25];                                // mouth interior
      if (y < f.my - f.open * 0.22) c = [200, 110, 120];   // gum band
      for (const t of Object.keys(teeth)) { const b = teeth[t]; if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) c = [220 + (t === '11' ? 0 : -6), 200, 150]; }
      if (y > f.my + f.open * 0.22 && y <= f.my + f.open * 0.27) c = [40, 25, 25];   // occlusal shadow
      if (y > f.my + f.open * 0.27) c = [205, 195, 160];   // lower teeth row
    }
    const n = (rnd() - 0.5) * 10, i = (y * W + x) * 4;
    d[i] = c[0] + n; d[i + 1] = c[1] + n; d[i + 2] = c[2] + n; d[i + 3] = 255;
  }
  return { w: W, h: H, data: d, teeth };
}
function defaultTeeth(f) {
  const cx = f.W / 2, c = 42 * f.sc, top = f.my - f.open * 0.36, L = f.open * 0.58;
  const ws = [c * 0.5, c * 0.72, c, c, c * 0.72, c * 0.5], names = ['13', '12', '11', '21', '22', '23'];
  let x = cx - c - c * 0.72 - c * 0.5; const T = {};
  names.forEach((t, i) => { T[t] = { x0: x + 1.5, x1: x + ws[i] - 1.5, y0: top + (i === 1 || i === 4 ? 4 : 0), y1: top + L * [0.9, 0.8, 1, 1, 0.8, 0.9][i] }; x += ws[i]; });
  return T;
}
module.exports = { face, render, defaultTeeth };
