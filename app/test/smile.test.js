const assert = require('assert');
const C = require('../src/core.js'), S = require('../src/smile.js'), Sim = require('../src/sim.js');
const { face, render } = require('./synth.js');
let pass = 0; const ok = (n, f) => { try { f(); pass++; console.log('✓', n); } catch (e) { console.log('✗', n, '\n  ', e.stack.split('\n').slice(0, 3).join('\n   ')); process.exitCode = 1; } };
const near = (a, b, e, m) => assert(Math.abs(a - b) <= e, `${m}: ${a} vs ${b}`);

ok('pose: frontal and rolled', () => {
  const f = face(); const p = S.pose(f.P); near(p.roll, 0, 0.01, 'roll'); near(p.yaw, 0, 0.01, 'yaw');
  near(S.pose(face(1200, 1600, { roll: 9 }).P).roll, 9, 0.05, 'roll 9');
  assert(Math.abs(S.pose(face(1200, 1600, { yaw: 0.2 }).P).yaw) > 0.1, 'yaw detect');
});
ok('upright turns from eye/mouth geometry', () => {
  const f = face(1200, 1200);
  const rot = (P, k) => P.map((p) => { let q = { ...p }; for (let i = 0; i < k; i++) q = { x: 1200 - q.y, y: q.x }; return q; });  // 90° cw each
  assert.strictEqual(S.uprightTurns(f.P), 0);
  for (const k of [1, 2, 3]) { const need = S.uprightTurns(rot(f.P, k)); assert.strictEqual((k + need) % 4, 0, `k=${k} need=${need}`); }
});
ok('quality checks pass on a good frame and explain failures', () => {
  const f = face(); const img = render(f); const st = S.imageStats(img.data, f.W, f.H, f.P);
  const q = S.qualityChecks(f.P, f.W, f.H, st);
  assert(q.ok, JSON.stringify(q.items.filter((i) => !i.ok)) + JSON.stringify(st));
  const small = face(1200, 1600, { scale: 0.45 }); const q2 = S.qualityChecks(small.P, small.W, small.H, S.imageStats(render(small).data, small.W, small.H, small.P));
  assert(!q2.items.find((i) => i.id === 'distance').ok);
  const dark = { ...st, faceL: 40 }; assert(!S.qualityChecks(f.P, f.W, f.H, dark).items.find((i) => i.id === 'light').ok);
  const tilted = face(1200, 1600, { roll: 8 }); assert(!S.qualityChecks(tilted.P, tilted.W, tilted.H, st).items.find((i) => i.id === 'roll').ok);
  const closed = face(1200, 1600, { open: 8 }); assert(!S.qualityChecks(closed.P, closed.W, closed.H, { ...st, teethFrac: 0.02 }).items.find((i) => i.id === 'smile').ok);
});
ok('mouth crop: centre, aspect, inverse, roll aligned', () => {
  const f = face(1200, 1600, { roll: 7 }); const cr = S.mouthCrop(f.P);
  near(cr.outW / cr.outH, 1.6, 0.01, 'aspect');
  const c = S.applyAffine(cr.M, { x: cr.outW / 2, y: cr.outH / 2 }); near(c.x, cr.center.x, 1e-6, 'cx'); near(c.y, cr.center.y, 1e-6, 'cy');
  const back = S.applyAffine(cr.Minv, S.applyAffine(cr.M, { x: 10, y: 20 })); near(back.x, 10, 1e-6, 'inv'); near(back.y, 20, 1e-6, 'inv');
  const a = S.applyAffine(cr.Minv, f.P[61]), b = S.applyAffine(cr.Minv, f.P[291]); near(a.y, b.y, 1e-6, 'commissures level in crop');
});
function cropOf(f) {
  const cr = S.mouthCrop(f.P), img = render(f);
  const crop = Sim.warpAffine(img, cr.M, cr.outW, cr.outH);
  const toC = (p) => S.applyAffine(cr.Minv, p);
  const lips = { upper: S.INNER_UPPER.map((i) => toC(f.P[i])), lower: S.INNER_LOWER.map((i) => toC(f.P[i])), poly: S.innerLipPolygon(f.P).map(toC) };
  const teethC = {}; for (const [t, b] of Object.entries(img.teeth)) { const p0 = toC({ x: b.x0, y: b.y0 }), p1 = toC({ x: b.x1, y: b.y1 }); teethC[t] = { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y }; }
  return { cr, img, crop, lips, teethC };
}
function labelsFrom(teethC, w, h, S0, extra = []) {
  const L = new Uint8Array(S0 * S0), conf = new Float32Array(S0 * S0).fill(0.95);
  const put = (b, c) => { for (let y = Math.floor(b.y0 / h * S0); y <= Math.ceil(b.y1 / h * S0); y++) for (let x = Math.floor(b.x0 / w * S0); x <= Math.ceil(b.x1 / w * S0); x++) if (x >= 0 && y >= 0 && x < S0 && y < S0) L[y * S0 + x] = c; };
  for (const [t, b] of Object.entries(teethC)) put(b, C.CLASS_OF[t]);
  for (const [b, c] of extra) put(b, c);
  return { L, conf };
}
ok('selectUpperTeeth keeps upper teeth and rejects lower teeth mislabelled as upper', () => {
  const f = face(); const { crop, lips, teethC } = cropOf(f);
  const lowY = Math.max(...lips.lower.map((p) => p.y));
  // (a) lower tooth as a separate blob below the occlusal shadow → rejected
  const lowerSep = [[{ x0: teethC['11'].x0, x1: teethC['21'].x1, y0: teethC['11'].y1 + 0.08 * (lowY - teethC['11'].y0), y1: lowY - 4 }, 3]];
  let { L, conf } = labelsFrom(teethC, crop.w, crop.h, 384, lowerSep);
  // carve the occlusal shadow rows out of the labels so the fake lower blob is separate
  for (let y = 0; y < 384; y++) { const cy = (y + 0.5) * crop.h / 384; if (cy > teethC['11'].y1 && cy < teethC['11'].y1 + 0.08 * (lowY - teethC['11'].y0)) for (let x = 0; x < 384; x++) L[y * 384 + x] = 0; }
  let sel = S.selectUpperTeeth(L, conf, 384, crop.w, crop.h, lips);
  assert.strictEqual(sel.presentCount, 6); assert(sel.orderOk);
  near(sel.teeth['11'].y1, teethC['11'].y1, 4, 'upper 11 kept'); assert(sel.rejected.some((r) => r.reason === 'lower'), 'lower blob rejected');
  // (b) model bleeds the label across the shadow into the lower teeth → one tall blob → cut at the dark occlusal line
  ({ L, conf } = labelsFrom(teethC, crop.w, crop.h, 384, [[{ x0: teethC['11'].x0, x1: teethC['11'].x1, y0: teethC['11'].y0, y1: lowY - 4 }, 3]]));
  sel = S.selectUpperTeeth(L, conf, 384, crop.w, crop.h, lips, { rgba: crop.data });
  assert(sel.teeth['11'].cut, 'cut applied'); near(sel.teeth['11'].y1, teethC['11'].y1, 5, 'cut at the occlusal line');
  const noCut = S.selectUpperTeeth(L, conf, 384, crop.w, crop.h, lips);
  assert(noCut.teeth['11'].y1 > teethC['11'].y1 + 10, 'without image the blob stays tall (control)');
  // (c) a single tooth with a darker translucent incisal edge must NOT be cut
  const im2 = { ...crop, data: new Uint8ClampedArray(crop.data) }; const b11 = teethC['11'];
  for (let y = Math.round(b11.y1) - 6; y <= Math.round(b11.y1) + 30; y++) for (let x = Math.round(b11.x0); x <= Math.round(b11.x1); x++) { const i = (y * crop.w + x) * 4; const dark = y > b11.y1 - 6; im2.data[i] = dark ? 150 : 220; im2.data[i + 1] = dark ? 150 : 200; im2.data[i + 2] = dark ? 160 : 150; }
  ({ L, conf } = labelsFrom(teethC, crop.w, crop.h, 384, [[{ x0: b11.x0, x1: b11.x1, y0: b11.y0, y1: b11.y1 + 0.3 * (b11.x1 - b11.x0) }, 3]]));
  const tr = S.selectUpperTeeth(L, conf, 384, crop.w, crop.h, lips, { rgba: im2.data });
  assert(!tr.teeth['11'].cut, 'translucent edge not cut');
});
ok('classifyAbsent: gap → missing, canine out of view → hidden', () => {
  const f = face(); const t = require('./synth.js').defaultTeeth(f); delete t['12'];
  const shown = Object.assign({}, t); delete shown['23'];          // canine really not in the picture
  const img = render(f, { teeth: shown }); const cr = S.mouthCrop(f.P); const crop = Sim.warpAffine(img, cr.M, cr.outW, cr.outH);
  const toC = (p) => S.applyAffine(cr.Minv, p); const teethC = {};
  for (const [k, b] of Object.entries(t)) { const p0 = toC({ x: b.x0, y: b.y0 }), p1 = toC({ x: b.x1, y: b.y1 }); teethC[k] = { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y }; }
  delete teethC['23'];
  const st = S.classifyAbsent({ teeth: teethC }, crop.data, crop.w, crop.h);
  assert.strictEqual(st['12'], 'missing'); assert.strictEqual(st['23'], 'hidden');
});
ok('classifyAbsent: canine visible but not labelled → estimated', () => {
  const f = face(); const t = require('./synth.js').defaultTeeth(f);
  const img = render(f, { teeth: t }); const cr = S.mouthCrop(f.P); const crop = Sim.warpAffine(img, cr.M, cr.outW, cr.outH);
  const toC = (p) => S.applyAffine(cr.Minv, p); const teethC = {};
  for (const [k, b] of Object.entries(t)) { const p0 = toC({ x: b.x0, y: b.y0 }), p1 = toC({ x: b.x1, y: b.y1 }); teethC[k] = { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y }; }
  delete teethC['23'];
  const st = S.classifyAbsent({ teeth: teethC }, crop.data, crop.w, crop.h);
  assert.strictEqual(st['23'], 'estimated');
});
ok('smile arc: incisal edges follow a rising lower lip, centrals stay longest', () => {
  const T = require('./synth.js').defaultTeeth(face());
  const lip = (x) => 900 - 0.002 * (x - 600) ** 2;   // curves up towards the corners
  const out = S.applySmileArc(T, lip, 0.1, {});
  assert(out['12'].y1 < out['11'].y1 && out['13'].y1 < out['11'].y1, 'laterals/canines shorter');
  near(out['11'].y1, out['21'].y1, 1e-9, 'centrals equal');
  const longer = S.applySmileArc(T, lip, 0.1, { lengthDeltaMm: 1 }); near(longer['11'].y1 - out['11'].y1, 10, 1e-6, '+1mm = 10px at 0.1mm/px');
});
ok('iris scale', () => { near(S.irisMmPerPx(face().P), 0.30, 1e-9, 'mm/px'); });
ok('lab round trip', () => { const l = [0, 0, 0], r = [0, 0, 0]; for (const c of [[0, 0, 0], [255, 255, 255], [220, 200, 150], [30, 120, 220]]) { Sim.rgb2lab(...c, l); Sim.lab2rgb(...l, r); for (let i = 0; i < 3; i++) near(r[i], c[i], 0.6, 'rt'); } });
ok('inpaint fills a hole smoothly', () => {
  const im = Sim.makeImg(64, 40); for (let y = 0; y < 40; y++) for (let x = 0; x < 64; x++) { const i = (y * 64 + x) * 4; im.data[i] = x * 4; im.data[i + 1] = 100; im.data[i + 2] = 50; im.data[i + 3] = 255; }
  const m = new Uint8Array(64 * 40); for (let y = 10; y < 30; y++) for (let x = 20; x < 44; x++) m[y * 64 + x] = 1;
  const out = Sim.inpaint(im, m); const v = out.data[(20 * 64 + 32) * 4]; assert(v > 80 && v < 170, 'centre ~ neighbours ' + v); assert.strictEqual(out.data[(20 * 64 + 32) * 4 + 1], 100);
});
ok('toothPolygon stays in its box; canine has a cusp', () => {
  const b = { x0: 10, x1: 60, y0: 5, y1: 80 };
  for (const s of ['square', 'oval', 'triangular']) for (const p of Sim.toothPolygon(s, b, false)) assert(p.x >= 9.9 && p.x <= 60.1 && p.y >= 4.9 && p.y <= 80.1, s);
  const c = Sim.toothPolygon('square', b, true); const bottom = c.filter((p) => p.y > 76); assert(bottom.every((p) => Math.abs(p.x - 35) < 12), 'cusp');
});
ok('simulate: lips/skin untouched, treated teeth whiter, gap closed', () => {
  const f = face(); const t = require('./synth.js').defaultTeeth(f); t['12'].x1 -= 10;   // narrow 12 → gap
  const img = render(f, { teeth: t }); const cr = S.mouthCrop(f.P); const crop = Sim.warpAffine(img, cr.M, cr.outW, cr.outH);
  const toC = (p) => S.applyAffine(cr.Minv, p), w = crop.w, h = crop.h;
  const cur = {}, masks = {};
  for (const [k, b] of Object.entries(t)) { const p0 = toC({ x: b.x0, y: b.y0 }), p1 = toC({ x: b.x1, y: b.y1 }); cur[k] = { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y };
    const m = new Uint8Array(w * h); for (let y = Math.ceil(p0.y); y <= Math.floor(p1.y); y++) for (let x = Math.ceil(p0.x); x <= Math.floor(p1.x); x++) m[y * w + x] = 1; masks[k] = m; }
  const mouth = S.rasterPoly(S.innerLipPolygon(f.P).map(toC), w, h);
  const meas = C.measure(cur, 0.3), design = C.proposeDesign(cur, meas, C.DEFAULT_RULES, { proportion: 'red70' });
  const lip = S.INNER_LOWER.map((i) => toC(f.P[i]));
  const target = S.limitChanges(cur, S.applySmileArc(design.target, (x) => S.chainY(lip, x), 0.3), 0.3);
  for (const k of C.TEETH) assert(cur[k].y1 - target[k].y1 <= 1.5 / 0.3 + 1e-6, 'shortening limited');
  const treat = Object.fromEntries(C.TEETH.map((k) => [k, 'veneer']));
  const r = Sim.simulate({ crop, mouthMask: mouth, toothMasks: masks, current: cur, target, treat, status: {}, shape: 'square', shadeHex: '#F3EFE6', whiten: 1 });
  // outside the mouth (lips, skin) nothing changes
  let maxOut = 0; for (let k = 0; k < w * h; k++) if (r.alpha[k] === 0) for (let c = 0; c < 3; c++) maxOut = Math.max(maxOut, Math.abs(r.img.data[k * 4 + c] - crop.data[k * 4 + c]));
  assert.strictEqual(maxOut, 0, 'outside changed');
  // treated teeth: whiter (lower b*) and brighter on average
  const lab = [0, 0, 0]; const stat = (im, b) => { let L = 0, B = 0, n = 0; for (let y = Math.ceil(b.y0 + 4); y < b.y1 - 4; y++) for (let x = Math.ceil(b.x0 + 4); x < b.x1 - 4; x++) { const i = (y * w + x) * 4; Sim.rgb2lab(im.data[i], im.data[i + 1], im.data[i + 2], lab); L += lab[0]; B += lab[2]; n++; } return [L / n, B / n]; };
  const [L0, B0] = stat(crop, cur['11']), [L1, B1] = stat(r.img, target['11']);
  assert(B1 < B0 - 5 && L1 >= L0 - 2, `b* ${B0.toFixed(1)}→${B1.toFixed(1)} L ${L0.toFixed(1)}→${L1.toFixed(1)}`);
  // the old gap next to 12 is now covered by tooth colour (bright), not dark mouth
  const gx = Math.round(cur['12'].x1 + 4), gy = Math.round((target['12'].y0 + target['12'].y1) / 2), gi = (gy * w + gx) * 4;
  assert(crop.data[gi] < 90 && r.img.data[gi] > 150, `gap ${crop.data[gi]} → ${r.img.data[gi]}`);
  for (let k = 0; k < r.img.data.length; k++) if (Number.isNaN(r.img.data[k])) throw new Error('NaN');
  // where an old tooth was removed and no new tooth covers it, the mouth is dark (no grey ghost of the old tooth)
  const b = cur['11'], tb = target['11'];
  if (b.y1 - tb.y1 > 4) { const yy = Math.round((tb.y1 + b.y1) / 2 + 1), xx = Math.round((b.x0 + b.x1) / 2), ii = (yy * w + xx) * 4; assert(r.img.data[ii] < 80, 'ghost ' + r.img.data[ii]); }
  module.exports.last = { crop, r, w, h };
});
console.log(`\n${pass} passed`);

// ---------- trauma cases ----------
const synth = require('./synth.js');
function cropAndMasks(f, teeth, opts = {}) {
  const img = synth.render(f, { teeth }); const cr = S.mouthCrop(f.P); const crop = Sim.warpAffine(img, cr.M, cr.outW, cr.outH);
  const toC = (p) => S.applyAffine(cr.Minv, p), w = crop.w, h = crop.h, cur = {}, masks = {};
  for (const [k, b] of Object.entries(teeth)) { const p0 = toC({ x: b.x0, y: b.y0 }), p1 = toC({ x: b.x1, y: b.y1 }); cur[k] = { x0: p0.x, y0: p0.y, x1: p1.x, y1: p1.y };
    const m = new Uint8Array(w * h); for (let y = Math.ceil(p0.y); y <= Math.floor(p1.y); y++) for (let x = Math.ceil(p0.x); x <= Math.floor(p1.x); x++) m[y * w + x] = 1; masks[k] = m; }
  const lips = { upper: S.INNER_UPPER.map((i) => toC(f.P[i])), lower: S.INNER_LOWER.map((i) => toC(f.P[i])), poly: S.innerLipPolygon(f.P).map(toC) };
  return { crop, cur, masks, lips, mouth: S.rasterPoly(lips.poly, w, h), w, h, toC, cr };
}
function fakeLibrary() {
  const lib = {};
  for (const t of C.TEETH) { const w = 40, h = 60, img = Sim.makeImg(w, h), m = new Uint8Array(w * h).fill(1);
    for (let k = 0; k < w * h; k++) { const n = ((k * 7919) % 13) - 6; img.data[k * 4] = 232 + n; img.data[k * 4 + 1] = 222 + n; img.data[k * 4 + 2] = 196 + n; img.data[k * 4 + 3] = 255; }
    lib[t] = { img, mask: m, stats: Sim.labStats(img, m) }; }
  return lib;
}
const meanL = (im, b, w) => { const lab = [0, 0, 0]; let s = 0, n = 0; for (let y = Math.ceil(b.y0 + 0.25 * (b.y1 - b.y0)); y < b.y1 - 0.2 * (b.y1 - b.y0); y++) for (let x = Math.ceil(b.x0 + 3); x < b.x1 - 3; x++) { const i = (y * w + x) * 4; Sim.rgb2lab(im.data[i], im.data[i + 1], im.data[i + 2], lab); s += lab[0]; n++; } return s / n; };

ok('lost tooth over pink healed gum is classified missing (not unsure)', () => {
  const f = face(); const t = synth.defaultTeeth(f); const lost = t['21']; delete t['21'];
  const r = cropAndMasks(f, t); const p0 = r.toC({ x: lost.x0, y: lost.y0 }), p1 = r.toC({ x: lost.x1, y: lost.y1 });
  for (let y = Math.round(p0.y); y <= Math.round(p1.y); y++) for (let x = Math.round(p0.x); x <= Math.round(p1.x); x++) { const i = (y * r.w + x) * 4; r.crop.data[i] = 205; r.crop.data[i + 1] = 120; r.crop.data[i + 2] = 125; }
  const st = S.classifyAbsent({ teeth: r.cur }, r.crop.data, r.w, r.h);
  assert.strictEqual(st['21'], 'missing');
});
ok('fractured incisor detected from its counterpart and from W/L', () => {
  const T = synth.defaultTeeth(face()); const st = Object.fromEntries(C.TEETH.map((k) => [k, 'present']));
  T['21'].y1 = T['21'].y0 + (T['21'].y1 - T['21'].y0) * 0.55;
  assert.deepStrictEqual([...S.detectBroken(T, st)], ['21']);
  const T2 = synth.defaultTeeth(face()); for (const k of ['11', '21']) T2[k].y1 = T2[k].y0 + (T2[k].x1 - T2[k].x0) / 1.6;
  assert.deepStrictEqual([...S.detectBroken(T2, st)].sort(), ['11', '21']);
  assert.deepStrictEqual([...S.detectBroken(synth.defaultTeeth(face()), st)], [], 'healthy arch: none');
  // measurement precision: a borderline shortening (ratio 0.74) is 'broken' on a sharp photo, 'unsure' on a coarse one;
  // a clear fracture (0.55) stays 'broken' either way
  const T3 = synth.defaultTeeth(face()); T3['21'].y1 = T3['21'].y0 + (T3['11'].y1 - T3['11'].y0) * 0.74;
  assert.deepStrictEqual([...S.detectBroken(T3, st, 0.3)], ['21']);
  const coarse = S.detectBroken(T3, st, 2);
  assert.deepStrictEqual([...coarse], []); assert.deepStrictEqual(coarse.unsure, ['21']);
  assert.deepStrictEqual([...S.detectBroken(T, st, 2)], ['21']);
});
ok('rebuilding a missing/broken tooth is not capped by the change limits', () => {
  const cur = { '11': { x0: 0, x1: 85, y0: 0, y1: 40 } }, tgt = { '11': { x0: 0, x1: 85, y0: 0, y1: 105 } };
  assert(S.limitChanges(cur, tgt, 0.1, {}, { '11': 'present' })['11'].y1 <= 40 + 25 + 1e-9);
  assert.strictEqual(S.limitChanges(cur, tgt, 0.1, {}, { '11': 'broken' })['11'].y1, 105);
});
ok('no front teeth: an ideal set is designed from the face (midline, mm scale, order)', () => {
  const f = face(); const r = cropAndMasks(f, {});
  const midX = r.w / 2 + 6, T = S.designFromFace(r.lips, midX, 0.1);
  near((T['11'].x1 + T['21'].x0) / 2, midX, 1e-6, 'on the facial midline');
  near(T['11'].x1 - T['11'].x0, 86, 1e-6, '8.6 mm at 0.1 mm/px');
  for (let i = 1; i < 6; i++) assert(T[C.TEETH[i]].x0 >= T[C.TEETH[i - 1]].x1 - 1e-6, 'order');
  assert(T['11'].y1 - T['11'].y0 > T['12'].y1 - T['12'].y0, 'centrals longest');
});
function trauma(teeth, statusOver, opts = {}) {
  const f = face(); const r = cropAndMasks(f, teeth);
  const mm = S.irisMmPerPx(f.P) * r.cr.scale;
  let cur = r.cur, status;
  if (opts.edentulous) { cur = S.designFromFace(r.lips, r.w / 2, mm); status = Object.fromEntries(C.TEETH.map((k) => [k, 'missing'])); }
  else { status = Object.fromEntries(C.TEETH.map((k) => [k, statusOver[k] || 'present'])); cur = C.fillMissing(Object.fromEntries(Object.entries(cur).filter(([k]) => status[k] !== 'missing')), r.w, r.h).teeth; }
  const masks = {}; for (const k of C.TEETH) if (r.masks[k] && status[k] !== 'missing') masks[k] = r.masks[k];
  const mStatus = Object.fromEntries(C.TEETH.map((k) => [k, status[k] === 'present' ? 'present' : status[k] === 'broken' ? 'broken' : 'missing']));
  const meas = C.measure(cur, mm, mStatus), d = C.proposeDesign(cur, meas, C.DEFAULT_RULES, { proportion: 'red70' });
  const target = S.limitChanges(cur, S.applySmileArc(d.target, (x) => S.chainY(r.lips.lower, x), mm), mm, {}, status);
  const treat = Object.fromEntries(C.TEETH.map((k) => [k, status[k] === 'missing' ? 'implant' : status[k] === 'broken' ? 'crown' : 'veneer']));
  const out = Sim.simulate({ crop: r.crop, mouthMask: r.mouth, toothMasks: masks, current: cur, target, treat, status, shape: 'oval', shadeHex: '#F0E9D8', whiten: 1, library: fakeLibrary() });
  return { ...r, out, target, status };
}
ok('simulation fills lost teeth (mirror tooth / library) and rebuilds a broken tooth in full', () => {
  const f = face(); const t = synth.defaultTeeth(f); delete t['11']; delete t['12'];      // two lost teeth
  t['21'].y1 = t['21'].y0 + (t['21'].y1 - t['21'].y0) * 0.5;                              // and a fractured 21
  const r = trauma(t, { '11': 'missing', '12': 'missing', '21': 'broken' });
  for (const k of ['11', '12', '21']) { const L = meanL(r.out.img, r.target[k], r.w); assert(L > 75, `${k} drawn as tooth, L*=${L.toFixed(1)}`); }
  const tb = r.target['21']; assert(tb.y1 - tb.y0 > 0.9 * (r.target['11'].y1 - r.target['11'].y0), 'broken 21 rebuilt to full length');
  let changedOutside = 0; for (let k = 0; k < r.w * r.h; k++) if (r.out.alpha[k] === 0 && r.out.img.data[k * 4] !== r.crop.data[k * 4]) changedOutside++;
  assert.strictEqual(changedOutside, 0, 'lips untouched');
  module.exports.trauma1 = r;
});
ok('simulation of a mouth with no front teeth at all', () => {
  const r = trauma({}, {}, { edentulous: true });
  for (const k of ['11', '21', '12', '22']) assert(meanL(r.out.img, r.target[k], r.w) > 75, k + ' drawn');
  module.exports.trauma2 = r;
});
ok('appearance check: a dark painted-over gap is taken out, dim photos and yellow teeth are not flagged', () => {
  const Sz = 120, lab = new Uint8Array(Sz * Sz), rgba = new Uint8Array(Sz * Sz * 4);
  const paint = (yellow, gap, dim) => { for (let y = 0; y < Sz; y++) for (let x = 0; x < Sz; x++) { const c = 1 + Math.floor(x / 20); lab[y * Sz + x] = y > 20 && y < 90 ? c : 0;
    const i = (y * Sz + x) * 4, k = dim ? 0.5 : 1; let v = [235, 230, 220]; if (yellow) v = [220, 200, 140]; if (c === 5 && gap) v = [40, 25, 25];
    rgba[i] = v[0] * (c === 5 && gap ? 1 : k); rgba[i + 1] = v[1] * (c === 5 && gap ? 1 : k); rgba[i + 2] = v[2] * (c === 5 && gap ? 1 : k); rgba[i + 3] = 255; } };
  paint(false, false, false); assert.deepStrictEqual(S.appearanceCheck(lab, Sz, rgba, Sz, Sz).verdict, {});
  paint(true, false, true); assert.deepStrictEqual(S.appearanceCheck(lab, Sz, rgba, Sz, Sz).verdict, {});
  paint(false, true, false); assert.deepStrictEqual(S.appearanceCheck(lab, Sz, rgba, Sz, Sz).verdict, { '22': 'missing' });
});

ok('numbering safety net: a tooth-sized gap or a split tooth asks for confirmation, a normal arch does not', () => {
  const st = Object.fromEntries(C.TEETH.map((k) => [k, 'present']));
  const T = synth.defaultTeeth(face());
  assert.deepStrictEqual(S.gapCheck(T, st), [], 'normal arch');
  const T2 = JSON.parse(JSON.stringify(T)); const w12 = T2['12'].x1 - T2['12'].x0;
  T2['12'] = { ...T2['13'] }; T2['13'] = { ...T2['13'], x0: T2['13'].x0 - w12, x1: T2['13'].x1 - w12 };   // 12 lost, canine renumbered 12
  assert.ok(S.gapCheck(T2, st).includes('12'), 'gap between "12" and 11');
  const T3 = JSON.parse(JSON.stringify(T)); T3['12'] = { ...T3['13'], x0: T3['13'].x0 + 4 };          // canine split into 13 and "12"
  const r3 = S.gapCheck(T3, st); assert.ok(r3.includes('12') && r3.includes('13'), 'split tooth');
});

console.log(`\n${pass} passed (with trauma cases)`);
