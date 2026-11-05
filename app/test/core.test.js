const C = require('../src/core.js');
const assert = require('assert');
let pass = 0;
const ok = (name, fn) => { try { fn(); pass++; console.log('✓', name); } catch (e) { console.log('✗', name, '\n  ', e.message); process.exitCode = 1; } };
const near = (a, b, eps, msg) => assert(Math.abs(a - b) <= eps, `${msg}: ${a} vs ${b}`);

// Build an ideal synthetic arch (px): central 90 wide, wl 0.8, RED 70%
function idealTeeth(mid = 500, c = 90, p = 0.7, y0 = 100) {
  const L = c / 0.8, w = [c, c * p, c * p * p], len = [L, L * 0.85, L * 0.95];
  const b = {}; let x = mid;
  ['11', '12', '13'].forEach((t, i) => { b[t] = { x0: x - w[i], x1: x, y0: y0 + (i === 1 ? 7 : 0), y1: y0 + (i === 1 ? 7 : 0) + len[i] }; x -= w[i]; });
  x = mid; ['21', '22', '23'].forEach((t, i) => { b[t] = { x0: x, x1: x + w[i], y0: y0 + (i === 1 ? 7 : 0), y1: y0 + (i === 1 ? 7 : 0) + len[i] }; x += w[i]; });
  return b;
}

ok('fitTilt recovers known slope', () => {
  const a = 8 * Math.PI / 180;
  const pts = [0, 1, 2, 3, 4, 5].map((i) => ({ x: i * 50, y: 10 + Math.tan(a) * i * 50 }));
  near(C.fitTilt(pts), 8, 1e-6, 'tilt');
});

ok('measure ideal arch ratios', () => {
  const m = C.measure(idealTeeth(), 0.1);
  near(m.ratios.central_wl, 0.8, 1e-9, 'wl');
  near(m.ratios.lat_central_w, 0.7, 1e-9, 'lat/central');
  near(m.ratios.can_lat_w, 0.7, 1e-9, 'can/lat');
  near(m.ratios.lat_central_len, 0.85, 1e-9, 'len');
  near(m.teeth['11'].wmm, 9.0, 1e-9, 'mm');
  assert.strictEqual(m.symmetry.central.dw, 0);
});

ok('rules all ok on ideal arch (calibrated)', () => {
  const T = idealTeeth();
  // make calibrated values within mm ranges: central 9mm wide, 11.25 long → outside 10-11.5? 11.25 ok
  const m = C.measure(T, 0.1);
  const r = C.evaluateRules(m, C.DEFAULT_RULES, { proportionP: 0.7, teeth: T });
  const bad = r.filter((x) => x.group !== 'face' && x.status !== 'ok');
  assert.deepStrictEqual(bad.map((x) => x.id + ':' + x.status + ':' + x.valueText), []);
  assert(r.filter((x) => x.group === 'face').every((x) => x.status === 'na'));
});

ok('rules uncalibrated → mm rules na', () => {
  const m = C.measure(idealTeeth(), null);
  const r = C.evaluateRules(m, C.DEFAULT_RULES, { proportionP: 0.7, teeth: idealTeeth() });
  assert(r.find((x) => x.id === 'central_len_mm').status === 'na');
  assert(r.find((x) => x.id === 'central_wl').status === 'ok');
});

ok('rules detect short wide central', () => {
  const T = idealTeeth(); T['11'].y1 = T['11'].y0 + 90; T['21'].y1 = T['21'].y0 + 90; // wl=1.0
  const r = C.evaluateRules(C.measure(T, null), C.DEFAULT_RULES, { proportionP: 0.7, teeth: T });
  const s = r.find((x) => x.id === 'central_wl');
  assert(s.status === 'fail' || s.status === 'warn', s.status);
});

ok('proposeDesign on ideal arch ≈ no change', () => {
  const T = idealTeeth();
  const d = C.proposeDesign(T, C.measure(T, 0.1), C.DEFAULT_RULES, { proportion: 'red70' });
  assert(d.ok);
  for (const t of C.TEETH) {
    near(d.changes[t].dW, 0, 1e-6, 'dW ' + t);
    near(d.changes[t].dL, 0, 1e-6, 'dL ' + t);
  }
});

ok('proposeDesign preserve-space fills span exactly', () => {
  const T = idealTeeth(); T['12'].x0 += 10; T['13'].x1 += 10; // distort
  const d = C.proposeDesign(T, C.measure(T, null), C.DEFAULT_RULES, { proportion: 'golden' });
  near(d.target['13'].x0, T['13'].x0, 1e-6, 'right edge');
  near(d.target['23'].x1, T['23'].x1, 1e-6, 'left edge');
  near((d.target['12'].x1 - d.target['12'].x0) / (d.target['11'].x1 - d.target['11'].x0), 0.618, 1e-9, 'golden');
});

ok('proposeDesign clamps central length to mm range', () => {
  const T = idealTeeth(500, 120); // central 12mm wide at 0.1 → L=15mm > 11.5
  const d = C.proposeDesign(T, C.measure(T, 0.1), C.DEFAULT_RULES, { spaceMode: 'central' });
  near(d.changes['11'].hmm, 11.5, 1e-6, 'clamped');
});

ok('postprocess: softmax, TTA flip, labelsToTeeth from synthetic mask', () => {
  const S = 64, Cn = 7, n = S * S;
  const lab = new Uint8Array(n);
  // six vertical bands rows 20..44
  const bands = [[4, 12], [12, 20], [20, 31], [31, 42], [42, 50], [50, 58]];
  bands.forEach(([a, b], i) => { for (let y = 20; y < 44; y++) for (let x = a; x < b; x++) lab[y * S + x] = i + 1; });
  lab[2 * S + 2] = 3; // stray pixel of class 3 → removed by largest component
  const logits = new Float32Array(Cn * n);
  for (let i = 0; i < n; i++) logits[lab[i] * n + i] = 6;
  // flipped image prediction: flip mask horizontally and swap classes
  const lf = new Float32Array(Cn * n);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) { const c = lab[y * S + x]; lf[C.FLIP_CHANNELS[c] * n + y * S + (S - 1 - x)] = 6; }
  const p = C.mergeFlipTTA(C.softmaxNCHW(logits, Cn, S), C.softmaxNCHW(lf, Cn, S), Cn, S);
  const { labels, conf } = C.argmaxNCHW(p, Cn, S);
  for (let i = 0; i < n; i++) if (i !== 2 * S + 2) assert.strictEqual(labels[i], lab[i], 'tta argmax ' + i);
  const clean = C.largestComponents(labels, S, Cn);
  assert.strictEqual(clean[2 * S + 2], 0);
  const det = C.labelsToTeeth(clean, conf, S, 640, 320, { minAreaFrac: 0.001 });
  assert.strictEqual(det.presentCount, 6); assert(det.orderOk); assert(det.meanConf > 0.9);
  near(det.teeth['11'].x0, 20 * 10 + 5, 11, 'x0 scale'); near(det.teeth['11'].y0, 20 * 5 + 2.5, 6, 'y0 scale');
  assert.strictEqual(C.qualityReport(det).level, 'ok');
});

ok('face analysis: midline deviation, cant, arc in tilted photo', () => {
  const a = 5 * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a);
  const R = (x, y) => ({ x: 500 + (x - 500) * cs - (y - 500) * sn, y: 500 + (x - 500) * sn + (y - 500) * cs });
  const pts = {
    pupilR: R(400, 300), pupilL: R(600, 300), glabella: R(500, 280), subnasale: R(500, 480), menton: R(500, 680),
    commR: R(420, 560), commL: R(580, 560), upperLip: R(503, 530), zenith11: R(503, 532), incisal11: R(503, 562),
    midlineDental: R(503, 560), cusp13: R(450, 555), cusp23: R(556, 555), lastR: R(432, 550), lastL: R(568, 550),
    lowerLipMid: R(503, 572), lowerLipR: R(450, 566), lowerLipL: R(556, 566)
  };
  const f = C.faceAnalysis(pts, { centralLenMm: 10 }); // 30px = 10mm
  near(f.metrics.ipl_angle, 5, 1e-6, 'ipl');
  near(f.metrics.commissure_cant, 0, 1e-6, 'cant');
  near(f.metrics.midline_dev, 1.0, 1e-6, 'midline mm');
  assert.strictEqual(f.metrics.smile_arc, 'consonant');
  near(f.metrics.lower_third, 1.0, 1e-9, 'thirds');
  near(f.metrics.buccal_corridor, 136 / 160, 1e-9, 'corridor');
});

ok('toothPath returns closed path', () => {
  for (const s of ['square', 'oval', 'triangular']) { const p = C.toothPath(s, { x0: 0, x1: 50, y0: 0, y1: 60 }); assert(p.startsWith('M') && p.endsWith('Z')); }
  assert(C.toothPath('oval', { x0: 0, x1: 50, y0: 0, y1: 60 }, true).endsWith('Z'));
});

ok('fillMissing: single gap between neighbours', () => {
  const T = idealTeeth(); const found = { ...T }; delete found['12'];
  const r = C.fillMissing(found, 1000, 600);
  assert.deepStrictEqual(r.filled, ['12']);
  near(r.teeth['12'].x0, T['13'].x1, 1e-6, 'x0'); near(r.teeth['12'].x1, T['11'].x0, 1e-6, 'x1');
});
ok('fillMissing: both centrals missing share gap 1:1', () => {
  const T = idealTeeth(); const found = { ...T }; delete found['11']; delete found['21'];
  const r = C.fillMissing(found, 1000, 600);
  near(r.teeth['11'].x1 - r.teeth['11'].x0, r.teeth['21'].x1 - r.teeth['21'].x0, 1e-6, 'equal');
  near(r.teeth['11'].x1, 500, 1e-6, 'midline');
});
ok('fillMissing: end tooth missing extends outward', () => {
  const T = idealTeeth(); const found = { ...T }; delete found['23'];
  const r = C.fillMissing(found, 1000, 600);
  near(r.teeth['23'].x0, T['22'].x1, 1e-6, 'adjacent');
  near(r.teeth['23'].x1 - r.teeth['23'].x0, 90 * 0.49, 1e-6, 'width');
});
ok('measure: missing excluded, broken length ignored', () => {
  const T = idealTeeth(); T['21'].y1 = T['21'].y0 + 40; // fractured
  const m = C.measure(T, 0.1, { '12': 'missing', '21': 'broken' });
  assert(!m.teeth['12']); assert.strictEqual(m.teeth['21'].wl, null);
  near(m.ratios.central_wl, 0.8, 1e-9, 'wl from 11 only');
  near(m.ratios.lat_central_w, 0.7, 1e-9, 'lat from 22');
  const r = C.evaluateRules(m, C.DEFAULT_RULES, { proportionP: 0.7, teeth: T });
  assert.strictEqual(r.find((x) => x.id === 'sym_lateral').status, 'na');
  assert.strictEqual(r.find((x) => x.id === 'central_len_mm').status, 'ok');
  assert.strictEqual(r.find((x) => x.id === 'incisal_step').status, 'ok');
});
ok('proposeDesign works with both centrals missing (central mode)', () => {
  const T = idealTeeth(); const st = { '11': 'missing', '21': 'missing' };
  const d = C.proposeDesign(T, C.measure(T, 0.1, st), C.DEFAULT_RULES, { spaceMode: 'central' });
  assert(d.ok); near(d.target['11'].x1 - d.target['11'].x0, 90, 1e-6, 'from boxes');
});
ok('restAnalysis: display in tilted photo', () => {
  const a = 4 * Math.PI / 180, cs = Math.cos(a), sn = Math.sin(a);
  const R = (x, y) => ({ x: 500 + (x - 500) * cs - (y - 500) * sn, y: 500 + (x - 500) * sn + (y - 500) * cs });
  const r = C.restAnalysis({ pupilR: R(400, 300), pupilL: R(526, 300), lipRest: R(463, 600), incisal11: R(460, 604) }, { ipdMm: 63 });
  near(r.display, 2.0, 1e-6, 'display'); assert(r.approx);
  assert.strictEqual(C.restAnalysis({ pupilR: R(400, 300), pupilL: R(526, 300), lipRest: R(463, 600) }, { hidden: true }).display, 0);
});
ok('safetyChecks blocks veneer on missing tooth and unreviewed teeth', () => {
  const T = idealTeeth();
  const x = C.safetyChecks({ teeth: T, status: { '12': 'missing' }, treat: { '12': 'veneer' }, reviewed: false, meas: C.measure(T, 0.1), consent: true });
  assert(x.some((i) => i.level === 'block' && i.text.includes('12')));
  assert(x.some((i) => i.level === 'block' && i.text.includes('مراجعة')));
  const y = C.safetyChecks({ teeth: T, status: {}, treat: {}, reviewed: true, meas: C.measure(T, 0.1), consent: true });
  assert(!y.some((i) => i.level === 'block'));
});

console.log(`\n${pass} passed`);
