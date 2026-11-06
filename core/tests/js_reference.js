// Produces reference outputs from the JavaScript app for random inputs (used by test_parity.py).
// Usage: node js_reference.js <n> <seed>  -> JSON on stdout
const path = require('path');
const SRC = path.join(__dirname, '..', '..', 'app', 'src');
const C = require(path.join(SRC, 'core.js'));
global.DSDCore = C;
const S = require(path.join(SRC, 'smile.js'));

const n = +(process.argv[2] || 200);
let seed = +(process.argv[3] || 1);
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
const U = (a, b) => a + (b - a) * rnd();
const TEETH = C.TEETH;

function teethCase() {
  const c = U(60, 120), mid = U(380, 420), top = U(90, 130), p = U(0.6, 0.8);
  const w = { '11': c * U(0.9, 1.1), '21': c * U(0.9, 1.1), '12': c * p * U(0.85, 1.15), '22': c * p * U(0.85, 1.15), '13': c * p * p * U(0.8, 1.2), '23': c * p * p * U(0.8, 1.2) };
  const T = {}; let x = mid;
  for (const t of ['11', '12', '13']) { T[t] = { x0: x - w[t], x1: x - U(0, 3) }; x -= w[t]; }
  x = mid; for (const t of ['21', '22', '23']) { T[t] = { x0: x + U(0, 3), x1: x + w[t] }; x += w[t]; }
  for (const t of TEETH) { const y0 = top + U(-8, 12); T[t].y0 = y0; T[t].y1 = y0 + c / U(0.72, 0.9) * (t[1] === '1' ? 1 : t[1] === '2' ? U(0.8, 0.9) : U(0.88, 1.0)); }
  return T;
}
const STATUSES = ['present', 'present', 'present', 'present', 'broken', 'missing'];
const out = [];
for (let i = 0; i < n; i++) {
  const T = teethCase();
  const status = {}; for (const t of TEETH) status[t] = rnd() < 0.25 ? STATUSES[Math.floor(rnd() * STATUSES.length)] : 'present';
  const mmpp = rnd() < 0.2 ? null : U(0.07, 0.12);
  const meas = C.measure(T, mmpp, status);
  const pupilR = { x: U(200, 260), y: U(-500, -470) }, pupilL = { x: U(540, 600), y: U(-500, -470) };
  const pts = { pupilR, pupilL, glabella: { x: U(390, 410), y: -420 }, subnasale: { x: U(390, 410), y: -40 }, menton: { x: U(385, 415), y: 380 },
    commR: { x: U(160, 200), y: U(150, 175) }, commL: { x: U(600, 640), y: U(150, 175) }, upperLip: { x: 400, y: U(80, 120) },
    zenith11: { x: (T['11'].x0 + T['11'].x1) / 2, y: T['11'].y0 }, incisal11: { x: (T['11'].x0 + T['11'].x1) / 2, y: T['11'].y1 },
    midlineDental: { x: (T['11'].x1 + T['21'].x0) / 2, y: T['11'].y1 - 5 }, cusp13: { x: (T['13'].x0 + T['13'].x1) / 2, y: T['13'].y1 },
    cusp23: { x: (T['23'].x0 + T['23'].x1) / 2, y: T['23'].y1 }, lastR: { x: U(200, 230), y: 180 }, lastL: { x: U(570, 600), y: 180 },
    lowerLipMid: { x: 400, y: U(250, 290) }, lowerLipR: { x: 230, y: U(220, 260) }, lowerLipL: { x: 570, y: U(220, 260) } };
  const ipdMm = U(58, 68);
  const face = C.faceAnalysis(pts, { ipdMm });
  const restPts = { pupilR, pupilL, lipRest: { x: 400, y: U(100, 140) }, incisal11: { x: 400, y: U(120, 160) } };
  const rest = C.restAnalysis(restPts, { ipdMm, ipdMeasured: true });
  const proportion = ['red70', 'golden', 'red80'][Math.floor(rnd() * 3)];
  const pP = C.PROPORTIONS[proportion].p;
  const rules = C.evaluateRules(meas, C.DEFAULT_RULES, { proportionP: pP, face, rest, teeth: T });
  const edgePx = +(0.3 + rnd() * 6).toFixed(3);
  const faceEdgePx = +(0.3 + rnd() * 4).toFixed(3);
  const rulesE = C.evaluateRules(meas, C.DEFAULT_RULES, { proportionP: pP, face, rest, teeth: T, edgePx, faceEdgePx });
  const opts = { proportion, spaceMode: rnd() < 0.5 ? 'preserve' : 'ideal', symmetric: rnd() < 0.7, gingivaMode: rnd() < 0.5 ? 'keep' : 'ideal',
    midline: rnd() < 0.5 ? 'current' : 'facial', facialMidlineX: U(390, 410) };
  const design = C.proposeDesign(T, meas, C.DEFAULT_RULES, opts);
  const lower = [{ x: 150, y: U(200, 230) }, { x: 280, y: U(240, 270) }, { x: 400, y: U(255, 290) }, { x: 520, y: U(240, 270) }, { x: 650, y: U(200, 230) }];
  const upper = [{ x: 150, y: U(130, 150) }, { x: 280, y: U(95, 115) }, { x: 400, y: U(85, 105) }, { x: 520, y: U(95, 115) }, { x: 650, y: U(130, 150) }];
  const lowerY = (x) => { const s = [...lower].sort((a, b) => a.x - b.x); if (x <= s[0].x) return s[0].y; if (x >= s[s.length - 1].x) return s[s.length - 1].y; for (let k = 1; k < s.length; k++) if (x <= s[k].x) { const u = (x - s[k - 1].x) / (s[k].x - s[k - 1].x); return s[k - 1].y + u * (s[k].y - s[k - 1].y); } };
  const lengthDeltaMm = rnd() < 0.5 ? 0 : U(-1, 1.5);
  const arc = design.ok ? S.applySmileArc(design.target, lowerY, mmpp, { lengthDeltaMm }) : null;
  const limited = arc ? S.limitChanges(T, arc, mmpp, {}, status) : null;
  const fromFace = S.designFromFace({ upper, lower }, U(390, 410), mmpp, {});
  const found = {}; for (const t of TEETH) if (rnd() < 0.6) found[t] = T[t];
  const filled = C.fillMissing(found, 800, 500);
  out.push({ T, status, mmpp, meas, pts, ipdMm, face, restPts, rest, proportion, rules, edgePx, faceEdgePx, rulesE, opts, design, lower, upper, lengthDeltaMm, arc, limited, fromFace, found, filled });
}
process.stdout.write(JSON.stringify(out));
