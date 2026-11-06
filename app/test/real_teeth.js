// Runs the smile pipeline (upper-teeth selection, absent/broken classification, design, morph) on REAL frontal tooth photos + REAL model output.
const fs = require('fs'), C = require('../src/core.js'), S = require('../src/smile.js'), Sim = require('../src/sim.js');
const D = process.env.REAL_TEETH_DIR || '/tmp/realteeth/', idx = JSON.parse(fs.readFileSync(D + 'index.json'));
function npy(f) { const b = fs.readFileSync(f), hl = b.readUInt16LE(8), hdr = b.slice(10, 10 + hl).toString(); const sh = hdr.match(/\((.*?)\)/)[1].split(',').filter(Boolean).map(Number); const data = b.slice(10 + hl); const t = hdr.includes('<f4') ? new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length)) : new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length)); return { sh, t }; }
const stats = { n: 0, six: 0, orderOk: 0, broken: {}, missing: {}, unsure: {}, cut: 0, simMs: [] }, samples = [];
for (const it of idx) {
  const lab = npy(D + it.name + '_lab.npy').t, conf = npy(D + it.name + '_conf.npy').t, rgba = npy(D + it.name + '_rgba.npy');
  const w = it.w, h = it.h, img = { w, h, data: new Uint8ClampedArray(rgba.t.buffer) };
  const m = 3, poly = [{ x: m, y: m }, { x: w - m, y: m }, { x: w - m, y: h - m }, { x: m, y: h - m }];
  const lips = { upper: [{ x: 0, y: 0 }, { x: w / 2, y: 0 }, { x: w, y: 0 }], lower: [{ x: 0, y: h }, { x: w / 2, y: h }, { x: w, y: h }], poly };
  const sel = S.selectUpperTeeth(lab, conf, 384, w, h, lips, { rgba: img.data });
  const absent = S.classifyAbsent(sel, img.data, w, h, lips);
  const status = Object.fromEntries(C.TEETH.map((t) => [t, sel.teeth[t] ? 'present' : absent[t]]));
  const teeth = C.fillMissing(sel.teeth, w, h).teeth;
  for (const t of S.detectBroken(teeth, status)) status[t] = 'broken';
  stats.n++; if (sel.presentCount === 6) stats.six++; if (sel.orderOk) stats.orderOk++;
  for (const t of C.TEETH) { if (status[t] === 'broken') stats.broken[t] = (stats.broken[t] || 0) + 1; if (status[t] === 'missing') stats.missing[t] = (stats.missing[t] || 0) + 1; if (status[t] === 'unsure') stats.unsure[t] = (stats.unsure[t] || 0) + 1; if (sel.teeth[t]?.cut) stats.cut++; }
  if (samples.length < 4 && it.name.startsWith('fdi_test')) {
    const masks = {}; for (const t of C.TEETH) { if (status[t] !== 'present' && status[t] !== 'broken') continue; const c = C.CLASS_OF[t], mk = new Uint8Array(w * h); let any = 0; for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (sel.clean[Math.min(383, Math.floor((y + 0.5) * 384 / h)) * 384 + Math.min(383, Math.floor((x + 0.5) * 384 / w))] === c) { mk[y * w + x] = 1; any = 1; } if (any) masks[t] = mk; }
    const mm = 8.6 / ((teeth['11'].x1 - teeth['11'].x0 + teeth['21'].x1 - teeth['21'].x0) / 2);     // assume average central width (no face in these photos)
    const mSt = Object.fromEntries(C.TEETH.map((t) => [t, status[t] === 'present' ? 'present' : status[t] === 'broken' ? 'broken' : 'missing']));
    const meas = C.measure(teeth, mm, mSt), d = C.proposeDesign(teeth, meas, C.DEFAULT_RULES, { proportion: 'red70' });
    const target = S.limitChanges(teeth, d.target, mm, {}, status);
    const treat = Object.fromEntries(C.TEETH.map((t) => [t, status[t] === 'missing' ? 'implant' : status[t] === 'broken' ? 'crown' : 'veneer']));
    const t0 = Date.now();
    const r = Sim.simulate({ crop: img, mouthMask: S.rasterPoly(poly, w, h), toothMasks: masks, current: teeth, target, treat, status, shape: 'oval', shadeHex: '#F0E9D8', whiten: 1 });
    stats.simMs.push(Date.now() - t0);
    samples.push({ name: it.name, w, h, before: Buffer.from(img.data.buffer), after: Buffer.from(r.img.data.buffer), status });
  }
}
fs.writeFileSync(D + 'samples.json', JSON.stringify(samples.map((s) => ({ name: s.name, w: s.w, h: s.h, status: s.status }))));
samples.forEach((s, i) => { fs.writeFileSync(D + `s${i}_before.raw`, s.before); fs.writeFileSync(D + `s${i}_after.raw`, s.after); });
console.log(JSON.stringify(stats));
