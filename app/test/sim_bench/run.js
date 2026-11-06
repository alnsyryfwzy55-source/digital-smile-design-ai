// step 2 of 3: runs the app's simulate() on each case, full photo and gap photo
const fs = require('fs'), path = require('path'), SIM = require(path.join(__dirname, '..', '..', 'src', 'sim.js'));
process.chdir(__dirname);
const TEETH = ['13', '12', '11', '21', '22', '23'];
const cases = JSON.parse(fs.readFileSync('cases.json'));
fs.mkdirSync('out', { recursive: true });
const t0 = Date.now();
for (const c of cases) {
  const mask = new Uint8Array(fs.readFileSync(`in/${c.k}_mask.u8`)), N = c.w * c.h;
  const tm = (keepMissing) => Object.fromEntries(TEETH.filter((t) => !(keepMissing && t === c.t)).map((t, i) => { const m = new Uint8Array(N), cls = 1 + TEETH.indexOf(t); for (let j = 0; j < N; j++) m[j] = mask[j] === cls ? 1 : 0; return [t, m]; }));
  for (const tag of ['full', 'gap']) {
    const crop = { w: c.w, h: c.h, data: new Uint8ClampedArray(fs.readFileSync(`in/${c.k}_${tag}.rgba`)) };
    const status = Object.fromEntries(TEETH.map((t) => [t, tag === 'gap' && t === c.t ? 'missing' : 'present']));
    const r = SIM.simulate({ crop, mouthMask: new Uint8Array(N).fill(1), toothMasks: tm(tag === 'gap'), current: c.boxes, target: c.boxes,
      treat: Object.fromEntries(TEETH.map((t) => [t, 'veneer'])), status, shape: 'square', shadeHex: '#F0E9D8', whiten: 1, library: null });
    fs.writeFileSync(`out/${c.k}_${tag}.rgba`, Buffer.from(r.img.data.buffer));
  }
}
console.log(cases.length, 'cases simulated in', ((Date.now() - t0) / 1000).toFixed(1), 's');
