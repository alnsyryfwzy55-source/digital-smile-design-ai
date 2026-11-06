/* Background worker: tooth segmentation (ONNX Runtime, WebGPU → WASM) and smile morphing. Keeps the UI responsive. */
'use strict';
self.importScripts('core.js', 'smile.js', 'sim.js');
const Sim = self.DSDSim;
let session = null, ep = null, simState = null, library = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function fetchCached(url, label) {
  let cache = null;
  try { cache = await caches.open('dsd-models-v2'); const hit = await cache.match(url); if (hit) return await hit.arrayBuffer(); } catch (e) { /* no cache storage */ }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`تعذّر تحميل ${label} (${res.status})`);
  const total = +res.headers.get('content-length') || 0;
  let buf;
  if (res.body && total) {
    const reader = res.body.getReader(), chunks = []; let got = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; post({ type: 'progress', label, got, total }); }
    buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
    buf = buf.buffer;
  } else buf = await res.arrayBuffer();
  try { if (cache) await cache.put(url, new Response(buf.slice(0), { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { /* quota */ }
  return buf;
}

async function init(m) {
  self.importScripts(m.ortUrl);
  self.ort.env.wasm.wasmPaths = m.ortBase;
  self.ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, self.navigator.hardwareConcurrency || 2) : 1;
  self.ort.env.logLevel = 'error';
  const buf = await fetchCached(m.modelUrl, 'model');
  modelBuf = buf; modelSize = m.size || 384; selfTest = m.selfTest || null;
  const tries = (m.webgpu !== false && self.navigator.gpu) ? ['webgpu', 'wasm'] : ['wasm'];
  let last = null;
  for (const e of tries) {
    try { const t0 = performance.now(); await createSession(e); return { ep, warmMs: Math.round(performance.now() - t0), fallback: last ? String(last.message || last) : null, selfTest: selfTestResult }; }
    catch (err) { last = err; session = null; }
  }
  throw last || new Error('تعذّر تشغيل النموذج');
}
let modelBuf = null, modelSize = 384, selfTest = null, selfTestResult = null;
// create a session on one execution provider and prove it works with a warm-up run (compiles GPU kernels)
async function createSession(e) {
  const s = await self.ort.InferenceSession.create(modelBuf, { executionProviders: [e], graphOptimizationLevel: 'all' });
  const S = modelSize;
  await s.run({ [s.inputNames[0]]: new self.ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]) });
  // self-test on a known tooth photo: some phone GPUs return wrong numbers without any error
  if (selfTest) {
    const out = (await s.run({ [s.inputNames[0]]: new self.ort.Tensor('float32', new Float32Array(selfTest.input), [1, 3, S, S]) }))[s.outputNames[0]].data;
    const n = S * S, cnt = new Array(7).fill(0);
    for (let i = 0; i < n; i++) { let b = 0, bv = out[i]; for (let c = 1; c < 7; c++) { const v = out[c * n + i]; if (v > bv) { bv = v; b = c; } } cnt[b]++; }
    const ok = [1, 2, 3, 4, 5, 6].every((c) => Math.abs(cnt[c] - selfTest.expect[c]) <= 0.2 * selfTest.expect[c] + 50);
    selfTestResult = { ep: e, ok, counts: cnt };
    if (!ok) throw new Error('self-test failed on ' + e + ': ' + cnt.join(','));
  }
  session = s; ep = e;
}

function softmaxArgmax(lo, flipLo, S) {
  const n = S * S, C = 7, labels = new Uint8Array(n), conf = new Float32Array(n), SW = [0, 6, 5, 4, 3, 2, 1], p = new Float32Array(C);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x, j = y * S + (S - 1 - x);
    let m = -Infinity; for (let c = 0; c < C; c++) m = Math.max(m, lo[c * n + i]);
    let s = 0; for (let c = 0; c < C; c++) { p[c] = Math.exp(lo[c * n + i] - m); s += p[c]; }
    for (let c = 0; c < C; c++) p[c] /= s;
    if (flipLo) {
      let m2 = -Infinity; for (let c = 0; c < C; c++) m2 = Math.max(m2, flipLo[c * n + j]);
      let s2 = 0; const q = new Float32Array(C); for (let c = 0; c < C; c++) { q[c] = Math.exp(flipLo[c * n + j] - m2); s2 += q[c]; }
      for (let c = 0; c < C; c++) p[c] = 0.5 * (p[c] + q[SW[c]] / s2);
    }
    let b = 0; for (let c = 1; c < C; c++) if (p[c] > p[b]) b = c;
    labels[i] = b; conf[i] = p[b];
  }
  return { labels, conf };
}

async function segment(m) {
  try { return await segmentOnce(m); }
  catch (err) {
    if (ep !== 'webgpu') throw err;
    await createSession('wasm');            // GPU failed on this device: continue on the CPU engine
    return await segmentOnce(m);
  }
}
async function segmentOnce(m) {
  const S = m.size, t0 = performance.now(), name = session.inputNames[0];
  const run = async (x) => (await session.run({ [name]: new self.ort.Tensor('float32', x, [1, 3, S, S]) }))[session.outputNames[0]];
  const out = await run(m.input);
  if (out.dims[1] !== 7 || out.dims[2] !== S) throw new Error('شكل مخرج النموذج غير متوقع: ' + out.dims.join('×'));
  let flip = null;
  if (m.tta) {
    const f = new Float32Array(m.input.length);
    for (let c = 0; c < 3; c++) for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) f[c * S * S + y * S + x] = m.input[c * S * S + y * S + (S - 1 - x)];
    flip = (await run(f)).data;
  }
  const r = softmaxArgmax(out.data, flip, S);
  return { labels: r.labels, conf: r.conf, ms: Math.round(performance.now() - t0), ep };
}

function buildLibrary(items) {
  const lib = {};
  for (const [t, it] of Object.entries(items || {})) {
    const img = { w: it.w, h: it.h, data: new Uint8ClampedArray(it.data) }, mask = new Uint8Array(it.w * it.h), hole = new Uint8Array(it.w * it.h);
    for (let k = 0; k < it.w * it.h; k++) { const a = img.data[k * 4 + 3] > 128; mask[k] = a ? 1 : 0; hole[k] = a ? 0 : 1; img.data[k * 4 + 3] = 255; }
    const core = Sim.erode(mask, it.w, it.h, 2), filled = Sim.inpaint(img, hole);
    lib[t] = { img: filled, mask: core, stats: Sim.labStats(filled, core) };
  }
  return lib;
}

function simulate(m) {
  const t0 = performance.now(), s = simState;
  const r = Sim.simulate({ crop: s.crop, mouthMask: s.mouth, toothMasks: s.toothMasks, current: s.current, status: m.status || s.status,
    target: m.target, treat: m.treat, shape: m.shape, shadeHex: m.shadeHex, whiten: m.whiten, library });
  const full = { w: s.full.w, h: s.full.h, data: new Uint8ClampedArray(s.full.data) };
  Sim.pasteBack(full, r.img, r.alpha, s.Minv, s.M);
  return { full: full.data.buffer, cropImg: r.img.data.buffer, ms: Math.round(performance.now() - t0) };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') post({ type: 'ready', id: m.id, ...(await init(m)) });
    else if (m.type === 'segment') { const r = await segment(m); post({ type: 'segmented', id: m.id, ...r }, [r.labels.buffer, r.conf.buffer]); }
    else if (m.type === 'cpu') { await createSession('wasm'); post({ type: 'ok', id: m.id, ep }); }
    else if (m.type === 'library') { library = buildLibrary(m.items); post({ type: 'ok', id: m.id }); }
    else if (m.type === 'simInit') {
      const toothMasks = {}; for (const [t, b] of Object.entries(m.toothMasks)) toothMasks[t] = new Uint8Array(b);
      simState = { crop: { w: m.cropW, h: m.cropH, data: new Uint8ClampedArray(m.crop) }, full: { w: m.fullW, h: m.fullH, data: new Uint8ClampedArray(m.full) },
        mouth: new Uint8Array(m.mouth), toothMasks, current: m.current, status: m.status, M: m.M, Minv: m.Minv };
      post({ type: 'ok', id: m.id });
    } else if (m.type === 'simulate') { const r = simulate(m); post({ type: 'simulated', id: m.id, ...r }, [r.full, r.cropImg]); }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
