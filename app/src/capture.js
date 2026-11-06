/* Guided capture: live face checks, instructions, automatic shutter. Then automatic analysis. */
'use strict';
const CHECK_LABELS = { distance: 'المسافة', center: 'التوسيط', roll: 'استقامة الرأس', frontal: 'النظر للأمام', smile: 'الابتسامة', light: 'الإضاءة', sharp: 'الوضوح', still: 'الثبات' };
const HOLD_MS = 700;

async function viewCapture(root, caseId) {
  const c = await DB.get('cases', caseId); if (!c) return go('#/');
  await withReady(() => openCamera(root, c));
}

async function openCamera(root, c) {
  let facing = 'environment', stream = null, running = true, goodSince = 0, lastRun = 0, busy = false;
  const hist = [];
  const video = h('video', { playsinline: true, muted: true, autoplay: true });
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('class', 'guide');
  const msg = h('div', { class: 'msg' }, 'جارٍ تشغيل الكاميرا…');
  const checks = h('div', { class: 'checks' });
  const ring = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); ring.setAttribute('viewBox', '0 0 100 100');
  ring.innerHTML = '<circle cx="50" cy="50" r="46" fill="none" stroke="rgba(255,255,255,.25)" stroke-width="6"/><circle class="arc" cx="50" cy="50" r="46" fill="none" stroke="#3db5a2" stroke-width="6" stroke-dasharray="289" stroke-dashoffset="289" stroke-linecap="round"/>';
  const shutter = h('button', { class: 'shutter', 'aria-label': 'التقاط يدوي', onclick: () => capture(true) }, ring);
  const flash = h('div', { class: 'flash' });
  const close = () => { stop(); go('#/case/' + c.id); };
  const el = h('div', { class: 'cam' }, video, svg,
    h('div', { class: 'top' }, h('button', { class: 'iconbtn', 'aria-label': 'إغلاق', onclick: close }, '✕'), h('div', { class: 'grow' }),
      h('button', { class: 'iconbtn', 'aria-label': 'من المعرض', onclick: () => { stop(); pickFromGallery(c.id); } }, '🖼'),
      h('button', { class: 'iconbtn', 'aria-label': 'تبديل الكاميرا', onclick: () => { facing = facing === 'environment' ? 'user' : 'environment'; startStream(); } }, '🔄')),
    msg, h('div', { class: 'bottom' }, checks, shutter), flash);
  root.append(el);
  cleanup = stop;

  async function startStream() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1440 } } });
    } catch (e) {
      msg.textContent = 'لا يمكن فتح الكاميرا: اسمح للتطبيق باستخدامها من إعدادات المتصفح، أو اختر صورة من المعرض.'; return;
    }
    video.srcObject = stream; await video.play().catch(() => {});
    video.classList.toggle('mirror', facing === 'user'); svg.classList.toggle('mirror', facing === 'user');
  }
  function stop() { running = false; if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; }

  // small canvases for per-frame pixel statistics (mouth region + face brightness) — keeps each frame cheap
  const mc = canvasOf(256, 160), mx = mc.getContext('2d', { willReadFrequently: true });
  const fc = canvasOf(48, 48), fx = fc.getContext('2d', { willReadFrequently: true });
  function frameStats(P) {
    const inner = SM.innerLipPolygon(P);
    let x0 = Math.min(...inner.map((p) => p.x)), x1 = Math.max(...inner.map((p) => p.x)), y0 = Math.min(...inner.map((p) => p.y)), y1 = Math.max(...inner.map((p) => p.y));
    const pad = (x1 - x0) * 0.08; x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
    const k = 256 / Math.max(1, x1 - x0), ch = Math.max(8, Math.min(256, Math.round((y1 - y0) * k)));
    if (mc.height !== ch) mc.height = ch;
    mx.drawImage(video, x0, y0, x1 - x0, y1 - y0, 0, 0, 256, ch);
    const md = mx.getImageData(0, 0, 256, ch).data;
    const [eR, eL] = SM.eyes(P), m = P[SM.LM.upperInner];
    fx.drawImage(video, Math.min(eR.x, eL.x), Math.min(eR.y, eL.y), Math.abs(eL.x - eR.x) || 1, Math.max(1, m.y - Math.min(eR.y, eL.y)), 0, 0, 48, 48);
    const fd = fx.getImageData(0, 0, 48, 48).data; let fs = 0; for (let i = 0; i < fd.length; i += 4) fs += 0.299 * fd[i] + 0.587 * fd[i + 1] + 0.114 * fd[i + 2];
    const Pm = P.map((p) => ({ x: (p.x - x0) * k, y: (p.y - y0) * k }));
    return SM.imageStats(md, 256, ch, Pm, { faceL: fs / (48 * 48) });
  }

  function drawGuide(P, vw, vh, ok) {
    // map video pixels to the cover-fitted screen
    const sw = el.clientWidth, sh = el.clientHeight, k = Math.max(sw / vw, sh / vh), ox = (sw - vw * k) / 2, oy = (sh - vh * k) / 2;
    svg.setAttribute('viewBox', `0 0 ${sw} ${sh}`);
    const col = ok ? '#3db5a2' : 'rgba(255,255,255,.75)';
    let out = `<ellipse cx="${sw / 2}" cy="${sh * 0.46}" rx="${sw * 0.33}" ry="${sh * 0.3}" fill="none" stroke="rgba(255,255,255,.35)" stroke-width="2" stroke-dasharray="8 8"/>`;
    if (P) {
      const pt = (p) => `${(ox + p.x * k).toFixed(1)},${(oy + p.y * k).toFixed(1)}`;
      out += `<polygon points="${SM.innerLipPolygon(P).map(pt).join(' ')}" fill="none" stroke="${col}" stroke-width="2.5"/>`;
      const [a, b] = SM.eyes(P); out += `<line x1="${pt(a).split(',')[0]}" y1="${pt(a).split(',')[1]}" x2="${pt(b).split(',')[0]}" y2="${pt(b).split(',')[1]}" stroke="${col}" stroke-width="2"/>`;
    }
    svg.innerHTML = out;
  }
  function setChecks(items) {
    checks.replaceChildren(...items.map((i) => h('span', { class: i.ok ? 'ok' : '' }, (i.ok ? '✓ ' : '') + (CHECK_LABELS[i.id] || i.label))));
  }
  function setRing(f) { ring.querySelector('.arc').setAttribute('stroke-dashoffset', String(289 * (1 - Math.max(0, Math.min(1, f))))); }

  async function capture(manual) {
    if (busy || !video.videoWidth) return; busy = true;
    const cv = canvasOf(video.videoWidth, video.videoHeight); cv.getContext('2d').drawImage(video, 0, 0);
    flash.style.opacity = '0.8'; setTimeout(() => { flash.style.opacity = '0'; }, 120);
    const blob = await toBlob(cv, 0.93);
    stop();
    startAnalysis(c.id, blob, { manual });
  }

  async function loop(now) {
    if (!running) return;
    if (video.readyState >= 2 && now - lastRun > 90 && !busy) {
      lastRun = now;
      try {
        const faces = await Engine.detectVideo(video, now);
        const vw = video.videoWidth, vh = video.videoHeight;
        if (faces.length !== 1) {
          goodSince = 0; setRing(0); drawGuide(null, vw, vh, false); hist.length = 0;
          msg.className = 'msg'; msg.textContent = faces.length ? 'أكثر من وجه في الصورة' : 'وجّه الكاميرا إلى وجه المريض';
          setChecks(Object.keys(CHECK_LABELS).map((id) => ({ id, ok: false })));
        } else {
          const P = SM.toPx(faces[0], vw, vh);
          const q = SM.qualityChecks(P, vw, vh, frameStats(P));
          // stability: landmarks barely moving over the last frames
          const fh = Math.hypot(P[10].x - P[152].x, P[10].y - P[152].y) || 1, key = [P[13], P[468] || P[33], P[473] || P[263]];
          hist.push(key); if (hist.length > 6) hist.shift();
          const move = hist.length < 4 ? 1 : Math.max(...hist.slice(1).map((k2, i) => Math.max(...k2.map((p, j) => Math.hypot(p.x - hist[i][j].x, p.y - hist[i][j].y))))) / fh;
          const still = move < 0.012;
          const items = [...q.items, { id: 'still', ok: still, label: 'الثبات', hint: 'ثبّت الجوال لحظة' }];
          const bad = items.find((i) => !i.ok);
          drawGuide(P, vw, vh, !bad); setChecks(items);
          if (!bad) {
            goodSince = goodSince || now; const f = (now - goodSince) / HOLD_MS; setRing(f);
            msg.className = 'msg good'; msg.textContent = 'ممتاز، ثبّت…';
            if (f >= 1) { capture(false); return; }
          } else { goodSince = 0; setRing(0); msg.className = 'msg'; msg.textContent = bad.hint || bad.label; }
        }
      } catch (e) { console.warn(e); }
    }
    requestAnimationFrame(loop);
  }
  await startStream();
  requestAnimationFrame(loop);
}

// ---------- automatic analysis ----------
const STEP_LABELS = [['face', 'تحديد الوجه واتجاه الصورة'], ['mouth', 'تحديد الفم والشفاه'], ['teeth', 'تحديد الأسنان العلوية'], ['design', 'القياسات والتصميم'], ['sim', 'محاكاة الابتسامة الجديدة']];
async function startAnalysis(caseId, blob, opts = {}) {
  go('#/case/' + caseId + '/analyzing');
  await new Promise((r) => setTimeout(r, 0));
  const root = $('#app'); root.innerHTML = '';
  const steps = h('ul', { class: 'steps' }, STEP_LABELS.map(([k, l]) => h('li', { 'data-k': k }, h('span', { class: 's' }), l)));
  const preview = h('canvas');
  root.append(appbar('جارٍ التحليل'), h('main', null, h('div', { class: 'mediawrap' }, h('div', { class: 'scan' }, preview, h('div', { class: 'line' }))), h('div', { class: 'card' }, steps)));
  try { const pc = await blobToCanvas(blob); preview.width = pc.width; preview.height = pc.height; preview.getContext('2d').drawImage(pc, 0, 0); } catch (e) { /* ignore */ }
  const mark = (k) => { let seen = false; for (const li of steps.children) { if (li.dataset.k === k) { li.className = 'on'; seen = true; } else if (!seen) { li.className = 'done'; li.querySelector('.s').textContent = '✓'; } } };
  await withReady(async () => {
    try {
      const c = await DB.get('cases', caseId);
      const { st, rotatedBlob } = await analyzePhoto(blob, S, mark);
      // store a normalised copy of the photo (same size the analysis used) so the case reopens identically
      const photoBlob = await toBlob(st.full, 0.92);
      for (const k of ['photoId', 'thumbId']) if (c[k]) await DB.del('blobs', c[k]);
      c.photoId = await putBlob(photoBlob);
      c.analysis = serializeAnalysis(st);
      c.reviewed = false; c.approval = null; c.updated = Date.now();
      await DB.put('cases', c);
      mark('sim');
      Results.pending = st;               // hand the live state to the results screen (no recomputation)
      if (!st.quality.ok && !opts.ignoreQuality) return qualityGate(caseId, st, blob);
      go('#/case/' + caseId);
    } catch (e) {
      console.error(e);
      const noFace = e.code === 'NO_FACE';
      root.querySelector('main').replaceChildren(h('div', { class: 'card' }, h('h2', null, noFace ? 'لم يُعثر على وجه' : 'تعذّر التحليل'),
        h('p', { class: 'muted' }, noFace ? 'يجب أن تكون صورة لوجه المريض كاملاً وهو يبتسم، والوجه مستقيم وواضح.' : String(e.message || e))),
        actionbar(h('button', { class: 'btn primary grow', onclick: () => go('#/capture/' + caseId) }, 'التقاط صورة جديدة'), h('button', { class: 'btn', onclick: () => pickFromGallery(caseId) }, 'من المعرض')));
    }
  });
}
function qualityGate(caseId, st, blob) {
  const root = $('#app'); root.innerHTML = '';
  const bad = st.quality.items.filter((i) => !i.ok);
  root.append(appbar('مراجعة الصورة', '#/'), h('main', null, h('div', { class: 'card' }, h('h2', null, 'الصورة قد تعطي نتيجة غير دقيقة'),
    bad.map((i) => h('div', { class: 'notice warn' }, h('b', null, i.label + ': '), i.hint)),
    h('p', { class: 'small muted' }, 'الأفضل التقاط صورة جديدة. الكاميرا ترشدك وتلتقط تلقائياً حين تكون الصورة مناسبة.'))),
  actionbar(h('button', { class: 'btn primary grow', onclick: () => go('#/capture/' + caseId) }, 'التقاط صورة جديدة'), h('button', { class: 'btn', onclick: () => go('#/case/' + caseId) }, 'متابعة رغم ذلك')));
}

function serializeAnalysis(st) {
  const g = st.geom;
  return {
    P: Float32Array.from(g.P.flatMap((p) => [p.x, p.y])), fullW: g.fullW, fullH: g.fullH, turns: g.turns, faces: g.faces,
    crop: { M: g.crop.M, Minv: g.crop.Minv, outW: g.crop.outW, outH: g.crop.outH, scale: g.crop.scale, roll: g.crop.roll },
    clean: g.clean, selTeeth: st.sel.teeth, orderOk: st.sel.orderOk, meanConf: st.sel.meanConf, info: st.sel.info,
    teeth: st.teeth, status: st.status, edentulous: st.edentulous, quality: { ok: st.quality.ok, items: st.quality.items, pose: st.quality.pose }, timings: st.timings, segMs: st.segMs, ep: st.ep, server: st.server || null, at: Date.now()
  };
}
async function restoreState(c) {
  const a = c.analysis, blob = await getBlob(c.photoId); if (!a || !blob) return null;
  const full = await blobToCanvas(blob);
  if (Math.abs(full.width - a.fullW) > 1 || Math.abs(full.height - a.fullH) > 1) throw new Error('حجم الصورة المحفوظة لا يطابق التحليل');
  const P = []; for (let i = 0; i < a.P.length; i += 2) P.push({ x: a.P[i], y: a.P[i + 1] });
  const crop = { ...a.crop };
  const geom = { P, crop, lips: lipsInCrop(P, crop), clean: a.clean, turns: a.turns, fullW: a.fullW, fullH: a.fullH, faces: a.faces };
  const fullImg = full.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, full.width, full.height);
  const st = buildState(full, fullImg, cropCanvasFrom(full, crop), geom, S, { teeth: a.teeth, selTeeth: a.selTeeth, status: a.status, orderOk: a.orderOk, meanConf: a.meanConf, info: a.info, edentulous: a.edentulous });
  if (c.toothMasksEdited) for (const [t, box] of Object.entries(c.toothMasksEdited)) st.toothMasks[t] = maskForBox(st, box);
  st.quality = a.quality; st.server = a.server || null; st.timings = a.timings; st.segMs = a.segMs; st.ep = a.ep; st.caliperMm = c.caliperMm || null; st.broken = c.broken || {};
  if (st.caliperMm) computeMeasures(st);
  return st;
}
function maskForBox(st, box) {
  const w = st.cropImg.width, h = st.cropImg.height, poly = Sim.toothPolygon('oval', box, false), m = Sim.rasterPoly(poly, w, h);
  for (let k = 0; k < m.length; k++) if (!st.mouthMask[k]) m[k] = 0;
  return m;
}
