/* Settings, encrypted backup, diagnostics, licences, startup */
'use strict';
async function viewSettings(root) {
  const main = h('main'); root.append(appbar('الإعدادات', '#/'), main);
  const txt = (k, label) => { const i = h('input', { type: 'text', id: 'set-' + k, value: S[k] || '' }); i.addEventListener('change', async () => { S[k] = i.value; await saveSettings(); }); return h('label', { class: 'field' }, h('span', null, label), i); };
  const theme = h('select', { id: 'set-theme' }, [['auto', 'حسب الجهاز'], ['light', 'فاتح'], ['dark', 'داكن']].map(([v, l]) => h('option', { value: v, selected: S.theme === v }, l)));
  theme.addEventListener('change', async () => { S.theme = theme.value; applyTheme(); await saveSettings(); });
  main.append(h('div', { class: 'card' }, h('h2', null, 'العيادة'), txt('clinicName', 'اسم العيادة'), txt('doctorName', 'اسم الطبيب'), txt('material', 'المادة الافتراضية للمعمل'), h('label', { class: 'field' }, h('span', null, 'المظهر'), theme)));

  const srvMsg = h('div', { class: 'small muted', id: 'srv-msg' });
  main.append(h('div', { class: 'card' }, h('h2', null, 'خادم التحليل (اختياري)'),
    h('p', { class: 'small muted' }, 'إن وُضع عنوان، تُرسل صورة الحالة إلى الخادم ليتحقق من تحديد الأسنان ويُخرج ملف المعمل بالملم. السن الذي يختلف فيه الخادم والجهاز يُعرض عليك للتأكيد. اتركه فارغاً ليعمل كل شيء على الجهاز.'),
    txt('serverUrl', 'العنوان (يبدأ بـ https://)'),
    h('div', { class: 'row' }, h('button', { class: 'btn', id: 'srv-test', onclick: async () => {
      srvMsg.textContent = 'جارٍ الفحص…';
      const r = await serverHealth(S.serverUrl);
      srvMsg.textContent = r.ok ? `متصل · الإصدار ${r.version}` : `غير متصل: ${r.error || 'لا يستجيب'}`;
    } }, 'فحص الاتصال')), srvMsg));

  main.append(h('div', { class: 'card' }, h('h2', null, 'الأمان والنسخ الاحتياطي'),
    h('p', { class: 'small muted' }, 'البيانات على هذا الجهاز فقط. النسخة الاحتياطية ملف مشفّر بكلمة مرور (AES-256) تحفظه خارج الجهاز.'),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', onclick: exportBackup }, 'تصدير نسخة احتياطية'),
      h('button', { class: 'btn', onclick: importBackup }, 'استيراد نسخة'),
      h('button', { class: 'btn', onclick: setPin }, S.pin ? 'تغيير رمز الدخول' : 'تفعيل رمز الدخول'))));

  const diag = h('div', { class: 'small muted' }, 'جارٍ الفحص…');
  (async () => {
    let est = null, persisted = false; try { est = await navigator.storage?.estimate?.(); persisted = await navigator.storage?.persisted?.(); } catch (e) { /* ignore */ }
    const last = (await DB.all('cases')).filter((c) => c.analysis).sort((a, b) => b.updated - a.updated)[0];
    diag.replaceChildren(h('div', null, `الإصدار ${APP_VERSION} · محرك النموذج: ${Engine.info.ep || 'لم يُحمَّل بعد'} · كاشف الوجه: ${Engine.info.face || 'لم يُحمَّل بعد'} · ملفات محلية: ${Engine.info.vendor ? 'نعم' : 'لا'}`),
      est ? h('div', null, `التخزين: ${(est.usage / 1048576).toFixed(0)} ميغابايت${persisted ? ' (دائم)' : ''}`) : null,
      last?.analysis?.timings ? h('div', null, `آخر تحليل: ${Object.entries(last.analysis.timings).map(([k, v]) => k + ' ' + v + 'ms').join(' · ')}${last.analysis.segMs ? ' · النموذج ' + last.analysis.segMs + 'ms' : ''}`) : null);
  })();
  main.append(h('div', { class: 'card' }, h('h2', null, 'التشخيص'), diag,
    h('div', { class: 'row', style: { marginTop: '10px' } }, h('button', { class: 'btn', onclick: exportDiagnostics }, 'تصدير تقرير تشخيص'),
      h('button', { class: 'btn', onclick: async () => { if (!await confirmBox('إعادة تحميل', 'حذف أدوات الذكاء الاصطناعي المخزّنة وإعادة تحميلها؟')) return; try { await caches.delete('dsd-models-v2'); } catch (e) { /* ignore */ } location.reload(); } }, 'إعادة تحميل الأدوات'))));

  const tb = h('tbody');
  for (const r of S.rules) {
    const en = h('input', { type: 'checkbox', checked: r.enabled, 'aria-label': 'تفعيل' }); en.addEventListener('change', async () => { r.enabled = en.checked; await saveSettings(); });
    const cell = (k) => { if (r[k] === undefined) return h('td', { class: 'muted' }, '—'); const i = h('input', { type: 'number', step: '0.01', value: r[k] ?? '', style: { minWidth: '68px' } }); i.addEventListener('change', async () => { r[k] = i.value === '' ? null : +i.value; await saveSettings(); }); return h('td', null, i); };
    tb.append(h('tr', null, h('td', null, en), h('td', null, r.label), cell('min'), cell('max'), cell('target')));
  }
  main.append(h('div', { class: 'card' }, h('details', { class: 'adv' }, h('summary', null, 'قواعد التصميم (للطبيب)'),
    h('p', { class: 'small muted' }, 'قيم افتراضية من أدبيات تصميم الابتسامة. عدّلها حسب معاييرك.'),
    h('div', { class: 'tablewrap' }, h('table', { class: 't' }, h('thead', null, h('tr', null, h('th', null, ''), h('th', null, 'القاعدة'), h('th', null, 'أدنى'), h('th', null, 'أعلى'), h('th', null, 'الهدف'))), tb)),
    h('button', { class: 'btn', onclick: async () => { if (await confirmBox('استعادة', 'إرجاع القواعد للقيم الافتراضية؟')) { S.rules = C.clone(C.DEFAULT_RULES); await saveSettings(); render(); } } }, 'استعادة الافتراضي'))));

  main.append(h('div', { class: 'card small muted' }, h('h2', null, 'حول النظام والمصادر'),
    h('p', null, 'أداة مساعدة للطبيب في تحليل الابتسامة وتصميمها وعرضها، وليست جهاز تشخيص. كل تصميم يعتمده الطبيب.'),
    h('p', null, 'نموذج تحديد الأسنان: U-Net + MobileNetV2، دُرّب على بيانات Roboflow Universe بترخيص CC BY 4.0: Teeth-detection-FDI-System (rizka-milandga-2k04u)، tooth_frontal (gradproject-g265b)، teeth (tooth-yqbyw). مكتبة الأسنان المدمجة من Teeth-detection-FDI-System.'),
    h('p', null, 'كاشف الوجه: Google MediaPipe Face Landmarker (Apache 2.0). محرك التشغيل: ONNX Runtime Web (MIT).'),
    h('p', null, 'المقاسات بالملم تقديرية من قطر قزحية العين (11.7 ملم في المتوسط) ما لم تُعايَر بعرض الثنية المقاس بالفرجار.')));
}

async function setPin() {
  const a = await promptBox('رمز الدخول', 'رمز جديد (4 أرقام على الأقل)، اتركه فارغاً لإلغائه', 'password'); if (a == null) return;
  if (!a) { S.pin = null; await saveSettings(); toast('أُلغي رمز الدخول'); render(); return; }
  if (a.length < 4) { toast('الرمز قصير'); return; }
  const b = await promptBox('تأكيد الرمز', 'أعد كتابة الرمز', 'password'); if (a !== b) { toast('غير متطابق'); return; }
  const salt = crypto.getRandomValues(new Uint8Array(16)); S.pin = { salt: b64.enc(salt), hash: await hashPin(a, salt) }; await saveSettings(); toast('فُعّل رمز الدخول'); render();
}

// typed arrays survive JSON via a tagged base64 form
const pack = (v) => JSON.stringify(v, (k, x) => ArrayBuffer.isView(x) ? { __ta: x.constructor.name, b: b64.enc(x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength)) } : x);
const unpack = (s) => JSON.parse(s, (k, x) => x && x.__ta ? new (self[x.__ta])(b64.dec(x.b).buffer) : x);

async function exportBackup() {
  const pw = await promptBox('كلمة مرور النسخة', 'كلمة مرور قوية (8 أحرف على الأقل). بدونها لا يمكن فتح النسخة.', 'password'); if (!pw) return;
  if (pw.length < 8) { toast('كلمة المرور قصيرة'); return; }
  if (pw !== await promptBox('تأكيد', 'أعد كتابة كلمة المرور', 'password')) { toast('غير متطابقة'); return; }
  toast('جارٍ التجهيز…');
  const blobs = []; for (const r of await DB.all('blobs')) blobs.push({ id: r.id, type: r.blob.type, b: b64.enc(await r.blob.arrayBuffer()) });
  const plain = new TextEncoder().encode(pack({ v: 2, patients: await DB.all('patients'), cases: await DB.all('cases'), blobs, settings: { ...S, pin: null }, at: Date.now() }));
  const salt = crypto.getRandomValues(new Uint8Array(16)), iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, await deriveKey(pw, salt), plain);
  const file = JSON.stringify({ format: 'dsd-backup', v: 2, iter: 250000, salt: b64.enc(salt), iv: b64.enc(iv), data: b64.enc(ct) });
  const name = `dsd-backup-${new Date().toISOString().slice(0, 10)}.dsdb`, f = new File([file], name, { type: 'application/octet-stream' });
  try { if (navigator.canShare && navigator.canShare({ files: [f] })) { await navigator.share({ files: [f], title: 'نسخة احتياطية' }); await DB.setKV('lastBackup', Date.now()); return; } } catch (e) { if (e.name === 'AbortError') return; }
  const u = URL.createObjectURL(f), a = h('a', { href: u, download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 4000);
  await DB.setKV('lastBackup', Date.now()); toast('حُفظت النسخة: انقلها إلى مكان آمن خارج الجهاز');
}
async function importBackup() {
  const inp = h('input', { type: 'file', accept: '.dsdb,application/octet-stream' });
  inp.addEventListener('change', async () => {
    const f = inp.files[0]; if (!f) return;
    let j; try { j = JSON.parse(await f.text()); if (j.format !== 'dsd-backup') throw 0; } catch (e) { toast('ملف غير صالح'); return; }
    const pw = await promptBox('كلمة مرور النسخة', 'كلمة المرور', 'password'); if (!pw) return;
    let data;
    try { data = unpack(new TextDecoder().decode(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: b64.dec(j.iv) }, await deriveKey(pw, b64.dec(j.salt), j.iter), b64.dec(j.data)))); }
    catch (e) { toast('كلمة مرور خاطئة أو ملف تالف'); return; }
    if (!await confirmBox('استيراد', `ستُضاف ${data.cases.length} حالة و${data.patients.length} مريض. السجلات بنفس المعرّف تُستبدل.`, 'استيراد')) return;
    for (const p of data.patients) await DB.put('patients', p);
    for (const c of data.cases) await DB.put('cases', c);
    for (const b of data.blobs) await DB.put('blobs', { id: b.id, blob: new Blob([b64.dec(b.b)], { type: b.type }) });
    toast('تم الاستيراد'); go('#/');
  });
  inp.click();
}
async function exportDiagnostics() {
  const cases = (await DB.all('cases')).filter((c) => c.analysis).slice(-5).map((c) => ({ quality: c.analysis.quality, timings: c.analysis.timings, segMs: c.analysis.segMs, ep: c.analysis.ep, status: c.analysis.status, meanConf: c.analysis.meanConf, orderOk: c.analysis.orderOk, reviewed: c.reviewed }));
  const rep = { app: APP_VERSION, ua: navigator.userAgent, engine: Engine.info, gpu: !!navigator.gpu, coi: self.crossOriginIsolated, cases, at: new Date().toISOString() };
  const f = new File([JSON.stringify(rep, null, 1)], 'dsd-diagnostics.json', { type: 'application/json' });
  try { if (navigator.canShare && navigator.canShare({ files: [f] })) { await navigator.share({ files: [f] }); return; } } catch (e) { if (e.name === 'AbortError') return; }
  const u = URL.createObjectURL(f), a = h('a', { href: u, download: f.name }); document.body.append(a); a.click(); a.remove();
}

(async function init() {
  try { await DB.open(); await loadSettings(); }
  catch (e) { document.body.innerHTML = '<p style="padding:20px">تعذّر فتح التخزين في هذا المتصفح. استخدم Chrome حديثاً خارج وضع التصفح الخفي.</p>'; return; }
  if ('serviceWorker' in navigator && location.protocol === 'https:' && !window.__NO_SW) navigator.serviceWorker.register('sw.js').catch(() => {});
  try { if (navigator.storage?.persist && !(await navigator.storage.persisted())) await navigator.storage.persist(); } catch (e) { /* ignore */ }
  await showLock();
  await render();
  showDisclaimer();
})();
