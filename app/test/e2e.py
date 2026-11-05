import asyncio, json, os, sys, subprocess, time, pathlib
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parent
DIST = ROOT.parent.parent / 'docs'
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'shots'); OUT.mkdir(parents=True, exist_ok=True)
PORT = 8123

HOOK = r"""
window.__NO_SW = true; window.__DSD_NO_PRINT = true;
// mock MediaPipe: 478 landmarks for the synthetic face (800x1100)
window.__DSD_TEST_FACE = (cv) => {
  if (cv.width !== 800) return null;
  const L = Array.from({length: 478}, () => ({x: 0.5, y: 0.6, z: 0}));
  const set = (i, x, y) => { L[i] = {x: x/800, y: y/1100, z: 0}; };
  set(468, 310, 430); set(473, 490, 430); set(9, 400, 420); set(2, 400, 600); set(152, 400, 875);
  set(61, 292, 700); set(291, 508, 700); set(13, 400, 688); set(14, 400, 738);
  const outer = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
  outer.forEach((i, k) => { if (i===61||i===291) return; const a = Math.PI * (k/20) * 2; set(i, 400 + 108*Math.cos(a), 712 + 34*Math.sin(a)); });
  [[78,300,705],[95,320,725],[88,340,733],[178,355,735],[87,380,737],[317,420,737],[402,445,735],[318,460,733],[324,480,725],[308,500,705]].forEach(([i,x,y]) => set(i,x,y));
  return L;
};
// mock SAM: flood-fill the bright region under the tap; also return the whole bright area (higher score) to test the picker
window.__DSD_TEST_SAM = (cv, pt) => {
  const W = cv.width, H = cv.height, d = cv.getContext('2d').getImageData(0, 0, W, H).data;
  const br = (i) => d[i*4] > 200 && d[i*4+1] > 200 && d[i*4+2] > 200;
  const comp = new Uint8Array(W*H), all = new Uint8Array(W*H);
  for (let i = 0; i < W*H; i++) all[i] = br(i) ? 1 : 0;
  const st = [Math.round(pt.y)*W + Math.round(pt.x)]; if (all[st[0]]) comp[st[0]] = 1;
  while (st.length) { const k = st.pop(); if (!all[k]) continue; const x = k % W, y = (k / W) | 0;
    for (const q of [x>0?k-1:-1, x<W-1?k+1:-1, y>0?k-W:-1, y<H-1?k+W:-1]) if (q>=0 && all[q] && !comp[q]) { comp[q] = 1; st.push(q); } }
  window.__samCalls = (window.__samCalls||0) + 1;
  return [{score: 0.97, w: W, h: H, has: (i) => !!all[i]}, {score: 0.9, w: W, h: H, has: (i) => !!comp[i]}];
};

window.__DSD_TEST_INFER = (t, Sz, flip) => {
  const n = Sz*Sz, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225];
  const bright = new Uint8Array(n);
  for (let i=0;i<n;i++){ let ok=true; for(let k=0;k<3;k++){ const v=t[k*n+i]*std[k]+mean[k]; if(v<0.78) ok=false; } bright[i]=ok?1:0; }
  const lab = new Int32Array(n).fill(-1); const comps=[];
  for (let i=0;i<n;i++){ if(!bright[i]||lab[i]>=0) continue; const st=[i]; const id=comps.length; lab[i]=id; let sx=0,cnt=0;
    while(st.length){ const k=st.pop(); sx+=k%Sz; cnt++; const x=k%Sz, y=(k/Sz)|0;
      for (const q of [x>0?k-1:-1, x<Sz-1?k+1:-1, y>0?k-Sz:-1, y<Sz-1?k+Sz:-1]) if(q>=0&&bright[q]&&lab[q]<0){lab[q]=id;st.push(q);} }
    comps.push({cx:sx/cnt,cnt,id}); }
  // tooth identity is encoded in the fixture's blue tint (B = 196 + 9k); a flipped view reads as the mirrored tooth, like a real model
  const bsum = new Float64Array(comps.length);
  for (let i=0;i<n;i++) if (lab[i]>=0) bsum[lab[i]] += t[2*n+i]*std[2]+mean[2];
  const big = comps.filter(c=>c.cnt>30).sort((a,b)=>b.cnt-a.cnt).slice(0,6);
  const cls = new Map(big.map((c)=>{ let k = Math.round((bsum[c.id]/c.cnt*255 - 196)/9); k = Math.min(6, Math.max(1, k)); return [c.id, flip ? 7-k : k]; }));
  const out = new Float32Array(7*n);
  for (let i=0;i<n;i++){ const c = lab[i]>=0 ? (cls.get(lab[i])||0) : 0; out[c*n+i]=8; }
  window.__inferCalls = (window.__inferCalls||0)+1;
  return out;
};
"""

errors = []
async def shot(page, name, full=True):
    await page.wait_for_timeout(250)
    await page.screenshot(path=str(OUT / f'{name}.png'), full_page=full)

async def tap_img(page, sel, x, y, W, H):
    svg = page.locator(sel).first
    await svg.scroll_into_view_if_needed()
    b = await svg.bounding_box()
    await page.mouse.click(b['x'] + x / W * b['width'], b['y'] + y / H * b['height'])

async def drag_img(page, sel, x0, y0, x1, y1, W, H):
    svg = page.locator(sel).first
    b = await svg.bounding_box()
    P = lambda x, y: (b['x'] + x / W * b['width'], b['y'] + y / H * b['height'])
    await page.mouse.move(*P(x0, y0)); await page.mouse.down()
    for k in range(1, 6): await page.mouse.move(*P(x0 + (x1 - x0) * k / 5, y0 + (y1 - y0) * k / 5))
    await page.mouse.up()

async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', str(DIST)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.8)
    res = {}
    try:
        async with async_playwright() as p:
            br = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox'])
            ctx = await br.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, has_touch=True, locale='ar', accept_downloads=True)
            await ctx.add_init_script(HOOK)
            await ctx.route('**/*', lambda r: r.abort() if not r.request.url.startswith(f'http://localhost:{PORT}') else r.continue_())
            page = await ctx.new_page()
            page.on('console', lambda m: errors.append(f'[{m.type}] {m.text}') if m.type in ('error', 'warning') else None)
            page.on('pageerror', lambda e: errors.append(f'[pageerror] {e}'))
            page.on('dialog', lambda d: asyncio.ensure_future(d.dismiss()))
            await page.goto(f'http://localhost:{PORT}/index.html')
            await page.get_by_text('فهمت وأوافق').click()
            await shot(page, '01_home')
            # patient
            await page.get_by_text('+ مريض جديد').click()
            await page.locator('.modal input').nth(0).fill('مريض تجريبي')
            await page.locator('.modal input').nth(1).fill('A-102')
            await page.locator('.modal input').nth(2).fill('34')
            await page.locator('.modal').get_by_text('حفظ').click()
            await page.wait_for_function("location.hash.includes('patient')")
            await page.get_by_text('+ حالة جديدة').click()
            await page.wait_for_function("location.hash.includes('photos')")
            # consent then upload
            await page.locator('label.check input').first.check()
            await page.wait_for_timeout(400)
            inputs = page.locator('input[type=file]:not([capture])')
            await inputs.nth(0).set_input_files(str(ROOT / 'fixtures/intra.jpg'))
            await page.wait_for_function("location.hash.includes('crop')")
            await page.wait_for_timeout(2500)
            await shot(page, '02_crop')
            res['autocrop_calls'] = await page.evaluate('window.__inferCalls')
            await page.get_by_text('تطبيق والمتابعة ←').click()
            await page.wait_for_function("location.hash.includes('teeth')")
            await page.wait_for_timeout(500)
            await shot(page, '03_teeth_empty')
            await page.get_by_text('🤖 كشف تلقائي').click()
            await page.wait_for_function('document.querySelectorAll(".stage svg rect").length >= 6', timeout=20000)
            await page.wait_for_timeout(600)
            res['scroll_dbg'] = await page.evaluate('(()=>{const w=document.querySelector(".stagewrap"),st=w.firstChild;return [w.scrollLeft,w.scrollWidth,w.clientWidth,st.style.width,st.clientWidth]})()')
            await page.locator('.stagewrap').scroll_into_view_if_needed()
            await shot(page, '04_teeth_detected', full=False)
            res['infer_calls'] = await page.evaluate('window.__inferCalls')
            case = await page.evaluate('''async () => { const id = location.hash.split('/')[2]; return await DB.get('cases', id); }''')
            res['det'] = case['intra']['det']
            res['teeth'] = case['intra']['teeth']
            # calibrate by tooth 11 width 8.6mm
            await page.locator('select[aria-label="طريقة المعايرة"]').select_option('tooth')
            await page.wait_for_timeout(200)
            cal = page.locator('.card', has=page.locator('select[aria-label="طريقة المعايرة"]'))
            await cal.locator('input[type=number]').fill('8.6')
            await cal.locator('input[type=number]').dispatch_event('input')
            await page.wait_for_timeout(700)
            await shot(page, '05_teeth_calibrated')
            # drag handle of tooth 11 incisal edge down a bit (select 11 chip first)
            await page.locator('.chips .chip', has_text='11').click()
            img_wh = await page.evaluate('[...document.querySelectorAll(".stage svg")].map(s=>s.viewBox.baseVal).map(v=>[v.width,v.height])[0]')
            t11 = res['teeth']['11']
            iw, ih = img_wh
            await drag_img(page, '.stage svg', (t11['x0'] + t11['x1']) / 2, t11['y1'], (t11['x0'] + t11['x1']) / 2, t11['y1'] + 10, iw, ih)
            await page.wait_for_timeout(700)
            # SAM: tap inside tooth 12 after shrinking its box
            await page.evaluate('''async () => { const id = location.hash.split('/')[2]; const c = await DB.get('cases', id); const b = c.intra.teeth['12']; c.intra.teeth['12'] = {x0:b.x0+15, x1:b.x1-15, y0:b.y0+20, y1:b.y1-20}; await DB.put('cases', c); render(); }''')
            await page.wait_for_timeout(900)
            await page.get_by_text('👆 تحديد بنقرة').click()
            await page.locator('.chips .chip', has_text='12').click()
            t12 = res['teeth']['12']
            await page.locator('.stagewrap').scroll_into_view_if_needed()
            await tap_img(page, '.stage svg', (t12['x0']+t12['x1'])/2, (t12['y0']+t12['y1'])/2, iw, ih)
            await page.wait_for_timeout(1200)
            case = await page.evaluate('''async () => { const id = location.hash.split('/')[2]; return await DB.get('cases', id); }''')
            res['sam_12'] = case['intra']['teeth']['12']; res['sam_12_before'] = t12
            await shot(page, '05b_sam', full=False)
            # tooth library from this case
            await page.get_by_text('📚 حفظ الأسنان في المكتبة').click()
            await page.locator('.modal').get_by_text('لدي موافقته').click()
            await page.locator('.modal').get_by_text('موافق').click()
            await page.wait_for_timeout(800)
            res['library'] = await page.evaluate("(async()=>{const L=await DB.getKV('toothLibrary',[]);return L.map(s=>[s.name,Object.keys(s.teeth).length])})()")
            await page.locator('label.check', has_text='راجعتُ بنفسي').locator('input').check()
            await page.wait_for_timeout(400)
            case = await page.evaluate('''async () => { const id = location.hash.split('/')[2]; return await DB.get('cases', id); }''')
            res['t11_after_drag'] = case['intra']['teeth']['11']; res['source'] = case['intra']['source']; res['reviewed'] = case['intra'].get('reviewed')
            # face: upload face photo via photos step
            cid = case['id']
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/photos')
            await page.wait_for_timeout(500)
            await page.locator('input[type=file]:not([capture])').nth(1).set_input_files(str(ROOT / 'fixtures/face.jpg'))
            await page.wait_for_timeout(1200)
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/face')
            await page.wait_for_timeout(800)
            await page.get_by_text('🤖 نقاط تلقائية').click()
            await page.wait_for_timeout(1500)
            res['face_auto_modal'] = await page.locator('.modal').inner_text()
            await page.locator('.modal').get_by_text('حسناً').click()
            res['face_auto_count'] = await page.evaluate(f"(async()=>Object.keys((await DB.get('cases','{cid}')).face.points).length)()")
            coords = {'zenith11':(393,692),'incisal11':(393,720),'midlineDental':(403,718),'cusp13':(340,716),'cusp23':(460,716),'lastR':(325,705),'lastL':(475,705)}
            missing = await page.evaluate(f"(async()=>{{const P=(await DB.get('cases','{cid}')).face.points; return C.FACE_POINTS.map(p=>p.id).filter(i=>!P[i]);}})()")
            pts = [coords[m] for m in missing if m in coords]
            for (x, y) in pts:
                await tap_img(page, '.stage svg', x, y, 800, 1100)
                await page.wait_for_timeout(60)
            await page.wait_for_timeout(700)
            await shot(page, '06_face')
            case = await page.evaluate(f'''async () => await DB.get('cases', '{cid}')''')
            res['face_points'] = len(case['face']['points'])
            res['face_metrics'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); const A = computeAll(c); return A.face && A.face.metrics; }}''')
            # rest photo
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/photos'); await page.wait_for_timeout(500)
            await page.locator('input[type=file]:not([capture])').nth(2).set_input_files(str(ROOT / 'fixtures/face.jpg'))
            await page.wait_for_timeout(1200)
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/rest'); await page.wait_for_timeout(800)
            await page.get_by_text('🤖 نقاط تلقائية').click(); await page.wait_for_timeout(900)
            await tap_img(page, '.stage svg', 400, 694, 800, 1100); await page.wait_for_timeout(700)
            await shot(page, '06b_rest')
            res['rest'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); const A = computeAll(c); return [A.rest, A.rulesNow.find(r=>r.id==='rest_display')]; }}''')
            # design
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/design')
            await page.wait_for_timeout(900)
            await page.locator('.stagewrap').scroll_into_view_if_needed()
            await shot(page, '07b_design_stage', full=False)
            await shot(page, '07_design_outline')
            await page.get_by_text('محاكاة بعد').click()
            await page.wait_for_timeout(400)
            await page.locator('.stagewrap').first.scroll_into_view_if_needed()
            await shot(page, '08_design_sim', full=False)
            await page.locator('label.field', has_text='مصدر الأسنان').locator('select').select_option(index=1)
            await page.wait_for_timeout(900)
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/design'); await page.wait_for_timeout(900)
            await page.get_by_text('محاكاة بعد').click(); await page.wait_for_timeout(500)
            await page.locator('.stagewrap').first.scroll_into_view_if_needed()
            await shot(page, '08b_design_library', full=False)
            lm = page.locator('label.field', has_text='طول الثنية').locator('select')
            res['lengthMode_available'] = await lm.count()
            if res['lengthMode_available']:
                await lm.select_option('rest'); await page.wait_for_timeout(700)
                res['restLen'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); const A = computeAll(c); return [A.design.restLen, A.design.changes['11'].hmm]; }}''')
            res['design'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); const A = computeAll(c); return {{ changes: A.design.changes, p: A.design.prop.p, wlAfter: A.design.measAfter.ratios.central_wl, rulesNow: A.rulesNow.map(r=>[r.id,r.status,r.valueText]), rulesAfter: A.design.rulesAfter.map(r=>[r.id,r.status,r.valueText]) }}; }}''')
            # change midline to facial
            opts = await page.locator('label.field', has_text='خط الوسط').locator('option').all_inner_texts(); res['midline_opts']=opts
            if len(opts) > 1: await page.locator('label.field', has_text='خط الوسط').locator('select').select_option('facial')
            await page.wait_for_timeout(700)
            res['facialX'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); const A = computeAll(c); return [A.design.facialX, A.design.prop.actualMid]; }}''')
            # output
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{cid}/output')
            await page.wait_for_timeout(600)
            await page.locator('label.check input').check()
            await page.get_by_text('اعتماد التصميم').click()
            await page.locator('.modal input').fill('د. أحمد')
            await page.locator('.modal').get_by_text('موافق').click()
            await page.wait_for_timeout(700)
            await shot(page, '09_output')
            res['safety_main'] = await page.evaluate(f'''async () => {{ const c = await DB.get('cases','{cid}'); return [computeAll(c).safety, !!c.approval]; }}''')
            await page.get_by_text('🖨 Case Sheet (طباعة / PDF)').click()
            await page.wait_for_function('!!window.__lastSheet', timeout=10000)
            sheet = await page.evaluate('window.__lastSheet')
            sp = await ctx.new_page(); await sp.set_viewport_size({'width': 794, 'height': 1123})
            css = (DIST / 'index.html').read_text().split('<style>')[1].split('</style>')[0]
            await sp.set_content(f'<html dir="rtl"><head><style>{css}</style></head><body style="background:#fff">{sheet}</body></html>')
            await sp.wait_for_timeout(500)
            await sp.pdf(path=str(OUT / 'case_sheet.pdf'), format='A4', margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'}, print_background=True)
            await sp.screenshot(path=str(OUT / '10_case_sheet.png'), full_page=True)
            await sp.close()
            # presentation
            await page.goto(f'http://localhost:{PORT}/index.html#/present/{cid}')
            await page.wait_for_timeout(900)
            await shot(page, '11_present', full=False)
            # second case: tooth 12 missing
            pid = (await page.evaluate(f"(async()=> (await DB.get('cases','{cid}')).patientId)()"))
            await page.goto(f'http://localhost:{PORT}/index.html#/patient/{pid}'); await page.wait_for_timeout(500)
            await page.get_by_text('+ حالة جديدة').click()
            await page.wait_for_function("location.hash.includes('photos')")
            await page.locator('label.check input').first.check(); await page.wait_for_timeout(400)
            await page.locator('input[type=file]:not([capture])').nth(0).set_input_files(str(ROOT / 'fixtures/intra_missing12.jpg'))
            await page.wait_for_function("location.hash.includes('crop')"); await page.wait_for_timeout(2500)
            await page.get_by_text('تطبيق والمتابعة ←').click()
            await page.wait_for_function("location.hash.includes('teeth')"); await page.wait_for_timeout(500)
            await page.get_by_text('🤖 كشف تلقائي').click()
            await page.wait_for_function('document.querySelectorAll(".stage svg rect").length >= 6', timeout=20000); await page.wait_for_timeout(700)
            await page.locator('.stagewrap').scroll_into_view_if_needed()
            await shot(page, '15_missing_teeth', full=False)
            await shot(page, '15b_missing_teeth_full')
            c2 = await page.evaluate("(async()=>{const id=location.hash.split('/')[2]; return await DB.get('cases',id);})()")
            res['missing_status'] = c2['intra']['status']; res['missing_treat'] = c2['design']['treat']; res['missing_issues'] = c2['intra']['det']['issues']
            res['missing_box12'] = c2['intra']['teeth']['12']; res['missing_nb'] = [c2['intra']['teeth']['13']['x1'], c2['intra']['teeth']['11']['x0']]
            c2id = c2['id']
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{c2id}/design'); await page.wait_for_timeout(900)
            await page.get_by_text('محاكاة بعد').click(); await page.wait_for_timeout(500)
            await page.locator('.stagewrap').first.scroll_into_view_if_needed()
            await shot(page, '16_missing_design', full=False)
            await page.goto(f'http://localhost:{PORT}/index.html#/case/{c2id}/output'); await page.wait_for_timeout(700)
            res['missing_approve_disabled'] = await page.locator('button', has_text='اعتماد التصميم').is_disabled()
            res['missing_safety'] = await page.evaluate(f"(async()=>computeAll(await DB.get('cases','{c2id}')).safety)()")
            await shot(page, '17_missing_output')
            # set treat to veneer on missing tooth via DB and verify block
            res['veneer_block'] = await page.evaluate(f"(async()=>{{const c=await DB.get('cases','{c2id}'); c.design.treat['12']='veneer'; c.intra.reviewed=true; return computeAll(c).safety.filter(x=>x.level==='block').map(x=>x.text);}})()")
            res['missing_sheet_ok'] = True
            await page.evaluate(f"(async()=>{{const c=await DB.get('cases','{c2id}'); window.__lastSheet=null; await printSheet(c, await DB.get('patients', c.patientId)); }})()")
            res['missing_sheet_has'] = await page.evaluate("window.__lastSheet.includes('مفقود') && window.__lastSheet.includes('زرعة')")
            # backup export/import roundtrip
            await page.goto(f'http://localhost:{PORT}/index.html#/settings')
            await page.wait_for_timeout(600)
            await shot(page, '12_settings')
            async with page.expect_download() as dl:
                await page.get_by_text('⬇ تصدير نسخة احتياطية مشفّرة').click()
                await page.locator('.modal input').fill('correct-horse-1'); await page.locator('.modal').get_by_text('موافق').click()
                await page.locator('.modal input').fill('correct-horse-1'); await page.locator('.modal').get_by_text('موافق').click()
            d = await dl.value; bpath = OUT / 'backup.dsdb'; await d.save_as(str(bpath))
            raw = json.loads(bpath.read_text()); res['backup_format'] = raw['format']; res['backup_plain_leak'] = 'مريض تجريبي' in bpath.read_text()
            # wipe DB then import
            await page.evaluate('''async () => { for (const s of ['patients','cases']) for (const r of await DB.all(s)) await DB.del(s, r.id); }''')
            async with page.expect_file_chooser() as fc:
                await page.get_by_text('⬆ استيراد نسخة').click()
            await (await fc.value).set_files(str(bpath))
            await page.locator('.modal input').fill('correct-horse-1'); await page.locator('.modal').get_by_text('موافق').click()
            await page.locator('.modal button', has_text='استيراد').click()
            await page.locator('.modal').get_by_text('إلغاء').click()
            await page.wait_for_timeout(800)
            res['after_import'] = await page.evaluate('''async () => [(await DB.all('patients')).length, (await DB.all('cases')).length, (await DB.all('cases')).some(c=>!!c.approval)]''')
            # PIN lock
            await page.goto(f'http://localhost:{PORT}/index.html#/settings'); await page.wait_for_timeout(400)
            await page.get_by_text('تفعيل رمز الدخول').click()
            await page.locator('.modal input').fill('2468'); await page.locator('.modal').get_by_text('موافق').click()
            await page.locator('.modal input').fill('2468'); await page.locator('.modal').get_by_text('موافق').click()
            await page.wait_for_timeout(600)
            await page.reload(); await page.wait_for_timeout(800)
            res['lock_shown'] = await page.locator('.lock').count() == 1
            await page.locator('.lock input').fill('1111'); await page.get_by_text('دخول').click(); await page.wait_for_timeout(500)
            res['lock_wrong_stays'] = await page.locator('.lock').count() == 1
            await page.locator('.lock input').fill('2468'); await page.get_by_text('دخول').click(); await page.wait_for_timeout(700)
            res['lock_open'] = await page.locator('.lock').count() == 0
            await shot(page, '13_home_after', full=True)
            # dark mode check
            dctx = await br.new_context(viewport={'width': 412, 'height': 915}, color_scheme='dark', device_scale_factor=2)
            await dctx.add_init_script(HOOK)
            dp = await dctx.new_page(); await dp.goto(f'http://localhost:{PORT}/index.html'); await dp.wait_for_timeout(500)
            await dp.screenshot(path=str(OUT / '14_dark.png'))
            await br.close()
    finally:
        srv.terminate()
    res['errors'] = [e for e in errors if 'fonts.googleapis' not in e and 'ERR_FAILED' not in e and 'net::' not in e]
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))

asyncio.run(main())
