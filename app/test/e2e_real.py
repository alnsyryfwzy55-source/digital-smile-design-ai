import asyncio, json, sys, subprocess, time, pathlib, shutil
from playwright.async_api import async_playwright
ROOT = pathlib.Path(__file__).resolve().parent; DIST = ROOT.parent.parent / 'docs'; OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PHOTOS = [a for a in sys.argv[2:] if not a.startswith('--')] or ['/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg']
PORT = 8131
async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', str(DIST)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
    logs, res = [], {}
    try:
        async with async_playwright() as p:
            br = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--enable-unsafe-webgpu'])
            ctx = await br.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, locale='ar', has_touch=True)
            await ctx.add_init_script('window.__NO_SW = true; window.__DSD_NO_PRINT = true;')
            await ctx.route('**/*', lambda r: r.continue_() if r.request.url.startswith(f'http://localhost:{PORT}') else r.abort())
            page = await ctx.new_page()
            page.on('console', lambda m: logs.append(f'[{m.type}] {m.text}') if m.type in ('error', 'warning') else None)
            page.on('pageerror', lambda e: logs.append(f'[pageerror] {e}'))
            await page.goto(f'http://localhost:{PORT}/index.html' + ('?cpu' if '--cpu' in sys.argv else ''))
            await page.get_by_text('فهمت').click()
            t0 = time.time()
            info = await page.evaluate('Engine.prepare().then(i => JSON.parse(JSON.stringify(i)))')
            res['prepare_s'] = round(time.time() - t0, 1); res['engine'] = info
            for n, photo in enumerate(PHOTOS):
                await page.goto(f'http://localhost:{PORT}/index.html' + ('?cpu' if '--cpu' in sys.argv else '') + '#/new'); await page.wait_for_timeout(300)
                await page.fill('#pname', f'اختبار {n+1}'); await page.check('#consent')
                async with page.expect_file_chooser() as fc:
                    await page.get_by_text('من المعرض').click()
                await (await fc.value).set_files(photo)
                t1 = time.time()
                await page.wait_for_function("document.querySelector('.ba') || document.body.innerText.includes('مراجعة الصورة') || document.body.innerText.includes('تعذّر') || document.body.innerText.includes('لم يُعثر')", timeout=120000)
                await page.screenshot(path=str(OUT / f'{n}_a_after_analysis.png'), full_page=True)
                gate = await page.evaluate("document.body.innerText.includes('مراجعة الصورة')")
                r = {'photo': pathlib.Path(photo).name, 'analysis_s': round(time.time() - t1, 1), 'quality_gate': gate}
                if not gate and not await page.locator('.ba').count():          # the analysis stopped with a message: record it, never stale state
                    r['error'] = (await page.inner_text('main'))[:300]; res[f'photo{n}'] = r; continue
                if gate:
                    r['gate_text'] = await page.inner_text('main')
                    await page.get_by_text('متابعة رغم ذلك').click()
                    await page.wait_for_function("document.querySelector('.ba') || document.body.innerText.includes('تعذّر') || document.body.innerText.includes('لا يمكن')", timeout=60000)
                await page.wait_for_function("!document.querySelector('.ba .busy') || document.querySelector('.ba .busy').classList.contains('hidden')", timeout=60000)
                await page.wait_for_timeout(500)
                await page.screenshot(path=str(OUT / f'{n}_b_results.png'), full_page=True)
                st = await page.evaluate('''() => { const L = Results.live; if (!L) return null; const s = L.st;
                    return { status: s.status, teeth: s.teeth, meanConf: s.sel && s.sel.meanConf, orderOk: s.sel && s.sel.orderOk, quality: s.quality, scale: s.scaleSource, mmpp: s.mmppCrop,
                      timings: s.timings, segMs: s.segMs, ep: s.ep, simMs: s.lastSimMs, face: s.face && s.face.metrics, pose: s.quality && s.quality.pose, crop: {w: s.geom.crop.outW, h: s.geom.crop.outH, roll: s.geom.crop.roll} }; }''')
                r['state'] = st
                # one design change → re-simulation time
                r['no_teeth_screen'] = await page.evaluate("document.body.innerText.includes('لا يمكن تصميم هذه الصورة')")
                if await page.get_by_text('بيضاوي').count():
                    t2 = time.time(); await page.get_by_text('بيضاوي').click()
                    await page.wait_for_timeout(200)
                    await page.wait_for_function("document.querySelector('.ba .busy').classList.contains('hidden')", timeout=30000)
                    r['resim_s'] = round(time.time() - t2, 2)
                if await page.locator('.ba').count(): await page.locator('.ba').first.scroll_into_view_if_needed()
                await page.screenshot(path=str(OUT / f'{n}_c_ba.png'))
                res[f'photo{n}'] = r
            await br.close()
    finally:
        srv.terminate()
    res['logs'] = [l for l in logs if 'fonts.googleapis' not in l and 'ERR_FAILED' not in l][:30]
    (OUT / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:9000])
asyncio.run(main())
