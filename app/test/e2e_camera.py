import asyncio, json, sys, subprocess, time, pathlib
from playwright.async_api import async_playwright
DIST = pathlib.Path(__file__).resolve().parent.parent.parent / 'docs'; OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); PORT = 8132
async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', str(DIST)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
    res = {}; logs = []
    try:
        async with async_playwright() as p:
            br = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', f'--use-file-for-fake-video-capture={sys.argv[2]}'])
            ctx = await br.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, permissions=['camera'])
            await ctx.add_init_script('window.__NO_SW = true;')
            await ctx.route('**/*', lambda r: r.continue_() if r.request.url.startswith(f'http://localhost:{PORT}') else r.abort())
            page = await ctx.new_page(); page.on('pageerror', lambda e: logs.append(str(e)))
            await page.goto(f'http://localhost:{PORT}/index.html'); await page.get_by_text('فهمت').click()
            await page.goto(f'http://localhost:{PORT}/index.html#/new'); await page.fill('#pname', 'كاميرا'); await page.check('#consent')
            await page.get_by_text('فتح الكاميرا').click()
            await page.wait_for_selector('.cam video', timeout=60000)
            msgs = set()
            for i in range(30):
                await page.wait_for_timeout(300)
                msgs.add(await page.inner_text('.cam .msg'))
            res['messages_seen'] = sorted(msgs)
            res['checks'] = await page.inner_text('.cam .checks')
            res['video'] = await page.evaluate("(() => { const v = document.querySelector('.cam video'); return [v.videoWidth, v.videoHeight]; })()")
            res['guide_drawn'] = await page.evaluate("document.querySelector('.cam svg.guide').innerHTML.includes('polygon')")
            await page.screenshot(path=str(OUT / 'camera.png'))
            await page.click('.cam .shutter')
            await page.wait_for_function("document.body.innerText.includes('مراجعة الصورة') || document.querySelector('.ba') || document.body.innerText.includes('لا يمكن')", timeout=120000)
            res['after_capture'] = (await page.inner_text('main'))[:200]
            await br.close()
    finally: srv.terminate()
    res['pageerrors'] = logs
    print(json.dumps(res, ensure_ascii=False, indent=1))
asyncio.run(main())
