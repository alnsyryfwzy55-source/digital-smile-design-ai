import asyncio, json, pathlib, sys, subprocess, time
from playwright.async_api import async_playwright
PORT = 8141
async def run(bogus):
    async with async_playwright() as p:
        br = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox'])
        ctx = await br.new_context()
        if bogus: await ctx.route('**/selftest.json', lambda r: r.fulfill(status=200, content_type='application/json', body=json.dumps([147456, 0, 0, 0, 0, 0, 0])))
        page = await ctx.new_page(); await page.goto(f'http://localhost:{PORT}/index.html?cpu')
        try: r = await page.evaluate('Engine.prepare().then(i => ({ ok: true, ep: i.ep, selfTest: i.selfTest && { ep: i.selfTest.ep, ok: i.selfTest.ok, counts: i.selfTest.counts } }))')
        except Exception as e: r = {'ok': False, 'error': str(e).split('\n')[0][:200]}
        await br.close(); return r
async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', str(pathlib.Path(__file__).resolve().parent.parent.parent / 'docs')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
    try:
        print('healthy engine :', json.dumps(await run(False)))
        print('wrong numbers  :', json.dumps(await run(True)))
    finally: srv.terminate()
asyncio.run(main())
