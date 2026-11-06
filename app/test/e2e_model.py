import asyncio, json, sys, subprocess, time, pathlib
from playwright.async_api import async_playwright
DIST = pathlib.Path(__file__).resolve().parent.parent.parent / 'docs'; PORT = 8133
JS = r"""async () => {
  const names = (await (await fetch('_t/')).text()).match(/[\w.-]+\.jpg/g).filter((v, i, a) => a.indexOf(v) === i);
  const inter = new Array(7).fill(0), ps = new Array(7).fill(0), gs = new Array(7).fill(0), times = [];
  for (const n of names) {
    const img = await createImageBitmap(await (await fetch('_t/' + n)).blob());
    const c = canvasOf(img.width, img.height); c.getContext('2d').drawImage(img, 0, 0);
    const t0 = performance.now(); const r = await Engine.segment(modelInput(c)); times.push(performance.now() - t0);
    const mb = await createImageBitmap(await (await fetch('_t/' + n.replace('.jpg', '.png'))).blob());
    const mc = canvasOf(384, 384), mx = mc.getContext('2d'); mx.imageSmoothingEnabled = false; mx.drawImage(mb, 0, 0, 384, 384);
    const md = mx.getImageData(0, 0, 384, 384).data;
    for (let i = 0; i < 384 * 384; i++) { const g = md[i * 4], p = r.labels[i]; if (g) gs[g]++; if (p) ps[p]++; if (g && g === p) inter[g]++; }
  }
  const dice = [1, 2, 3, 4, 5, 6].map((c) => 2 * inter[c] / Math.max(1, ps[c] + gs[c]));
  times.sort((a, b) => a - b);
  return { n: names.length, dice: dice.map((v) => +v.toFixed(3)), mean: +(dice.reduce((a, b) => a + b) / 6).toFixed(4), medianMs: Math.round(times[times.length >> 1]), ep: Engine.info.ep, warmMs: Engine.info.warmMs };
}"""
async def run(nogpu):
    async with async_playwright() as p:
        br = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=['--no-sandbox', '--enable-unsafe-webgpu'])
        ctx = await br.new_context()
        await ctx.add_init_script('window.__NO_SW = true;')
        await ctx.route('**/*', lambda r: r.continue_() if r.request.url.startswith(f'http://localhost:{PORT}') else r.abort())
        page = await ctx.new_page(); errs = []; page.on('pageerror', lambda e: errs.append(str(e)))
        await page.goto(f'http://localhost:{PORT}/index.html' + ('?cpu' if nogpu else ''))
        t0 = time.time(); await page.evaluate('Engine.prepare()'); prep = round(time.time() - t0, 1)
        r = await page.evaluate(JS); r['prepare_s'] = prep; r['errors'] = errs
        await br.close(); return r
async def main():
    srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', str(DIST)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(0.8)
    try:
        if '--gpu' in sys.argv: print('WebGPU:', json.dumps(await run(False)))
        print('WASM  :', json.dumps(await run(True)))
    finally: srv.terminate()
asyncio.run(main())
