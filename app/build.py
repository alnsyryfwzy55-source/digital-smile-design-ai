import pathlib, shutil, subprocess
R = pathlib.Path(__file__).parent; S = R/'src'; D = R.parent/'docs'; D.mkdir(exist_ok=True)
t = (S/'index.template.html').read_text()
app = '\n'.join((S/f).read_text() for f in ['app1.js','app2.js','app4.js','app3.js'])
t = t.replace('/*CSS*/', (S/'style.css').read_text()).replace('/*CORE*/', (S/'core.js').read_text()).replace('/*APP*/', app)
(D/'index.html').write_text(t)
for f in ['sw.js','manifest.webmanifest','icon.svg']: shutil.copy(S/f, D/f)
import cv2, numpy as np
svg = (S/'icon.svg').read_text()
# raster icons via headless chrome
for n in (192, 512):
    subprocess.run(['google-chrome','--headless=new','--no-sandbox','--disable-gpu',f'--window-size={n},{n}',f'--screenshot={D}/icon-{n}.png','--default-background-color=00000000',f'data:text/html,<style>html,body{{margin:0}}</style><img src="data:image/svg+xml;utf8,{svg.replace("#","%23").replace(chr(10),"")}" width="{n}" height="{n}">'],capture_output=True)
print('built', (D/'index.html').stat().st_size)
