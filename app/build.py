"""Builds the app into ../docs (served by GitHub Pages). The AI runtimes live in ../docs/vendor; put teeth_unet_v2.onnx in ../docs."""
import pathlib, shutil, time
R = pathlib.Path(__file__).parent; S = R / 'src'; D = R.parent / 'docs'
D.mkdir(exist_ok=True)
shutil.copy(S / 'index.template.html', D / 'index.html')
for f in ['selftest.jpg', 'selftest.json', 'style.css', 'core.js', 'smile.js', 'sim.js', 'library.js', 'engine.js', 'app.js', 'capture.js', 'results.js', 'settings.js', 'worker.js', 'manifest.webmanifest', 'icon.svg', 'icon-192.png', 'icon-512.png']:
    shutil.copy(S / f, D / f)
if (D / 'annotate').exists(): shutil.rmtree(D / 'annotate')
shutil.copytree(S / 'annotate', D / 'annotate')   # tooth annotation tool (docs/annotate/)
(D / 'sw.js').write_text((S / 'sw.js').read_text().replace('__BUILD__', time.strftime('%Y%m%d%H%M%S')))
# stamp the model's fingerprint into its URL (cache-busting: the offline cache keeps model files forever)
import hashlib
if (D / 'teeth_unet_v2.onnx').exists():
    sha = hashlib.sha256((D / 'teeth_unet_v2.onnx').read_bytes()).hexdigest()[:12]
    (D / 'engine.js').write_text((D / 'engine.js').read_text().replace('__MODEL_SHA__', sha))
missing = [p for p in ['vendor/ort.webgpu.min.js', 'vendor/face_landmarker.task', 'vendor/mediapipe/vision_bundle.mjs', 'teeth_unet_v2.onnx'] if not (D / p).exists()]
print('built ../docs' + (' | missing: ' + ', '.join(missing) if missing else ''))
