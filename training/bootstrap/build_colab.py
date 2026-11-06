"""Builds run_in_colab.ipynb: the whole bootstrap task in one notebook (code embedded), for Google Colab."""
import base64, io, json, os, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(HERE):
        if '__pycache__' in root or '/models' in root or root.endswith('models'):
            continue
        for f in files:
            if f.endswith(('.py', '.txt', '.md')) and f not in ('build_colab.py',):
                p = os.path.join(root, f); z.write(p, os.path.relpath(p, HERE))
B64 = base64.b64encode(buf.getvalue()).decode()

md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
cells = [
md("""# مسودات تعليم الأسنان من صور الإنترنت

**قبل التشغيل:** من القائمة Runtime ← Change runtime type ← اختر **T4 GPU** ← Save.

**التشغيل:** Runtime ← **Run all**. سيطلب الإذن بالوصول إلى Google Drive، فوافق، لتُحفظ الصور والنتائج هناك.

- يستغرق نحو ساعة: تنزيل حتى 1500 صورة بترخيص حر من Openverse، ثم تعليمها تلقائياً.
- إذا انقطع الاتصال أو أُغلقت الصفحة، شغّل **Run all** مرة أخرى. يكمل من حيث توقف ولا يعيد تنزيل ما نزل.
- في النهاية يُنزَّل ملف `bootstrap-results.zip`. أرسله كما هو.
"""),
code("""N_PHOTOS = 1500      # عدد الصور المطلوب
USE_DRIVE = True     # حفظ العمل في Google Drive ليستأنف بعد أي انقطاع
"""),
code("""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mediapipe', 'transformers>=4.56', 'onnxruntime'], check=True)
print('packages ready')
"""),
code("""import os, sys, subprocess
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/smile_bootstrap'
else:
    WORK = os.path.abspath('smile_bootstrap')
os.makedirs(WORK, exist_ok=True)
print('working folder:', WORK)
"""),
code("""# the task code (bootstrap.py + our post-processing core), embedded so nothing else has to be uploaded
import base64, io, zipfile, shutil
PAYLOAD = '""" + B64 + """'
CODE = os.path.join(WORK, 'code')
models = os.path.join(CODE, 'models')
keep = None
if os.path.isdir(models):                       # keep downloaded models between runs
    keep = os.path.join(WORK, '_models'); shutil.rmtree(keep, ignore_errors=True); shutil.move(models, keep)
shutil.rmtree(CODE, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(CODE)
if keep: shutil.move(keep, models)
print(sorted(os.listdir(CODE)))
"""),
code("""# download (resumes) + automatic labelling; the log streams below
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU: slower but works)')
OUT = os.path.join(WORK, 'run2')
p = subprocess.Popen([sys.executable, '-u', 'bootstrap.py', '--openverse', str(N_PHOTOS), '--out', OUT], cwd=CODE,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if 'Warning' not in line and 'warn(' not in line:
        print(line, end='')
p.wait()
print('exit code', p.returncode)
"""),
code("""# one zip to send back
import json
res = os.path.join(WORK, 'bootstrap-results.zip')
with zipfile.ZipFile(res, 'w', zipfile.ZIP_STORED) as z:
    for f in ('drafts.zip', 'report.json', 'log.txt'):
        if os.path.exists(os.path.join(OUT, f)):
            z.write(os.path.join(OUT, f), f)
    z.writestr('environment.json', json.dumps(dict(gpu=torch.cuda.is_available(), python=sys.version.split()[0], torch=torch.__version__)))
print(open(os.path.join(OUT, 'report.json')).read() if os.path.exists(os.path.join(OUT, 'report.json')) else 'no report')
if IN_COLAB:
    from google.colab import files
    files.download(res)
"""),
]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3'), accelerator='GPU', colab=dict(provenance=[], gpuType='T4')))
out = os.path.join(HERE, 'run_in_colab.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out, os.path.getsize(out))
