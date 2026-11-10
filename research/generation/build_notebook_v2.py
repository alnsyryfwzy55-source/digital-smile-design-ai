"""Builds generation_v2.ipynb (Colab, free T4): clones the repository and runs run_v2.py."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
cells = [
    md("""# تجربة التوليد — النسخة 2

**الفرق عن النسخة 1:** الأسنان في الرسم الأولي **أسنان حقيقية** من بيانات التدريب (ترخيص CC BY 4.0) بشكلها التشريحي وملمسها، ولونها من لون أسنان الشخص نفسه مع تبييض معتدل. نموذج التوليد يمرّ عليها مروراً خفيفاً لتوحيد الإضاءة فقط، ونقارن SD 1.5 مع **SDXL**. والفحص الآلي يقيس الآن **الواقعية** أيضاً: الملمس، واللون، والحواف.

**التشغيل:** Runtime ← Change runtime type ← **T4 GPU**، ثم شغّل الخلايا بالترتيب. المدة نحو 45–60 دقيقة (ينزّل نحو 12 GB من النماذج أول مرة).

**الصور:** نفس `photos.zip` من التجربة الأولى (صور من الإنترنت، لا صور مرضى)."""),
    code("""!nvidia-smi -L
!rm -rf /content/repo && git clone -q --depth 1 https://github.com/alnsyryfwzy55-source/digital-smile-design-ai /content/repo
!pip -q install diffusers transformers accelerate mediapipe onnxruntime opencv-python-headless
print('ready')"""),
    code("""import os, zipfile, glob
from google.colab import files
PHOTO_DIR = '/content/photos'
if not glob.glob(PHOTO_DIR + '/**/*.*', recursive=True):
    up = files.upload()                       # اختر photos.zip
    for n in up:
        zipfile.ZipFile(n).extractall(PHOTO_DIR)
print(len(glob.glob(PHOTO_DIR + '/**/*.*', recursive=True)), 'files')"""),
    code("""%cd /content/repo/research/generation
!python -u run_v2.py --photos /content/photos --out /content/results_v2 2>&1 | grep -v -e W0000 -e 'INFO:'"""),
    code("""from IPython.display import Image, display
import glob
for p in sorted(glob.glob('/content/results_v2/case_*_sheet.jpg'))[:6]:
    print(p.split('/')[-1], '— قبل | التصميم | أسنان حقيقية بلا توليد | بعد التوليد'); display(Image(p, width=1100))"""),
    code("""!cd /content && rm -f results_v2.zip && zip -qr results_v2.zip results_v2
from google.colab import files
files.download('/content/results_v2.zip')"""),
]
nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[]), kernelspec=dict(name='python3', display_name='Python 3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=0)
json.dump(nb, open(os.path.join(HERE, 'generation_v2.ipynb'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote generation_v2.ipynb')
