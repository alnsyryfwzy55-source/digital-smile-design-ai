"""Builds generation_poc.ipynb (self-contained: dsd_gen.py is written into the Colab session by the first cells)."""
import json, pathlib
HERE = pathlib.Path(__file__).parent
cells = []
md = lambda s: cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n')})
code = lambda s: cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n')})

md(r'''
# تجربة إثبات جدوى: توليد صورة «بعد» تلتزم بالتصميم

**السؤال الذي يجيب عنه هذا الدفتر:** هل يستطيع نموذج توليد مفتوح ومجاني أن يرسم أسناناً واقعية **داخل حدود التصميم الهندسي بالضبط**، دون أن يغيّر الشفاه أو الأسنان السفلية؟

**شرط النجاح** (على 20 صورة):
- الفحص الآلي: تطابق حدود الأسنان المولّدة مع التصميم (IoU) ≥ 0.80، وتغيّر خارج المنطقة المسموحة < 1، ولا أسنان زائدة، في **80% من الصور على الأقل**.
- الواقعية: تقييم أعمى من الطبيب (حزمة التقييم تُنشأ في آخر الدفتر).

**الخصوصية:** لا تستخدم صور مرضى في هذه التجربة. استخدم صورك أو صور أشخاص وافقوا، أو صوراً عامة مرخّصة.

**التشغيل:** Runtime ← Change runtime type ← **T4 GPU**، ثم شغّل الخلايا بالترتيب. المدة نحو 30–45 دقيقة.
''')

code(r'''
# 1) التجهيز
import subprocess, sys, os
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
!pip -q install mediapipe diffusers accelerate onnxruntime
''')

code('%%writefile /content/dsd_gen.py\n' + (HERE / 'dsd_gen.py').read_text())

code(r'''
# 2) ملفات النماذج من مستودع المشروع (نموذج تحديد الأسنان + كاشف الوجه)
import os, urllib.request
REPO = 'https://raw.githubusercontent.com/alnsyryfwzy55-source/digital-smile-design-ai/main/docs/'
FILES = {'teeth_unet_v2.onnx': REPO + 'teeth_unet_v2.onnx',
         'face_landmarker.task': REPO + 'vendor/face_landmarker.task'}
for name, url in FILES.items():
    if not os.path.exists(name):
        try:
            urllib.request.urlretrieve(url, name)
        except Exception as e:
            print(f'تعذّر تنزيل {name}: {e}\nارفعه يدوياً من مجلد docs في المستودع إلى /content')
    print(name, os.path.getsize(name) if os.path.exists(name) else 'غير موجود')
# raw.githubusercontent لا يخدم ملفات Git LFS؛ إن كان حجم النموذج صغيراً جداً فهو مؤشر LFS
assert os.path.getsize('teeth_unet_v2.onnx') > 10_000_000, 'ملف النموذج غير صحيح'
''')

md(r'''
## 3) الصور
ارفع ملفاً مضغوطاً `photos.zip` فيه 20 صورة ابتسامة أمامية تظهر فيها الأسنان العلوية (JPG أو PNG)،
أو ضع الصور في Google Drive وغيّر `PHOTO_DIR`.
''')

code(r'''
import glob, zipfile
PHOTO_DIR = '/content/photos'
os.makedirs(PHOTO_DIR, exist_ok=True)
if not glob.glob(PHOTO_DIR + '/*'):
    from google.colab import files
    up = files.upload()                       # اختر photos.zip
    for n in up:
        if n.endswith('.zip'):
            zipfile.ZipFile(n).extractall(PHOTO_DIR)
PHOTOS = sorted(p for p in glob.glob(PHOTO_DIR + '/**/*', recursive=True) if p.lower().endswith(('.jpg', '.jpeg', '.png')))
print(len(PHOTOS), 'صورة')
''')

code(r'''
# 4) التحليل والتصميم الهندسي (قبل أي توليد)
import numpy as np, cv2, json, matplotlib.pyplot as plt
import dsd_gen as G
face = G.FaceFinder('face_landmarker.task')
seg = G.Segmenter('teeth_unet_v2.onnx')
SHAPE, PROPORTION, SHADE = 'square', 'red70', 'B1'

cases, skipped = [], []
for p in PHOTOS:
    rgb = G.load_rgb(p)
    r = G.analyse(rgb, face, seg, shape=SHAPE, proportion=PROPORTION, shade=SHADE)
    if not r.get('ok'):
        skipped.append((os.path.basename(p), r.get('reason') or r.get('problems')))
        continue
    r['path'], r['rgb'] = p, rgb
    cases.append(r)
print('جاهزة للتوليد:', len(cases), '| مستبعدة:', len(skipped))
for s in skipped: print('  ', s)

n = min(6, len(cases))
fig, ax = plt.subplots(n, 2, figsize=(12, 3.2 * n))
for i in range(n):
    c = cases[i]
    ax[i, 0].imshow(G.overlay(c['crop'], c['design'], c['current'], c['mouth_poly'])); ax[i, 0].set_title('التصميم (أخضر) فوق الأسنان الحالية (أحمر)')
    ax[i, 1].imshow(c['inputs']['prepaint']); ax[i, 1].set_title('مدخل التوليد')
    for a in ax[i]: a.axis('off')
plt.tight_layout(); plt.show()
''')

code(r'''
# 5) نموذج التوليد: Stable Diffusion inpainting + ControlNet (حواف التصميم)
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

BASES = {'A_inpaint': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
         'B_base':    'stable-diffusion-v1-5/stable-diffusion-v1-5'}
controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny', torch_dtype=torch.float16)
_pipes = {}
def get_pipe(key):
    if key not in _pipes:
        for k in list(_pipes):                 # one pipeline in GPU memory at a time
            del _pipes[k]
        torch.cuda.empty_cache()
        p = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            BASES[key], controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        _pipes[key] = p.to('cuda')
    return _pipes[key]

PROMPT = ('close-up photo of a natural smile, healthy natural teeth, realistic enamel texture, '
          'subtle translucency at the edges, natural lighting, sharp focus, high detail')
NEGATIVE = ('cartoon, painting, 3d render, deformed teeth, extra teeth, fused teeth, missing teeth, gap, braces, '
            'metal, blurry, lowres, lipstick, distorted lips, tongue')

def match_color(gen, ref, mask, k=0.6):
    """Pull the generated teeth's mean colour towards the chosen shade (LAB), keeping texture."""
    g = cv2.cvtColor(gen, cv2.COLOR_RGB2LAB).astype(np.float32); r = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 50: return gen
    g[m] += k * (r[m].mean(0) - g[m].mean(0))
    return cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def render(case, base='A_inpaint', strength=0.6, cn_scale=0.8, seed=0, steps=30):
    inp, crop = case['inputs'], case['crop']
    h, w = crop.shape[:2]
    pipe = get_pipe(base)
    out = pipe(PROMPT, negative_prompt=NEGATIVE,
               image=Image.fromarray(inp['prepaint']), mask_image=Image.fromarray(inp['mask'] * 255),
               control_image=Image.fromarray(np.dstack([inp['edges']] * 3)),
               height=h, width=w, strength=strength, num_inference_steps=steps, guidance_scale=7.0,
               controlnet_conditioning_scale=cn_scale, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
    gen = np.asarray(out.resize((w, h)))
    gen = match_color(gen, inp['prepaint'], inp['design_mask'])
    after, alpha = G.composite(crop, gen, inp['mask'])
    chk = G.check_generated(seg, crop, after, inp, case['mouth_poly'])
    return after, alpha, chk
''')

code(r'''
# 6) مسح الإعدادات على أول 5 صور: أي إعداد يلتزم بالتصميم أكثر؟
import itertools, time, pandas as pd
SWEEP = list(itertools.product(['A_inpaint', 'B_base'], [0.45, 0.6, 0.75], [0, 1]))
rows = []
t0 = time.time()
for base, s, seed in SWEEP:
    for i, c in enumerate(cases[:5]):
        _, _, chk = render(c, base=base, strength=s, seed=seed)
        rows.append(dict(base=base, strength=s, seed=seed, case=i, **{k: v for k, v in chk.items() if k != 'iou'}))
df = pd.DataFrame(rows)
summary = df.groupby(['base', 'strength']).agg(pass_rate=('pass', 'mean'), iou_union=('iou_union', 'mean'),
                                               iou_min=('iou_min', 'mean'), outside=('outside_change', 'mean')).reset_index()
summary = summary.sort_values(['pass_rate', 'iou_union'], ascending=False)
print(f'{time.time() - t0:.0f} ث'); display(summary)
BEST = summary.iloc[0][['base', 'strength']].to_dict()
print('أفضل إعداد:', BEST)
''')

code(r'''
# 7) الإعداد الأفضل على كل الصور: 3 بذور لكل صورة، ويُختار أعلاها التزاماً بالتصميم
OUT = '/content/results'; os.makedirs(OUT, exist_ok=True)
results = []
for i, c in enumerate(cases):
    best = None
    for seed in range(3):
        after, alpha, chk = render(c, base=BEST['base'], strength=float(BEST['strength']), seed=seed)
        if best is None or chk['iou_union'] > best[2]['iou_union']:
            best = (after, alpha, chk, seed)
    after, alpha, chk, seed = best
    name = f'case_{i:02d}'
    full_after = G.paste_back(c['rgb'], after, c['M'], alpha)
    sheet = np.concatenate([c['crop'], G.overlay(c['crop'], c['design']), after], 1)
    cv2.imwrite(f'{OUT}/{name}_sheet.jpg', cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{OUT}/{name}_after_full.jpg', cv2.cvtColor(full_after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    results.append(dict(case=name, photo=os.path.basename(c['path']), seed=seed, **{k: v for k, v in chk.items() if k != 'iou'},
                        iou_per_tooth=chk['iou']))
res = pd.DataFrame(results)
res.drop(columns=['iou_per_tooth']).to_csv(f'{OUT}/metrics.csv', index=False)
print(f"اجتاز الفحص الآلي: {res['pass'].sum()} من {len(res)} ({100 * res['pass'].mean():.0f}%)  |  الشرط: 80%")
display(res[['case', 'iou_union', 'iou_min', 'teeth_found', 'outside_change', 'extra_tooth_px', 'pass']])
''')

code(r'''
# 8) عرض النتائج: قبل | التصميم | بعد
for r in results[:8]:
    im = cv2.cvtColor(cv2.imread(f"{OUT}/{r['case']}_sheet.jpg"), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 3.6)); plt.imshow(im); plt.axis('off')
    plt.title(f"{r['case']}  IoU={r['iou_union']:.2f}  {'✓' if r['pass'] else '✗'}"); plt.show()
''')

code(r'''
# 9) حزمة التقييم الأعمى للطبيب: صور حقيقية ومولّدة مخلوطة بأسماء عشوائية
import random, shutil, csv
BLIND = f'{OUT}/blind_test'; os.makedirs(BLIND, exist_ok=True)
items = []
for i, c in enumerate(cases):
    after = cv2.imread(f'{OUT}/case_{i:02d}_sheet.jpg')
    w3 = after.shape[1] // 3
    items.append(('real', i, cv2.cvtColor(c['crop'], cv2.COLOR_RGB2BGR)))
    items.append(('generated', i, after[:, 2 * w3:]))
random.seed(7); random.shuffle(items)
key = []
for k, (kind, i, im) in enumerate(items):
    fn = f'img_{k:03d}.jpg'; cv2.imwrite(f'{BLIND}/{fn}', im); key.append((fn, kind, i))
with open(f'{OUT}/blind_key_DO_NOT_SHARE.csv', 'w', newline='') as f: csv.writer(f).writerows([('file', 'kind', 'case')] + key)
with open(f'{BLIND}/rating_sheet.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['file', 'realism_1_to_5', 'looks_generated_yes_no', 'notes'])
    for fn, _, _ in key: w.writerow([fn, '', '', ''])
print(len(key), 'صورة في', BLIND, '— أرسل المجلد للطبيب دون ملف المفتاح')
''')

code(r'''
# 10) تنزيل النتائج + ملخص ترسله لي
summary_out = dict(best_config=BEST, n=len(res), pass_rate=float(res['pass'].mean()),
                   iou_union_mean=float(res['iou_union'].mean()), iou_min_mean=float(res['iou_min'].mean()),
                   outside_change_max=float(res['outside_change'].max()), skipped=len(skipped),
                   sweep=summary.round(3).to_dict(orient='records'))
json.dump(summary_out, open(f'{OUT}/summary.json', 'w'), indent=1)
print(json.dumps(summary_out, indent=1))
shutil.make_archive('/content/generation_results', 'zip', OUT)
from google.colab import files; files.download('/content/generation_results.zip')
''')

nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
      'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:
    s = c['source']; lines = s.split('\n'); c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
(HERE / 'generation_poc.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('wrote generation_poc.ipynb with', len(cells), 'cells')
