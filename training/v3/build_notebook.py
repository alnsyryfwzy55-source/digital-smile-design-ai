"""Builds train_v3.ipynb (Google Colab). The data module v3data.py is embedded, so the notebook is self-contained."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
V3DATA = open(os.path.join(HERE, 'v3data.py'), encoding='utf-8').read()
REPO = 'https://raw.githubusercontent.com/alnsyryfwzy55-source/digital-smile-design-ai/main/'

md = lambda s: dict(cell_type='markdown', metadata={}, source=s)
code = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)

cells = [
md("""# تدريب النسخة الثالثة من نموذج الأسنان

**ما الجديد عن النسخة الثانية:**
1. **ابتسامات حقيقية:** صور الإنترنت بعد مراجعتك في أداة التعليم، مع الشفاه والأسنان السفلية والظلال والحالات الصعبة.
2. **تعليم النموذج ألا يرسم سناً في الفراغ:** أثناء التدريب يُمحى سن عشوائياً ويُرسم مكانه فراغ داكن أو لثة ملتئمة، ويُعلَّم خلفية. النسخة الثانية ترسم سناً في الفراغ في 95% من الحالات.
3. **مقارنة عادلة مع النسخة الثانية** على المقاييس نفسها، وقرار مكتوب: تُعتمد الثالثة فقط إن كانت أفضل.

**قبل التشغيل:**
- Runtime ← Change runtime type ← **T4 GPU**.
- ضع ملفات المراجعة (ZIP من أداة التعليم، زر «تصدير») في مجلد `smile_reviews` في Google Drive. أو ارفعها حين تطلبها الخلية.

**التشغيل:** Runtime ← **Run all**. يستغرق نحو 40 دقيقة على T4.
"""),
code("""EPOCHS = 60
P_GAP = 0.2            # نسبة صور التدريب التي يُمحى فيها سن ويُرسم فراغ مكانه
USE_DRIVE = True       # قراءة ملفات المراجعة من Drive/smile_reviews وحفظ النتائج هناك
"""),
code("""!pip -q install segmentation-models-pytorch albumentations onnx onnxruntime opencv-python-headless
"""),
code("""# data helpers (also tested outside Colab): reading reviewed exports, gap augmentation, metrics
open('v3data.py', 'w', encoding='utf-8').write(""" + repr(V3DATA) + """)
import importlib, v3data as V
importlib.reload(V)
print('v3data ready')
"""),
code("""import os, glob, zipfile, urllib.request, sys
IN_COLAB = 'google.colab' in sys.modules
REPO = '""" + REPO + """'
def fetch(url, dst):
    if os.path.exists(dst): return True
    try:
        urllib.request.urlretrieve(url, dst); return os.path.getsize(dst) > 1000
    except Exception as e:
        print('could not download', url, e); return False

# 1) the v2 dataset (retractor photos) + the v2 model to compare against + the app's self-test photo
if not fetch(REPO + 'data/smile_teeth_v2.zip', 'smile_teeth_v2.zip') and IN_COLAB:
    from google.colab import files
    print('ارفع smile_teeth_v2.zip'); files.upload()
zipfile.ZipFile('smile_teeth_v2.zip').extractall('.')
V2 = os.path.abspath('smile_teeth_v2')
assert fetch(REPO + 'docs/teeth_unet_v2.onnx', 'teeth_unet_v2.onnx'), 'teeth_unet_v2.onnx is needed for the comparison'
fetch(REPO + 'docs/selftest.jpg', 'selftest.jpg')

# 2) the reviewed smiles
WORK = '.'
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/smile_reviews'; os.makedirs(WORK, exist_ok=True)
is_review = lambda z: 'manifest.csv' in zipfile.ZipFile(z).namelist()      # exports of the annotation tool only
REVIEWS = [z for z in sorted(glob.glob(f'{WORK}/*.zip')) if is_review(z)]
if not REVIEWS and IN_COLAB:
    from google.colab import files
    print('ارفع ملفات المراجعة (ZIP من أداة التعليم)')
    REVIEWS = [os.path.abspath(n) for n in files.upload()]
print('review files:', [os.path.basename(r) for r in REVIEWS])
"""),
code("""# keep only fully reviewed smiles, split them (per photo for internet photos, per patient for clinic photos)
import shutil
items, why = V.read_reviewed(REVIEWS)
print('smiles kept:', len(items), '| skipped:', dict(why))
shutil.rmtree('smiles', ignore_errors=True)
print('split:', V.write_split(items, 'smiles'))
SM = os.path.abspath('smiles')
sm_train, sm_val, sm_test = V.items_of(SM, 'train'), V.items_of(SM, 'val'), V.items_of(SM, 'test')
v2_train, v2_val, v2_test = V.items_of(V2, 'train'), V.items_of(V2, 'val'), V.items_of(V2, 'test')
print('v2:', len(v2_train), len(v2_val), len(v2_test), '| smiles:', len(sm_train), len(sm_val), len(sm_test))
if len(sm_test) < 5:
    print('تنبيه: صور اختبار الابتسامات قليلة جداً، فرقم الابتسامات تقريبي. راجع مزيداً من المسودات.')
"""),
code("""import random, time, copy
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)
IMG, NUM_CLASSES, BATCH, FREEZE_EPOCHS = 384, 7, 16, 3
LR_ENC, LR_DEC, DEC = 1e-4, 5e-4, (128, 64, 32, 16, 16)
SWAP = [0, 6, 5, 4, 3, 2, 1]

geo_color = A.Compose([
    A.Affine(rotate=(-20, 20), scale=(0.85, 1.15), translate_percent=(-0.06, 0.06), p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.RandomGamma(gamma_limit=(70, 140), p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.RGBShift(r_shift_limit=15, g_shift_limit=12, b_shift_limit=20, p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.25),
    A.CLAHE(p=0.2),
])
to_tensor = A.Compose([A.Normalize(), ToTensorV2()])
RNG = np.random.default_rng(SEED)

def degrade(img):
    h, w = img.shape[:2]
    if random.random() < 0.5:
        _, enc = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 90)])
        img = cv2.imdecode(enc, 1)[:, :, ::-1]
    if random.random() < 0.3:
        img = np.clip(img + np.random.normal(0, random.uniform(2, 8), img.shape), 0, 255)
    if random.random() < 0.3:
        g = np.linspace(random.uniform(0.6, 1.0), random.uniform(1.0, 1.3), w)
        if random.random() < 0.5: g = g[::-1]
        img = np.clip(img * g[None, :, None], 0, 255)
    return np.ascontiguousarray(img.astype(np.uint8))

def flip_swap(img, msk):
    img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
    tooth = (msk > 0) & (msk < NUM_CLASSES)                      # 255 (ignore) stays 255
    return img, np.where(tooth, NUM_CLASSES - msk, msk).astype(np.uint8)

class TeethDS(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        img, msk = V.load_pair(*self.items[i])
        if self.train:
            if random.random() < P_GAP: img, msk = V.gap_augment(img, msk, np.random.default_rng(random.getrandbits(32)))
            if random.random() < 0.5: img, msk = flip_swap(img, msk)
            o = geo_color(image=img, mask=msk); img, msk = o['image'], o['mask']
            img = degrade(img)
        o = to_tensor(image=img, mask=msk)
        return o['image'], o['mask'].long()

# smiles are repeated so they weigh about as much as the retractor photos in every epoch
REPEAT = max(1, min(8, round(len(v2_train) / max(1, len(sm_train))))) if sm_train else 0
train_ds = ConcatDataset([TeethDS(v2_train, True)] + [TeethDS(sm_train, True)] * REPEAT)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_v2_dl = DataLoader(TeethDS(v2_val, False), batch_size=BATCH, num_workers=2)
val_sm_dl = DataLoader(TeethDS(sm_val, False), batch_size=BATCH, num_workers=2) if sm_val else None
print('training samples per epoch:', len(train_ds), '(smiles repeated x%d)' % REPEAT)
"""),
code("""def build_model(pretrained=True):
    return smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet' if pretrained else None,
                    in_channels=3, classes=NUM_CLASSES, decoder_channels=DEC).to(DEVICE)

# 255 = ignore: regions a reviewer marked as 'learn nothing here' (an edge canine that could not be outlined)
dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True, ignore_index=255)
ce_loss = nn.CrossEntropyLoss(ignore_index=255)
def criterion(lg, y): return dice_loss(lg, y) + ce_loss(lg, y)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    inter = np.zeros(NUM_CLASSES); ps = np.zeros(NUM_CLASSES); gs = np.zeros(NUM_CLASSES)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE); p = model(x).argmax(1); valid = y != 255
        for c in range(1, NUM_CLASSES):
            pc, yc = (p == c) & valid, (y == c)
            inter[c] += (pc & yc).sum().item(); ps[c] += pc.sum().item(); gs[c] += yc.sum().item()
    return float((2 * inter / np.maximum(ps + gs, 1))[1:].mean())

def val_score(model):
    a = evaluate(model, val_v2_dl)
    b = evaluate(model, val_sm_dl) if val_sm_dl else a
    return (a + b) / 2, a, b          # both kinds of photo count equally when choosing the best epoch

def train_model(model, epochs, ckpt):
    enc_ids = {id(p) for p in model.encoder.parameters()}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': LR_ENC}, {'params': rest, 'lr': LR_DEC}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = DEVICE == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best = -1.0
    for ep in range(epochs):
        freeze = ep < FREEZE_EPOCHS
        for p in model.encoder.parameters(): p.requires_grad = not freeze
        model.train()
        if freeze: model.encoder.eval()
        t0, run, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                lg = model(x)
            loss = criterion(lg.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        m, a, b = val_score(model); tag = ''
        if m > best:
            best = m; torch.save(model.state_dict(), ckpt); tag = '  <- best'
        print(f'epoch {ep+1:02d}/{epochs} loss {run/n:.4f} | val retractor {a:.4f} smiles {b:.4f}{tag} | {time.time()-t0:.0f}s')
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return best

torch.manual_seed(SEED)
model = build_model()
best = train_model(model, EPOCHS, 'best_v3.pt')
print('best validation score', round(best, 4))
"""),
code("""# v2 versus v3 on the same test photos and the same measures
@torch.no_grad()
def v3_predict(img):
    model.eval()
    x = to_tensor(image=img)['image'].unsqueeze(0).to(DEVICE)
    return model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
v2_predict = V.onnx_predictor('teeth_unet_v2.onnx')

rows = {}
for name, fn in (('v2', v2_predict), ('v3', v3_predict)):
    r = {}
    r['retractor_test_dice'], r['retractor_per_tooth'] = V.dice_of(fn, v2_test)
    if sm_test: r['smile_test_dice'], r['smile_per_tooth'] = V.dice_of(fn, sm_test)
    r['gap_hallucination'], r['gap_trials'] = V.hallucination_rate(fn, v2_test + v2_val + sm_test)
    rows[name] = r
print('                     v2       v3')
for k in ('retractor_test_dice', 'smile_test_dice', 'gap_hallucination'):
    if k in rows['v2']: print(f'{k:20s} {rows["v2"][k]:.3f}    {rows["v3"][k]:.3f}')
a, b = rows['v2'], rows['v3']
adopt = (b['retractor_test_dice'] >= a['retractor_test_dice'] - 0.01
         and b['gap_hallucination'] <= a['gap_hallucination']
         and (('smile_test_dice' not in a) or b['smile_test_dice'] >= a['smile_test_dice']))
print('\\nالقرار:', 'اعتماد النسخة الثالثة' if adopt else 'إبقاء النسخة الثانية (الثالثة لم تتفوق على كل المقاييس)')
"""),
code("""# export (same input/output as v2, so the app and the server take it without code changes)
import onnxruntime as ort, json
m_cpu = model.to('cpu').eval(); dummy = torch.randn(1, 3, IMG, IMG)
try:
    torch.onnx.export(m_cpu, dummy, 'teeth_unet_v3.onnx', input_names=['image'], output_names=['logits'], opset_version=17, dynamo=False)
except TypeError:
    torch.onnx.export(m_cpu, dummy, 'teeth_unet_v3.onnx', input_names=['image'], output_names=['logits'], opset_version=17)
sess = ort.InferenceSession('teeth_unet_v3.onnx', providers=['CPUExecutionProvider'])
with torch.no_grad(): ref = m_cpu(dummy).numpy()
diff = float(np.abs(ref - sess.run(None, {'image': dummy.numpy()})[0]).max())
print(f'ONNX {os.path.getsize("teeth_unet_v3.onnx")/1e6:.1f} MB, max difference vs PyTorch {diff:.2e}')
p3 = V.onnx_predictor('teeth_unet_v3.onnx')
selftest = V.selftest_counts(p3, 'selftest.jpg') if os.path.exists('selftest.jpg') else None
json.dump(selftest, open('selftest_v3.json', 'w'))
report = dict(adopt=adopt, compare=rows, onnx_max_diff=diff, smiles=dict(kept=len(items), skipped=dict(why), train=len(sm_train), val=len(sm_val), test=len(sm_test)),
              epochs=EPOCHS, p_gap=P_GAP, repeat=REPEAT, device=DEVICE, torch=torch.__version__, selftest=selftest)
json.dump(report, open('report_v3.json', 'w'), indent=1, ensure_ascii=False, default=float)
with zipfile.ZipFile('v3-results.zip', 'w') as z:
    for f in ('teeth_unet_v3.onnx', 'selftest_v3.json', 'report_v3.json'): z.write(f)
if IN_COLAB and USE_DRIVE: shutil.copy('v3-results.zip', WORK)
if IN_COLAB:
    from google.colab import files
    files.download('v3-results.zip')
print(json.dumps(report, indent=1, ensure_ascii=False, default=float)[:2000])
"""),
md("""## ما الذي ترسله لي؟
ملف `v3-results.zip`، وفيه النموذج وتقرير المقارنة وأرقام الفحص الذاتي للتطبيق. ونسخة منه تُحفظ في Drive أيضاً.

إذا ظهر خطأ، أرسل نصه كاملاً.
"""),
]
nb = dict(nbformat=4, nbformat_minor=5, cells=cells,
          metadata=dict(kernelspec=dict(name='python3', display_name='Python 3'), accelerator='GPU', colab=dict(provenance=[], gpuType='T4')))
out = os.path.join(HERE, 'train_v3.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(out, os.path.getsize(out))
