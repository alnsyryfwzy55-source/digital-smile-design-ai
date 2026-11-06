"""Data for the v3 tooth model (numpy/OpenCV only, so it is tested without a GPU).

- read the annotation tool's exports (reviewed smiles): images/*.jpg, masks/*.png (0..6), meta/*.json
- keep only fully reviewed photos: a photo where any tooth is still 'unsure' would teach 'background' on a tooth
- split by patient (clinic ids C001_...) or per photo (internet ids), stable across runs
- gap augmentation: paint a dark gap over one tooth and erase its label, so the network learns NOT to paint a
  tooth into an empty space from context (the failure the appearance check catches today)
- metrics: Dice per tooth, and the hallucination rate (blacked-out lateral still predicted as a tooth)
"""
import hashlib, io, json, os, re, zipfile
from collections import Counter

import numpy as np
import cv2

TEETH = ['13', '12', '11', '21', '22', '23']
IMG = 384
IGNORE = 255                     # 'learn nothing here' (e.g. an edge canine the reviewer could not outline)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def case_of(img_id):
    """Clinic photos C001_smile / C001_rest share a patient; internet photos (ov_..., px_...) are each their own."""
    m = re.match(r'^([A-Za-z]{0,3}\d{2,})_', img_id)
    return m.group(1) if m else img_id


def split_of(key, val=0.15, test=0.15):
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if u < test else 'val' if u < test + val else 'train'


def read_reviewed(zips):
    """-> list of dict(id, img RGB uint8, mask uint8 0..6, status), Counter of skip reasons."""
    out, why, seen = [], Counter(), set()
    for zp in zips:
        z = zipfile.ZipFile(zp); names = set(z.namelist())
        for n in sorted(names):
            if not (n.startswith('images/') and n.endswith('.jpg')):
                continue
            i = n[7:-4]
            if i in seen:
                why['duplicate'] += 1; continue
            if f'masks/{i}.png' not in names:
                why['not reviewed (no mask)'] += 1; continue
            meta = json.loads(z.read(f'meta/{i}.json')) if f'meta/{i}.json' in names else {}
            status = meta.get('status') or {}
            if any(v == 'unsure' for v in status.values()):
                why['a tooth still unsure'] += 1; continue
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            m = cv2.imdecode(np.frombuffer(z.read(f'masks/{i}.png'), np.uint8), cv2.IMREAD_UNCHANGED)
            if m is not None and m.ndim == 3:
                m = m[..., 0]
            if img is None or m is None or img.shape[:2] != m.shape[:2]:
                why['unreadable or size mismatch'] += 1; continue
            if ((m > 6) & (m != IGNORE)).any():
                why['mask value not 0-6/255'] += 1; continue
            if not (m > 0).any() and not any(v == 'missing' for v in status.values()):
                why['empty mask'] += 1; continue
            seen.add(i)
            out.append(dict(id=i, img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask=m, status=status, reviewer=meta.get('reviewer') or 'human'))
    return out, why


def write_split(items, root):
    """Write reviewed smiles as root/{train,val,test}/{images,masks} (same layout as smile_teeth_v2)."""
    counts = Counter()
    for sp in ('train', 'val', 'test'):
        os.makedirs(f'{root}/{sp}/images', exist_ok=True); os.makedirs(f'{root}/{sp}/masks', exist_ok=True)
    for it in items:
        # labels checked only by an AI reviewer train the model but never judge it: test/val stay human-reviewed
        sp = 'train' if it.get('reviewer') == 'gemini' else split_of(case_of(it['id']))
        cv2.imwrite(f'{root}/{sp}/images/{it["id"]}.jpg', cv2.cvtColor(it['img'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f'{root}/{sp}/masks/{it["id"]}.png', it['mask'])
        counts[sp] += 1
    return dict(counts)


def gap_augment(img, msk, rng, classes=(1, 2, 5, 6, 3, 4), min_px=150):
    """Replace one visible tooth by a dark gap (lost tooth) and set its label to background."""
    present = [c for c in classes if (msk == c).sum() >= min_px]
    if not present:
        return img, msk
    c = present[int(rng.integers(len(present)))]
    reg = cv2.dilate((msk == c).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    reg &= (msk == 0) | (msk == c)                       # never paint over a neighbour
    h, w = msk.shape
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    base = rng.uniform(18, 75)                           # open gap: dark red mouth cavity
    gb = rng.uniform(0.45, 0.8)
    tint = np.array([1.0, gb, gb * rng.uniform(0.95, 1.15)], np.float32)
    fill = np.clip((base * (0.8 + 0.4 * ys)) * tint + rng.normal(0, 4, (h, w, 3)), 0, 255)
    if rng.random() < 0.35:                              # healed ridge: gum on top, the cavity shows below it
        ry = np.nonzero(reg.any(1))[0]; y0, y1 = ry.min(), ry.max()
        cut = y0 + rng.uniform(0.25, 0.6) * (y1 - y0)
        gum = np.array([175, 95, 105], np.float32) * rng.uniform(0.7, 1.1)
        yy = np.arange(h, dtype=np.float32)[:, None, None]
        t = np.clip((yy - cut) / max(3.0, 0.12 * (y1 - y0)) + 0.5, 0, 1)          # soft, rounded ridge edge
        gfill = np.clip(gum * (1.0 - 0.15 * (yy - y0) / max(1, y1 - y0)) + rng.normal(0, 5, (h, w, 3)), 0, 255)
        fill = gfill * (1 - t) + fill * t
    soft = cv2.GaussianBlur(reg.astype(np.float32), (5, 5), 0)[..., None]
    out = img.astype(np.float32) * (1 - soft) + fill * soft
    m2 = msk.copy(); m2[reg] = 0
    return np.ascontiguousarray(out.astype(np.uint8)), m2


def load_pair(ip, mp):
    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
    msk = cv2.imread(mp, 0)
    return (cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA),
            cv2.resize(msk, (IMG, IMG), interpolation=cv2.INTER_NEAREST))


def items_of(root, split):
    import glob
    return [(p, p.replace('/images/', '/masks/')[:-4] + '.png') for p in sorted(glob.glob(f'{root}/{split}/images/*.jpg'))]


def dice_of(predict, items):
    """predict(img_u8 384x384x3) -> labels 384x384. Mean Dice over the 6 teeth (pooled over the set)."""
    inter = np.zeros(7); ps = np.zeros(7); gs = np.zeros(7)
    for ip, mp in items:
        img, msk = load_pair(ip, mp); p = predict(img); valid = msk != IGNORE
        for c in range(1, 7):
            pc, yc = (p == c) & valid, msk == c
            inter[c] += (pc & yc).sum(); ps[c] += pc.sum(); gs[c] += yc.sum()
    d = 2 * inter / np.maximum(ps + gs, 1)
    return float(d[1:].mean()), [round(float(x), 3) for x in d[1:]]


def hallucination_rate(predict, items, classes=(2, 5), min_px=200):
    """Black out a lateral incisor's box; a model that still labels >= 30% of it as that tooth hallucinates."""
    bad = n = 0
    for ip, mp in items:
        img, msk = load_pair(ip, mp)
        for c in classes:
            m = msk == c
            if m.sum() < min_px:
                continue
            ys, xs = np.nonzero(m); g = img.copy(); g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = (40, 25, 25)
            p = predict(g); n += 1
            bad += (p[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == c).sum() >= 0.3 * m.sum()
    return (bad / n if n else float('nan')), n


def onnx_predictor(path):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider']); name = s.get_inputs()[0].name
    def predict(img):
        x = ((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
        return s.run(None, {name: x})[0][0].argmax(0).astype(np.uint8)
    return predict


def selftest_counts(predict, jpg_path):
    """The app checks a new model on docs/selftest.jpg against these per-class pixel counts (selftest.json)."""
    img = cv2.cvtColor(cv2.imread(jpg_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    p = predict(img)
    return [int((p == c).sum()) for c in range(7)]
