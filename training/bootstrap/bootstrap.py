"""Build DRAFT labels for smile photos from the internet, to be corrected in the annotation tool (docs/annotate/).

    python bootstrap.py --pexels-key KEY --n 1500 --out drafts      # download + filter + label
    python bootstrap.py --photos my_photos/ --out drafts            # label photos you already have
    python bootstrap.py --photos my_photos/ --out drafts --fast     # no face-parsing / SAM 2 (pipeline check)

Per photo:
  1. face landmarks (MediaPipe) -> quality filter (one frontal face, open smile, teeth visible)
  2. mouth crop aligned to the eye line (same crop as the app and the annotation tool)
  3. lips + mouth opening: face-parsing SegFormer (CelebAMask-HQ classes 10 mouth, 11 upper lip, 12 lower lip)
  4. teeth: our U-Net (docs/teeth_unet_v2.onnx) -> core/dsd/postprocess (no overlap, one region per tooth,
     FDI order, nothing outside the mouth) -> each tooth refined by SAM 2 (box prompt), kept only where it agrees
  5. polygons + priority for review (teeth where model and SAM 2 disagree are reviewed first)

Output: drafts/drafts.zip (images/<id>.jpg, meta/<id>.json in the annotation-tool format, manifest.csv,
SOURCES.csv with photographer + URL for every photo) and drafts/report.json.
"""
import argparse, csv, glob, hashlib, io, json, math, os, random, sys, time, zipfile
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, 'core'), os.path.join(HERE, '..', '..', 'core')):
    if os.path.isdir(os.path.join(p, 'dsd')):
        sys.path.insert(0, p); break
from dsd.postprocess import teeth_from_probs, softmax, appearance_check, vertical_check   # noqa: E402
from dsd.constants import TEETH                                 # noqa: E402

REPO_RAW = 'https://raw.githubusercontent.com/alnsyryfwzy55-source/digital-smile-design-ai/main/docs/'
MODELS = {'teeth_unet_v2.onnx': [REPO_RAW + 'teeth_unet_v2.onnx'], 'face_landmarker.task': [REPO_RAW + 'vendor/face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task']}
IRIS_A, IRIS_B, COMM_A, COMM_B, NOSE, EDGE_A, EDGE_B = 468, 473, 61, 291, 1, 234, 454
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
QUERIES = ['big smile portrait', 'toothy smile', 'smiling man teeth', 'smiling woman teeth', 'laughing portrait',
           'smile close up', 'happy face smile', 'elderly person smiling', 'young man smiling', 'girl smiling teeth',
           'african man smiling', 'african woman smiling', 'arab man smiling', 'arab woman smiling', 'asian man smiling',
           'asian woman smiling', 'indian man smiling', 'latino smiling', 'gap teeth smile', 'crooked teeth smile',
           'missing tooth smile', 'yellow teeth smile', 'gummy smile', 'smiling businessman portrait', 'grandfather smiling',
           'broken front tooth', 'chipped tooth smile', 'braces smile', 'dentures smile']
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------------------------------- download
def pexels_download(key, n, out_dir, log):
    import requests
    os.makedirs(out_dir, exist_ok=True)
    rows, seen = [], set()
    per_query = max(10, math.ceil(n / len(QUERIES)))
    for q in QUERIES:
        got, page = 0, 1
        while got < per_query and page <= 5:
            r = requests.get('https://api.pexels.com/v1/search', headers={'Authorization': key},
                             params={'query': q, 'per_page': 80, 'page': page, 'orientation': 'portrait'}, timeout=30)
            if r.status_code == 429:
                log('rate limited, waiting 60 s'); time.sleep(60); continue
            r.raise_for_status()
            photos = r.json().get('photos', [])
            if not photos:
                break
            for ph in photos:
                if ph['id'] in seen or got >= per_query:
                    continue
                seen.add(ph['id'])
                url = ph['src'].get('large2x') or ph['src']['large']
                fn = f"px{ph['id']}.jpg"
                try:
                    img = requests.get(url, timeout=60).content
                    open(os.path.join(out_dir, fn), 'wb').write(img)
                except Exception as e:                      # one failed download never stops the run
                    log(f'download failed {fn}: {e}'); continue
                rows.append(dict(file=fn, url=ph['url'], photographer=ph.get('photographer', ''),
                                 photographer_url=ph.get('photographer_url', ''), query=q, licence='Pexels License'))
                got += 1
                time.sleep(0.15)
            page += 1
            time.sleep(0.4)
        log(f'{q}: {got}')
    with open(os.path.join(out_dir, 'SOURCES.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'url', 'photographer', 'photographer_url', 'query', 'licence'])
        w.writeheader(); w.writerows(rows)
    return rows


def _get(url, log, tries=4, **kw):
    """GET with patience: Openverse answers 504 under load and Flickr drops SSL connections now and then."""
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, timeout=kw.pop('timeout', 60) if k == 0 else 60, **kw)
            if r.status_code in (500, 502, 503, 504):
                raise requests.exceptions.HTTPError(f'HTTP {r.status_code}')
            return r
        except requests.exceptions.RequestException as e:
            if k == tries - 1:
                log(f'gave up on {url[:80]}: {e}'); return None
            time.sleep(5 * 3 ** k)                      # 5, 15, 45 s
    return None


def openverse_download(n, out_dir, log, min_width=800):
    """Openly licensed photos (Creative Commons / public domain) through the Openverse API; no key needed.
    Resumable: photos already listed in out_dir/SOURCES.csv are kept and each query only tops up what it lacks,
    so running the same command again fetches only what failed last time."""
    os.makedirs(out_dir, exist_ok=True)
    rows, seen = [], set()
    sp = os.path.join(out_dir, 'SOURCES.csv')
    if os.path.exists(sp):
        rows = [r for r in csv.DictReader(open(sp, encoding='utf-8')) if os.path.exists(os.path.join(out_dir, r['file']))]
        seen = {r['file'] for r in rows}
        log(f'resuming: {len(rows)} photos already here')
    per_query = max(1, math.ceil(n / len(QUERIES)))
    ua = {'User-Agent': 'smile-design-research/1.0 (graduation project; non-commercial)'}
    for q in QUERIES:
        got = sum(r['query'] == q for r in rows)
        for page in range(1, 13):
            if got >= per_query:
                break
            r = _get('https://api.openverse.org/v1/images/', log, headers=ua, timeout=30,
                     params={'q': q, 'page': page, 'page_size': 20, 'mature': 'false'})
            if r is None:
                break
            if r.status_code == 429:
                log('Openverse quota reached; stopping downloads here (run again later to resume)')
                return _save_sources(rows, out_dir)
            if r.status_code != 200:
                log(f'{q} page {page}: HTTP {r.status_code}'); break
            res = r.json().get('results', [])
            if not res:
                break
            for it in res:
                fn = f"ov_{it['id'][:12]}.jpg"
                if fn in seen or got >= per_query or (it.get('width') or 0) < min_width:
                    continue
                seen.add(fn)
                img = _get(it['url'], log, tries=2, headers=ua)
                if img is None or img.status_code != 200 or len(img.content) < 20_000:
                    continue
                open(os.path.join(out_dir, fn), 'wb').write(img.content)
                lic = f"CC {it.get('license', '').upper()} {it.get('license_version', '')}".strip()
                rows.append(dict(file=fn, url=it.get('foreign_landing_url') or it['url'], photographer=it.get('creator', ''),
                                 photographer_url=it.get('creator_url', ''), query=q, licence=lic))
                got += 1
            time.sleep(1.0)
        log(f'{q}: {got}')
        _save_sources(rows, out_dir)                    # keep the record even if the run is interrupted
    return _save_sources(rows, out_dir)


def _save_sources(rows, out_dir):
    with open(os.path.join(out_dir, 'SOURCES.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'url', 'photographer', 'photographer_url', 'query', 'licence'])
        w.writeheader(); w.writerows(rows)
    return rows


def ensure_models(d):
    import urllib.request
    os.makedirs(d, exist_ok=True)
    for name, url in MODELS.items():
        p = os.path.join(d, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            for u in url:                                   # first working source wins (official MediaPipe as fallback)
                try:
                    urllib.request.urlretrieve(u, p); break
                except Exception as e:
                    print('download failed', u, e)
    return {k: os.path.join(d, k) for k in MODELS}


# ------------------------------------------------------------------------------------------- models
class Face:
    def __init__(self, task):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task), num_faces=3))

    def __call__(self, rgb):
        r = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        h, w = rgb.shape[:2]
        return [np.array([[p.x * w, p.y * h] for p in f]) for f in r.face_landmarks]


class Teeth:
    def __init__(self, onnx):
        import onnxruntime as ort
        self.s = ort.InferenceSession(onnx, providers=['CPUExecutionProvider']); self.n = self.s.get_inputs()[0].name

    def probs(self, crop):
        h, w = crop.shape[:2]
        x = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        lo = self.s.run(None, {self.n: ((x - MEAN) / STD).transpose(2, 0, 1)[None]})[0][0]
        p = softmax(lo)
        return np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in p])


class Parser:
    """face-parsing SegFormer (CelebAMask-HQ); non-commercial research licence."""
    def __init__(self, device):
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        self.t = torch; self.dev = device
        self.p = SegformerImageProcessor.from_pretrained('jonathandinu/face-parsing')
        self.m = SegformerForSemanticSegmentation.from_pretrained('jonathandinu/face-parsing').to(device).eval()

    def __call__(self, face_rgb):
        from PIL import Image
        inp = self.p(images=Image.fromarray(face_rgb), return_tensors='pt').to(self.dev)
        with self.t.no_grad():
            lo = self.m(**inp).logits
        lo = self.t.nn.functional.interpolate(lo, size=face_rgb.shape[:2], mode='bilinear', align_corners=False)
        return lo.argmax(1)[0].cpu().numpy().astype(np.uint8)


class Sam:
    def __init__(self, device, model_id='facebook/sam2.1-hiera-small'):
        import torch
        from transformers import Sam2Processor, Sam2Model
        self.t = torch; self.dev = device
        self.p = Sam2Processor.from_pretrained(model_id)
        self.m = Sam2Model.from_pretrained(model_id).to(device).eval()

    def boxes(self, rgb, boxes):
        from PIL import Image
        inp = self.p(images=Image.fromarray(rgb), input_boxes=[boxes], return_tensors='pt').to(self.dev)
        with self.t.no_grad():
            out = self.m(**inp, multimask_output=False)
        masks = self.p.post_process_masks(out.pred_masks.cpu(), inp['original_sizes'])[0]
        return masks[:, 0].numpy().astype(bool), out.iou_scores.cpu().numpy().reshape(-1)


# ------------------------------------------------------------------------------------------- geometry
def load_rgb(path):
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))


def eyes(P):
    return sorted([P[IRIS_A], P[IRIS_B]], key=lambda p: p[0])


def pose(P):
    a, b = eyes(P)
    roll = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    l, r = sorted([P[EDGE_A], P[EDGE_B]], key=lambda p: p[0])
    dl, dr = abs(P[NOSE][0] - l[0]), abs(r[0] - P[NOSE][0])
    return roll, (dl - dr) / max(1, dl + dr)


def inner_poly(P):
    return np.concatenate([P[INNER_UPPER], P[INNER_LOWER[1:-1]]])


def mouth_crop(P, out_w=768, aspect=1.6, widen=1.18):
    a, b = eyes(P)
    roll = math.atan2(b[1] - a[1], b[0] - a[0])
    c = inner_poly(P).mean(0)
    wc = np.linalg.norm(P[COMM_A] - P[COMM_B]) * widen
    out_h = int(round(out_w / aspect / 8) * 8)
    s = wc / out_w
    cs, sn = math.cos(roll) * s, math.sin(roll) * s
    M = np.array([[cs, -sn, c[0] - cs * out_w / 2 + sn * out_h / 2], [sn, cs, c[1] - sn * out_w / 2 - cs * out_h / 2]])
    return M, cv2.invertAffineTransform(M), (out_w, out_h), s


def to_crop(Minv, pts):
    return np.asarray(pts, np.float64) @ Minv[:, :2].T + Minv[:, 2]


def poly_mask(poly, w, h):
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1); return m


def largest_contour(mask, eps_frac=0.006, min_pts=6):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    eps = eps_frac * cv2.arcLength(c, True)
    a = cv2.approxPolyDP(c, eps, True)
    while len(a) < min_pts and eps > 0.3:
        eps /= 2; a = cv2.approxPolyDP(c, eps, True)
    return [dict(x=float(p[0][0]) + 0.5, y=float(p[0][1]) + 0.5) for p in a]


def dhash(rgb, n=16):
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).ravel()


# ------------------------------------------------------------------------------------------- one photo
FILTERS = dict(max_roll=25.0, max_yaw=0.30, min_mouth_px=120, min_open=0.035, min_teeth_frac=0.22,
               closeup_min_px=220)
# the crop is rotated to the eye line, so a tilted head is fine up to 25 degrees; a turned face is not
# (the side teeth are foreshortened). Openverse photos are ~1024 px wide, so a portrait mouth is often
# 120-180 px: still enough for the 384 px network input.


def detect_faces(face, rgb):
    """Faces, largest mouth first. A face that fills the whole frame is often missed: retry on a padded copy."""
    faces = face(rgb)
    if not faces:
        h, w = rgb.shape[:2]; pad = int(0.35 * max(h, w))
        big = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        faces = [P - pad for P in face(big)]
    return sorted(faces, key=lambda P: -np.linalg.norm(P[COMM_A] - P[COMM_B]))


def crop_affine(c, roll, wc, out_w=768, aspect=1.6):
    out_h = int(round(out_w / aspect / 8) * 8)
    s = wc / out_w
    cs, sn = math.cos(roll) * s, math.sin(roll) * s
    M = np.array([[cs, -sn, c[0] - cs * out_w / 2 + sn * out_h / 2], [sn, cs, c[1] - sn * out_w / 2 - cs * out_h / 2]])
    return M, cv2.invertAffineTransform(M), (out_w, out_h), s


def closeup_geometry(lab, F):
    """Mouth close-ups (no face for the landmarker): place the crop from the face-parsing lips instead.
    lab: parsing labels of the whole photo (10 opening, 11 upper lip, 12 lower lip)."""
    h, w = lab.shape
    lips = np.isin(lab, (10, 11, 12))
    if (lab == 11).sum() < 0.002 * h * w or (lab == 12).sum() < 0.002 * h * w or (lab == 10).sum() < 0.003 * h * w:
        return None, 'closeup: no lips found'
    n, cc, st, _ = cv2.connectedComponentsWithStats(lips.astype(np.uint8), 8)
    lips = cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lips)
    if xs.min() <= 1 or xs.max() >= w - 2:
        return None, 'closeup: mouth cut off'
    pts = np.stack([xs, ys], 1).astype(np.float64); c0 = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - c0).T))
    ax = evec[:, 1] if evec[0, 1] >= 0 else -evec[:, 1]          # main axis, pointing right
    roll = math.atan2(ax[1], ax[0])
    if abs(math.degrees(roll)) > F['max_roll']:
        return None, 'tilted head'
    proj = (pts - c0) @ ax
    wc = float(proj.max() - proj.min())                           # corner to corner
    if wc < F['closeup_min_px']:
        return None, 'mouth too small in photo'
    up, lo = np.argwhere(lab == 11).mean(0), np.argwhere(lab == 12).mean(0)
    nrm = np.array([-ax[1], ax[0]])                               # points down in the image when roll = 0
    if (lo[::-1] - up[::-1]) @ nrm <= 0:
        return None, 'closeup: lips upside down'
    oy, ox = np.nonzero(lab == 10)
    c = np.array([ox.mean(), oy.mean()])
    return crop_affine(c, roll, wc * 1.12), 'ok'


def warp_labels(lab, M, W, H):
    return cv2.warpAffine(lab, M, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)


def process(rgb, face, teeth, parser=None, sam=None, F=FILTERS):
    faces = detect_faces(face, rgb)
    regions, yaw = {}, 0.0
    if faces:
        mode = 'face'
        P = faces[0]                                              # several faces: the largest mouth
        roll, yaw = pose(P)
        if abs(roll) > F['max_roll']:
            return None, 'tilted head'
        if abs(yaw) > F['max_yaw']:
            return None, 'face turned'
        if np.linalg.norm(P[COMM_A] - P[COMM_B]) < F['min_mouth_px']:
            return None, 'mouth too small in photo'
        M, Minv, (W, H), s = mouth_crop(P)
        lip_poly = to_crop(Minv, inner_poly(P))
        mouth = poly_mask(lip_poly, W, H)
        if parser is not None:                       # pixel-accurate lips and opening from face parsing
            xs, ys = P[:, 0], P[:, 1]
            fx0, fy0 = max(0, xs.min() - 0.25 * (xs.max() - xs.min())), max(0, ys.min() - 0.25 * (ys.max() - ys.min()))
            fx1, fy1 = min(rgb.shape[1], xs.max() + 0.25 * (xs.max() - xs.min())), min(rgb.shape[0], ys.max() + 0.15 * (ys.max() - ys.min()))
            lab = parser(rgb[int(fy0):int(fy1), int(fx0):int(fx1)])
            A = np.array([[1, 0, -int(fx0)], [0, 1, -int(fy0)]], np.float64)
            Acrop = (np.vstack([A, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2]   # crop -> face-crop pixels
            lab_c = warp_labels(lab, Acrop, W, H)
            mouth_p = (lab_c == 10).astype(np.uint8)
            if mouth_p.sum() > 0.5 * mouth.sum():                  # trust parsing when it found the opening
                mouth = mouth_p
            for k, name in ((10, 'mouth'), (11, 'upper_lip'), (12, 'lower_lip')):
                regions[name] = largest_contour(lab_c == k, 0.004)
        else:
            regions['mouth'] = [dict(x=float(x), y=float(y)) for x, y in lip_poly]
    else:
        if parser is None:
            return None, 'faces=0'
        mode = 'closeup'
        lab = parser(rgb)
        geo, why = closeup_geometry(lab, F)
        if geo is None:
            return None, why
        M, Minv, (W, H), s = geo
        roll = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        lab_c = warp_labels(lab, M, W, H)
        mouth = (lab_c == 10).astype(np.uint8)
        for k, name in ((10, 'mouth'), (11, 'upper_lip'), (12, 'lower_lip')):
            regions[name] = largest_contour(lab_c == k, 0.004)
        lip_poly = np.array([[q['x'], q['y']] for q in (regions['mouth'] or [])], np.float64).reshape(-1, 2)
    crop = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    if mouth.sum() < F['min_open'] * W * H:
        return None, 'mouth closed'
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    teethish = (hsv[..., 2] > 120) & (hsv[..., 1] < 90)
    if teethish[mouth > 0].mean() < F['min_teeth_frac']:
        return None, 'teeth not visible'
    prob = teeth.probs(crop)
    mouth_d = cv2.dilate(mouth, np.ones((5, 5), np.uint8))
    post = teeth_from_probs(prob, mouth=mouth_d)
    labels = post['labels'].copy()
    polys, status, review = {}, dict(post['status']), {}
    looks, _ = appearance_check(labels, crop)          # a dark 'tooth' is a gap the network filled in
    for t, v in looks.items():
        labels[labels == 1 + TEETH.index(t)] = 0
        status[t] = v
    placed, _ = vertical_check(labels, (mouth > 0) | (labels > 0))   # lower tooth or half a mask under an upper label
    for t in placed:
        labels[labels == 1 + TEETH.index(t)] = 0
        status[t] = 'unsure'
    present = [t for t in TEETH if (labels == 1 + TEETH.index(t)).any()]
    masks = {t: labels == 1 + TEETH.index(t) for t in present}
    if sam is not None and present:
        boxes = []
        for t in present:
            ys, xs = np.nonzero(masks[t])
            boxes.append([float(xs.min()) - 3, float(ys.min()) - 3, float(xs.max()) + 4, float(ys.max()) + 4])
        sm, sc = sam.boxes(crop, boxes)
        claim = np.zeros((len(present), H, W), bool)
        for i, t in enumerate(present):
            near = cv2.dilate(masks[t].astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
            cand = sm[i] & near & (mouth_d > 0)
            iou = (cand & masks[t]).sum() / max(1, (cand | masks[t]).sum())
            review[t] = round(float(iou), 3)
            claim[i] = cand if iou >= 0.6 else masks[t]
        # a pixel claimed by two teeth goes to the one the segmentation model prefers
        pref = np.stack([prob[1 + TEETH.index(t)] for t in present])
        pref[~claim] = -1
        win = pref.argmax(0)
        any_claim = claim.any(0)
        for i, t in enumerate(present):
            masks[t] = any_claim & (win == i)
    for t in present:
        poly = largest_contour(masks[t])
        if poly:
            polys[t] = poly
        else:
            status[t] = 'missing'
    for t in TEETH:
        if t not in polys and status.get(t) == 'present':
            status[t] = 'missing'
    # a tooth the model did not find is 'hidden' only if nothing tooth-like is visible where it should be
    if polys:
        xs_lab = [q['x'] for poly in polys.values() for q in poly]
        unit = max(8.0, (max(xs_lab) - min(xs_lab)) / max(1, len(polys)))
        cols = (teethish & (mouth > 0)).sum(0) / np.maximum(1, (mouth > 0).sum(0))   # share of the opening that looks like enamel
        def visible(x0, x1):
            x0, x1 = int(max(0, x0)), int(min(W, x1))
            return x1 > x0 and cols[x0:x1].mean() > 0.25
        left, right = min(xs_lab), max(xs_lab)
        lt = [t for t in TEETH if t in polys][0]; rt = [t for t in TEETH if t in polys][-1]
        for t in TEETH[:TEETH.index(lt)]:
            if status.get(t) == 'hidden' and visible(left - 0.9 * unit, left - 0.1 * unit):
                status[t] = 'unsure'
        for t in TEETH[TEETH.index(rt) + 1:]:
            if status.get(t) == 'hidden' and visible(right + 0.1 * unit, right + 0.9 * unit):
                status[t] = 'unsure'
    else:                                                   # enamel is visible (filter above) but no tooth recognised
        status = {t: 'unsure' for t in TEETH}
    conf = {t: round(float(v), 3) for t, v in post['confidence'].items()}
    priority = 1.0 if any(v == 'unsure' for v in status.values()) else 1.0 - min([*review.values(), *conf.values()] or [0.0])
    meta = dict(mode=mode, width=W, height=H, crop_to_image=M.tolist(), teeth=polys, status=status,
                lips=dict(poly=[dict(x=float(x), y=float(y)) for x, y in lip_poly]), regions=regions,
                draft=dict(source='bootstrap', model_conf=conf, sam_agreement=review, priority=round(priority, 3),
                           roll=round(roll, 2), yaw=round(yaw, 3)))
    return dict(crop=crop, meta=meta), 'ok'


# ------------------------------------------------------------------------------------------- main
def main(argv=None, face=None, teeth=None, parser=None, sam=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--photos'); ap.add_argument('--pexels-key'); ap.add_argument('--openverse', type=int, default=0, help='download this many CC photos from Openverse (no key)'); ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--out', default='drafts'); ap.add_argument('--fast', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='process at most this many photos')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'log.txt'), 'a', encoding='utf-8')
    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()
    t0 = time.time()
    src_dir = a.photos
    sources = {}
    if a.pexels_key:
        src_dir = os.path.join(a.out, 'downloaded')
        for r in pexels_download(a.pexels_key, a.n, src_dir, log):
            sources[r['file']] = r
    elif a.openverse:
        src_dir = os.path.join(a.out, 'downloaded')
        for r in openverse_download(a.openverse, src_dir, log):
            sources[r['file']] = r
    elif a.photos and os.path.exists(os.path.join(a.photos, 'SOURCES.csv')):
        sources = {r['file']: r for r in csv.DictReader(open(os.path.join(a.photos, 'SOURCES.csv'), encoding='utf-8'))}
    if not src_dir:
        ap.error('give --photos, --openverse N or --pexels-key')
    if face is None or teeth is None:
        mp = ensure_models(os.path.join(HERE, 'models'))
        face = face or Face(mp['face_landmarker.task'])
        teeth = teeth or Teeth(mp['teeth_unet_v2.onnx'])
    if not a.fast and (parser is None or sam is None):
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        log(f'device: {dev}')
        parser = parser or Parser(dev)
        sam = sam or Sam(dev)
    files = sorted(p for p in glob.glob(os.path.join(src_dir, '**', '*'), recursive=True)
                   if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
    if a.limit:
        files = files[:a.limit]
    zpath = os.path.join(a.out, 'drafts.zip')
    z = zipfile.ZipFile(zpath, 'w', zipfile.ZIP_STORED)
    reasons, kept, hashes, rows, modes = {}, 0, [], [], {}
    for k, f in enumerate(files):
        name = os.path.basename(f)
        try:
            rgb = load_rgb(f)
            h = dhash(rgb)
            if any((h != q).sum() <= 12 for q in hashes):
                reasons['duplicate'] = reasons.get('duplicate', 0) + 1; continue
            res, why = process(rgb, face, teeth, None if a.fast else parser, None if a.fast else sam)
        except Exception as e:
            res, why = None, f'error: {type(e).__name__}: {e}'
        if res is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        hashes.append(h)
        img_id = os.path.splitext(name)[0]
        res['meta'].update(id=img_id, source_file=name, source=sources.get(name))
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(res['crop'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        z.writestr(f'images/{img_id}.jpg', buf.tobytes())
        z.writestr(f'meta/{img_id}.json', json.dumps(res['meta'], ensure_ascii=False))
        st = res['meta']['status']
        rows.append([img_id, res['meta']['draft']['priority'], sum(v == 'present' for v in st.values()),
                     ' '.join(f'{t}:{st[t]}' for t in TEETH)])
        kept += 1
        modes[res['meta']['mode']] = modes.get(res['meta']['mode'], 0) + 1
        if k % 25 == 0:
            log(f'{k + 1}/{len(files)} kept {kept} ({time.time() - t0:.0f}s)')
    rows.sort(key=lambda r: -r[1])                       # most uncertain first
    z.writestr('manifest.csv', '﻿id,priority,teeth_found,status\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')
    if sources:
        out = io.StringIO(); w = csv.DictWriter(out, fieldnames=['file', 'url', 'photographer', 'photographer_url', 'query', 'licence'])
        w.writeheader(); w.writerows(sources.values()); z.writestr('SOURCES.csv', out.getvalue())
    z.close()
    report = dict(photos=len(files), kept=kept, kept_by_mode=modes, skipped=reasons, minutes=round((time.time() - t0) / 60, 1),
                  mode='fast' if a.fast else 'full', drafts=zpath)
    json.dump(report, open(os.path.join(a.out, 'report.json'), 'w'), indent=1, ensure_ascii=False)
    log(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
