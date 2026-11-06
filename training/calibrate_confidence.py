"""Calibrate the per-tooth confidence threshold under which the doctor is asked to confirm (CONFIRM_BELOW).

    python training/calibrate_confidence.py [--model docs/teeth_unet_v3.onnx]

Runs the model on the labelled test/val photos and 7 hard variants of each (blur, dark, JPEG 12, rotation, upper lip
covering a third of the teeth, yellow cast), marks each detected tooth right (IoU >= 0.5 with the label) or wrong,
and prints, per threshold, how many wrong teeth are caught and how many correct teeth the doctor is asked about.
"""
import argparse, os, sys, glob, zipfile, tempfile, json, numpy as np, cv2, onnxruntime as ort
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(R, 'core'))
from dsd.postprocess import teeth_from_probs, softmax, appearance_check
from dsd.constants import TEETH
ap = argparse.ArgumentParser(); ap.add_argument('--model', default=os.path.join(R, 'docs', 'teeth_unet_v2.onnx'))
ap.add_argument('--data', default=os.path.join(R, 'data', 'smile_teeth_v2.zip')); A = ap.parse_args()
tmp = tempfile.mkdtemp(); zipfile.ZipFile(A.data).extractall(tmp, [n for n in zipfile.ZipFile(A.data).namelist() if '/test/' in n or '/val/' in n])
s = ort.InferenceSession(A.model, providers=['CPUExecutionProvider']); nm = s.get_inputs()[0].name
M, SD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
probs = lambda rgb: softmax(s.run(None, {nm: ((rgb.astype(np.float32) / 255 - M) / SD).transpose(2, 0, 1)[None]})[0][0])
rng = np.random.default_rng(0)
def variants(rgb, gt):
    yield 'orig', rgb, gt
    yield 'blur', cv2.GaussianBlur(rgb, (0, 0), 3), gt
    yield 'dark', np.clip(rgb * 0.45, 0, 255).astype(np.uint8), gt
    _, e = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 12]); yield 'jpeg12', cv2.imdecode(e, 1), gt
    Rm = cv2.getRotationMatrix2D((192, 192), 18, 1.0)
    yield 'rot18', cv2.warpAffine(rgb, Rm, (384, 384), borderMode=cv2.BORDER_REFLECT), cv2.warpAffine(gt, Rm, (384, 384), flags=cv2.INTER_NEAREST)
    lip = rgb.copy(); ys = np.nonzero((gt > 0).any(1))[0]
    if len(ys): lip[: ys.min() + int(0.35 * (ys.max() - ys.min()))] = (150, 70, 80)   # upper lip covering a third of the teeth
    yield 'lipcover', lip, gt
    cast = np.clip(rgb * np.array([1.15, 1.0, 0.7]), 0, 255).astype(np.uint8); yield 'yellow', cast, gt
rows = []
for f in sorted(glob.glob(f'{tmp}/*/*/images/*.jpg')):
    rgb = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (384, 384), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(cv2.imread(f.replace('images', 'masks').replace('.jpg', '.png'), 0), (384, 384), interpolation=cv2.INTER_NEAREST)
    for name, im, g in variants(rgb, gt):
        pr = probs(im); r = teeth_from_probs(pr); lab = r['labels']; mx = pr.max(0)
        look, _ = appearance_check(lab, im)
        for i, t in enumerate(TEETH):
            pm, gm = lab == 1 + i, g == 1 + i
            if not pm.any(): continue
            iou = (pm & gm).sum() / max(1, (pm | gm).sum())
            rows.append(dict(img=f.split('/')[-1], var=name, t=t, conf=r['confidence'].get(t, 0.0), jsconf=float(mx[pm].mean()), iou=float(iou), caught=t in look))
wrong = [r for r in rows if r['iou'] < 0.5]; ok = [r for r in rows if r['iou'] >= 0.5]; clean = [r for r in ok if r['var'] == 'orig']
print(f'{len(rows)} detected teeth, {len(wrong)} wrong')
print('threshold | wrong caught (server / app) | correct asked: clean photos / all variants')
for t in (0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96):
    cs = sum(r['conf'] < t or r['caught'] for r in wrong); cj = sum(r['jsconf'] < t or r['caught'] for r in wrong)
    print(f'  {t:.2f}    | {cs}/{len(wrong)} / {cj}/{len(wrong)} | {sum(r["conf"] < t for r in clean)}/{len(clean)} / {sum(r["conf"] < t for r in ok) / max(1, len(ok)):.3f}')
