"""Realism benchmark for the simulation (rebuilding a missing tooth), step 1 of 3.

    python app/test/sim_bench/prep.py && node app/test/sim_bench/run.js && python app/test/sim_bench/metrics.py

Labelled test photos; one tooth (12, 22 or 11) is erased with a realistic gap (dark cavity or healed gum, only where
the tooth was). The simulation then rebuilds it, and metrics.py compares the rebuilt tooth with the real one treated
the same way (full photo through the same simulation): colour, texture, edge visibility, black remnants, and seams
the simulation adds inside the teeth. Real masks are used, so this measures the painting, not the segmentation.
"""
import sys, glob, zipfile, tempfile, json, os, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'training', 'v3')); import v3data as V
os.chdir(HERE)
TEETH = ['13', '12', '11', '21', '22', '23']
tmp = tempfile.mkdtemp(); zipfile.ZipFile(os.path.join(ROOT, 'data', 'smile_teeth_v2.zip')).extractall(tmp)
cases = []
os.makedirs('in', exist_ok=True)
rng = np.random.default_rng(1)
for f in sorted(glob.glob(f'{tmp}/smile_teeth_v2/test/images/*.jpg')) + sorted(glob.glob(f'{tmp}/smile_teeth_v2/val/images/*.jpg')):
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); m = cv2.imread(f.replace('images', 'masks').replace('.jpg', '.png'), 0)
    W = 640; H = int(round(img.shape[0] * W / img.shape[1]))
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA); m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if any((m == c).sum() < 300 for c in range(1, 7)):
        continue
    for t in ('12', '22', '11'):
        c = 1 + TEETH.index(t)
        g, m2 = V.gap_augment(img, m, np.random.default_rng(int(rng.integers(1 << 30))), classes=(c,))
        k = len(cases); name = f'{os.path.basename(f)[:-4]}_{t}'
        boxes = {}
        for i, tt in enumerate(TEETH):
            ys, xs = np.nonzero(m == 1 + i); boxes[tt] = dict(x0=float(xs.min()), x1=float(xs.max() + 1), y0=float(ys.min()), y1=float(ys.max() + 1))
        for tag, im in (('full', img), ('gap', g)):
            np.dstack([im, np.full((H, W), 255, np.uint8)]).astype(np.uint8).tofile(f'in/{k}_{tag}.rgba')
        m.astype(np.uint8).tofile(f'in/{k}_mask.u8')
        cases.append(dict(k=k, name=name, t=t, w=W, h=H, boxes=boxes))
json.dump(cases, open('cases.json', 'w'))
print(len(cases), 'cases')
