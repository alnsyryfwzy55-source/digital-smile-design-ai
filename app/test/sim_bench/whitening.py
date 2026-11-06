"""Whitening naturalness on the full photos (run after prep.py and run.js): brightness gain, enamel texture kept,
true loss to white (all three channels clipped), and colour cast in highlights (should stay warm-neutral)."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, cv2
cases = json.load(open('cases.json'))
def load(p, c): return np.fromfile(p, np.uint8).reshape(c['h'], c['w'], 4)[..., :3]
def lab(x): return (cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32) * np.float32([100/255, 1, 1]) - np.float32([0, 128, 128]))
seen = set(); rows = []
for c in cases:
    key = c['name'].rsplit('_', 1)[0]
    if key in seen: continue
    seen.add(key)
    I, R = load(f'in/{c["k"]}_full.rgba', c), load(f'out/{c["k"]}_full.rgba', c)
    m = cv2.erode((np.fromfile(f'in/{c["k"]}_mask.u8', np.uint8).reshape(c['h'], c['w']) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    LI, LR = lab(I), lab(R)
    rows.append(dict(dL=float(LR[m][:, 0].mean() - LI[m][:, 0].mean()), tex=float(LR[m][:, 0].std() / max(0.5, LI[m][:, 0].std())),
                     clipR=float((R[m].min(1) >= 250).mean()), clipI=float((I[m].min(1) >= 250).mean()), LIm=float(LI[m][:, 0].mean())))
A = lambda k: np.median([r[k] for r in rows])
print(f"brightness +{A('dL'):.1f} L* | texture kept {A('tex'):.2f} | lost to white: median {100*A('clipR'):.2f}% worst {100*max(r['clipR'] for r in rows):.1f}% (the photo itself: worst {100*max(r['clipI'] for r in rows):.1f}%)")
# colour cast in highlights: pixels already bright before (L* > 85) should stay neutral, not turn blue/green
casts = []
seen = set()
for c in cases:
    key = c['name'].rsplit('_', 1)[0]
    if key in seen: continue
    seen.add(key)
    I, R = load(f'in/{c["k"]}_full.rgba', c), load(f'out/{c["k"]}_full.rgba', c)
    m = cv2.erode((np.fromfile(f'in/{c["k"]}_mask.u8', np.uint8).reshape(c['h'], c['w']) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    LI, LR = lab(I), lab(R); hl = m & (LI[..., 0] > 85)
    if hl.sum() > 50:
        casts.append((float(LR[hl][:, 1].mean()), float(LR[hl][:, 2].mean()), float(LI[hl][:, 1].mean()), float(LI[hl][:, 2].mean())))
cs = np.array(casts)
print(f"highlights ({len(cs)} photos): a* {np.median(cs[:,2]):.1f} -> {np.median(cs[:,0]):.1f} (green if < 0) | b* {np.median(cs[:,3]):.1f} -> {np.median(cs[:,1]):.1f} (blue if < 0) | photos with a cold cast (b* < 0): {(cs[:,1] < 0).sum()}")
