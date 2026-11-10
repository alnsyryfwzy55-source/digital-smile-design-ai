"""Real-tooth bank: anatomical shape and enamel texture for the prepaint (experiment v2).

The v1 prepaint painted flat, identical blocks; the generator kept them, so results looked like plastic.
Here every designed tooth is drawn from a REAL tooth of the same FDI class, taken from one photo of the
labelled set smile_teeth_v2 (Roboflow Universe, CC BY 4.0, see data/README.md), so the six teeth share one
lighting and one shade, keep their own outline (canine pointed, lateral smaller and rounder) and real texture.
The patient's own tooth colour then sets the shade (with moderate whitening).

    bank = build_bank('data/smile_teeth_v2.zip')       # list of sets, each {t: (rgb, alpha)} + metadata
    s = pick_set(bank, design)                          # the set whose width proportions are closest to the design
    paint_teeth(pre, design, s, target_lab)             # in place, returns the per-tooth shape masks
"""
import io, math, zipfile
import numpy as np
import cv2

TEETH = ['13', '12', '11', '21', '22', '23']


def _component(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m.astype(np.uint8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def build_bank(zip_path, min_side=500, max_sets=60):
    """Sets of six real teeth from photos that show all six cleanly at good resolution."""
    z = zipfile.ZipFile(zip_path)
    names = sorted(n for n in z.namelist() if '/masks/' in n and n.endswith('.png'))
    bank = []
    for mn in names:
        im_n = mn.replace('/masks/', '/images/').replace('.png', '.jpg')
        if im_n not in z.namelist():
            continue
        mask = cv2.imdecode(np.frombuffer(z.read(mn), np.uint8), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(im_n), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if mask is None or rgb is None or min(rgb.shape[:2]) < min_side or set(range(1, 7)) - set(np.unique(mask).tolist()):
            continue
        teeth, ok = {}, True
        for i, t in enumerate(TEETH):
            m = _component(mask == i + 1)
            ys, xs = np.nonzero(m)
            if len(xs) < 400:
                ok = False; break
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            a = cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (3, 3), 0)       # soft 1-px edge
            teeth[t] = dict(rgb=rgb[y0:y1, x0:x1].copy(), alpha=a, w=float(x1 - x0), h=float(y1 - y0),
                            cx=float((x0 + x1) / 2))
        if not ok:
            continue
        # order and plausibility: left->right, laterals narrower than centrals, no very dark (decayed/stained) tooth
        cxs = [teeth[t]['cx'] for t in TEETH]
        wc = (teeth['11']['w'] + teeth['21']['w']) / 2
        lab = {t: cv2.cvtColor(teeth[t]['rgb'], cv2.COLOR_RGB2LAB)[teeth[t]['alpha'] > 0.5].astype(np.float32) for t in TEETH}
        L = {t: float(lab[t][:, 0].mean()) for t in TEETH}
        if cxs != sorted(cxs) or not all(0.45 < teeth[t]['w'] / wc < 0.95 for t in ('12', '22')) \
                or min(L.values()) < 0.72 * max(L.values()) or max(abs(teeth['11']['w'] - teeth['21']['w']) / wc, 0) > 0.18:
            continue
        mean_lab = np.concatenate(list(lab.values())).mean(0)
        bank.append(dict(id=mn.split('/')[-1][:-4], teeth=teeth, wc=wc, lab=mean_lab,
                         ratio={t: teeth[t]['w'] / wc for t in TEETH},
                         wl={t: teeth[t]['w'] / teeth[t]['h'] for t in TEETH}))
    bank.sort(key=lambda s: -min(min(s['teeth'][t]['rgb'].shape[:2]) for t in TEETH))   # sharpest first
    return bank[:max_sets]


def pick_set(bank, design):
    """Closest width proportions and width/length ratios to the design (log-space)."""
    wc = np.mean([design[t]['x1'] - design[t]['x0'] for t in ('11', '21') if t in design])
    best, cost_best = None, 1e9
    for s in bank:
        cost = 0.0
        for t, b in design.items():
            w, h = b['x1'] - b['x0'], b['y1'] - b['y0']
            cost += abs(math.log((w / wc) / s['ratio'][t])) + 0.5 * abs(math.log((w / h) / s['wl'][t]))
        if cost < cost_best:
            best, cost_best = s, cost
    return best


def target_shade(crop, ref_mask, whiten=6.0, fallback=(205.0, 129.0, 140.0)):
    """Patient's own tooth colour (OpenCV Lab, 0..255) with moderate whitening: lighter, less yellow, never paper white."""
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[ref_mask.astype(bool)]
    if len(px) < 200:
        ref = np.array(fallback, np.float32)
    else:
        lo, hi = np.percentile(px[:, 0], [30, 95])                    # enamel, not shadows or speculars
        px = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
        ref = px.mean(0)
    L = min(ref[0] + whiten, 228.0)                                    # ~ L* 89 at most
    a = 128 + 0.7 * (ref[1] - 128)
    b = 128 + 0.75 * (ref[2] - 128)
    b = max(b, 133.0)                                                  # keep a natural warm tint (b* >= ~5)
    return np.array([L, min(a, 131.0), b], np.float32)


def paint_teeth(pre, design, s, target_lab, poly_fn=None):
    """Draw the set's real teeth into their design boxes (in place). Returns {t: shape mask uint8}.
    Each tooth keeps its own outline and texture; the set's colour is moved to target_lab as a whole
    (one transform for all six, so their relative shading stays natural)."""
    h, w = pre.shape[:2]
    shapes = {}
    for t, b in design.items():
        src = s['teeth'][t]
        x0, y0 = int(round(b['x0'])), int(round(b['y0']))
        tw, th = max(4, int(round(b['x1'] - b['x0']))), max(4, int(round(b['y1'] - b['y0'])))
        rgb = cv2.resize(src['rgb'], (tw, th), interpolation=cv2.INTER_AREA if tw < src['rgb'].shape[1] else cv2.INTER_CUBIC)
        a = cv2.resize(src['alpha'], (tw, th), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        own = lab[a > 0.5].mean(0) if (a > 0.5).sum() > 20 else s['lab']
        lab = lab + target_lab - (0.4 * s['lab'] + 0.6 * own)        # mostly per tooth, so the six match each other
        rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + tw), min(h, y0 + th)
        if xb <= xa or yb <= ya:
            continue
        sx, sy = xa - x0, ya - y0
        aa = a[sy:sy + yb - ya, sx:sx + xb - xa][..., None]
        pre[ya:yb, xa:xb] = pre[ya:yb, xa:xb] * (1 - aa) + rgb[sy:sy + yb - ya, sx:sx + xb - xa] * aa
        m = np.zeros((h, w), np.uint8)
        m[ya:yb, xa:xb] = (aa[..., 0] > 0.5)
        shapes[t] = m
    return shapes
