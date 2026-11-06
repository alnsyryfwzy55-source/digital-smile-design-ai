"""step 3 of 3: realism metrics (see prep.py)."""
import json, os, sys, numpy as np, cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
TEETH = ['13', '12', '11', '21', '22', '23']
cases = json.load(open('cases.json'))
def load(p, c): return np.fromfile(p, np.uint8).reshape(c['h'], c['w'], 4)[..., :3]
def lab(x): return (cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32) * np.float32([100 / 255, 1, 1]) - np.float32([0, 128, 128])).astype(np.float32)
def ssim(a, b, m):
    a, b = a[m], b[m]; C1, C2 = (0.01 * 100) ** 2, (0.03 * 100) ** 2
    ma, mb = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a - ma) * (b - mb)).mean()
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
rows = []
for c in cases:
    mask = np.fromfile(f'in/{c["k"]}_mask.u8', np.uint8).reshape(c['h'], c['w']) == 1 + TEETH.index(c['t'])
    k3 = np.ones((7, 7), np.uint8)
    core = cv2.erode(mask.astype(np.uint8), k3) > 0
    ring = (cv2.dilate(mask.astype(np.uint8), k3) > 0) & ~core
    R, G = lab(load(f'out/{c["k"]}_full.rgba', c)), lab(load(f'out/{c["k"]}_gap.rgba', c))
    gr = lambda L: np.hypot(cv2.Sobel(np.ascontiguousarray(L), cv2.CV_32F, 1, 0), cv2.Sobel(np.ascontiguousarray(L), cv2.CV_32F, 0, 1))
    dE = float(np.linalg.norm(G[core].mean(0) - R[core].mean(0)))
    tex = float(G[core][:, 0].std() / max(0.5, R[core][:, 0].std()))
    edge = float(gr(G[..., 0])[ring].mean() / max(0.5, gr(R[..., 0])[ring].mean()))
    # how much of the real tooth's area the painted tooth covers (tooth-coloured = close to the full result's colour)
    covered = float((np.linalg.norm(G[core] - R[core].mean(0), axis=1) < 15).mean())
    # black remnants: around the rebuilt tooth, pixels still dark in the gap result but tooth/gum in the full result
    near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    remn = float(((G[..., 0] < 35) & (R[..., 0] > 50) & near).sum() / max(1, near.sum()))
    # seams: edges added inside the teeth that the photo does not have (full result vs the photo itself)
    I = lab(load(f'in/{c["k"]}_full.rgba', c)); allm = np.fromfile(f'in/{c["k"]}_mask.u8', np.uint8).reshape(c['h'], c['w'])
    inner = np.zeros_like(mask)
    for cl in range(1, 7):
        inner |= cv2.erode((allm == cl).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    seams = float(np.maximum(0, gr(R[..., 0]) - gr(I[..., 0]) - 4)[inner].mean())
    # halo: colour just outside the tooth (2-6 px) against the reference, where gum / embrasures should look the same.
    # Only pixels the synthetic gap left untouched (it paints a little beyond the tooth): those measure the simulation.
    outer = (cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & ~(cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    same = np.all(load(f'in/{c["k"]}_gap.rgba', c) == load(f'in/{c["k"]}_full.rgba', c), axis=2) & outer
    halo = float(np.linalg.norm(G[same] - R[same], axis=1).mean()) if same.any() else float('nan')
    rows.append(dict(name=c['name'], t=c['t'], halo=halo, remnants=remn, seams=seams, dE=dE, ssim=float(ssim(G[..., 0], R[..., 0], core)), tex=tex, edge=edge, covered=covered))
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else 'metrics.json', 'w'))
A = lambda k: np.median([r[k] for r in rows])
print(f"cases {len(rows)} | colour ΔE median {A('dE'):.1f} (p90 {np.percentile([r['dE'] for r in rows], 90):.1f}) | SSIM {A('ssim'):.2f} | texture ratio {A('tex'):.2f} | edge ratio {A('edge'):.2f} | area covered {A('covered'):.2f} | black remnants {100*np.mean([r['remnants'] for r in rows]):.1f}% | added seams {np.mean([r['seams'] for r in rows]):.2f} | halo ΔE {np.nanmedian([r['halo'] for r in rows]):.1f} (2.3 = just noticeable)")
