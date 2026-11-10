"""Generation proof-of-concept: everything except the diffusion model itself.

Pipeline for one smile photo:
    face landmarks -> mouth crop (eye-line aligned) -> tooth segmentation (our U-Net, ONNX)
    -> geometric design (one polygon per tooth, in crop pixels) -> design validation
    -> prepaint + inpaint mask + edge map  (inputs for the diffusion model)
    -> [diffusion model in the notebook]
    -> composite back + automatic check of the generated teeth against the design.

The design is decided here, before any generation. The generator only renders it,
and check_generated() measures whether it kept to it.
"""
import math
import numpy as np
import cv2

TEETH = ['13', '12', '11', '21', '22', '23']          # model classes 1..6, image left -> right
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SHADES = {'BL2': '#F3EFE6', 'B1': '#F0E9D8', 'A1': '#EEE6D4', 'A2': '#E9DDC4'}

# MediaPipe FaceLandmarker indices (478-point mesh)
IRIS_A, IRIS_B = 468, 473
COMM_A, COMM_B = 61, 291
SUBNASALE = 2
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


# ---------------------------------------------------------------- face + crop
def load_rgb(path):
    """Read a photo with its EXIF rotation applied (phone photos are often stored sideways)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    return np.asarray(im)


class FaceFinder:
    def __init__(self, task_path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.det = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task_path), num_faces=1))

    def __call__(self, rgb):
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float64)


def inner_lip_polygon(P):
    return np.concatenate([P[INNER_UPPER], P[INNER_LOWER[1:-1]]])


def mouth_crop(P, aspect=1.6, widen=1.18, out_w=768):
    """Affine crop around the mouth, rotated so the eye line is horizontal.
    Returns M (crop -> image) and Minv (image -> crop), both 2x3."""
    a, b = sorted([P[IRIS_A], P[IRIS_B]], key=lambda p: p[0])
    roll = math.atan2(b[1] - a[1], b[0] - a[0])
    c = inner_lip_polygon(P).mean(0)
    wc = np.linalg.norm(P[COMM_A] - P[COMM_B]) * widen
    out_h = int(round(out_w / aspect / 8) * 8)
    s = wc / out_w
    cs, sn = math.cos(roll) * s, math.sin(roll) * s
    M = np.array([[cs, -sn, c[0] - cs * out_w / 2 + sn * out_h / 2],
                  [sn, cs, c[1] - sn * out_w / 2 - cs * out_h / 2]])
    return M, cv2.invertAffineTransform(M), (out_w, out_h)


def warp_crop(rgb, M, size):
    return cv2.warpAffine(rgb, M, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def to_crop(Minv, pts):
    pts = np.asarray(pts, np.float64)
    return pts @ Minv[:, :2].T + Minv[:, 2]


def paste_back(full, crop_new, M, mask_crop):
    """Put the edited crop back into the full photo, only where mask_crop (0..1) says so."""
    h, w = full.shape[:2]
    Minv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(crop_new, Minv, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    alpha = cv2.warpAffine(mask_crop.astype(np.float32), Minv, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)[..., None]
    return (full * (1 - alpha) + warped * alpha).astype(np.uint8)


# ---------------------------------------------------------------- segmentation
class Segmenter:
    """Our U-Net (MobileNetV2 encoder, 7 classes) exported to ONNX, input 1x3x384x384."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.name = self.s.get_inputs()[0].name

    def labels(self, crop_rgb):
        h, w = crop_rgb.shape[:2]
        x = cv2.resize(crop_rgb, (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        lo = self.s.run(None, {self.name: x})[0][0]
        return cv2.resize(lo.argmax(0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    def tooth_masks(self, crop_rgb, mouth=None, min_frac=0.002):
        lab = self.labels(crop_rgb)
        if mouth is not None:
            lab[mouth == 0] = 0
        out = {}
        for k, t in enumerate(TEETH, start=1):
            m = largest_component((lab == k).astype(np.uint8))
            if m.sum() >= min_frac * m.size:
                out[t] = m
        return out


def largest_component(m):
    n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.zeros_like(m)
    return (cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def box(m):
    ys, xs = np.nonzero(m)
    return dict(x0=float(xs.min()), x1=float(xs.max() + 1), y0=float(ys.min()), y1=float(ys.max() + 1))


def poly_mask(poly, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m


# ---------------------------------------------------------------- geometric design
PROPORTION = {'red70': (1.0, 0.70, 0.49), 'golden': (1.0, 0.618, 0.382), 'natural': (1.0, 0.78, 0.62)}


def propose_design(teeth, mouth_poly, midline_x, shape='square', proportion='red70', wl=0.80, keep_span=True):
    """Target box for each of the six teeth, in crop pixels.

    teeth: current tooth boxes {'11': {x0,x1,y0,y1}, ...} (any subset; empty = no teeth visible)
    widths follow the chosen apparent-width proportion, centred on the facial midline;
    length from width/length ratio wl; incisal edges follow a smile arc (laterals shorter)."""
    p = PROPORTION[proportion]
    xs = mouth_poly[:, 0]
    inner_w = xs.max() - xs.min()
    cur_w = {t: b['x1'] - b['x0'] for t, b in teeth.items()}
    # central width from three independent estimates; the face-based one (about 0.155 of the commissure width)
    # bounds the others, so a mislabelled tooth cannot shrink or blow up the whole design
    wc_face = 0.155 * inner_w
    est = []
    if keep_span and '13' in teeth and '23' in teeth:
        est.append((teeth['23']['x1'] - teeth['13']['x0']) / (2 * sum(p)))
    if '11' in cur_w or '21' in cur_w:
        est.append(float(np.mean([cur_w[t] for t in ('11', '21') if t in cur_w])))
    est = [e for e in est if 0.75 * wc_face <= e <= 1.3 * wc_face]
    wc = float(np.median(est)) if est else wc_face
    widths = {'11': wc * p[0], '21': wc * p[0], '12': wc * p[1], '22': wc * p[1], '13': wc * p[2], '23': wc * p[2]}
    L = wc / wl
    cen = [teeth[t]['y1'] for t in ('11', '21') if t in teeth]
    edge = float(np.mean(cen)) if cen else float(np.min(mouth_poly[:, 1]) + 0.9 * L)
    top_c = edge - L
    raise_ = {'11': 0, '21': 0, '12': 0.10 * L, '22': 0.10 * L, '13': 0.05 * L, '23': 0.05 * L}   # smile arc
    drop = {'11': 0, '21': 0, '12': 0.07 * L, '22': 0.07 * L, '13': 0, '23': 0}                  # gingival line
    out, x = {}, midline_x
    for t in ('21', '22', '23'):
        out[t] = dict(x0=x, x1=x + widths[t]); x += widths[t]
    x = midline_x
    for t in ('11', '12', '13'):
        out[t] = dict(x0=x - widths[t], x1=x); x -= widths[t]
    for t, b in out.items():
        b['y0'] = top_c + drop[t]
        b['y1'] = edge - raise_[t]
        b['shape'] = shape
    return out


def tooth_polygon(t, b, n=24):
    """Outline of one tooth inside its box: curved cervical line with the zenith slightly distal,
    rounded incisal corners (sharper mesially), canines with a cusp tip."""
    x0, x1, y0, y1, shape = b['x0'], b['x1'], b['y0'], b['y1'], b.get('shape', 'square')
    w, h = x1 - x0, y1 - y0
    right_side = t[0] == '2'                       # patient's left = image right
    mesial_is_left = right_side                     # mesial faces the midline
    r_dist = {'square': 0.18, 'oval': 0.40, 'triangular': 0.30}[shape] * w
    r_mes = {'square': 0.08, 'oval': 0.30, 'triangular': 0.15}[shape] * w
    taper = {'square': 0.06, 'oval': 0.10, 'triangular': 0.18}[shape] * w
    rl, rr = (r_mes, r_dist) if mesial_is_left else (r_dist, r_mes)
    pts = []
    # cervical arc, left to right, zenith shifted distally
    zx = x0 + w * (0.58 if mesial_is_left else 0.42)
    for i in range(n + 1):
        u = i / n
        xx = x0 + taper + (w - 2 * taper) * u
        k = (xx - zx) / (w / 2)
        pts.append((xx, y0 + 0.10 * h * min(1.0, k * k)))
    # right side down, incisal edge right to left, left side up
    pts.append((x1, y0 + 0.45 * h))
    for i in range(n // 2 + 1):
        a = math.pi / 2 * i / (n // 2)
        pts.append((x1 - rr + rr * math.cos(a), y1 - rr + rr * math.sin(a)))
    if t in ('13', '23'):                          # canine cusp
        pts.append((x0 + w / 2, y1 + 0.06 * h))
    for i in range(n // 2 + 1):
        a = math.pi / 2 + math.pi / 2 * i / (n // 2)
        pts.append((x0 + rl + rl * math.cos(a), y1 - rl + rl * math.sin(a)))
    pts.append((x0, y0 + 0.45 * h))
    return np.array(pts, np.float64)


def validate_design(design, mouth_poly, w, h):
    """Checks that run BEFORE any generation. Returns (ok, problems)."""
    problems = []
    mouth = poly_mask(mouth_poly, w, h)
    if len(design) != 6:
        problems.append(f'expected 6 teeth, got {len(design)}')
    xs = [(t, b['x0'], b['x1']) for t, b in design.items()]
    order = [t for t, _, _ in sorted(xs, key=lambda r: r[1])]
    if order != TEETH:
        problems.append('tooth order is not 13,12,11,21,22,23')
    for t, b in design.items():
        if b['x1'] - b['x0'] < 4 or b['y1'] - b['y0'] < 4:
            problems.append(f'{t}: degenerate size')
    c11 = (design['11']['x1'] - design['11']['x0']); c21 = (design['21']['x1'] - design['21']['x0'])
    if abs(c11 - c21) > 0.03 * max(c11, c21):
        problems.append('centrals differ in width by more than 3%')
    vis = {t: (poly_mask(tooth_polygon(t, b), w, h) & mouth).sum() for t, b in design.items()}
    for t in ('11', '21'):
        if vis.get(t, 0) < 50:
            problems.append(f'{t}: not visible inside the mouth opening')
    return (len(problems) == 0), problems


# ---------------------------------------------------------------- generator inputs
def hex_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


def prepare_inputs(crop, design, cur_masks, mouth_poly, shade='B1', dilate=5, bank_set=None):
    """Prepaint, inpaint mask, and edge map.

    v1 (bank_set=None): design teeth painted flat with simple shading inside the design polygons.
    v2 (bank_set from tooth_bank): real teeth of one labelled photo, each in its design box with its own
    outline and texture, recoloured to the patient's own shade with moderate whitening.
    The mask covers the old upper teeth and the new design, inside the lip opening and above
    the incisal line, so the lips, gums outside the mask and the lower teeth stay original."""
    h, w = crop.shape[:2]
    mouth = poly_mask(mouth_poly, w, h)
    old = np.zeros((h, w), np.uint8)
    for m in cur_masks.values():
        old |= m
    design_m = np.zeros((h, w), np.uint8)
    tooth_ms = {}
    target_lab = None
    ref_mask = old & mouth
    if bank_set is not None:
        import tooth_bank as TB
        tooth_ms = TB.paint_teeth(np.zeros((h, w, 3), np.float32), design, bank_set, bank_set['lab'])   # shapes only
        ref = old & mouth
        if ref.sum() < 200:                                    # no upper teeth left: use the lower teeth of the opening
            low = np.zeros((h, w), np.uint8)
            ys = [b['y1'] for b in design.values()]
            low[int(max(ys)) + 4:, :] = 1
            g = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 0]
            ref = (low & mouth & (g > np.percentile(g[mouth > 0], 70))).astype(np.uint8)
        target_lab = TB.target_shade(crop, ref)
        ref_mask = ref
    else:
        for t, b in design.items():
            tooth_ms[t] = poly_mask(tooth_polygon(t, b), w, h)
    for m in tooth_ms.values():
        design_m |= m
    region = cv2.dilate(design_m | old, np.ones((2 * dilate + 1,) * 2, np.uint8))
    # incisal line per column: nothing below max(design edge, old edge) + 3 px  (keeps the lower teeth)
    cut = np.full(w, -1, np.int32)
    for m in [design_m, old]:
        ys, xs = np.nonzero(m)
        if len(xs):
            np.maximum.at(cut, xs, ys)
    cut = cv2.dilate(cut.reshape(1, -1).astype(np.float32), np.ones((1, 2 * dilate + 1), np.uint8)).ravel()
    yy = np.arange(h)[:, None]
    region &= (yy <= (cut[None, :] + 3)).astype(np.uint8)
    mask = region & mouth

    base = hex_rgb(SHADES[shade])
    # where old teeth were but the design has none: gum above the new cervical line, dark embrasure below it
    hole = ((mask == 1) & (design_m == 0)).astype(np.uint8)
    pre = cv2.inpaint(crop, cv2.dilate(hole, np.ones((3, 3), np.uint8)), 7, cv2.INPAINT_TELEA).astype(np.float32)
    gum_line = np.full(w, np.inf)
    for t, b in design.items():
        xa, xb = int(max(0, b['x0'] - 4)), int(min(w, b['x1'] + 4))
        gum_line[xa:xb] = np.minimum(gum_line[xa:xb], b['y0'] + (0.5 if bank_set is not None else 0.22) * (b['y1'] - b['y0']))
    dark = (hole == 1) & (yy > gum_line[None, :])
    pre[dark] = (45, 28, 28)
    detail = crop.astype(np.float32) - cv2.GaussianBlur(crop, (0, 0), 3).astype(np.float32)
    if bank_set is not None:
        TB.paint_teeth(pre, design, bank_set, target_lab)
    for t, m in ({} if bank_set is not None else tooth_ms).items():
        b = design[t]
        ys, xs = np.nonzero(m & mask)
        if not len(xs):
            continue
        u = (xs - b['x0']) / max(1, b['x1'] - b['x0'])               # 0..1 across
        v = (ys - b['y0']) / max(1, b['y1'] - b['y0'])               # 0..1 down
        shade_k = 0.86 + 0.10 * np.sin(np.pi * np.clip(u, 0, 1)) + 0.06 * np.clip(v, 0, 1) - 0.08 * np.clip(0.25 - v, 0, 1) * 4
        col = base[None, :] * shade_k[:, None]
        inc = np.clip((v - 0.85) / 0.15, 0, 1)[:, None]                # translucent incisal edge
        col = col * (1 - 0.25 * inc) + np.array([150, 160, 175], np.float32) * 0.25 * inc
        on_old = old[ys, xs][:, None].astype(np.float32)               # keep the patient's own enamel texture
        pre[ys, xs] = col + 0.6 * on_old * detail[ys, xs]
    edges = np.zeros((h, w), np.uint8)
    for t, b in design.items():
        if bank_set is not None and t in tooth_ms:
            cs, _ = cv2.findContours(tooth_ms[t], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(edges, cs, -1, 255, 2)
        else:
            cv2.polylines(edges, [np.round(tooth_polygon(t, b)).astype(np.int32)], True, 255, 2)
    edges &= (mouth * 255).astype(np.uint8)
    # thin dark contact lines only BETWEEN neighbouring teeth (no dark rim at the gum line or the lips)
    k3 = np.ones((5, 5), np.uint8)
    order = [t for t in ('13', '12', '11', '21', '22', '23') if t in tooth_ms]
    contact = np.zeros((h, w), bool)
    for a, b in zip(order, order[1:]):
        contact |= (cv2.dilate(tooth_ms[a], k3) > 0) & (cv2.dilate(tooth_ms[b], k3) > 0)
    if bank_set is None:
        pre[contact & (mask == 1)] *= 0.62
    else:                                                   # real teeth bring their own contacts; just a soft shadow
        pre[contact & (mask == 1) & (design_m == 0)] *= 0.55
    return dict(prepaint=np.clip(pre, 0, 255).astype(np.uint8), mask=mask, edges=edges, target_lab=target_lab, ref_mask=ref_mask,
                design_mask=design_m & mouth, tooth_masks={t: m & mouth for t, m in tooth_ms.items()})


def feather(mask, px=4):
    m = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, px // 2)).astype(np.float32)
    k = 2 * px + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def composite(crop, generated, mask, px=4):
    a = feather(mask, px)[..., None]
    return (crop * (1 - a) + generated * a).astype(np.uint8), a[..., 0]


# ---------------------------------------------------------------- automatic check after generation
def check_generated(seg, before, after, inputs, mouth_poly):
    """Did the generated image keep to the design?  Returns a dict of metrics and a verdict."""
    h, w = after.shape[:2]
    mouth = poly_mask(mouth_poly, w, h)
    got = seg.tooth_masks(after, mouth)
    ious = {}
    for t, dm in inputs['tooth_masks'].items():
        if dm.sum() < 30:
            continue
        g = got.get(t, np.zeros_like(dm))
        ious[t] = float((g & dm).sum() / max(1, (g | dm).sum()))
    gen_all = np.zeros((h, w), np.uint8)
    for m in got.values():
        gen_all |= m
    dm_all = inputs['design_mask']
    union_iou = float((gen_all & dm_all).sum() / max(1, (gen_all | dm_all).sum()))
    outside = (inputs['mask'] == 0)
    drift = float(np.abs(after.astype(np.int16) - before.astype(np.int16))[outside].mean())
    extra = int(((gen_all == 1) & (cv2.dilate(dm_all, np.ones((9, 9), np.uint8)) == 0)).sum())
    r = dict(iou_mean=float(np.mean(list(ious.values()))) if ious else 0.0,
             iou_min=float(min(ious.values())) if ious else 0.0,
             iou_union=union_iou, teeth_found=len(got), outside_change=drift, extra_tooth_px=extra,
             iou=ious)
    r['pass_geometry'] = bool(r['iou_union'] >= 0.80 and r['outside_change'] < 1.0 and extra < 0.01 * dm_all.sum() + 50)
    r.update(realism(before, after, inputs))
    r['pass'] = bool(r['pass_geometry'] and r['pass_realism'])
    return r


def _lab_cie(lab):
    return np.array([lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128], np.float32)


def realism(before, after, inputs):
    """v1 passed the geometry check by copying flat blocks. These measure what it missed:
    texture  - fine detail inside the new teeth vs the patient's real teeth (Laplacian spread); flat = low
    colour   - CIE dE of the new teeth from the planned shade (patient's own, moderately whitened)
    seam     - edge strength on the border of the edited area vs the same border before; a halo or rim = high"""
    k = np.ones((5, 5), np.uint8)
    new = cv2.erode(inputs['design_mask'], k)
    ref = cv2.erode(inputs.get('ref_mask', np.zeros_like(new)), np.ones((3, 3), np.uint8))
    ga, gb = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    la, lb = cv2.Laplacian(ga, cv2.CV_32F, ksize=3), cv2.Laplacian(gb, cv2.CV_32F, ksize=3)
    out = {}
    if new.sum() > 100:
        tex_new = float(la[new > 0].std())
        tex_ref = float(lb[ref > 0].std()) if ref.sum() > 100 else 12.0
        out['texture_ratio'] = tex_new / max(tex_ref, 1e-3)
        lab = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float32)[new > 0].mean(0)
        tgt = inputs.get('target_lab')
        if tgt is None:
            tgt = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float32)[ref > 0].mean(0) if ref.sum() > 100 else lab
        out['color_dE'] = float(np.linalg.norm(_lab_cie(lab) - _lab_cie(tgt)))
        out['L_star'] = float(lab[0] * 100 / 255)
    else:
        out.update(texture_ratio=0.0, color_dE=99.0, L_star=0.0)
    m = inputs['mask']
    ring = (cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0
    gxa, gya = cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1)
    gxb, gyb = cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    out['seam_ratio'] = float(np.hypot(gxa, gya)[ring].mean() / max(np.hypot(gxb, gyb)[ring].mean(), 1e-3)) if ring.any() else 1.0
    out['pass_realism'] = bool(0.45 <= out['texture_ratio'] <= 2.2 and out['color_dE'] <= 10 and out['seam_ratio'] <= 1.8)
    return out


# ---------------------------------------------------------------- one photo, up to the generator
def analyse(rgb, face, seg, shape='square', proportion='red70', shade='B1', out_w=768, bank=None):
    P = face(rgb)
    if P is None:
        return dict(ok=False, reason='no face found')
    M, Minv, size = mouth_crop(P, out_w=out_w)
    crop = warp_crop(rgb, M, size)
    mouth_poly = to_crop(Minv, inner_lip_polygon(P))
    mouth = poly_mask(mouth_poly, *size)
    if mouth.sum() < 0.02 * mouth.size:
        return dict(ok=False, reason='mouth closed or smile too narrow')
    eyes = to_crop(Minv, np.array([P[IRIS_A], P[IRIS_B]]))
    mid_x = float(0.5 * eyes[:, 0].mean() + 0.5 * to_crop(Minv, P[[SUBNASALE]])[0, 0])
    cur = seg.tooth_masks(crop, mouth)
    cur_boxes = {t: box(m) for t, m in cur.items()}
    design = propose_design(cur_boxes, mouth_poly, mid_x, shape=shape, proportion=proportion)
    ok, problems = validate_design(design, mouth_poly, *size)
    bank_set = None
    if bank:
        import tooth_bank as TB
        bank_set = TB.pick_set(bank, design)
    inputs = prepare_inputs(crop, design, cur, mouth_poly, shade=shade, bank_set=bank_set)
    return dict(ok=ok, problems=problems, P=P, M=M, crop=crop, mouth_poly=mouth_poly, midline_x=mid_x,
                current=cur_boxes, design=design, inputs=inputs, bank_id=bank_set['id'] if bank_set else None)


def overlay(crop, design, current=None, mouth_poly=None):
    """Picture for the doctor: current teeth (red), design outlines (green), lip opening (blue)."""
    v = crop.copy()
    if mouth_poly is not None:
        cv2.polylines(v, [np.round(mouth_poly).astype(np.int32)], True, (60, 120, 255), 1)
    for b in (current or {}).values():
        cv2.rectangle(v, (int(b['x0']), int(b['y0'])), (int(b['x1']), int(b['y1'])), (230, 60, 60), 1)
    for t, b in design.items():
        cv2.polylines(v, [np.round(tooth_polygon(t, b)).astype(np.int32)], True, (40, 220, 90), 2)
    return v
