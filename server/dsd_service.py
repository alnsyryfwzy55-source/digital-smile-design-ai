"""Analysis service: one smile photo + the face landmarks the app already computes -> a validated design.

The phone runs MediaPipe (live capture guidance); the server receives the photo and the 478 landmarks and does
everything heavy: tooth segmentation with structural guarantees, scale, face metrics, design, validation and the
lab files. Framework-agnostic: http_api.py (standard library) and app_fastapi.py both call analyze().

    from dsd_service import Service
    svc = Service('docs/teeth_unet_v2.onnx')
    result = svc.analyze(jpeg_bytes, landmarks, options={'caliper_central_mm': 8.6})
"""
import base64, math, os, sys, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'core'))
import dsd                                                        # noqa: E402
from dsd.postprocess import teeth_from_probs, softmax, appearance_check, vertical_check             # noqa: E402
from dsd.pipeline import plan_case                                # noqa: E402
from dsd.constants import TEETH, SHADES                           # noqa: E402

IRIS_A, IRIS_B, COMM_A, COMM_B, SUBNASALE = 468, 473, 61, 291, 2
INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
INNER_LOWER = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CONFIRM_BELOW = 0.95            # model v3, calibrated on the 43 labelled test/val photos x 6 hard variants (blur, dark,
                                # JPEG 12, rotation, lip covering, yellow cast): catches 43/47 wrong teeth, asks about
                                # 3/257 correct teeth in clean photos (training/calibrate_confidence.py)
VERSION = '0.1.0'


class AnalysisError(Exception):
    def __init__(self, code, message):
        super().__init__(message); self.code = code; self.message = message


# ----------------------------------------------------------------------------- geometry
def mouth_crop(P, out_w=768, aspect=1.6, widen=1.18):
    a, b = sorted([P[IRIS_A], P[IRIS_B]], key=lambda p: p[0])
    roll = math.atan2(b[1] - a[1], b[0] - a[0])
    inner = np.concatenate([P[INNER_UPPER], P[INNER_LOWER[1:-1]]])
    c = inner.mean(0)
    wc = np.linalg.norm(P[COMM_A] - P[COMM_B]) * widen
    out_h = int(round(out_w / aspect / 8) * 8)
    s = wc / out_w                                                 # image pixels per crop pixel
    cs, sn = math.cos(roll) * s, math.sin(roll) * s
    M = np.array([[cs, -sn, c[0] - cs * out_w / 2 + sn * out_h / 2], [sn, cs, c[1] - sn * out_w / 2 - cs * out_h / 2]])
    return M, cv2.invertAffineTransform(M), (out_w, out_h), s, math.degrees(roll)


def apply(A, pts):
    return np.asarray(pts, np.float64) @ A[:, :2].T + A[:, 2]


def poly_mask(poly, w, h):
    m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1); return m


def to_pts(arr):
    return [dict(x=float(x), y=float(y)) for x, y in arr]


def iris_diameter_px(P):
    d = [np.linalg.norm(P[a] - P[b]) for a, b in ((469, 471), (474, 476)) if len(P) > max(a, b)]
    return float(np.mean(d)) if d else None


def detect_broken(boxes, status, edge_px=0.0):
    """A fractured incisor is much shorter than its counterpart, or far too wide for its length (same rule as the app).
    A verdict closer to the threshold than twice its measurement error is returned in `unsure` instead."""
    ok = lambda t: t in boxes and status.get(t) == 'present'
    H = lambda t: boxes[t]['y1'] - boxes[t]['y0']
    W = lambda t: boxes[t]['x1'] - boxes[t]['x0']
    s2 = math.sqrt(2) * edge_px
    sig = lambda r, A, B: r * math.hypot(s2 / A, s2 / B) if edge_px else 0.0
    out, unsure = [], []
    for a, b in (('11', '21'), ('12', '22')):
        if ok(a) and ok(b):
            lo, hi = (a, b) if H(a) <= H(b) else (b, a)
            r = H(lo) / max(1, H(hi)); sg = sig(r, H(lo), H(hi))
            if r < 0.78 - 2 * sg: out.append(lo)
            elif r < 0.78: unsure.append(lo)
    for t in ('11', '21'):
        if ok(t) and t not in out and t not in unsure:
            r = W(t) / max(1, H(t)); sg = sig(r, W(t), H(t))
            if r > 1.35 + 2 * sg: out.append(t)
            elif r > 1.35: unsure.append(t)
    return out, unsure


# ----------------------------------------------------------------------------- service
class Service:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name

    def tooth_probs(self, crop):
        h, w = crop.shape[:2]
        x = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        p = softmax(self.sess.run(None, {self.inp: ((x - MEAN) / STD).transpose(2, 0, 1)[None]})[0][0])
        return np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) for c in p])

    def analyze(self, image, landmarks, options=None, with_overlay=True):
        """image: JPEG/PNG bytes or an RGB array (already upright: the app fixes orientation).
        landmarks: 478 points, normalized (0..1) or in pixels. options: caliper_central_mm, proportion, shape, shade,
        midline ('facial'|'current'), length_delta_mm, patient_goals, case_id."""
        t0 = time.time()
        o = dict(proportion='red70', shape='square', shade='B1', midline='facial', length_delta_mm=0.0, case_id='')
        o.update(options or {})
        rgb = image if isinstance(image, np.ndarray) else cv2.cvtColor(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rgb is None or rgb.size == 0:
            raise AnalysisError('BAD_IMAGE', 'تعذّرت قراءة الصورة')
        Hh, Ww = rgb.shape[:2]
        P = np.asarray(landmarks, np.float64)[:, :2]
        if P.shape[0] < 478:
            raise AnalysisError('NO_LANDMARKS', 'نقاط الوجه ناقصة (المطلوب 478 نقطة مع القزحية)')
        if P.max() <= 1.5:
            P = P * [Ww, Hh]
        M, Minv, (W, H), s, roll = mouth_crop(P)
        crop = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        upper = apply(Minv, P[INNER_UPPER]); lower = apply(Minv, P[INNER_LOWER])
        lip_poly = np.concatenate([upper, lower[1:-1]])
        mouth = poly_mask(lip_poly, W, H)
        if mouth.sum() < 0.02 * W * H:
            raise AnalysisError('MOUTH_CLOSED', 'الفم مغلق أو الابتسامة ضيقة: لا تظهر الأسنان')
        t_seg = time.time()
        prob = self.tooth_probs(crop)
        post = teeth_from_probs(prob, mouth=cv2.dilate(mouth, np.ones((7, 7), np.uint8)))
        seg_ms = int((time.time() - t_seg) * 1000)
        labels = post['labels'].copy()
        status = dict(post['status'])
        # a 'tooth' much darker than the other teeth is a gap the network filled in from context (lost tooth)
        looks, look_stats = appearance_check(labels, crop)
        for t, v in looks.items():
            labels[labels == 1 + TEETH.index(t)] = 0
            status[t] = v
        # a 'tooth' starting low in the opening and reaching the lower lip is a lower tooth (or half a mask)
        placed, place_stats = vertical_check(labels, (mouth > 0) | (labels > 0))
        for t in placed:
            labels[labels == 1 + TEETH.index(t)] = 0
            status[t] = 'unsure'
        boxes = {}
        for t in TEETH:
            ys, xs = np.nonzero(labels == 1 + TEETH.index(t))
            if len(xs):
                boxes[t] = dict(x0=float(np.percentile(xs, 1)), x1=float(np.percentile(xs, 99) + 1),
                                y0=float(np.percentile(ys, 1)), y1=float(np.percentile(ys, 99) + 1))
        broken, too_close = detect_broken(boxes, status, max(0.75 * W / 384, 1.0 / s))
        for t in broken:
            status[t] = 'broken'
        for t in too_close:                                         # too close to call at this resolution: the doctor decides
            status[t] = 'unsure'
        # scale: caliper on a central incisor beats the population iris estimate
        iris_px = iris_diameter_px(P)
        if o.get('caliper_central_mm') and ('11' in boxes or '21' in boxes):
            ref = boxes.get('11') or boxes['21']
            scale = dsd.scale_from_caliper(ref, float(o['caliper_central_mm']))
        elif iris_px:
            scale = dsd.scale_from_iris(iris_px / s)                 # iris diameter in crop pixels
        else:
            scale = dsd.Scale()
        # face metrics in image pixels (mm per image pixel = mm per crop pixel / s)
        fp = dsd.face_points_from_landmarks(P)
        toimg = lambda x, y: dict(zip(('x', 'y'), apply(M, [[x, y]])[0]))
        cx = lambda b: (b['x0'] + b['x1']) / 2
        if '11' in boxes:
            fp['zenith11'] = toimg(cx(boxes['11']), boxes['11']['y0']); fp['incisal11'] = toimg(cx(boxes['11']), boxes['11']['y1'])
        if '11' in boxes and '21' in boxes:
            fp['midlineDental'] = toimg((boxes['11']['x1'] + boxes['21']['x0']) / 2, (boxes['11']['y1'] + boxes['21']['y1']) / 2)
        for t, k in (('13', 'cusp13'), ('23', 'cusp23')):
            if t in boxes:
                fp[k] = toimg(cx(boxes[t]), boxes[t]['y1'])
        low_img = apply(M, lower)
        if 'cusp13' in fp:
            fp['lowerLipR'] = dict(x=fp['cusp13']['x'], y=float(np.interp(fp['cusp13']['x'], *zip(*sorted(low_img.tolist())))))
        if 'cusp23' in fp:
            fp['lowerLipL'] = dict(x=fp['cusp23']['x'], y=float(np.interp(fp['cusp23']['x'], *zip(*sorted(low_img.tolist())))))
        eyes_c = apply(Minv, [P[IRIS_A], P[IRIS_B]]); sub_c = apply(Minv, [P[SUBNASALE]])[0]
        facial_mid_x = float(0.5 * eyes_c[:, 0].mean() + 0.5 * sub_c[0])
        # precision of one tooth edge in crop pixels: the coarser of the model grid (384 across the crop) and the photo's own pixels
        edge_px = max(0.75 * W / 384, 1.0 / s)
        plan = plan_case(boxes, status, to_pts(lower), to_pts(upper), face_points=fp, scale=scale,
                         proportion=o['proportion'], shape=o['shape'], midline=o['midline'],
                         facial_midline_x=facial_mid_x, length_delta_mm=float(o['length_delta_mm']), frame=(W, H),
                         face_mm_per_px=(scale.mm_per_px / s) if scale.calibrated else None, edge_px=edge_px,
                         face_edge_px=edge_px * s)
        if not plan.get('target'):
            raise AnalysisError('NO_DESIGN', plan.get('error', 'تعذّر التصميم'))
        face = plan.get('face')
        confirm = sorted({t for t, c in post['confidence'].items() if c < CONFIRM_BELOW and t in boxes}
                         | {t for t, v in status.items() if v in ('unsure', 'missing', 'broken')})
        design = dsd.design_json(o['case_id'], plan['target'], scale.mm_per_px, scale.source, scale.rel_err,
                                 shape=o['shape'], shade=o['shade'], proportion=o['proportion'], status=status,
                                 changes=plan['changes'], validation=plan['validation'], patient_goals=o.get('patient_goals'))
        out = dict(version=VERSION, ok=plan['validation']['ok'], crop=dict(width=W, height=H, crop_to_image=M.tolist(), roll_deg=roll),
                   teeth=dict(boxes=boxes, status=status, confidence=post['confidence'], needs_confirmation=confirm, appearance=look_stats, placement=place_stats),
                   scale=dict(mm_per_px=scale.mm_per_px, source=scale.source, rel_err=scale.rel_err, note=scale.note),
                   measurements=plan['measurements'], rules=plan['rules'], face=face['metrics'] if face else None,
                   design=design, validation=plan['validation'], design_svg=dsd.design_svg(plan['target'], scale.mm_per_px, o['shape'], o['case_id']),
                   lips=dict(upper=to_pts(upper), lower=to_pts(lower)), facial_midline_x=facial_mid_x,
                   timing_ms=dict(segmentation=seg_ms, total=int((time.time() - t0) * 1000)))
        if with_overlay:
            out['overlay_jpg_b64'] = base64.b64encode(overlay(crop, labels, plan['target'], o['shape'], lip_poly, facial_mid_x)).decode()
        return out


COLORS = {'13': (255, 159, 67), '12': (254, 202, 87), '11': (72, 219, 251), '21': (29, 209, 161), '22': (255, 107, 107), '23': (197, 108, 240)}


def overlay(crop, labels, target, shape, lip_poly, mid_x):
    """For the doctor: found teeth (tinted), designed outlines (green), lip opening (blue), facial midline (white)."""
    v = crop.copy()
    for t in TEETH:
        m = labels == 1 + TEETH.index(t)
        if m.any():
            v[m] = (0.65 * v[m] + 0.35 * np.array(COLORS[t])).astype(np.uint8)
    cv2.polylines(v, [np.round(lip_poly).astype(np.int32)], True, (60, 120, 255), 1)
    cv2.line(v, (int(mid_x), 0), (int(mid_x), v.shape[0]), (255, 255, 255), 1)
    for t in TEETH:
        if t in target:
            pts = np.round(np.array(dsd.tooth_polygon(t, target[t], shape))).astype(np.int32)
            cv2.polylines(v, [pts], True, (40, 220, 90), 2)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()
