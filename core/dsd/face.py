"""Face and lip analysis in a frame aligned to the interpupillary line.

points: {name: {'x':..,'y':..}} in image pixels. Names as in the JavaScript app:
pupilR, pupilL (patient's right = image left), glabella, subnasale, menton, commR, commL, upperLip,
zenith11, incisal11, midlineDental, cusp13, cusp23, lastR, lastL, lowerLipMid, lowerLipR, lowerLipL.
"""
import math

from .constants import IRIS_MM


def line_angle_deg(a, b):
    return math.degrees(math.atan2(b['y'] - a['y'], b['x'] - a['x']))


def norm_angle(a):
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


def _rotator(P):
    ipl = line_angle_deg(P['pupilR'], P['pupilL'])
    a = -math.radians(ipl)
    cs, sn, o = math.cos(a), math.sin(a), P['pupilR']

    def T(p):
        return None if p is None else {'x': o['x'] + (p['x'] - o['x']) * cs - (p['y'] - o['y']) * sn,
                                       'y': o['y'] + (p['x'] - o['x']) * sn + (p['y'] - o['y']) * cs}
    return ipl, T


def face_analysis(points, mm_per_px=None, scale_source=None):
    P = points or {}
    has = lambda *ids: all(P.get(i) for i in ids)
    r = dict(has={}, metrics={}, scale=mm_per_px, scale_source=scale_source)
    if not has('pupilR', 'pupilL'):
        return r
    ipl, T = _rotator(P)
    Q = {k: T(v) for k, v in P.items()}
    mm = (lambda px: px * mm_per_px) if mm_per_px else (lambda px: None)
    M = r['metrics']
    M['ipl_angle'] = ipl
    if has('commR', 'commL'):
        M['commissure_cant'] = norm_angle(line_angle_deg(P['commR'], P['commL']) - ipl)
        M['commissure_span_px'] = math.hypot(P['commL']['x'] - P['commR']['x'], P['commL']['y'] - P['commR']['y'])
    if has('cusp13', 'cusp23'):
        M['incisal_cant'] = norm_angle(line_angle_deg(P['cusp13'], P['cusp23']) - ipl)
        M['incisal_span_px'] = math.hypot(P['cusp23']['x'] - P['cusp13']['x'], P['cusp23']['y'] - P['cusp13']['y'])
    if has('alarR', 'alarL'):                        # DSD reference: interalar line (at rest)
        M['alar_cant'] = norm_angle(line_angle_deg(P['alarR'], P['alarL']) - ipl)
        M['interalar_px'] = abs(Q['alarL']['x'] - Q['alarR']['x'])
        M['interalar'] = mm(M['interalar_px'])
        if has('cusp13', 'cusp23'):                  # classic denture guide: canine tips under the alae
            M['canine_alar_ratio'] = abs(Q['cusp23']['x'] - Q['cusp13']['x']) / max(1e-9, M['interalar_px'])
    mids = [Q[k] for k in ('glabella', 'subnasale', 'menton') if Q.get(k)]
    if len(mids) >= 2 and Q.get('midlineDental'):
        my = sum(p['y'] for p in mids) / len(mids)
        mx = sum(p['x'] for p in mids) / len(mids)
        syy = sum((p['y'] - my) ** 2 for p in mids)
        sxy = sum((p['y'] - my) * (p['x'] - mx) for p in mids)
        b = sxy / syy if syy > 1e-9 else 0
        x_at = mx + b * (Q['midlineDental']['y'] - my)
        dpx = Q['midlineDental']['x'] - x_at          # + : toward image right (patient's left)
        M['midline_dev_px'] = dpx
        M['midline_dev'] = mm(abs(dpx))
        M['midline_dir'] = '' if abs(dpx) < 1e-6 else ('إلى يسار المريض' if dpx > 0 else 'إلى يمين المريض')
        M['facial_midline_angle'] = math.degrees(math.atan(b))
        M['facial_midline_x_at_dental'] = x_at
    if has('upperLip', 'zenith11'):
        d = Q['zenith11']['y'] - Q['upperLip']['y']
        M['gingival_display_px'] = -d                   # + = gum visible above the zenith
        M['gingival_display'] = mm(-d)
    if has('lastR', 'lastL', 'commR', 'commL'):
        M['dentition_span_px'] = abs(Q['lastL']['x'] - Q['lastR']['x'])
        M['smile_width_px'] = abs(Q['commL']['x'] - Q['commR']['x'])
        M['buccal_corridor'] = M['dentition_span_px'] / M['smile_width_px']
    if has('incisal11', 'cusp13', 'cusp23', 'lowerLipMid', 'lowerLipR', 'lowerLipL'):
        inc = Q['incisal11']['y'] - (Q['cusp13']['y'] + Q['cusp23']['y']) / 2
        lip = Q['lowerLipMid']['y'] - (Q['lowerLipR']['y'] + Q['lowerLipL']['y']) / 2
        ref = (abs(Q['commL']['x'] - Q['commR']['x']) if has('commR', 'commL') else 0) or abs(Q['cusp23']['x'] - Q['cusp13']['x'])
        incN, lipN = inc / ref, lip / ref
        if incN < -0.01:
            cls = 'reverse'
        elif incN < 0.015:
            cls = 'flat'
        elif lipN > 0.005 and 0.5 <= incN / lipN <= 2:
            cls = 'consonant'
        else:
            cls = 'nonparallel'
        M.update(smile_arc=cls, smile_arc_inc=incN, smile_arc_lip=lipN, smile_arc_ref_px=ref,
                 incisal_to_lip=mm(Q['lowerLipMid']['y'] - Q['incisal11']['y']))
    if has('glabella', 'subnasale', 'menton'):
        M['lower_third'] = (Q['menton']['y'] - Q['subnasale']['y']) / (Q['subnasale']['y'] - Q['glabella']['y'])
    r['has'] = {k: v is not None for k, v in M.items()}
    return r


def rest_analysis(points, mm_per_px, approx=True, hidden=False):
    """Lips-at-rest photo: how much of the central incisor shows below the upper lip (mm).
    points: pupilR, pupilL, lipRest (lower border of the upper lip at the midline), incisal11."""
    P = points or {}
    if not (P.get('pupilR') and P.get('pupilL') and P.get('lipRest')) or not mm_per_px:
        return None
    _, T = _rotator(P)
    r = dict(scale=mm_per_px, approx=approx, display=None, hidden=hidden)
    if hidden:
        r['display'] = 0.0
        r['hidden_note'] = 'الحافة مغطاة بالشفة: الظهور 0 ملم أو أقل'
        return r
    if not P.get('incisal11'):
        return r
    r['display'] = (T(P['incisal11'])['y'] - T(P['lipRest'])['y']) * mm_per_px
    return r


def iris_diameter_px(lm):
    """Mean horizontal iris diameter from MediaPipe iris ring landmarks (469/471 and 474/476)."""
    d = []
    for a, b in ((469, 471), (474, 476)):
        if len(lm) > max(a, b):
            d.append(math.dist(lm[a], lm[b]))
    return sum(d) / len(d) if d else None


def iris_mm_per_px(lm):
    px = iris_diameter_px(lm)
    return IRIS_MM / px if px and px > 4 else None


# MediaPipe FaceLandmarker (478 points) -> named points used by face_analysis.
# Alae: 129 / 358 (outer nostril wings). Checked on two public-domain frontal photos with MediaPipe: they sit on the
# outermost edge of the wings (48/278 and 64/294 fall inside it). The LINE through them is reliable (follows an 8 degree
# head rotation to within 0.4 degree of the eye line), so `alar_cant` uses it. The WIDTH is not: on the iris scale it
# came out 40-42 mm, above typical adult alar widths, because the mesh points sit in the alar-facial groove. So the
# canine-alar rule stays disabled until it is calibrated against caliper measurements of real patients.
# Teeth points come from the tooth boxes, not from the face mesh.
LANDMARKS = dict(irisA=468, irisB=473, glabella=9, subnasale=2, menton=152, commA=61, commB=291,
                 upperLip=13, lowerLipMid=14, alarA=129, alarB=358)


def face_points_from_landmarks(lm):
    """lm: list of (x, y) in image pixels (478 points). Patient's right = image left."""
    pt = lambda i: {'x': float(lm[i][0]), 'y': float(lm[i][1])}
    L = LANDMARKS
    a, b = sorted([pt(L['irisA']), pt(L['irisB'])], key=lambda p: p['x'])
    ca, cb = sorted([pt(L['commA']), pt(L['commB'])], key=lambda p: p['x'])
    aa, ab = sorted([pt(L['alarA']), pt(L['alarB'])], key=lambda p: p['x'])
    return dict(pupilR=a, pupilL=b, glabella=pt(L['glabella']), subnasale=pt(L['subnasale']), menton=pt(L['menton']),
                commR=ca, commL=cb, upperLip=pt(L['upperLip']), lowerLipMid=pt(L['lowerLipMid']), alarR=aa, alarL=ab)
