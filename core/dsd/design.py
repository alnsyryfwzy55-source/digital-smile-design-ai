"""Geometric smile design: one target box (and outline polygon) per tooth.

All coordinates are in the same image/crop pixels as the input boxes. The design is fixed here;
any image generation later only renders it.
"""
import math

from .constants import TEETH, PROPORTIONS, IDEAL_W, IDEAL_L


def _rule(rules, rid):
    return next((r for r in rules if r['id'] == rid), {})


def propose_design(teeth, meas, rules, proportion='red70', custom_p=0.7, wl=None, space_mode='preserve',
                   symmetric=True, gingiva_mode='keep', midline='current', facial_midline_x=None,
                   central_width_mm=None, central_len_mm=None):
    """Same algorithm as the JavaScript proposeDesign (kept identical; tested for parity)."""
    p = float(custom_p) if proportion == 'custom' else PROPORTIONS.get(proportion, {}).get('p', 0.7)
    wl = wl if wl is not None else _rule(rules, 'central_wl').get('target', 0.8)
    lat_len = _rule(rules, 'lat_central_len').get('target', 0.85)
    can_len = _rule(rules, 'can_central_len').get('target', 0.95)
    mmpp = meas.get('mm_per_px')
    T = teeth
    if not all(T.get(k) for k in TEETH):
        return dict(ok=False, error='يجب تحديد الأسنان الستة قبل التصميم')
    actual_mid = (T['11']['x1'] + T['21']['x0']) / 2
    mid = facial_midline_x if (midline == 'facial' and facial_midline_x is not None) else actual_mid
    if space_mode == 'preserve':
        k = 1 + p + p * p
        cR, cL = (mid - T['13']['x0']) / k, (T['23']['x1'] - mid) / k
        if symmetric:
            cR = cL = (cR + cL) / 2
    else:
        cw = [meas['teeth'][k]['w'] for k in ('11', '21') if k in meas['teeth']]
        c = sum(cw) / len(cw) if cw else (T['11']['x1'] - T['11']['x0'] + T['21']['x1'] - T['21']['x0']) / 2
        if central_width_mm and mmpp:
            c = central_width_mm / mmpp
        cR = cL = c
    Lc = ((cR + cL) / 2) / wl
    if central_len_mm and mmpp:
        Lc = central_len_mm / mmpp
    elif mmpp:
        r = _rule(rules, 'central_len_mm')
        if r.get('enabled') and r.get('min') is not None and r.get('max') is not None:
            Lc = min(max(Lc, r['min'] / mmpp), r['max'] / mmpp)
    length = {'11': Lc, '21': Lc, '12': Lc * lat_len, '22': Lc * lat_len, '13': Lc * can_len, '23': Lc * can_len}
    wR = {'11': cR, '12': cR * p, '13': cR * p * p}
    wLft = {'21': cL, '22': cL * p, '23': cL * p * p}
    X = {}
    X['11'] = (mid - wR['11'], mid); X['12'] = (X['11'][0] - wR['12'], X['11'][0]); X['13'] = (X['12'][0] - wR['13'], X['12'][0])
    X['21'] = (mid, mid + wLft['21']); X['22'] = (X['21'][1], X['21'][1] + wLft['22']); X['23'] = (X['22'][1], X['22'][1] + wLft['23'])
    zc = (T['11']['y0'] + T['21']['y0']) / 2
    lat_drop = 0.75 / mmpp if mmpp else Lc * 0.07
    target, changes = {}, {}
    for k in TEETH:
        y0 = T[k]['y0'] if gingiva_mode == 'keep' else (zc + lat_drop if k in ('12', '22') else zc)
        target[k] = dict(x0=X[k][0], x1=X[k][1], y0=y0, y1=y0 + length[k])
        changes[k] = change_of(T[k], target[k], mmpp)
    return dict(ok=True, p=p, wl=wl, mid=mid, actual_mid=actual_mid, target=target, changes=changes)


def change_of(a, b, mmpp=None):
    c = dict(dW=(b['x1'] - b['x0']) - (a['x1'] - a['x0']), dL=(b['y1'] - b['y0']) - (a['y1'] - a['y0']),
             dIncisal=b['y1'] - a['y1'], dGingival=b['y0'] - a['y0'], w=b['x1'] - b['x0'], h=b['y1'] - b['y0'])
    if mmpp:
        for f in list(c):
            c[f + 'mm'] = c[f] * mmpp
    return c


def describe_change(c, mm):
    v = (lambda x: f'{abs(x):.1f} ملم') if mm else (lambda x: f'{abs(x):.0f} بكسل')
    dW, dI, dG = (c['dWmm'], c['dIncisalmm'], c['dGingivalmm']) if mm else (c['dW'], c['dIncisal'], c['dGingival'])
    th = 0.2 if mm else 2
    out = []
    if dW > th: out.append('توسيع ' + v(dW))
    elif dW < -th: out.append('تضييق ' + v(dW))
    if dI > th: out.append('إطالة الحافة ' + v(dI))
    elif dI < -th: out.append('تقصير الحافة ' + v(dI))
    if dG < -th: out.append('رفع اللثة (قطع) ' + v(dG))
    elif dG > th: out.append('إنزال حد التاج ' + v(dG))
    return '، '.join(out) if out else 'بلا تغيير يُذكر'


def default_teeth(W, H):
    total, k = W * 0.8, 1 + 0.7 + 0.49
    c = total / 2 / k
    mid, y0, L = W / 2, H * 0.25, c / 0.8
    b, wR = {}, [c, c * 0.7, c * 0.49]
    x = mid
    for i, t in enumerate(('11', '12', '13')):
        b[t] = dict(x0=x - wR[i], x1=x, y0=y0 + (L * 0.05 if i == 1 else 0), y1=y0 + L * [1, 0.85, 0.95][i]); x -= wR[i]
    x = mid
    for i, t in enumerate(('21', '22', '23')):
        b[t] = dict(x0=x, x1=x + wR[i], y0=y0 + (L * 0.05 if i == 1 else 0), y1=y0 + L * [1, 0.85, 0.95][i]); x += wR[i]
    return b


def fill_missing(found, W, H):
    """Boxes for teeth the model did not find, inside the gap between found neighbours."""
    out = {t: dict(found[t]) for t in TEETH if found.get(t)}
    have = [t for t in TEETH if t in out]
    if not have:
        return dict(teeth=default_teeth(W, H), filled=list(TEETH))
    unitW = sum((out[t]['x1'] - out[t]['x0']) / IDEAL_W[t] for t in have) / len(have)
    unitL = sum((out[t]['y1'] - out[t]['y0']) / IDEAL_L[t] for t in have) / len(have)
    top = sum(out[t]['y0'] for t in have) / len(have)
    filled, i = [], 0
    while i < 6:
        if TEETH[i] in out:
            i += 1; continue
        j = i
        while j < 6 and TEETH[j] not in out:
            j += 1
        run = TEETH[i:j]
        L = TEETH[i - 1] if i > 0 else None
        R = TEETH[j] if j < 6 else None
        weights = [IDEAL_W[t] for t in run]; sw = sum(weights)
        if L and R:
            x0, x1 = out[L]['x1'], out[R]['x0']
            if x1 - x0 < unitW * sw * 0.3:
                c = (x0 + x1) / 2; x0, x1 = c - unitW * sw * 0.5, c + unitW * sw * 0.5
        elif L:
            x0 = out[L]['x1']; x1 = x0 + unitW * sw
        else:
            x1 = out[R]['x0']; x0 = x1 - unitW * sw
        x = x0
        for k, t in enumerate(run):
            w = (x1 - x0) * weights[k] / sw
            out[t] = dict(x0=x, x1=x + w, y0=top, y1=top + unitL * IDEAL_L[t]); x += w; filled.append(t)
        i = j
    for t in TEETH:
        b = out[t]
        b['x0'] = max(0, b['x0']); b['x1'] = min(W, max(b['x1'], b['x0'] + 4)); b['y1'] = min(H, b['y1'])
    return dict(teeth=out, filled=filled)


def chain_y(chain, x):
    """y of a polyline (list of {x,y}, any direction) at x, by linear interpolation; clamped at the ends."""
    pts = sorted(chain, key=lambda p: p['x'])
    if x <= pts[0]['x']:
        return pts[0]['y']
    for a, b in zip(pts, pts[1:]):
        if x <= b['x']:
            u = (x - a['x']) / max(1e-9, b['x'] - a['x'])
            return a['y'] + u * (b['y'] - a['y'])
    return pts[-1]['y']


def design_from_face(upper_lip, mid_x, mmpp=None, central_width_mm=8.6, central_len_mm=10.5, p=0.7):
    """No front teeth at all (trauma, extraction): an ideal anterior set placed from the face itself.
    upper_lip: inner upper-lip chain [{x,y}]; mid_x: facial midline at the mouth."""
    xs = [q['x'] for q in upper_lip]
    mouth_w = max(xs) - min(xs)
    cW = central_width_mm / mmpp if mmpp else mouth_w * 0.155
    cL = central_len_mm / mmpp if mmpp else cW / 0.8
    ws = [k * cW for k in (p * p, p, 1, 1, p, p * p)]
    lens = [0.95, 0.85, 1, 1, 0.85, 0.95]
    out, x = {}, mid_x - cW - p * cW - p * p * cW
    for i, t in enumerate(TEETH):
        cx = x + ws[i] / 2
        top = chain_y(upper_lip, cx) - (1 / mmpp if mmpp else cL * 0.1)      # the neck sits just under the upper lip
        out[t] = dict(x0=x, x1=x + ws[i], y0=top, y1=top + cL * lens[i]); x += ws[i]
    return out


def apply_smile_arc(target, lower_y, mmpp=None, length_delta_mm=0.0, follow=0.85):
    """Incisal edges into an arc parallel to the lower lip. lower_y(x): lower-lip y."""
    out = {t: dict(target[t]) for t in TEETH if target.get(t)}
    if '11' not in out or '21' not in out:
        return out
    mid_x = (out['11']['x1'] + out['21']['x0']) / 2
    dL = (length_delta_mm / mmpp) if mmpp else 0
    yc = (out['11']['y1'] + out['21']['y1']) / 2 + dL
    step = {'12': 0.75, '22': 0.75, '13': 0.25, '23': 0.25}
    lip_mid = lower_y(mid_x)
    L11 = yc - (out['11']['y0'] + out['21']['y0']) / 2
    for t in TEETH:
        b = out.get(t)
        if not b:
            continue
        cx = (b['x0'] + b['x1']) / 2
        y1 = yc + min(0, lower_y(cx) - lip_mid) * follow        # side teeth rise with the lip, never below the centrals
        if t in step:
            y1 -= step[t] / mmpp if mmpp else L11 * 0.07 * step[t] / 0.75
        if t in ('11', '21'):
            y1 = yc
        y1 = min(y1, yc)
        out[t]['y1'] = max(b['y0'] + 0.55 * L11, y1)
    return out


def limit_changes(current, target, mmpp, status=None, shorten_mm=1.5, lengthen_mm=2.5):
    """Keep automatic incisal changes within what veneers/crowns usually achieve.
    Missing or broken teeth are rebuilt in full; the doctor can go further on purpose."""
    st = status or {}
    out = {}
    for t in TEETH:
        a, b = (current or {}).get(t), target.get(t)
        if not b:
            continue
        out[t] = dict(b)
        if not a or not mmpp or (st.get(t) and st[t] != 'present'):
            continue
        out[t]['y1'] = min(max(b['y1'], a['y1'] - shorten_mm / mmpp), a['y1'] + lengthen_mm / mmpp)
    return out


PAPILLA = 0.40          # DSD: papilla height about 40% of crown height


def papilla_points(target):
    """Tip of the interdental papilla between neighbouring teeth: 40% of the crown height below the gingival line."""
    out = []
    for a, b in zip(TEETH, TEETH[1:]):
        A, B = target.get(a), target.get(b)
        if not (A and B):
            continue
        y0 = (A['y0'] + B['y0']) / 2
        h = ((A['y1'] - A['y0']) + (B['y1'] - B['y0'])) / 2
        out.append(dict(between=f'{a}-{b}', x=(A['x1'] + B['x0']) / 2, y=y0 + PAPILLA * h))
    return out


def tooth_polygon(t, b, shape='square', n=24):
    """Outline of one tooth in its box: curved cervical line with the zenith slightly distal,
    rounded incisal corners (sharper mesially), and a cusp tip on canines."""
    x0, x1, y0, y1 = b['x0'], b['x1'], b['y0'], b['y1']
    w, h = x1 - x0, y1 - y0
    mesial_is_left = t[0] == '2'                    # patient's left teeth are on the image right; mesial faces the midline
    r_dist = {'square': 0.18, 'oval': 0.40, 'triangular': 0.30}[shape] * w
    r_mes = {'square': 0.08, 'oval': 0.30, 'triangular': 0.15}[shape] * w
    taper = {'square': 0.06, 'oval': 0.10, 'triangular': 0.18}[shape] * w
    rl, rr = (r_mes, r_dist) if mesial_is_left else (r_dist, r_mes)
    zx = x0 + w * (0.58 if mesial_is_left else 0.42)
    pts = []
    for i in range(n + 1):
        xx = x0 + taper + (w - 2 * taper) * i / n
        k = (xx - zx) / (w / 2)
        pts.append((xx, y0 + 0.10 * h * min(1.0, k * k)))
    pts.append((x1, y0 + PAPILLA * h))
    for i in range(n // 2 + 1):
        a = math.pi / 2 * i / (n // 2)
        pts.append((x1 - rr + rr * math.cos(a), y1 - rr + rr * math.sin(a)))
    if t in ('13', '23'):
        pts.append((x0 + w / 2, y1 + 0.06 * h))
    for i in range(n // 2 + 1):
        a = math.pi / 2 + math.pi / 2 * i / (n // 2)
        pts.append((x0 + rl + rl * math.cos(a), y1 - rl + rl * math.sin(a)))
    pts.append((x0, y0 + PAPILLA * h))
    return pts
