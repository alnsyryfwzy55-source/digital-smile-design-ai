"""Design validation (runs before any image is generated) and the safety checks before approval.

Every item: {level: 'block'|'warn', code, text}. A design with any 'block' item must not be rendered
or exported. Thresholds are starting values for the doctor to review, kept in one place (LIMITS).
"""
import math

from .constants import TEETH
from .measure import mean2

LIMITS = dict(
    central_w_mm=(7.0, 10.5), central_len_mm=(9.0, 12.5), central_wl=(0.70, 0.90),
    proportion_tol=0.03, centrals_width_diff=0.03, overlap_px=1.0,
    midline_mm=2.0, midline_frac_uncal=0.25, incisal_cant_deg=2.0,
    width_change_mm=2.0, shorten_mm=2.0, lengthen_mm=2.5,
)


def _item(level, code, text):
    return dict(level=level, code=code, text=text)


def validate_design(target, current=None, mm_per_px=None, p=None, facial_midline_x=None, status=None,
                    mouth_poly=None, limits=None):
    L = dict(LIMITS, **(limits or {}))
    st = status or {}
    items = []
    if not target or not all(target.get(t) for t in TEETH):
        return dict(ok=False, items=[_item('block', 'incomplete', 'التصميم لا يحتوي الأسنان الستة.')])
    W = {t: target[t]['x1'] - target[t]['x0'] for t in TEETH}
    H = {t: target[t]['y1'] - target[t]['y0'] for t in TEETH}
    for t in TEETH:
        if W[t] < 2 or H[t] < 2:
            items.append(_item('block', 'degenerate', f'السن {t}: أبعاد غير صالحة.'))
    for a, b in zip(TEETH, TEETH[1:]):
        if target[a]['x0'] >= target[b]['x0']:
            items.append(_item('block', 'order', f'ترتيب الأسنان غير سليم بين {a} و{b}.'))
        ov = target[a]['x1'] - target[b]['x0']
        if ov > L['overlap_px']:
            items.append(_item('block', 'overlap', f'السنّان {a} و{b} متداخلان ({ov:.1f} بكسل).'))
    if any(i['level'] == 'block' for i in items):
        return dict(ok=False, items=items)

    cw = (W['11'] + W['21']) / 2
    if abs(W['11'] - W['21']) / max(W['11'], W['21']) > L['centrals_width_diff']:
        items.append(_item('warn', 'centrals_asym', 'عرض الثنيتين غير متساوٍ في التصميم.'))
    if p:
        for (a, b, name) in (('12', '11', 'الرباعية/الثنية'), ('22', '21', 'الرباعية/الثنية'), ('13', '12', 'الناب/الرباعية'), ('23', '22', 'الناب/الرباعية')):
            r = W[a] / W[b]
            if abs(r - p) > L['proportion_tol']:
                items.append(_item('warn', 'proportion', f'نسبة {name} للسن {a} = {r:.2f} والهدف {p:.2f}.'))
    wl = cw / ((H['11'] + H['21']) / 2)
    if not (L['central_wl'][0] <= wl <= L['central_wl'][1]):
        items.append(_item('warn', 'central_wl', f'نسبة عرض الثنية إلى طولها {wl:.2f} خارج {L["central_wl"][0]}–{L["central_wl"][1]}.'))
    if mm_per_px:
        wmm, hmm = cw * mm_per_px, (H['11'] + H['21']) / 2 * mm_per_px
        if not (L['central_w_mm'][0] <= wmm <= L['central_w_mm'][1]):
            items.append(_item('warn', 'central_w_mm', f'عرض الثنية المصمم {wmm:.1f} ملم غير معتاد.'))
        if not (L['central_len_mm'][0] <= hmm <= L['central_len_mm'][1]):
            items.append(_item('warn', 'central_len_mm', f'طول الثنية المصمم {hmm:.1f} ملم غير معتاد.'))
    if facial_midline_x is not None:
        dev = abs((target['11']['x1'] + target['21']['x0']) / 2 - facial_midline_x)
        lim = L['midline_mm'] / mm_per_px if mm_per_px else L['midline_frac_uncal'] * cw
        if dev > lim:
            txt = f'{dev * mm_per_px:.1f} ملم' if mm_per_px else f'{dev:.0f} بكسل'
            items.append(_item('warn', 'midline', f'خط الوسط في التصميم منحرف عن خط الوجه بمقدار {txt}.'))
    cant = math.degrees(math.atan2(target['23']['y1'] - target['13']['y1'],
                                   (target['23']['x0'] + target['23']['x1']) / 2 - (target['13']['x0'] + target['13']['x1']) / 2))
    if abs(cant) > L['incisal_cant_deg']:
        items.append(_item('warn', 'incisal_cant', f'مستوى الحواف القاطعة مائل {cant:.1f}° عن خط العينين.'))
    if mean2(target['11']['y1'], target['21']['y1']) < mean2(target['13']['y1'], target['23']['y1']):
        items.append(_item('warn', 'smile_arc', 'الثنيتان أقصر من النابين: قوس الابتسامة معكوس.'))
    if current and mm_per_px:
        for t in TEETH:
            a = current.get(t)
            if not a or st.get(t, 'present') != 'present':
                continue
            dW = ((target[t]['x1'] - target[t]['x0']) - (a['x1'] - a['x0'])) * mm_per_px
            dI = (target[t]['y1'] - a['y1']) * mm_per_px
            if abs(dW) > L['width_change_mm']:
                items.append(_item('warn', 'width_change', f'تغيير عرض السن {t} بمقدار {dW:+.1f} ملم كبير — قد يحتاج تقويماً أو تحضيراً كبيراً.'))
            if dI < -L['shorten_mm']:
                items.append(_item('warn', 'shorten', f'تقصير السن {t} بمقدار {-dI:.1f} ملم كبير.'))
            if dI > L['lengthen_mm']:
                items.append(_item('warn', 'lengthen', f'إطالة السن {t} بمقدار {dI:.1f} ملم كبيرة.'))
    if mouth_poly:
        for t in ('11', '21'):
            cx, cy = (target[t]['x0'] + target[t]['x1']) / 2, target[t]['y1'] - 0.2 * H[t]
            if not point_in_poly(cx, cy, mouth_poly):
                items.append(_item('warn', 'not_visible', f'الحافة القاطعة للسن {t} خارج فتحة الفم في الصورة.'))
    return dict(ok=not any(i['level'] == 'block' for i in items), items=items)


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        a, b = poly[i], poly[j]
        if (a['y'] > y) != (b['y'] > y) and x < (b['x'] - a['x']) * (y - a['y']) / (b['y'] - a['y']) + a['x']:
            inside = not inside
        j = i
    return inside


def safety_checks(teeth=None, reviewed=False, status=None, treat=None, meas=None, rest=None, design_changes=None,
                  det_level=None, source=None):
    """Checks shown before the doctor approves (as in the JavaScript app, without a consent gate:
    the patient knows the clinic photographs them for treatment)."""
    out = []
    st = status or {}
    if not teeth or not all(teeth.get(t) for t in TEETH):
        out.append(_item('block', 'teeth', 'حدود الأسنان الستة غير مكتملة.'))
    if not reviewed:
        out.append(_item('block', 'review', 'لم تُؤكَّد مراجعة حدود الأسنان.'))
    for t in TEETH:
        tr = (treat or {}).get(t)
        if st.get(t) == 'missing' and tr in ('veneer', 'composite'):
            out.append(_item('block', 'treat', f"السن {t} مفقود ولا يصلح له {'فينير' if tr == 'veneer' else 'كومبوزيت'} — اختر زرعة أو جسراً أو بلا علاج."))
        if st.get(t) == 'broken' and tr == 'none':
            out.append(_item('warn', 'treat', f'السن {t} مكسور ولم يُحدَّد له علاج.'))
        if st.get(t) != 'missing' and tr == 'pontic':
            out.append(_item('warn', 'treat', f'السن {t} موجود لكنه محدد كسن معلّق في جسر — هل سيُخلع؟'))
    if meas and meas.get('calibrated'):
        w = mean2(meas['teeth'].get('11', {}).get('wmm'), meas['teeth'].get('21', {}).get('wmm'))
        if w is not None and (w < 6.5 or w > 11):
            out.append(_item('warn', 'calib', f'عرض الثنية المحسوب {w:.1f} ملم غير معتاد — راجع المعايرة.'))
    else:
        out.append(_item('warn', 'calib', 'الصورة غير معايرة: لا مقاسات بالملم في ورقة المعمل.'))
    if det_level and det_level != 'ok' and source == 'auto':
        out.append(_item('warn', 'detection', 'الكشف التلقائي أعطى تنبيهات ولم تُعدَّل الحدود.'))
    if rest and rest.get('display') is not None and (rest['display'] < -3 or rest['display'] > 8):
        out.append(_item('warn', 'rest', f"ظهور الثنية في الراحة {rest['display']:.1f} ملم قيمة غير معتادة — راجع صورة الراحة."))
    for t in TEETH:
        c = (design_changes or {}).get(t)
        if not c or c.get('dWmm') is None:
            continue
        if abs(c['dWmm']) > 2 and st.get(t) != 'missing':
            out.append(_item('warn', 'width_change', f"تغيير عرض السن {t} بمقدار {c['dWmm']:.1f} ملم كبير — قد يحتاج تقويماً أو تحضيراً كبيراً."))
        if c['dIncisalmm'] < -2 and st.get(t) != 'missing':
            out.append(_item('warn', 'shorten', f"تقصير السن {t} بمقدار {-c['dIncisalmm']:.1f} ملم كبير."))
    return out
