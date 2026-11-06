"""Aesthetic rules engine: each rule gives value, target range and ok / warn / fail / na."""
import math

from .constants import TEETH, SMILE_ARC_LABEL
from .measure import mean2

SYM_KEY = {'sym_central': 'central', 'sym_lateral': 'lateral', 'sym_canine': 'canine'}


def _n(v):
    """Number as JavaScript prints it (1.0 -> '1')."""
    return f'{v:g}'


def _fmt(v, d=2):
    return '—' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{d}f}'


def evaluate_rules(meas, rules, face=None, rest=None, teeth=None, proportion_p=0.70, edge_px=None, face_edge_px=None):
    """meas: from measure(); face: from face_analysis(); rest: from rest_analysis(); teeth: current boxes.
    edge_px: uncertainty (one standard deviation) of one tooth edge, in the boxes' pixels. A value outside the range
    by less than two standard deviations cannot be told from a normal one at this resolution: status 'uncertain'."""
    res = []
    p = proportion_p
    fm = (face or {}).get('metrics', {}) if face else {}
    t = meas['teeth']
    mmpp = meas['mm_per_px']
    e = edge_px or 0
    sz = math.sqrt(2) * e

    def avg(a, b, k):
        v = [t[x][k] for x in (a, b) if x in t and (k == 'w' or not t[x].get('broken'))]
        return sum(v) / len(v) if v else None

    def sig_ratio(r, A, B):
        return r * math.hypot(sz / A, sz / B) if (e and r is not None and A and B) else 0
    fe = face_edge_px or 0                       # the same edge uncertainty in photo pixels (face measurements)
    fsc = (face or {}).get('scale') or 0
    sig_mm = math.sqrt(2) * fe * fsc if fe and fsc else 0

    def sig_deg(span):
        return math.degrees(math.sqrt(2) * fe / span) if fe and span else 0
    S2 = {k: avg(a, b, q) for k, (a, b, q) in dict(cW=('11', '21', 'w'), cH=('11', '21', 'h'), lW=('12', '22', 'w'), lH=('12', '22', 'h'),
                                                     kW=('13', '23', 'w'), kH=('13', '23', 'h')).items()}
    for rule in rules:
        if not rule.get('enabled'):
            continue
        out = dict(id=rule['id'], group=rule['group'], label=rule['label'], value=None, target='', status='na', msg='')
        rr = dict(rule)

        def rng(v, unit='', sig=0):
            out['value'] = v
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out['status'] = 'na'
                return
            lo = rr.get('min', -math.inf)
            hi = rr.get('max', math.inf)
            if sig > 0:
                out['sigma'] = sig
            if rr.get('min') is not None and rr.get('max') is not None:
                out['target'] = f"{_n(rr['min'])}–{_n(rr['max'])}{unit}"
            elif rr.get('max') is not None:
                out['target'] = f"≤ {_n(rr['max'])}{unit}"
            else:
                out['target'] = f"≥ {_n(rr.get('min'))}{unit}"
            span = (hi - lo) if (math.isfinite(hi) and math.isfinite(lo)) else (abs(hi if math.isfinite(hi) else lo) or 1)
            if lo <= v <= hi:
                out['status'] = 'ok'
            else:
                out['status'] = 'fail' if (v < lo - span * 0.5 or v > hi + span * 0.5) else 'warn'
            if sig > 0 and out['status'] != 'ok' and ((lo - v) if v < lo else (v - hi)) < 2 * sig:
                out['status'] = 'uncertain'

        needs = rule.get('needs')
        if needs == 'rest' and not (rest and rest.get('display') is not None):
            out['msg'] = 'يحتاج صورة الشفاه في الراحة'; res.append(out); continue
        if needs == 'calib' and not meas['calibrated']:
            out['msg'] = 'يحتاج معايرة بالملم'; res.append(out); continue
        if needs == 'face' and not face:
            out['msg'] = 'يحتاج تحليل صورة الوجه'; res.append(out); continue

        rid = rule['id']
        if rid == 'central_wl':
            gd, gdp = fm.get('gingival_display'), fm.get('gingival_display_px')
            if face and (gd < -0.5 if gd is not None else gdp is not None and gdp < -3):
                # the lip hides the gum line: only part of the crown is visible, so W/L cannot be measured
                out['msg'] = 'الشفة تغطي حد اللثة، فالطول الظاهر أقصر من طول السن الحقيقي ولا تُقاس النسبة من هذه الصورة'
            else:
                rng(meas['ratios']['central_wl'], '', sig_ratio(meas['ratios']['central_wl'], S2['cW'], S2['cH']))
                out['msg'] = f"11: {_fmt(t.get('11', {}).get('wl'))} · 21: {_fmt(t.get('21', {}).get('wl'))}"
        elif rid in ('lat_central_w', 'can_lat_w'):
            tol = rule.get('tol', 0.06 if rid == 'lat_central_w' else 0.08)
            rr['min'], rr['max'] = round(p - tol, 3), round(p + tol, 3)
            A, B = (S2['lW'], S2['cW']) if rid == 'lat_central_w' else (S2['kW'], S2['lW'])
            rng(meas['ratios'][rid], '', sig_ratio(meas['ratios'][rid], A, B)); out['msg'] = f'الهدف {p}'
        elif rid in ('lat_central_len', 'can_central_len'):
            A = S2['lH'] if rid == 'lat_central_len' else S2['kH']
            rng(meas['ratios'][rid], '', sig_ratio(meas['ratios'][rid], A, S2['cH']))
        elif rid in SYM_KEY:
            s = meas['symmetry'][SYM_KEY[rid]]
            if s:
                v = max(s['dw'], s['dh'])
                A, B = t[s['a']], t[s['b']]
                den = max(A['w'], B['w']) if s['dw'] >= s['dh'] else max(A['h'], B['h'])
                rng(v, '', math.sqrt(2) * sz / den if e and den else 0); out['target'] = f"≤ {rule['max'] * 100:.0f}%"
                out['msg'] = f"فرق العرض {s['dw'] * 100:.1f}% · فرق الطول {s['dh'] * 100:.1f}%"
                out['value_text'] = f'{v * 100:.1f}%'
        elif rid == 'central_dominance':
            cs = [t[k]['w'] for k in ('11', '21') if k in t]
            others = [t[k]['w'] for k in ('12', '22', '13', '23') if k in t]
            if cs and others:
                ok = min(cs) >= max(others)
                out.update(status='ok' if ok else 'warn', value_text='نعم' if ok else 'لا', target='نعم')
        elif rid == 'central_len_mm':
            hv = lambda k: t[k]['hmm'] if k in t and not t[k]['broken'] else None
            rng(mean2(hv('11'), hv('21')), ' ملم')
        elif rid == 'central_w_mm':
            rng(mean2(t.get('11', {}).get('wmm'), t.get('21', {}).get('wmm')), ' ملم')
        elif rid in ('zenith_lateral', 'zenith_canine', 'zenith_sym', 'incisal_step'):
            st = meas.get('status') or {}
            if teeth:
                T = {k: teeth[k] for k in TEETH if teeth.get(k) and st.get(k) != 'missing'
                     and not (rid == 'incisal_step' and st.get(k) == 'broken')}
                y = lambda k, f: T[k][f] if k in T else None
                zc = mean2(y('11', 'y0'), y('21', 'y0'))
                if rid == 'zenith_sym':
                    if '11' in T and '21' in T:
                        rng(abs(T['11']['y0'] - T['21']['y0']) * mmpp, ' ملم')
                elif rid == 'incisal_step':
                    ic, il = mean2(y('11', 'y1'), y('21', 'y1')), mean2(y('12', 'y1'), y('22', 'y1'))
                    if ic is not None and il is not None:
                        rng((ic - il) * mmpp, ' ملم'); out['msg'] = 'موجب = الرباعية أقصر'
                else:
                    ids = ('12', '22') if rid == 'zenith_lateral' else ('13', '23')
                    z = mean2(y(ids[0], 'y0'), y(ids[1], 'y0'))
                    if zc is not None and z is not None:
                        rng((z - zc) * mmpp, ' ملم'); out['msg'] = 'موجب = أنزل (أقرب للحافة القاطعة)'
        elif rid == 'midline_dev':
            rng(fm.get('midline_dev'), ' ملم', sig_mm)
            if fm.get('midline_dir'):
                out['msg'] = fm['midline_dir']
        elif rid in ('commissure_cant', 'incisal_cant'):
            if fm.get(rid) is not None:
                rng(abs(fm[rid]), '°', sig_deg(fm.get('commissure_span_px' if rid == 'commissure_cant' else 'incisal_span_px')))
        elif rid == 'gingival_display':
            rng(fm.get('gingival_display'), ' ملم', sig_mm); out['msg'] = 'سالب = الشفة تغطي جزءاً من السن'
        elif rid == 'buccal_corridor':
            bc, dsp, sw = fm.get('buccal_corridor'), fm.get('dentition_span_px'), fm.get('smile_width_px')
            rng(bc, '', bc * math.hypot(math.sqrt(2) * fe / dsp, math.sqrt(2) * fe / sw) if fe and bc is not None and dsp and sw else 0)
        elif rid == 'smile_arc':
            if fm.get('smile_arc'):
                a = fm['smile_arc']
                out.update(status='ok' if a == 'consonant' else 'fail' if a == 'reverse' else 'warn',
                           value_text=SMILE_ARC_LABEL[a], target=SMILE_ARC_LABEL['consonant'])
                # a class decided by a hair: the measured curve is closer to a class boundary than twice its error
                if out['status'] != 'ok' and fe and fm.get('smile_arc_ref_px'):
                    si = math.sqrt(1.5) * fe / fm['smile_arc_ref_px']; inc, lip = fm['smile_arc_inc'], fm['smile_arc_lip']
                    near = lambda v, b, sg: abs(v - b) < 2 * sg
                    fragile = near(inc, -0.01, si) or near(inc, 0.015, si)
                    if not fragile and inc >= 0.015:
                        fragile = near(lip, 0.005, si)
                        if not fragile and lip > 0.005:
                            r = inc / lip; sr = r * math.hypot(si / inc, si / lip)
                            fragile = near(r, 0.5, sr) or near(r, 2, sr)
                    if fragile:
                        out['status'] = 'uncertain'; out['sigma'] = si
        elif rid == 'alar_cant':
            if fm.get('alar_cant') is not None:
                rng(abs(fm['alar_cant']), '°')
        elif rid == 'canine_alar':
            rng(fm.get('canine_alar_ratio'))
        elif rid == 'lower_third':
            rng(fm.get('lower_third'))
        elif rid == 'rest_display':
            rng(rest['display'], ' ملم')
            if rest.get('approx'):
                out['msg'] = 'المقياس تقريبي'
        if out.get('value_text') is None and out['value'] is not None:
            out['value_text'] = _fmt(out['value'], 1 if rule.get('unit') in ('deg', 'mm') else 2)
        if out['status'] == 'uncertain':
            u, b2 = rule.get('unit'), 2 * out['sigma']
            band = f'{b2 * 100:.1f}%' if u == 'pct' else f'{b2:.1f}°' if u == 'deg' else f'{b2:.1f} ملم' if u == 'mm' else f'{b2:.2f}'
            note = 'قريب من الحد بين تصنيفين ضمن هامش خطأ القياس — صورة أقرب تحسم' if u == 'class' else f'ضمن هامش خطأ القياس (±{band}) — صورة أقرب تحسم'
            out['msg'] = (out['msg'] + ' · ' if out['msg'] else '') + note
        res.append(out)
    return res
