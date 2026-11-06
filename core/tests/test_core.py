"""Behaviour of the parts that are new in the Python core (uncertainty, validation, pipeline, export)."""
import json, math, os, sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dsd
from dsd.pipeline import plan_case


def ideal_teeth(c=90.0, mid=400.0, top=100.0, p=0.7):
    w = {'11': c, '21': c, '12': c * p, '22': c * p, '13': c * p * p, '23': c * p * p}
    L = {'11': c / 0.8, '21': c / 0.8, '12': c / 0.8 * 0.85, '22': c / 0.8 * 0.85, '13': c / 0.8 * 0.95, '23': c / 0.8 * 0.95}
    T, x = {}, mid
    for t in ('11', '12', '13'):
        T[t] = dict(x0=x - w[t], x1=x, y0=top, y1=top + L[t]); x -= w[t]
    x = mid
    for t in ('21', '22', '23'):
        T[t] = dict(x0=x, x1=x + w[t], y0=top, y1=top + L[t]); x += w[t]
    return T


LOWER = [dict(x=150, y=260), dict(x=400, y=300), dict(x=650, y=260)]
UPPER = [dict(x=150, y=120), dict(x=400, y=90), dict(x=650, y=120)]


def test_scale_uncertainty():
    T = ideal_teeth()
    s = dsd.scale_from_caliper(T['11'], 8.5)                 # 90 px -> 8.5 mm
    assert math.isclose(s.mm_per_px, 8.5 / 90)
    m = dsd.measure(T, s)
    assert math.isclose(m['teeth']['11']['wmm'], 8.5)
    assert 0.1 < m['teeth']['11']['wmm_err'] < 0.4          # caliper: a few tenths of a millimetre
    si = dsd.scale_from_iris(110)
    mi = dsd.measure(T, si)
    assert mi['teeth']['11']['wmm_err'] > 0.08 * mi['teeth']['11']['wmm']   # iris: about ±10%
    assert dsd.measure(T)['teeth']['11']['wmm'] is None and dsd.measure(T)['teeth']['11']['wmm_err'] is None


def test_validate_ideal_design_is_clean():
    T = ideal_teeth()
    v = dsd.validate_design(T, current=T, mm_per_px=8.6 / 90, p=0.7, facial_midline_x=400)
    assert v['ok'] and v['items'] == [], v


def test_validate_blocks_and_warnings():
    T = ideal_teeth()
    bad = json.loads(json.dumps(T)); bad['12']['x1'] += 10          # 12 overlaps 11
    v = dsd.validate_design(bad)
    assert not v['ok'] and any(i['code'] == 'overlap' for i in v['items'])
    swapped = json.loads(json.dumps(T)); swapped['13'], swapped['12'] = swapped['12'], swapped['13']
    assert not dsd.validate_design(swapped)['ok']
    v = dsd.validate_design(T, mm_per_px=8.6 / 90, p=0.7, facial_midline_x=400 + 40)   # 3.8 mm off the face midline
    assert v['ok'] and any(i['code'] == 'midline' for i in v['items'])
    rev = json.loads(json.dumps(T))
    for t in ('11', '21'):
        rev[t]['y1'] = rev['13']['y1'] - 10
    assert any(i['code'] == 'smile_arc' for i in dsd.validate_design(rev)['items'])
    big = json.loads(json.dumps(T)); big['11']['y1'] += 40           # +3.8 mm
    assert any(i['code'] == 'lengthen' for i in dsd.validate_design(big, current=T, mm_per_px=8.6 / 90)['items'])
    assert not dsd.validate_design({'11': T['11']})['ok']


def test_pipeline_normal_case():
    T = ideal_teeth()
    sc = dsd.scale_from_caliper(T['11'], 8.6)
    r = plan_case(T, {}, LOWER, UPPER, scale=sc, facial_midline_x=400)
    assert r['ok'] and r['source'] == 'teeth'
    tw = {t: (b['x1'] - b['x0']) * sc.mm_per_px for t, b in r['target'].items()}
    assert math.isclose(tw['11'], tw['21'], rel_tol=1e-9)
    assert math.isclose(tw['12'] / tw['11'], 0.7, rel_tol=1e-6)
    assert r['target']['12']['y1'] < r['target']['11']['y1']         # laterals shorter: smile arc


def test_pipeline_missing_lateral_and_broken_central():
    T = ideal_teeth(); del T['22']
    T['21']['y1'] -= 40                                                  # fractured
    st = {'22': 'missing', '21': 'broken'}
    sc = dsd.scale_from_caliper(T['11'], 8.6)
    r = plan_case(T, st, LOWER, UPPER, scale=sc, facial_midline_x=400)
    assert r['source'] == 'teeth' and set(r['target']) == set(dsd.TEETH)
    assert math.isclose(r['target']['21']['y1'], r['target']['11']['y1'], rel_tol=1e-9)   # broken central rebuilt in full
    assert r['status']['22'] == 'missing' and '22' not in r['changes']


def test_pipeline_no_front_teeth_designs_from_face():
    r = plan_case({}, {}, LOWER, UPPER, scale=dsd.Scale(0.1, 'iris', 0.1), facial_midline_x=400)
    assert r['source'] == 'face' and set(r['target']) == set(dsd.TEETH)
    assert math.isclose((r['target']['11']['x1'] - r['target']['11']['x0']) * 0.1, 8.6, rel_tol=1e-9)


def test_export_json_and_svg_are_valid_and_to_scale():
    T = ideal_teeth()
    mmpp = 8.6 / 90
    j = dsd.design_json('C001', T, mmpp, 'caliper', 0.03, shade='B1', proportion='red70')
    s = json.dumps(j, ensure_ascii=False)
    assert j['schema'] == 'dsd-design/1' and j['teeth']['11']['width_mm'] == 8.6 and 'محاكاة' in s
    svg = dsd.design_svg(T, mmpp, case_id='C001')
    root = ET.fromstring(svg)
    assert root.get('width').endswith('mm')
    polys = [e for e in root.iter() if e.tag.endswith('polygon')]
    assert len(polys) == 6
    xs = [float(p.split(',')[0]) for p in polys[2].get('points').split()]    # tooth 11
    assert math.isclose(max(xs) - min(xs), 8.6, rel_tol=0.02)
    assert 'NOT TO SCALE' in dsd.design_svg(T, None)


def test_dsd_additions_papillae_alar_goals():
    T = ideal_teeth()
    pp = dsd.papilla_points(T)
    assert [p['between'] for p in pp] == ['13-12', '12-11', '11-21', '21-22', '22-23']
    mid = pp[2]; h = T['11']['y1'] - T['11']['y0']
    assert math.isclose(mid['x'], 400) and math.isclose(mid['y'], T['11']['y0'] + 0.4 * h)
    pts = dict(pupilR=dict(x=250, y=-480), pupilL=dict(x=550, y=-480), alarR=dict(x=345, y=-60), alarL=dict(x=455, y=-58),
               cusp13=dict(x=345, y=210), cusp23=dict(x=455, y=210))
    f = dsd.face_analysis(pts, 0.1)['metrics']
    assert abs(f['alar_cant'] - math.degrees(math.atan2(2, 110))) < 1e-9 and math.isclose(f['canine_alar_ratio'], 1.0)
    rules = [r for r in dsd.evaluate_rules(dsd.measure(T), dsd.DEFAULT_RULES, face={'metrics': f}) if r['id'] == 'alar_cant']
    assert rules and rules[0]['status'] == 'ok'
    j = dsd.design_json('C1', T, 0.1, patient_goals={'dislikes': 'الفراغ بين الأسنان'})
    assert len(j['papillae']) == 5 and j['patient_goals']['dislikes']
    lm = [(0.0, 0.0)] * 478
    lm = list(lm); lm[468], lm[473], lm[129], lm[358] = (300, 100), (200, 100), (260, 300), (240, 300)
    fp = dsd.face_points_from_landmarks(lm)
    assert fp['pupilR']['x'] < fp['pupilL']['x'] and fp['alarR']['x'] < fp['alarL']['x']


def test_rules_respect_measurement_precision():
    """A deviation smaller than the measurement error is 'uncertain', a large one stays a finding, no precision = unchanged."""
    import dsd
    def boxes(w11, w21):
        x, T = 0.0, {}
        for t, w in (('13', 60), ('12', 70), ('11', w11), ('21', w21), ('22', 70), ('23', 60)):
            T[t] = dict(x0=x, x1=x + w, y0=0.0, y1=120.0); x += w + 1
        return T
    st = {t: 'present' for t in dsd.TEETH}
    def sym(T, e):
        m = dsd.measure(T, None, st)
        return next(r for r in dsd.evaluate_rules(m, dsd.DEFAULT_RULES, teeth=T, edge_px=e) if r['id'] == 'sym_central')
    small = boxes(100, 92)                      # 8% difference: beyond the 5% limit, but within 2 sigma at 3 px edges
    assert sym(small, None)['status'] in ('warn', 'fail')
    r = sym(small, 3.0)
    assert r['status'] == 'uncertain' and 'هامش خطأ القياس' in r['msg']
    assert sym(small, 0.5)['status'] in ('warn', 'fail')          # a sharp photo still decides
    big = boxes(100, 70)                        # 30% difference: a real finding at any of these resolutions
    assert sym(big, 3.0)['status'] == 'fail'
