"""The Python core must give the same numbers as the JavaScript app it was ported from."""
import json, math, os, shutil, subprocess, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import dsd

HERE = os.path.dirname(__file__)
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


@pytest.fixture(scope='module')
def cases():
    out = subprocess.run(['node', os.path.join(HERE, 'js_reference.js'), '300', '7'], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def close(a, b, tol=1e-7):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def boxes_close(A, B):
    assert set(A) == set(B)
    for t in A:
        for k in ('x0', 'x1', 'y0', 'y1'):
            assert close(A[t][k], B[t][k]), (t, k, A[t][k], B[t][k])


def scale_of(c):
    return dsd.Scale(c['mmpp'], 'test', 0.0) if c['mmpp'] else dsd.Scale()


def test_measure(cases):
    for c in cases:
        m = dsd.measure(c['T'], scale_of(c), c['status'])
        j = c['meas']
        assert set(m['teeth']) == set(j['teeth'])
        for t, v in j['teeth'].items():
            for k in ('w', 'h', 'wmm', 'hmm', 'wl'):
                assert close(m['teeth'][t][k], v.get(k)), (t, k)
        for k, v in j['ratios'].items():
            assert close(m['ratios'][k], v), k
        for k, v in j['symmetry'].items():
            if v is None:
                assert m['symmetry'][k] is None
            else:
                assert close(m['symmetry'][k]['dw'], v['dw']) and close(m['symmetry'][k]['dh'], v['dh'])


def ipd_scale(c):
    p, q = c['pts']['pupilR'], c['pts']['pupilL']
    return c['ipdMm'] / math.hypot(q['x'] - p['x'], q['y'] - p['y'])


def test_face_and_rest(cases):
    rename = {'facial_midline_xAtDental': 'facial_midline_x_at_dental'}
    for c in cases:
        f = dsd.face_analysis(c['pts'], ipd_scale(c))
        for k, v in c['face']['metrics'].items():
            pk = rename.get(k, k)
            if isinstance(v, (int, float)):
                assert close(f['metrics'][pk], v), k
            else:
                assert f['metrics'][pk] == v, k
        r = dsd.rest_analysis(c['restPts'], ipd_scale(c), approx=False)
        assert close(r['display'], c['rest']['display'])


def test_rules(cases):
    for c in cases:
        m = dsd.measure(c['T'], scale_of(c), c['status'])
        face = dsd.face_analysis(c['pts'], ipd_scale(c))
        rest = dsd.rest_analysis(c['restPts'], ipd_scale(c), approx=False)
        p = dsd.PROPORTIONS[c['proportion']]['p']
        res = dsd.evaluate_rules(m, dsd.DEFAULT_RULES, face=face, rest=rest, teeth=c['T'], proportion_p=p)
        js_ids = [r['id'] for r in c['rules']]
        res = [r for r in res if r['id'] in js_ids]          # rules added after the port (alar line) are tested elsewhere
        assert [r['id'] for r in res] == js_ids
        for a, b in zip(res, c['rules']):
            assert a['status'] == b['status'], (a['id'], a['status'], b['status'])
            assert close(a['value'], b['value']), a['id']
            assert a['target'] == b['target'] or (not a['target'] and not b['target']), (a['id'], a['target'], b['target'])


def test_rules_with_measurement_precision(cases):
    """Same rules with an edge uncertainty: statuses (incl. 'uncertain'), sigmas and messages match JS exactly."""
    n_unc = n_face = 0
    for c in cases:
        m = dsd.measure(c['T'], scale_of(c), c['status'])
        face = dsd.face_analysis(c['pts'], ipd_scale(c))
        rest = dsd.rest_analysis(c['restPts'], ipd_scale(c), approx=False)
        p = dsd.PROPORTIONS[c['proportion']]['p']
        res = dsd.evaluate_rules(m, dsd.DEFAULT_RULES, face=face, rest=rest, teeth=c['T'], proportion_p=p, edge_px=c['edgePx'], face_edge_px=c['faceEdgePx'])
        js = {r['id']: r for r in c['rulesE']}
        for a in res:
            if a['id'] not in js:
                continue
            b = js[a['id']]
            assert a['status'] == b['status'], (a['id'], a['status'], b['status'], c['edgePx'])
            assert close(a.get('sigma'), b.get('sigma')), a['id']
            if a['status'] == 'uncertain':
                assert a['msg'] == b['msg'], (a['msg'], b['msg']); n_unc += 1
                n_face += a['group'] == 'face'
    assert n_unc > 50 and n_face > 20                    # both the tooth and the face branches are exercised                                    # the new branch is actually exercised


def test_design_arc_limits(cases):
    for c in cases:
        m = dsd.measure(c['T'], scale_of(c), c['status'])
        o = c['opts']
        d = dsd.propose_design(c['T'], m, dsd.DEFAULT_RULES, proportion=o['proportion'], space_mode=o['spaceMode'],
                               symmetric=o['symmetric'], gingiva_mode=o['gingivaMode'], midline=o['midline'],
                               facial_midline_x=o['facialMidlineX'])
        assert d['ok'] == c['design']['ok']
        if not d['ok']:
            continue
        boxes_close(d['target'], c['design']['target'])
        lower_y = lambda x: dsd.chain_y(c['lower'], x)
        arc = dsd.apply_smile_arc(d['target'], lower_y, c['mmpp'], length_delta_mm=c['lengthDeltaMm'])
        boxes_close(arc, c['arc'])
        lim = dsd.limit_changes(c['T'], arc, c['mmpp'], c['status'])
        boxes_close(lim, c['limited'])


def test_from_face_and_fill(cases):
    for c in cases:
        mid = (c['fromFace']['11']['x1'])           # the JS midline is where 11 ends and 21 begins
        f = dsd.design_from_face(c['upper'], mid, c['mmpp'])
        boxes_close(f, c['fromFace'])
        r = dsd.fill_missing(c['found'], 800, 500)
        boxes_close(r['teeth'], c['filled']['teeth'])
        assert r['filled'] == c['filled']['filled']
