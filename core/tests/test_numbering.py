"""Global FDI numbering: order, duplicates, missing teeth, hidden canines, premolars."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dsd.numbering import assign_fdi
from dsd.constants import TEETH, IDEAL_W


def row(c=80.0, gaps=None, drop=(), probs=None, extra_left=0, extra_right=0, start=100.0):
    """Synthetic regions left->right. probs: {tooth: {label: p}} overrides."""
    out, x = [], start
    for _ in range(extra_left):
        out.append(dict(x0=x, x1=x + 0.45 * c, probs={'13': 0.6, 'other': 0.3})); x += 0.45 * c + 2
    for t in TEETH:
        w = IDEAL_W[t] * c
        if t in drop:
            x += w if (gaps or {}).get(t, True) else 0
            continue
        p = {k: 0.02 for k in TEETH}; p[t] = 0.9
        if probs and t in probs:
            p = dict(probs[t])
        out.append(dict(x0=x, x1=x + w, probs=p, truth=t)); x += w + 1
    for _ in range(extra_right):
        out.append(dict(x0=x, x1=x + 0.45 * c, probs={'23': 0.6, 'other': 0.3})); x += 0.45 * c + 2
    return out


def labels(regs):
    return assign_fdi(regs)['labels']


def test_clean_six():
    r = row()
    res = assign_fdi(r)
    assert res['labels'] == TEETH and all(v == 'present' for v in res['status'].values())


def test_swapped_probabilities_are_put_back_in_order():
    p11 = {k: 0.02 for k in TEETH}; p11['12'] = 0.8; p11['11'] = 0.15     # the model calls 11 "12"
    p12 = {k: 0.02 for k in TEETH}; p12['11'] = 0.8; p12['12'] = 0.15     # and 12 "11"
    r = row(probs={'11': p11, '12': p12})
    assert labels(r) == TEETH                                            # order + widths decide


def test_duplicate_label_never_survives():
    p = {k: 0.02 for k in TEETH}; p['11'] = 0.9
    r = row(probs={'21': p})                                             # two regions say "11"
    out = labels(r)
    assert len(set(out)) == len(out) and out == TEETH


def test_missing_lateral_with_a_gap():
    r = row(drop=('12',))
    res = assign_fdi(r)
    assert res['labels'] == ['13', '11', '21', '22', '23']
    assert res['status']['12'] == 'missing'


def test_no_skip_without_room():
    """Two adjacent regions cannot be 13 and 11 when there is no space for 12 between them."""
    p = {k: 0.02 for k in TEETH}; p['13'] = 0.9
    c = 80.0
    r = [dict(x0=100, x1=100 + 0.7 * c, probs=p),                        # really a lateral, model says canine
         dict(x0=100 + 0.7 * c + 1, x1=100 + 1.7 * c, probs={**{k: 0.02 for k in TEETH}, '11': 0.9}),
         dict(x0=100 + 1.7 * c + 2, x1=100 + 2.7 * c, probs={**{k: 0.02 for k in TEETH}, '21': 0.9})]
    res = assign_fdi(r, unit=c)
    assert res['labels'][0] == '12' and res['status']['13'] == 'hidden'


def test_hidden_canines_at_the_ends():
    r = [x for x in row() if x['truth'] not in ('13', '23')]
    res = assign_fdi(r)
    assert res['labels'] == ['12', '11', '21', '22'] and res['status']['13'] == 'hidden' and res['status']['23'] == 'hidden'


def test_premolars_at_the_ends_become_other():
    r = row(extra_left=1, extra_right=1)
    out = labels(r)
    assert out[0] == 'other' and out[-1] == 'other' and out[1:-1] == TEETH


def test_empty_and_order_of_input_does_not_matter():
    assert assign_fdi([])['labels'] == []
    r = row()
    rev = list(reversed(r))
    assert labels(rev) == list(reversed(TEETH))


def test_vertical_check_flags_lower_tooth_only():
    np = __import__('pytest').importorskip('numpy'); __import__('pytest').importorskip('cv2')
    from dsd.postprocess import vertical_check
    H, W = 200, 400
    mouth = np.zeros((H, W), np.uint8); mouth[40:160, 20:380] = 1
    lab = np.zeros((H, W), np.uint8)
    for k, (x0, x1) in enumerate([(40, 90), (95, 140), (145, 195), (205, 255), (260, 305), (310, 360)]):
        lab[42:120, x0:x1] = 1 + k                 # upper teeth hang from the upper lip
    lab[42:120, 145:195] = 0; lab[105:160, 150:190] = 3   # 11 missing: the network labels the lower incisor
    lab[70:120, 310:360] = 6                             # gummy-looking 23: top at 25% of the opening -> fine
    v, st = vertical_check(lab, mouth)
    assert v == {'11': 'misplaced'}
    assert st['23']['top'] < 0.45
