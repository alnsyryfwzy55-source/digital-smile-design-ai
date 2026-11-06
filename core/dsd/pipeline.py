"""One call from analysed photo data to a validated design: what the server will run for each case.

Input is what the image models produce (tooth boxes + status, lip chains, face points, scale);
no images are needed here, so this part is fully testable without a GPU.
"""
from .constants import TEETH, DEFAULT_RULES, PROPORTIONS
from .measure import Scale, measure
from .face import face_analysis
from .rules import evaluate_rules
from .design import propose_design, apply_smile_arc, limit_changes, design_from_face, fill_missing, chain_y, change_of
from .validate import validate_design


def plan_case(teeth, status, lower_lip, upper_lip, face_points=None, scale=None, rest=None, rules=None,
              proportion='red70', shape='square', midline='facial', facial_midline_x=None, length_delta_mm=0.0,
              gingiva_mode='keep', space_mode='preserve', frame=(800, 500), face_mm_per_px=None, edge_px=None, face_edge_px=None):
    """teeth: boxes found (any subset); status: per tooth; lower_lip/upper_lip: inner-lip chains [{x,y}].

    Returns measurements, rule results, the design (target boxes), per-tooth changes and validation.
    face_points may be in another pixel frame (e.g. the full photo): give face_mm_per_px for that frame.
    Teeth with no box are placed by fill_missing; with no front teeth at all the design comes from the face."""
    rules = rules or DEFAULT_RULES
    sc = scale or Scale()
    st = {t: (status or {}).get(t, 'present' if (teeth or {}).get(t) else 'missing') for t in TEETH}
    mmpp = sc.mm_per_px
    p = PROPORTIONS.get(proportion, {}).get('p', 0.7)
    present = [t for t in TEETH if (teeth or {}).get(t) and st[t] in ('present', 'broken', 'estimated')]
    lower_y = lambda x: chain_y(lower_lip, x)
    if not present:
        mid = facial_midline_x if facial_midline_x is not None else sum(q['x'] for q in upper_lip) / len(upper_lip)
        target = design_from_face(upper_lip, mid, mmpp, p=p)
        source, boxes = 'face', {}
    else:
        boxes = fill_missing({t: teeth[t] for t in present}, *frame)['teeth']
        meas0 = measure(boxes, sc, st)
        d = propose_design(boxes, meas0, rules, proportion=proportion, space_mode=space_mode, gingiva_mode=gingiva_mode,
                           midline=midline if facial_midline_x is not None else 'current', facial_midline_x=facial_midline_x)
        if not d['ok']:
            return dict(ok=False, error=d['error'])
        target, source = d['target'], 'teeth'
    target = apply_smile_arc(target, lower_y, mmpp, length_delta_mm=length_delta_mm)
    if source == 'teeth':
        target = limit_changes(boxes, target, mmpp, st)
    meas = measure(teeth or {}, sc, st)
    face = face_analysis(face_points, face_mm_per_px if face_mm_per_px is not None else mmpp) if face_points else None
    rule_res = evaluate_rules(meas, rules, face=face, rest=rest, teeth=teeth, proportion_p=p, edge_px=edge_px, face_edge_px=face_edge_px)
    changes = {t: change_of(teeth[t], target[t], mmpp) for t in TEETH if (teeth or {}).get(t) and st[t] != 'missing'}
    val = validate_design(target, current={t: teeth[t] for t in present} if present else None, mm_per_px=mmpp, p=p,
                          facial_midline_x=facial_midline_x, status=st)
    return dict(ok=val['ok'], source=source, status=st, measurements=meas, face=face, rules=rule_res,
                target=target, changes=changes, validation=val, proportion=proportion, shape=shape,
                scale=dict(mm_per_px=mmpp, source=sc.source, rel_err=sc.rel_err))
