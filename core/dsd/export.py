"""Design files for the lab: JSON (all numbers) and SVG (outlines; 1:1 in millimetres when calibrated)."""
import json
from datetime import date

from .constants import TEETH, TOOTH_NAME
from .design import tooth_polygon, papilla_points

SCHEMA = 'dsd-design/1'


def design_json(case_id, target, mm_per_px=None, scale_source=None, scale_rel_err=None, shape='square', shade=None,
                proportion=None, status=None, treat=None, changes=None, validation=None, notes='', patient_goals=None):
    """Everything the lab and a later 3-D step need, in millimetres where calibrated (FDI numbering)."""
    teeth = {}
    for t in TEETH:
        b = target[t]
        w, h = b['x1'] - b['x0'], b['y1'] - b['y0']
        teeth[t] = dict(name=TOOTH_NAME[t], status=(status or {}).get(t, 'present'), treatment=(treat or {}).get(t),
                        box_px=dict(b), width_mm=round(w * mm_per_px, 2) if mm_per_px else None,
                        length_mm=round(h * mm_per_px, 2) if mm_per_px else None,
                        change=(changes or {}).get(t))
    return dict(schema=SCHEMA, case=case_id, date=date.today().isoformat(), numbering='FDI',
                scale=dict(mm_per_px=mm_per_px, source=scale_source, rel_err=scale_rel_err),
                shape=shape, shade=shade, proportion=proportion, teeth=teeth,
                papillae=[dict(p, y_px=p['y'], x_px=p['x']) for p in papilla_points(target)],
                patient_goals=patient_goals or {},
                validation=validation, notes=notes,
                disclaimer='محاكاة تقريبية للتخطيط والتواصل، وليست ضماناً للنتيجة السريرية.')


def design_svg(target, mm_per_px=None, shape='square', case_id='', margin_mm=6.0):
    """Outlines of the six designed teeth. With a scale, the SVG is in millimetres (prints 1:1, with a
    10 mm check bar); without one it is in pixels and says it is not to scale."""
    k = mm_per_px or 1.0
    unit = 'mm' if mm_per_px else 'px'
    xs = [target[t][c] for t in TEETH for c in ('x0', 'x1')]
    ys = [target[t][c] for t in TEETH for c in ('y0', 'y1')]
    ox, oy = min(xs) * k - margin_mm, min(ys) * k - margin_mm
    W = (max(xs) - min(xs)) * k + 2 * margin_mm
    H = (max(ys) - min(ys)) * k + 2 * margin_mm + 14
    fs = 2.2 if mm_per_px else 12
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}{unit}" height="{H:.2f}{unit}" viewBox="0 0 {W:.2f} {H:.2f}">',
             f'<title>Smile design {case_id}</title>']
    for t in TEETH:
        pts = ' '.join(f'{x * k - ox:.2f},{y * k - oy:.2f}' for x, y in tooth_polygon(t, target[t], shape))
        b = target[t]
        cx = ((b['x0'] + b['x1']) / 2) * k - ox
        parts.append(f'<polygon id="tooth-{t}" points="{pts}" fill="none" stroke="#000" stroke-width="{0.2 if mm_per_px else 1.5}"/>')
        parts.append(f'<text x="{cx:.2f}" y="{b["y1"] * k - oy + fs * 1.6:.2f}" font-size="{fs}" text-anchor="middle" font-family="sans-serif">{t}</text>')
        if mm_per_px:
            parts.append(f'<text x="{cx:.2f}" y="{b["y1"] * k - oy + fs * 3.0:.2f}" font-size="{fs * 0.8:.2f}" text-anchor="middle" font-family="sans-serif">'
                         f'{(b["x1"] - b["x0"]) * k:.1f}×{(b["y1"] - b["y0"]) * k:.1f}</text>')
    for p in papilla_points(target):
        parts.append(f'<circle cx="{p["x"] * k - ox:.2f}" cy="{p["y"] * k - oy:.2f}" r="{0.35 if mm_per_px else 3}" fill="#000"/>')
    if mm_per_px:
        y = H - 3
        parts.append(f'<line x1="{margin_mm:.2f}" y1="{y:.2f}" x2="{margin_mm + 10:.2f}" y2="{y:.2f}" stroke="#000" stroke-width="0.3"/>')
        parts.append(f'<text x="{margin_mm + 11:.2f}" y="{y + 0.8:.2f}" font-size="2" font-family="sans-serif">10 mm — check after printing (100% scale)</text>')
    else:
        parts.append(f'<text x="4" y="{H - 4:.2f}" font-size="11" font-family="sans-serif">NOT TO SCALE (no calibration)</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def save(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=1)
