"""Measurements from tooth boxes, with the uncertainty of every millimetre value.

A box is a dict {x0, x1, y0, y1} in image pixels. status maps a tooth to
'present' | 'broken' | 'missing' | 'hidden' | 'estimated' | 'unsure'.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from .constants import TEETH, IRIS_MM


@dataclass
class Scale:
    """Millimetres per pixel and how much it can be trusted.

    source: 'caliper' (a tooth measured in the mouth), 'iris' (population mean iris diameter) or None.
    rel_err: relative standard uncertainty of the scale itself.
    edge_px: uncertainty of one tooth boundary in pixels (segmentation + blur).
    """
    mm_per_px: Optional[float] = None
    source: Optional[str] = None
    rel_err: float = 0.0
    edge_px: float = 1.5
    note: str = ''

    @property
    def calibrated(self):
        return bool(self.mm_per_px and self.mm_per_px > 0)

    def mm(self, px):
        return None if not self.calibrated or px is None else px * self.mm_per_px

    def err_mm(self, px, edges=2):
        """Standard uncertainty (mm) of a length of `px` pixels bounded by `edges` boundaries."""
        if not self.calibrated or px is None:
            return None
        v = px * self.mm_per_px
        return math.hypot(v * self.rel_err, math.sqrt(edges) * self.edge_px * self.mm_per_px)


def scale_from_caliper(tooth_box, tooth_mm, which='w', edge_px=1.5):
    """Scale from a tooth measured with a caliper (width 'w' or length 'h').
    The reference tooth's own edges set the scale's uncertainty; perspective adds about 2%."""
    px = (tooth_box['x1'] - tooth_box['x0']) if which == 'w' else (tooth_box['y1'] - tooth_box['y0'])
    if px <= 4 or not tooth_mm:
        return Scale()
    rel = math.hypot(math.sqrt(2) * edge_px / px, 0.02)
    return Scale(tooth_mm / px, 'caliper', rel, edge_px, f'عرض السن المقاس بالفرجار {tooth_mm} ملم')


def scale_from_iris(iris_diam_px, edge_px=1.5):
    """Scale from the visible iris diameter (population mean 11.7 mm, about ±5% between people;
    the iris is also not in the plane of the teeth, so the total is taken as ±10%)."""
    if not iris_diam_px or iris_diam_px <= 4:
        return Scale()
    return Scale(IRIS_MM / iris_diam_px, 'iris', 0.10, edge_px, 'تقديري من قطر القزحية (±10%)')


def measure(teeth, scale=None, status=None):
    """Per-tooth sizes and the ratios the rules use. Missing teeth are left out; broken teeth keep
    their width but not their length (their length is not the real crown length)."""
    sc = scale or Scale()
    st = status or {}
    m = dict(calibrated=sc.calibrated, mm_per_px=sc.mm_per_px, scale_source=sc.source, teeth={}, missing=[], broken=[], status=st)
    for t in TEETH:
        b = (teeth or {}).get(t)
        if not b:
            continue
        if st.get(t) == 'missing':
            m['missing'].append(t)
            continue
        broken = st.get(t) == 'broken'
        if broken:
            m['broken'].append(t)
        w, h = b['x1'] - b['x0'], b['y1'] - b['y0']
        m['teeth'][t] = dict(w=w, h=h, broken=broken, wmm=sc.mm(w), hmm=sc.mm(h),
                             wmm_err=sc.err_mm(w), hmm_err=sc.err_mm(h),
                             wl=(w / h) if h > 0 and not broken else None)
    g = m['teeth'].get
    gw = lambda t: g(t)['w'] if g(t) else None
    gh = lambda t: g(t)['h'] if g(t) and not g(t)['broken'] else None
    cW, cH = mean2(gw('11'), gw('21')), mean2(gh('11'), gh('21'))
    lW, lH = mean2(gw('12'), gw('22')), mean2(gh('12'), gh('22'))
    kW, kH = mean2(gw('13'), gw('23')), mean2(gh('13'), gh('23'))
    m['ratios'] = dict(
        central_wl=cW / cH if cW and cH else None,
        lat_central_w=lW / cW if lW and cW else None,
        can_lat_w=kW / lW if kW and lW else None,
        lat_central_len=lH / cH if lH and cH else None,
        can_central_len=kH / cH if kH and cH else None)

    def side(a, b):
        if not (g(a) and g(b)):
            return None
        dh = abs(gh(a) - gh(b)) / max(gh(a), gh(b)) if gh(a) is not None and gh(b) is not None else 0
        return dict(a=a, b=b, dw=abs(g(a)['w'] - g(b)['w']) / max(g(a)['w'], g(b)['w']), dh=dh)
    m['symmetry'] = dict(central=side('11', '21'), lateral=side('12', '22'), canine=side('13', '23'))
    if teeth and teeth.get('11') and teeth.get('21'):
        m['midline_x'] = (teeth['11']['x1'] + teeth['21']['x0']) / 2
    return m


def mean2(a, b):
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b
