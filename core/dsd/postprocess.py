"""From the segmentation model's probabilities to a label map with structural guarantees.

Guarantees (hold whatever the model outputs):
  G1  no tooth pixel outside the mouth opening (when a mouth mask is given);
  G2  one label per pixel (no overlap between teeth);
  G3  each tooth label is a single connected region;
  G4  labels run 13,12,11,21,22,23 from left to right, each at most once (assign_fdi);
  G5  every tooth gets a confidence; the caller decides what needs the doctor's confirmation.
Needs numpy and OpenCV (server side).
"""
import numpy as np
import cv2

from .constants import TEETH
from .numbering import assign_fdi


def softmax(logits, axis=0):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def teeth_from_probs(prob, mouth=None, min_area_frac=0.0015, other_prob=None):
    """prob: (7, H, W) softmax over [background, 13, 12, 11, 21, 22, 23].
    mouth: optional (H, W) bool/uint8 mask of the lip opening.
    Returns dict(labels (H,W) uint8 with 0..6, status, confidence per tooth, instances)."""
    C, H, W = prob.shape
    arg = prob.argmax(0).astype(np.uint8)
    if mouth is not None:
        arg[np.asarray(mouth) == 0] = 0                                     # G1
    min_area = max(20, int(min_area_frac * H * W))
    inst, masks = [], []
    for c in range(1, 7):                                                  # candidate regions: every component of every tooth class
        n, cc, st, _ = cv2.connectedComponentsWithStats((arg == c).astype(np.uint8), 8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < min_area:
                continue
            m = cc == k
            ys, xs = np.nonzero(m)
            probs = {t: float(prob[1 + i][m].mean()) for i, t in enumerate(TEETH)}
            if other_prob is not None:
                probs['other'] = float(other_prob[m].mean())
            inst.append(dict(x0=float(xs.min()), x1=float(xs.max() + 1), y0=float(ys.min()), y1=float(ys.max() + 1),
                             area=int(m.sum()), probs=probs))
            masks.append(m)
    res = assign_fdi(inst)
    out = np.zeros((H, W), np.uint8)
    conf = {}
    for r, m, lab, cf in zip(inst, masks, res['labels'], res['confidence']):
        if lab is None or lab == 'other':
            continue
        out[m] = 1 + TEETH.index(lab)                                       # G2: one label per pixel
        conf[lab] = cf
    # G3: if one label ended up in pieces (cannot after assign_fdi, but keep the invariant explicit)
    for c in range(1, 7):
        n, cc, st, _ = cv2.connectedComponentsWithStats((out == c).astype(np.uint8), 8)
        if n > 2:
            keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            out[(out == c) & (cc != keep)] = 0
    status = dict(res['status'])
    for t in gap_check(out, status):
        status[t] = 'unsure'
    return dict(labels=out, status=status, confidence=conf, instances=inst, assignment=res['labels'])


NEIGHBOURS = (('13', '12'), ('12', '11'), ('21', '22'), ('22', '23'))   # (outer, inner) on each side; 11|21 = midline


def gap_check(labels, status, frac=0.6, overlap=0.5):
    """Numbering safety net: two teeth that are neighbours in the sequence cannot have a tooth-sized empty space
    between them. If they do, a tooth is missing there and the outer one may carry the wrong number (a network that
    hesitates between 13 and 12 can split the canine in two and so 'use up' the number of a lost lateral). The outer
    tooth is returned for confirmation. Two neighbours overlapping over more than half the narrower one's width are
    the same tooth split in two: both are returned. (A diastema between 11 and 21 is common and is not checked.)"""
    ext = {}
    for i, t in enumerate(TEETH):
        xs = np.nonzero((labels == 1 + i).any(0))[0]
        if status.get(t) == 'present' and len(xs) > 2:
            ext[t] = (float(np.percentile(xs, 2)), float(np.percentile(xs, 98)))
    if len(ext) < 2:
        return []
    w = float(np.median([b - a for a, b in ext.values()]))
    out = []
    for outer, inner in NEIGHBOURS:
        if outer in ext and inner in ext:
            (a0, a1), (b0, b1) = ext[outer], ext[inner]
            gap = b0 - a1 if outer in ('13', '12') else a0 - b1
            if gap > frac * w:
                out.append(outer)
            # neighbours sit side by side: one lying mostly within the other's span is the same tooth split in two
            ov = min(a1, b1) - max(a0, b0)
            if ov > overlap * min(a1 - a0, b1 - b0):
                out += [t for t in (outer, inner) if t not in out]
    return out


def check_invariants(labels, mouth=None):
    """True if G1-G4 hold for a label map (used in tests and as a runtime assertion)."""
    if mouth is not None and np.any((labels > 0) & (np.asarray(mouth) == 0)):
        return False, 'tooth outside mouth'
    xs = []
    for c in range(1, 7):
        m = (labels == c).astype(np.uint8)
        if not m.any():
            continue
        n, _, _, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 2:
            return False, f'class {c} in {n - 1} pieces'
        xs.append((c, np.nonzero(m)[1].mean()))
    cs = [c for c, _ in sorted(xs, key=lambda v: v[1])]
    if cs != sorted(cs):
        return False, f'order {cs}'
    return True, ''


def appearance_check(labels, rgb, rel_bright=0.55, dark_missing=0.50):
    """Catch a 'tooth' that is not a tooth: segmentation networks can paint a tooth into a dark gap from context
    alone (a lost tooth). A region is suspicious when its brightness is far below that of the other teeth in the
    SAME photo (relative, so dim photos and yellow teeth are not flagged). Returns ({tooth: 'missing'|'unsure'},
    per-tooth stats). 'missing' when most of the region is dark mouth cavity."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    V = hsv[..., 2].astype(np.float32)
    dark = V < 70
    med, stats = {}, {}
    for i, t in enumerate(TEETH):
        m = labels == 1 + i
        if m.any():
            med[t] = float(np.median(V[m]))
            stats[t] = dict(brightness=round(med[t], 1), dark=round(float(dark[m].mean()), 3))
    verdict = {}
    if len(med) < 2:
        return verdict, stats
    for t, v in med.items():
        others = [w for k, w in med.items() if k != t]
        ref = float(np.median(others))
        stats[t]['relative'] = round(v / max(1.0, ref), 3)
        if v < rel_bright * ref:
            verdict[t] = 'missing' if stats[t]['dark'] >= dark_missing else 'unsure'
    return verdict, stats


def vertical_check(labels, mouth, top_frac=0.45, bottom_frac=0.20, min_cols=3):
    """Catch a LOWER tooth labelled as an upper one (happens when upper incisors are missing: the network
    takes the lower incisors behind the gap). Upper teeth hang from the upper lip: their top edge sits in
    the upper part of the opening (a gummy smile still keeps it well above the middle). A region whose
    top is past `top_frac` of the local opening height AND whose bottom reaches the lower lip is a lower
    tooth, or only the lower half of an upper tooth (a broken mask). Either way the label is wrong.
    Returns ({tooth: 'misplaced'}, per-tooth stats)."""
    mouth = np.asarray(mouth) > 0
    cols = np.nonzero(mouth.any(0))[0]
    if len(cols) == 0:
        return {}, {}
    H = mouth.shape[0]
    idx = np.arange(H)[:, None]
    top = np.where(mouth, idx, H).min(0)
    bot = np.where(mouth, idx, -1).max(0)
    tall = 0.4 * float((bot - top)[cols].max())       # near the corners the opening is a sliver: no reliable verdict there
    verdict, stats = {}, {}
    for i, t in enumerate(TEETH):
        m = labels == 1 + i
        xs = [x for x in np.nonzero(m.any(0))[0] if bot[x] - top[x] > max(4, tall)]
        if len(xs) < min_cols:
            continue
        xs = np.array(xs)
        ytop = np.where(m[:, xs], idx, H).min(0)
        ybot = np.where(m[:, xs], idx, -1).max(0)
        h = (bot[xs] - top[xs]).astype(np.float32)
        tf = float(np.median((ytop - top[xs]) / h)); bf = float(np.median((bot[xs] - ybot) / h))
        stats[t] = dict(top=round(tf, 3), bottom=round(bf, 3))
        if tf > top_frac and bf < bottom_frac:
            verdict[t] = 'misplaced'
    return verdict, stats
