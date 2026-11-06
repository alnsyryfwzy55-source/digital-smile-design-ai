"""FDI numbering of the visible upper front teeth as one global decision.

The segmentation model gives, for each tooth-shaped region (instance), a probability for each of
13 12 11 21 22 23. Taken one by one these can contradict each other (two "11"s, a "12" left of a "13").
Here the labels are chosen together so that:

  * they run strictly 13 -> 12 -> 11 -> 21 -> 22 -> 23 from image left to right (no duplicates, no swaps);
  * regions beyond the canines may only be "other" (premolars), never a front tooth;
  * a skipped tooth is allowed only where there is room for it (a real gap) -> 'missing';
    a tooth skipped at either end of the smile is 'hidden' (behind the lip corner);
  * widths agree with the expected apparent proportions (central 1, lateral 0.7, canine 0.5).

The best assignment is found exactly by dynamic programming (at most ~12 regions, 6 labels).
Pure Python, no image libraries.
"""
import math

from .constants import TEETH, IDEAL_W

W_SIGMA = {t: 0.25 * IDEAL_W[t] + 0.08 for t in TEETH}
P_FLOOR = 1e-4


def _unit_width(insts):
    """Width of one central incisor, estimated robustly from the regions' widths and their most likely labels."""
    cands = []
    for r in insts:
        t, p = max(((t, r['probs'].get(t, 0.0)) for t in TEETH), key=lambda kv: kv[1])
        if p > 0.3:
            cands.append((r['x1'] - r['x0']) / IDEAL_W[t])
    if not cands:
        cands = [(r['x1'] - r['x0']) / 0.8 for r in insts]
    cands.sort()
    return cands[len(cands) // 2]


def assign_fdi(instances, unit=None, other_logp=math.log(0.05), skip_penalty=6.0, missed_gap_penalty=2.0):
    """instances: [{'x0','x1', 'probs': {'13':p, ..., '23':p, 'other':p?}}] in any order.

    Returns {'labels': [tooth label, 'other' (premolar) or None (dropped fragment) per input instance], 'status': {tooth: 'present'|'missing'|'hidden'},
             'confidence': [p of the chosen label per instance], 'unit': central width used, 'score': total}."""
    n = len(instances)
    if n == 0:
        return dict(labels=[], status={t: 'hidden' for t in TEETH}, confidence=[], unit=unit, score=0.0)
    order = sorted(range(n), key=lambda i: (instances[i]['x0'] + instances[i]['x1']) / 2)
    R = [instances[i] for i in order]
    u = unit or _unit_width(R)
    lp = lambda r, t: math.log(max(r['probs'].get(t, 0.0), P_FLOOR))

    def width_term(r, t):
        z = ((r['x1'] - r['x0']) / u - IDEAL_W[t]) / W_SIGMA[t]
        return -0.5 * z * z

    def other_term(r):
        # a premolar seen from the front is narrow (about 0.45 of a central)
        z = ((r['x1'] - r['x0']) / u - 0.45) / 0.2
        base = math.log(max(r['probs'].get('other', 0.0), P_FLOOR)) if 'other' in r['probs'] else other_logp
        return base - 0.5 * z * z

    def edge_term(r_out, r_in, skipped):
        """A premolar is visible next to the labelled run: the front teeth skipped in between cannot be
        hidden behind the lip, so they must be missing and need room."""
        if not skipped:
            return 0.0
        gap = (r_in['x0'] - r_out['x1']) if r_out['x1'] <= r_in['x0'] else (r_out['x0'] - r_in['x1'])
        need = u * sum(IDEAL_W[t] for t in skipped)
        return 0.0 if gap >= 0.45 * need else -skip_penalty * (1 - max(0.0, gap) / max(1e-9, 0.45 * need))

    def gap_term(prev, r, j, k):
        """prev got label j, r gets label k > j. Skipped teeth need room; no skip but a big gap is suspicious."""
        gap = r['x0'] - prev['x1']
        skipped = TEETH[j + 1:k]
        if skipped:
            need = u * sum(IDEAL_W[t] for t in skipped)
            return 0.0 if gap >= 0.45 * need else -skip_penalty * (1 - gap / max(1e-9, 0.45 * need))
        return -missed_gap_penalty if gap > 0.6 * u * min(IDEAL_W[TEETH[j]], IDEAL_W[TEETH[k]]) else 0.0

    def drop_term(r):
        """A small fragment (a sliver of a tooth, a reflection) may be left unlabelled instead of being forced
        into a tooth label. Only regions narrower than 0.35 of a central can be dropped."""
        w = (r['x1'] - r['x0']) / u
        return (-1.0 - 4.0 * w) if w < 0.35 else None

    NEG = -1e18
    drop = [drop_term(r) for r in R]
    # unlabelled regions at the ends: premolar ('other') or dropped fragment, whichever fits better
    free, free_kind = [], []
    for r, d in zip(R, drop):
        o = other_term(r)
        if d is not None and d > o:
            free.append(d); free_kind.append('drop')
        else:
            free.append(o); free_kind.append('other')
    pre = [0.0]
    for v in free:
        pre.append(pre[-1] + v)
    last_other_before = [None] * (n + 1)
    for i in range(n):
        last_other_before[i + 1] = i if free_kind[i] == 'other' else last_other_before[i]
    first_other_after = [None] * (n + 1)
    for i in range(n - 1, -1, -1):
        first_other_after[i] = i if free_kind[i] == 'other' else first_other_after[i + 1]

    base = [[lp(R[i], TEETH[k]) + width_term(R[i], TEETH[k]) for k in range(6)] for i in range(n)]
    dp = [[NEG] * 6 for _ in range(n)]
    bp = [[None] * 6 for _ in range(n)]
    for i in range(n):
        lo = last_other_before[i]
        for k in range(6):
            v = pre[i] + base[i][k] + (edge_term(R[lo], R[i], TEETH[:k]) if lo is not None else 0.0)
            if v > dp[i][k]:
                dp[i][k], bp[i][k] = v, None
        dsum = 0.0
        for i2 in range(i - 1, -1, -1):                       # previous labelled region i2; all between are dropped
            if i2 < i - 1:
                d = drop[i2 + 1]
                if d is None:
                    break
                dsum += d
            for k in range(6):
                for j in range(k):
                    if dp[i2][j] <= NEG / 2:
                        continue
                    v = dp[i2][j] + dsum + base[i][k] + gap_term(R[i2], R[i], j, k)
                    if v > dp[i][k]:
                        dp[i][k], bp[i][k] = v, (i2, j)
    best, end_state = pre[n], None                            # nothing labelled at all
    for i in range(n):
        fo = first_other_after[i + 1]
        tail = pre[n] - pre[i + 1]
        for k in range(6):
            if dp[i][k] <= NEG / 2:
                continue
            v = dp[i][k] + tail + (edge_term(R[fo], R[i], TEETH[k + 1:]) if fo is not None else 0.0)
            if v > best:
                best, end_state = v, (i, k)
    score = best
    chosen = {}
    st = end_state
    while st is not None:
        i, k = st
        chosen[i] = k
        st = bp[i][k]
    seq_idx = sorted(chosen)
    first = seq_idx[0] if seq_idx else n
    last = seq_idx[-1] if seq_idx else -1
    lab_sorted = []
    for i in range(n):
        if i in chosen:
            lab_sorted.append(TEETH[chosen[i]])
        elif i < first or i > last:
            lab_sorted.append(free_kind[i] if free_kind[i] == 'other' else None)
        else:
            lab_sorted.append(None)                           # dropped fragment inside the run
    seq = [chosen[i] for i in seq_idx]
    labels = [None] * n
    for pos, i in enumerate(order):
        labels[i] = lab_sorted[pos]
    used = [TEETH[k] for k in seq]
    status = {}
    for t in TEETH:
        if t in used:
            status[t] = 'present'
        elif not used:
            status[t] = 'hidden'
        else:
            k = TEETH.index(t)
            lo, hi = TEETH.index(used[0]), TEETH.index(used[-1])
            status[t] = 'missing' if lo < k < hi else 'hidden'
    conf = [None if labels[i] is None else instances[i]['probs'].get(labels[i], 0.0) for i in range(n)]
    return dict(labels=labels, status=status, confidence=conf, unit=u, score=score)
