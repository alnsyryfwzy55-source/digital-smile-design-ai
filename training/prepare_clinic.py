"""Turn exports of the annotation tool (docs/annotate/) into a training set in the same format as smile_teeth_v2.

    python prepare_clinic.py annotations_2026-11-10.zip [more.zip ...] --out smile_clinic

- Split by PATIENT, not by photo: every photo whose id starts with the same case number (C001_smile, C001_rest,
  C001_retracted ...) goes to the same split, so the test set never contains a patient seen in training.
- Split 60/20/20 by a stable hash of the case number (re-running gives the same split).
- Checks every mask: same size as its image, values 0-6 only. Bad pairs are reported and skipped.
"""
import argparse, csv, hashlib, io, json, os, re, shutil, sys, zipfile
from collections import Counter

import numpy as np
import cv2


def case_of(img_id):
    """C001_smile / C001_rest share a patient; internet photos (ov_..., px_...) are each their own."""
    m = re.match(r'^([A-Za-z]{0,3}\d{2,})_', img_id)
    return m.group(1) if m else img_id


def split_of(case, val=0.2, test=0.2):
    u = int(hashlib.sha1(case.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if u < test else 'val' if u < test + val else 'train'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('zips', nargs='+')
    ap.add_argument('--out', default='smile_clinic')
    a = ap.parse_args(argv)
    shutil.rmtree(a.out, ignore_errors=True)
    for sp in ('train', 'val', 'test'):
        os.makedirs(f'{a.out}/{sp}/images'); os.makedirs(f'{a.out}/{sp}/masks')
    rows, problems, seen = [], [], set()
    for zp in a.zips:
        z = zipfile.ZipFile(zp)
        names = set(z.namelist())
        for n in sorted(names):
            if not (n.startswith('images/') and n.endswith('.jpg')):
                continue
            img_id = n[len('images/'):-4]
            mn = f'masks/{img_id}.png'
            if mn not in names:
                problems.append((img_id, 'no mask')); continue
            if img_id in seen:
                problems.append((img_id, 'duplicate id (later copy skipped)')); continue
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            m = cv2.imdecode(np.frombuffer(z.read(mn), np.uint8), cv2.IMREAD_UNCHANGED)
            if m is not None and m.ndim == 3:
                m = m[..., 0]
            if img is None or m is None or img.shape[:2] != m.shape[:2]:
                problems.append((img_id, 'unreadable or size mismatch')); continue
            if m.max() > 6:
                problems.append((img_id, f'mask value {int(m.max())} > 6')); continue
            meta = json.loads(z.read(f'meta/{img_id}.json')) if f'meta/{img_id}.json' in names else {}
            case = case_of(img_id)
            sp = split_of(case)
            cv2.imwrite(f'{a.out}/{sp}/images/{img_id}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f'{a.out}/{sp}/masks/{img_id}.png', m)
            seen.add(img_id)
            rows.append(dict(split=sp, id=img_id, case=case, mode=meta.get('mode', ''), teeth=int(sum((m == c).any() for c in range(1, 7))),
                             width=img.shape[1], height=img.shape[0], reviewer=meta.get('reviewer', '')))
    with open(f'{a.out}/manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'id', 'case', 'mode', 'teeth', 'width', 'height', 'reviewer'])
        w.writeheader(); w.writerows(rows)
    cnt = Counter(r['split'] for r in rows)
    cases = {sp: len({r['case'] for r in rows if r['split'] == sp}) for sp in ('train', 'val', 'test')}
    print(f"photos: train {cnt['train']}  val {cnt['val']}  test {cnt['test']}  |  patients: {cases}")
    for p in problems:
        print('skipped', *p)
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
