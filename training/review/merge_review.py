"""Turn a reviewer's per-tooth verdicts into training-ready labels (annotation-tool export format).

    python merge_review.py ai_verdicts.json mapping.json drafts_dir out.zip

Only photos whose drawn outlines were all judged right are kept. A canine at the edge of the mouth that is visible but
unmarked (or badly outlined) does not cost the photo: its polygon is dropped and the mouth region beyond the lateral
incisor on that side is set to IGNORE (255), so training learns nothing there instead of learning 'no tooth'.
"""
import json, sys, zipfile
import numpy as np, cv2

TEETH = ['13', '12', '11', '21', '22', '23']
IGNORE = 255


def main(vpath, mpath, ddir, out):
    V, M = json.load(open(vpath)), json.load(open(mpath))
    reviewer = V.get('reviewer', 'ai')
    z = zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED); rows = []; n_ign = 0
    for code, tv in sorted(V['keep'].items()):
        did = M[code]; meta = json.load(open(f'{ddir}/meta/{did}.json')); W, H = meta['width'], meta['height']
        polys, status, mask, ign = {}, {}, np.zeros((H, W), np.uint8), []
        for t in TEETH:
            v = tv[t]
            if v in ('ok', 'broken'):
                assert t in meta['teeth'], (code, t); polys[t] = meta['teeth'][t]; status[t] = 'broken' if v == 'broken' else 'present'
            elif v in ('missing', 'hidden'):
                status[t] = v
            else:
                assert t in ('13', '23'), f'{code}: only canines may be ignored, not {t}'
                status[t] = 'present'; ign.append(t)
        for t, p in polys.items():
            cv2.fillPoly(mask, [np.array([[q['x'], q['y']] for q in p], np.int32)], 1 + TEETH.index(t))
        if ign:
            mouth = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mouth, [np.array([[q['x'], q['y']] for q in meta['regions']['mouth']], np.int32)], 1)
            xs = np.arange(W)[None, :].repeat(H, 0)
            for t in ign:
                if t == '13':
                    ref = next(p for k, p in polys.items() if k in ('12', '11'))
                    edge = min(q['x'] for q in ref); side = xs < edge
                else:
                    ref = next(p for k, p in polys.items() if k in ('22', '21'))
                    edge = max(q['x'] for q in ref); side = xs > edge
                mask[(mouth > 0) & side & (mask == 0)] = IGNORE; n_ign += 1
        m2 = dict(meta, teeth=polys, status=status, reviewed=True, who=reviewer, reviewer=reviewer, ignored=ign)
        z.write(f'{ddir}/images/{did}.jpg', f'images/{did}.jpg')
        z.writestr(f'masks/{did}.png', cv2.imencode('.png', mask)[1].tobytes())
        z.writestr(f'meta/{did}.json', json.dumps(m2, ensure_ascii=False)); rows.append(did)
    z.writestr('manifest.csv', '﻿id,reviewer\n' + ''.join(f'{d},{reviewer}\n' for d in rows))
    z.write(f'{ddir}/SOURCES.csv', 'SOURCES.csv'); z.close()
    print(f'{len(rows)} photos kept, {n_ign} canine regions ignored, {sum(len(v) for v in V["drop"].values())} dropped')


if __name__ == '__main__':
    main(*sys.argv[1:5])
