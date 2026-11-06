"""Merge an AI reviewer's verdicts (Gemini, on the numbered contact sheets) into the draft labels.

    python merge_gemini.py verdicts.json mapping.json drafts_dir out_dir

verdicts: [{"id": "001", "exclude": false, "reason": "", "teeth": {"13": "ok", "12": "outline", ...}}, ...]
per tooth: ok | outline (marked, edge wrong) | number (wrong tooth number) | unmarked (visible, no outline) |
           missing | hidden | broken
Writes two ZIPs in the annotation tool's formats:
  ai_reviewed.zip   photos where every tooth is decided and every drawn outline was judged right: training-ready
                    (images + masks 0-6 + meta), marked reviewer="gemini" so a human-only test set can exclude them
  needs_human.zip   drafts with the doubtful teeth set to 'unsure' (the tool will not accept them until decided)
"""
import csv, io, json, os, sys, zipfile
import numpy as np, cv2

TEETH = ['13', '12', '11', '21', '22', '23']
OK_DRAWN = {'ok', 'broken'}
HUMAN = {'outline', 'number', 'unmarked'}
VALID = OK_DRAWN | HUMAN | {'missing', 'hidden'}


def load_verdicts(path):
    txt = open(path, encoding='utf-8').read().strip()
    if txt.startswith('```'):
        txt = txt.strip('`').split('\n', 1)[1]
    try:
        v = json.loads(txt)
    except json.JSONDecodeError:                                   # several JSON arrays pasted one after another
        v = []
        for part in txt.replace('][', ']\n[').split('\n['):
            part = part if part.startswith('[') else '[' + part
            v += json.loads(part[:part.rindex(']') + 1])
    return {str(x['id']).lstrip('#').zfill(3): x for x in v}


def main(vpath, mpath, ddir, out):
    V, M = load_verdicts(vpath), json.load(open(mpath))
    os.makedirs(out, exist_ok=True)
    ok_rows, human_rows, report = [], [], dict(ai_reviewed=0, needs_human=0, excluded=0, no_verdict=0, bad_values=0)
    za = zipfile.ZipFile(os.path.join(out, 'ai_reviewed.zip'), 'w', zipfile.ZIP_STORED)
    zh = zipfile.ZipFile(os.path.join(out, 'needs_human.zip'), 'w', zipfile.ZIP_STORED)
    for code, did in sorted(M.items()):
        meta = json.load(open(f'{ddir}/meta/{did}.json'))
        v = V.get(code)
        if v is None:
            report['no_verdict'] += 1; v = {'teeth': {}}
        if v.get('exclude'):
            report['excluded'] += 1; continue
        tv = {t: str(v.get('teeth', {}).get(t, '')).strip().lower() for t in TEETH}
        bad = [t for t in TEETH if tv[t] not in VALID]
        report['bad_values'] += bool(bad)
        status, polys, doubt = {}, dict(meta['teeth']), []
        for t in TEETH:
            x = tv[t]
            if x in ('missing', 'hidden'):
                status[t] = x; polys.pop(t, None)
            elif x in OK_DRAWN and t in polys:
                status[t] = 'broken' if x == 'broken' else 'present'
            else:                                                  # outline / number / unmarked / ok-but-not-drawn / unknown
                status[t] = 'unsure'; doubt.append(t)
        img = open(f'{ddir}/images/{did}.jpg', 'rb').read()
        if not doubt:
            W, H = meta['width'], meta['height']; mask = np.zeros((H, W), np.uint8)
            for t, p in polys.items():
                cv2.fillPoly(mask, [np.array([[q['x'], q['y']] for q in p], np.int32)], 1 + TEETH.index(t))
            m2 = dict(meta, teeth=polys, status=status, reviewed=True, who='gemini', reviewer='gemini')
            za.writestr(f'images/{did}.jpg', img); za.writestr(f'masks/{did}.png', cv2.imencode('.png', mask)[1].tobytes())
            za.writestr(f'meta/{did}.json', json.dumps(m2, ensure_ascii=False)); ok_rows.append(did); report['ai_reviewed'] += 1
        else:
            m2 = dict(meta, teeth={t: p for t, p in polys.items() if t not in doubt or tv[t] == 'outline'}, status=status,
                      ai_notes={t: tv[t] for t in doubt})
            zh.writestr(f'images/{did}.jpg', img); zh.writestr(f'meta/{did}.json', json.dumps(m2, ensure_ascii=False))
            human_rows.append((did, doubt)); report['needs_human'] += 1
    za.writestr('manifest.csv', '﻿id,reviewer\n' + ''.join(f'{d},gemini\n' for d in ok_rows))
    zh.writestr('manifest.csv', '﻿id,priority,teeth_found,status\n' + ''.join(f'{d},1.0,0,{" ".join(t + ":unsure" for t in dt)}\n' for d, dt in human_rows))
    for z in (za, zh):
        if os.path.exists(f'{ddir}/SOURCES.csv'):
            z.write(f'{ddir}/SOURCES.csv', 'SOURCES.csv')
        z.close()
    json.dump(report, open(os.path.join(out, 'merge_report.json'), 'w'), indent=1)
    print(report)


if __name__ == '__main__':
    main(*sys.argv[1:5])
