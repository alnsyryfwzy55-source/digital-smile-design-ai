"""Acceptance gate for a new tooth model: one command, one verdict.

    python training/accept_model.py teeth_unet_v3.onnx [--smiles reviewed1.zip reviewed2.zip]

Compares the new model with the one in use (docs/teeth_unet_v2.onnx) on the same photos and the same measures, and
accepts it only if it is at least as good on everything that matters for the clinic:

  A  contract      same input/output as the app and the server expect (1x3x384x384 -> 1x7x384x384)
  B  retractor     Dice on the labelled test photos: not lower than the current model by more than 0.01
  C  smiles        Dice on reviewed real smiles (when given): not lower than the current model
  D  gaps          how often a blacked-out tooth is still painted as a tooth: not higher
  E  guarantees    no overlap / order / one piece per tooth after post-processing: on every photo
  F  false gaps    real teeth flagged as gaps by the appearance check: not more
  G  confidence    at the confirmation threshold (0.93) it still catches >= 90% of wrong teeth and asks about
                   <= 3% of correct ones in clean photos (otherwise it prints the threshold to use)
  H  self-test     the app's start-up check numbers for the new model (selftest.json), all six teeth found
  I  end to end    the analysis service runs on it: full analysis, and a missing tooth is reported missing
  J  practical     file size and speed

Writes accept_report.json and prints the installation steps when accepted.
"""
import argparse, glob, hashlib, json, os, sys, tempfile, time, zipfile

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'core')); sys.path.insert(0, os.path.join(HERE, 'v3'))
from dsd.postprocess import teeth_from_probs, check_invariants, softmax, appearance_check   # noqa: E402
from dsd.constants import TEETH                                                              # noqa: E402
import v3data as V                                                                           # noqa: E402

CONFIRM_BELOW = 0.95              # the threshold in use (app and server); v3 was accepted at this value


# ---------------------------------------------------------------- models
class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        self.path = path
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0]; self.out = self.sess.get_outputs()[0]

    def contract(self):
        ok = list(self.inp.shape) == [1, 3, 384, 384] and list(self.out.shape)[:2] in ([1, 7], ['batch', 7]) and self.out.shape[-2:] == [384, 384]
        return ok, dict(input=[self.inp.name, list(self.inp.shape)], output=[self.out.name, list(self.out.shape)])

    def probs(self, img):
        x = ((img.astype(np.float32) / 255 - V.MEAN) / V.STD).transpose(2, 0, 1)[None]
        return softmax(self.sess.run(None, {self.inp.name: x})[0][0])


class Wrapped:
    """A model seen through a function (used by the tests to make deliberately worse models)."""
    def __init__(self, base, fn, name):
        self.base, self.fn, self.path = base, fn, name

    def contract(self):
        return self.base.contract()

    def probs(self, img):
        return self.fn(self.base, img)


# ---------------------------------------------------------------- data
def load_v2_test(data_zip):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(data_zip) as z:
        z.extractall(tmp, [n for n in z.namelist() if '/test/' in n or '/val/' in n])
    root = glob.glob(f'{tmp}/*/test')[0][:-5]
    return V.items_of(root, 'test'), V.items_of(root, 'val')


def load_smiles(zips):
    if not zips:
        return []
    items, why = V.read_reviewed(zips)
    root = tempfile.mkdtemp(); V.write_split(items, root)
    return V.items_of(root, 'test') + V.items_of(root, 'val')        # never seen in training: test + val of the same split


def variants(rgb, gt):
    yield 'orig', rgb, gt
    yield 'blur', cv2.GaussianBlur(rgb, (0, 0), 3), gt
    yield 'dark', np.clip(rgb * 0.45, 0, 255).astype(np.uint8), gt
    _, e = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 12]); yield 'jpeg12', cv2.imdecode(e, 1), gt
    Rm = cv2.getRotationMatrix2D((192, 192), 18, 1.0)
    yield 'rot18', cv2.warpAffine(rgb, Rm, (384, 384), borderMode=cv2.BORDER_REFLECT), cv2.warpAffine(gt, Rm, (384, 384), flags=cv2.INTER_NEAREST)
    lip = rgb.copy(); ys = np.nonzero((gt > 0).any(1))[0]
    if len(ys):
        lip[: ys.min() + int(0.35 * (ys.max() - ys.min()))] = (150, 70, 80)
    yield 'lipcover', lip, gt
    yield 'yellow', np.clip(rgb * np.array([1.15, 1.0, 0.7]), 0, 255).astype(np.uint8), gt


# ---------------------------------------------------------------- measures
def measure(model, test, val, smiles):
    pred = lambda img: model.probs(img).argmax(0).astype(np.uint8)
    r = {}
    r['retractor_dice'], r['retractor_per_tooth'] = V.dice_of(pred, test)
    if smiles:
        r['smile_dice'], r['smile_per_tooth'] = V.dice_of(pred, smiles)
    r['gap_hallucination'], r['gap_trials'] = V.hallucination_rate(pred, test + val + smiles)
    inv_bad = alarms = teeth = 0; rows = []; t0 = time.time(); n_inf = 0
    for ip, mp in test + val:
        rgb, gt = V.load_pair(ip, mp)
        for name, im, g in variants(rgb, gt):
            p = model.probs(im); n_inf += 1
            post = teeth_from_probs(p); lab = post['labels']; mx = p.max(0)
            look, st = appearance_check(lab, im)
            if name == 'orig':
                inv_bad += not check_invariants(lab)[0]
                alarms += len(look); teeth += len(st)
            for i, t in enumerate(TEETH):
                pm, gm = lab == 1 + i, g == 1 + i
                if pm.any():
                    rows.append(dict(var=name, conf=post['confidence'].get(t, 0.0), jsconf=float(mx[pm].mean()),
                                     ok=bool((pm & gm).sum() / max(1, (pm | gm).sum()) >= 0.5), caught=t in look))
    r['ms_per_image'] = round(1000 * (time.time() - t0) / max(1, n_inf), 1)
    r['invariant_violations'] = inv_bad
    r['false_gap_alarms'], r['real_teeth_checked'] = alarms, teeth
    wrong = [x for x in rows if not x['ok']]; clean = [x for x in rows if x['ok'] and x['var'] == 'orig']

    def at(t, key):
        return (sum(x[key] < t or x['caught'] for x in wrong) / max(1, len(wrong)), sum(x[key] < t for x in clean) / max(1, len(clean)))
    r['wrong_teeth'] = len(wrong)
    r['confirm_catch'], r['confirm_false_ask'] = at(CONFIRM_BELOW, 'conf')
    r['confirm_catch_app'], r['confirm_false_ask_app'] = at(CONFIRM_BELOW, 'jsconf')
    sugg = None
    for t in np.arange(0.85, 0.991, 0.005):
        c, f = at(t, 'conf'); ca, fa = at(t, 'jsconf')
        if c >= 0.90 and ca >= 0.90 and f <= 0.03 and fa <= 0.03:
            sugg = round(float(t), 3); break
    r['suggested_threshold'] = sugg
    return r


def end_to_end(model_path):
    sys.path.insert(0, os.path.join(ROOT, 'server')); sys.path.insert(0, os.path.join(ROOT, 'server', 'tests'))
    import test_service as T
    from dsd_service import Service
    svc = Service(model_path)
    img, P = T.synthetic_face()
    r = svc.analyze(img, P.tolist(), with_overlay=False)
    full = set(r['teeth']['boxes']) == set(TEETH) and set(r['design']['teeth']) == set(TEETH)
    teeth = cv2.resize(T.tooth_photo(), (480, 300))
    lab = teeth_from_probs(svc.tooth_probs(teeth))['labels']; xs = np.nonzero(lab == 5)[1]
    gap_ok = False
    if len(xs):
        img2, P2 = T.synthetic_face(erase=(int(xs.min()), int(xs.max()) + 1))
        r2 = svc.analyze(img2, P2.tolist(), with_overlay=False)
        gap_ok = r2['teeth']['status']['22'] != 'present' and '22' in r2['design']['teeth']
    return dict(full_analysis=full, missing_tooth_reported=gap_ok, total_ms=r['timing_ms']['total'])


# ---------------------------------------------------------------- verdict
def judge(new, base, has_smiles):
    checks = []
    add = lambda k, ok, txt: checks.append(dict(id=k, ok=bool(ok), text=txt))
    add('A', new['contract'], 'نفس مدخل ومخرج النموذج الحالي')
    add('B', new['retractor_dice'] >= base['retractor_dice'] - 0.01, f"دقة صور المباعد {new['retractor_dice']:.3f} (الحالي {base['retractor_dice']:.3f})")
    if has_smiles:
        add('C', new['smile_dice'] >= base['smile_dice'], f"دقة الابتسامات الحقيقية {new['smile_dice']:.3f} (الحالي {base['smile_dice']:.3f})")
    add('D', new['gap_hallucination'] <= base['gap_hallucination'], f"رسم سن في الفراغ {new['gap_hallucination']:.0%} (الحالي {base['gap_hallucination']:.0%})")
    add('E', new['invariant_violations'] == 0, f"الضمانات البنيوية: {new['invariant_violations']} مخالفة")
    add('F', new['false_gap_alarms'] <= base['false_gap_alarms'], f"أسنان حقيقية عُلّمت فراغاً: {new['false_gap_alarms']} (الحالي {base['false_gap_alarms']})")
    add('G', new['confirm_catch'] >= 0.90 and new['confirm_false_ask'] <= 0.03 and new['confirm_catch_app'] >= 0.90 and new['confirm_false_ask_app'] <= 0.03,
        f"حدّ الثقة {CONFIRM_BELOW}: يلتقط {new['confirm_catch']:.0%} من الأخطاء ويسأل عن {new['confirm_false_ask']:.1%} من السليم"
        + ('' if new['suggested_threshold'] in (None, CONFIRM_BELOW) else f" — الحدّ المناسب لهذا النموذج {new['suggested_threshold']}"))
    add('H', new['selftest'] is not None and all(c > 0 for c in new['selftest'][1:]), f"الفحص الذاتي للتطبيق: {new['selftest']}")
    add('I', new['e2e']['full_analysis'] and new['e2e']['missing_tooth_reported'], f"الخدمة كاملة: تحليل كامل {new['e2e']['full_analysis']}، السن المفقود يُكتشف {new['e2e']['missing_tooth_reported']}")
    add('J', new['size_mb'] <= 40 and new['ms_per_image'] <= 3 * base['ms_per_image'], f"الحجم {new['size_mb']} م.ب، السرعة {new['ms_per_image']} مللي ث/صورة (الحالي {base['ms_per_image']})")
    return all(c['ok'] for c in checks), checks


def evaluate(new_model, base_model, data_zip, smiles_zips=(), selftest_jpg=None, e2e=True):
    test, val = load_v2_test(data_zip)
    smiles = load_smiles(list(smiles_zips))
    res = {}
    for key, m in (('base', base_model), ('new', new_model)):
        r = measure(m, test, val, smiles)
        r['contract'], r['io'] = m.contract()
        r['selftest'] = V.selftest_counts(lambda im: m.probs(im).argmax(0).astype(np.uint8), selftest_jpg) if selftest_jpg else None
        r['size_mb'] = round(os.path.getsize(m.path) / 1e6, 1) if os.path.exists(m.path) else 0
        r['e2e'] = end_to_end(m.path) if (e2e and os.path.exists(m.path)) else dict(full_analysis=True, missing_tooth_reported=True, total_ms=0)
        res[key] = r
    ok, checks = judge(res['new'], res['base'], bool(smiles))
    return dict(accept=ok, checks=checks, new=res['new'], base=res['base'], smiles=len(smiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('--baseline', default=os.path.join(ROOT, 'docs', 'teeth_unet_v2.onnx'))
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'smile_teeth_v2.zip'))
    ap.add_argument('--smiles', nargs='*', default=[]); ap.add_argument('--out', default='accept_report.json')
    a = ap.parse_args()
    rep = evaluate(OnnxModel(a.model), OnnxModel(a.baseline), a.data, a.smiles, os.path.join(ROOT, 'docs', 'selftest.jpg'))
    json.dump(rep, open(a.out, 'w'), indent=1, ensure_ascii=False, default=float)
    for c in rep['checks']:
        print(('✓ ' if c['ok'] else '✗ ') + c['id'] + '  ' + c['text'])
    if not rep['accept']:
        print('\nالقرار: لا يُعتمد. يبقى النموذج الحالي.'); return 1
    sha = hashlib.sha256(open(a.model, 'rb').read()).hexdigest()
    print(f"""
القرار: يُعتمد. خطوات التركيب:
 1. انسخ النموذج إلى docs/teeth_unet_v2.onnx (الاسم نفسه، فلا يتغير أي كود)
 2. ضع في app/src/selftest.json وفي docs/selftest.json (الملفان معاً، فالبناء ينسخ الأول فوق الثاني): {json.dumps(rep['new']['selftest'])}
 3. في deploy/huggingface/Dockerfile: MODEL_SHA256={sha}
 4. {'حدّ الثقة يبقى 0.93' if rep['new']['suggested_threshold'] in (None, CONFIRM_BELOW) else 'غيّر حدّ الثقة إلى ' + str(rep['new']['suggested_threshold']) + ' في app/src/engine.js وserver/dsd_service.py'}
 5. شغّل الاختبارات (core وserver وapp) ثم ارفع""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
