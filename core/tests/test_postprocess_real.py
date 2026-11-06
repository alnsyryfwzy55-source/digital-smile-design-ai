"""Post-processing on the real labelled test/val photos (43): guarantees hold and accuracy does not drop."""
import glob, os, sys, tempfile, zipfile
import pytest

np = pytest.importorskip('numpy'); cv2 = pytest.importorskip('cv2'); ort = pytest.importorskip('onnxruntime')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dsd.postprocess import teeth_from_probs, check_invariants, softmax

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
MODEL = os.path.join(ROOT, 'docs', 'teeth_unet_v2.onnx')
DATA = os.path.join(ROOT, 'data', 'smile_teeth_v2.zip')
pytestmark = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(DATA)), reason='model or data not present')


@pytest.fixture(scope='module')
def results():
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(DATA) as z:
        z.extractall(tmp, [n for n in z.namelist() if '/test/' in n or '/val/' in n])
    s = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider']); name = s.get_inputs()[0].name
    M, SD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
    dice = lambda a, b: np.mean([2 * ((a == c) & (b == c)).sum() / max(1, (a == c).sum() + (b == c).sum()) for c in range(1, 7)])
    out = []
    for f in sorted(glob.glob(f'{tmp}/smile_teeth_v2/*/images/*.jpg')):
        img = cv2.imread(f)
        gt = cv2.resize(cv2.imread(f.replace('images', 'masks').replace('.jpg', '.png'), 0), (384, 384), interpolation=cv2.INTER_NEAREST)
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        p = softmax(s.run(None, {name: ((x - M) / SD).transpose(2, 0, 1)[None]})[0][0])
        raw, post = p.argmax(0).astype(np.uint8), teeth_from_probs(p)['labels']
        sw = p.copy(); sw[[3, 2]] = sw[[2, 3]]                 # the model confusing 11 and 12
        er = p.copy(); m = gt == 2
        if m.sum() > 200:
            er[:, m] = 0; er[0, m] = 1
        out.append(dict(raw=dice(raw, gt), post=dice(post, gt), ok=check_invariants(post)[0],
                        sw=dice(teeth_from_probs(sw)['labels'], gt), sw_ok=check_invariants(teeth_from_probs(sw)['labels'])[0],
                        erased=(teeth_from_probs(er)['status']['12'] in ('missing', 'hidden')) if m.sum() > 200 else None))
    return out


def test_guarantees_always_hold(results):
    assert all(r['ok'] and r['sw_ok'] for r in results)


def test_accuracy_does_not_drop(results):
    assert np.mean([r['post'] for r in results]) >= np.mean([r['raw'] for r in results]) - 0.002


def test_numbering_recovers_from_swapped_classes(results):
    assert np.mean([r['sw'] for r in results]) > 0.93


def test_erased_tooth_is_reported_absent(results):
    e = [r['erased'] for r in results if r['erased'] is not None]
    assert len(e) >= 30 and all(e)


def test_appearance_check_no_false_alarms_and_catches_gaps():
    """A blacked-out lateral (a lost tooth) is caught, whether the network paints it or renumbers a neighbour into its
    place; intact photos raise (almost) no flags."""
    from dsd.postprocess import appearance_check
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(DATA) as z:
        z.extractall(tmp, [n for n in z.namelist() if ('/test/' in n or '/val/' in n) and '/images/' in n])
    s = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider']); name = s.get_inputs()[0].name
    M, SD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
    run = lambda rgb: teeth_from_probs(softmax(s.run(None, {name: ((rgb.astype(np.float32) / 255 - M) / SD).transpose(2, 0, 1)[None]})[0][0]))
    alarms = teeth_n = caught = trials = flags = 0
    for f in sorted(glob.glob(f'{tmp}/smile_teeth_v2/*/images/*.jpg')):
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (384, 384), interpolation=cv2.INTER_AREA)
        r = run(rgb); v, st = appearance_check(r['labels'], rgb); alarms += len(v); teeth_n += len(st)
        flags += sum(x == 'unsure' for x in r['status'].values())          # numbering safety net on intact photos
        for c, t in ((2, '12'), (5, '22')):
            m = r['labels'] == c
            if m.sum() < 200:
                continue
            ys, xs = np.nonzero(m); g = rgb.copy(); g[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = (40, 25, 25)
            r2 = run(g); v2, _ = appearance_check(r2['labels'], g); trials += 1
            caught += (r2['status'][t] != 'present') or (t in v2)
    assert alarms == 0 and teeth_n > 200
    assert flags <= 0.02 * teeth_n                     # the gap/overlap check asks about at most 2% of intact teeth
    assert trials >= 60 and caught / trials >= 0.95    # an erased lateral is never silently replaced (painted or renumbered)
