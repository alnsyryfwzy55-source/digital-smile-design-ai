"""The acceptance gate must reject worse models (slow: ~8 minutes on CPU)."""
import os, sys
import numpy as np, cv2, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
import accept_model as G   # noqa: E402

MODEL = os.path.join(ROOT, 'docs', 'teeth_unet_v2.onnx'); DATA = os.path.join(ROOT, 'data', 'smile_teeth_v2.zip')
pytestmark = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(DATA)), reason='model or data missing')
JPG = os.path.join(ROOT, 'docs', 'selftest.jpg')


def failed(rep):
    return {c['id'] for c in rep['checks'] if not c['ok']}


@pytest.fixture(scope='module')
def v2():
    return G.OnnxModel(MODEL)


def test_blurry_model_rejected_on_accuracy(v2):
    worse = G.Wrapped(v2, lambda b, img: b.probs(cv2.GaussianBlur(img, (0, 0), 4)), 'blurry')
    rep = G.evaluate(worse, v2, DATA, selftest_jpg=JPG, e2e=False)
    assert not rep['accept'] and 'B' in failed(rep)


def test_more_gap_painting_rejected(v2):
    # a baseline that never paints teeth into dark gaps; v2 (paints them 95% of the time) must not replace it
    def careful(b, img):
        p = b.probs(img); dark = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 2] < 70
        p[1:, dark] = 0; p[0, dark] = 1
        return p
    rep = G.evaluate(v2, G.Wrapped(v2, careful, 'careful'), DATA, selftest_jpg=JPG, e2e=False)
    assert not rep['accept'] and 'D' in failed(rep)


def test_overconfident_model_rejected_on_confirmation(v2):
    def overconfident(b, img):              # same answers, but always ~100% sure: errors would no longer be shown to the doctor
        p = b.probs(img); q = np.zeros_like(p); a = p.argmax(0)
        for c in range(7):
            q[c] = np.where(a == c, 0.999, 0.001 / 6)
        return q
    rep = G.evaluate(G.Wrapped(v2, overconfident, 'overconfident'), v2, DATA, selftest_jpg=JPG, e2e=False)
    assert not rep['accept'] and 'G' in failed(rep)


def test_wrong_contract_rejected(v2):
    class BadIO(G.Wrapped):
        def contract(self):
            return False, dict(input=['x', [1, 3, 256, 256]], output=['y', [1, 2, 256, 256]])
    rep = G.evaluate(BadIO(v2, lambda b, img: b.probs(img), 'bad-io'), v2, DATA, selftest_jpg=JPG, e2e=False)
    assert not rep['accept'] and failed(rep) == {'A'}
