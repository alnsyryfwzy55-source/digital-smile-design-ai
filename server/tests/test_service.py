"""End-to-end tests of the analysis service and its HTTP API on a synthetic face photo built from a real tooth photo."""
import base64, json, math, os, sys, threading, urllib.request, zipfile, tempfile
import numpy as np
import cv2
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))
from dsd_service import Service, AnalysisError, INNER_UPPER, INNER_LOWER   # noqa: E402

MODEL = os.path.join(ROOT, 'docs', 'teeth_unet_v2.onnx')
DATA = os.path.join(ROOT, 'data', 'smile_teeth_v2.zip')
pytestmark = pytest.mark.skipif(not (os.path.exists(MODEL) and os.path.exists(DATA)), reason='model or data missing')


def tooth_photo(name='smile_teeth_v2/test/images/fdi_test_001.jpg'):
    with zipfile.ZipFile(DATA) as z:
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 1), cv2.COLOR_BGR2RGB)


def synthetic_face(roll=8.0, mouth_open=True, erase=None):
    """A grey 'face' with a real tooth photo where the mouth is, rotated by `roll`, and matching landmarks."""
    teeth = cv2.resize(tooth_photo(), (480, 300))
    if erase is not None:                       # paint a dark gap over one tooth region (x0, x1) in mouth pixels
        teeth[:, erase[0]:erase[1]] = (40, 25, 25)
    img = np.full((900, 1200, 3), 175, np.uint8); img[400:700, 360:840] = teeth
    R = cv2.getRotationMatrix2D((600, 550), roll, 1.0)
    img = cv2.warpAffine(img, R, (1200, 900), borderValue=(175, 175, 175))
    rot = lambda p: R[:, :2] @ np.array(p, float) + R[:, 2]
    P = np.tile(rot((600, 500)), (478, 1))
    for i, p in {468: (480, 250), 473: (720, 250), 1: (600, 360), 2: (600, 380), 9: (600, 200), 152: (600, 820),
                 61: (380, 550), 291: (820, 550), 13: (600, 430), 14: (600, 670), 129: (560, 370), 358: (640, 370),
                 469: (430, 250), 471: (530, 250), 474: (670, 250), 476: (770, 250)}.items():
        P[i] = rot(p)
    xs = np.linspace(380, 820, 11)
    ylo = 670 if mouth_open else 432
    for k, i in enumerate(INNER_UPPER): P[i] = rot((xs[k], 430))
    for k, i in enumerate(INNER_LOWER): P[i] = rot((xs[::-1][k], ylo))
    P[13], P[14] = rot((600, 430)), rot((600, ylo))
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes(), P


@pytest.fixture(scope='module')
def svc():
    return Service(MODEL)


def test_full_analysis(svc):
    img, P = synthetic_face()
    r = svc.analyze(img, P.tolist())
    assert abs(r['crop']['roll_deg'] - (-8.0)) < 0.5 or abs(r['crop']['roll_deg'] - 8.0) < 0.5
    assert set(r['teeth']['boxes']) == {'13', '12', '11', '21', '22', '23'}
    assert set(r['design']['teeth']) == {'13', '12', '11', '21', '22', '23'}
    assert r['scale']['source'] == 'iris' and r['scale']['rel_err'] == 0.10
    w = r['design']['teeth']['11']['width_mm']
    assert 5.0 < w < 13.0                                         # plausible with an iris-based scale
    assert '<polygon id="tooth-11"' in r['design_svg']
    ov = cv2.imdecode(np.frombuffer(base64.b64decode(r['overlay_jpg_b64']), np.uint8), 1)
    assert ov.shape[1] == r['crop']['width']
    assert r['timing_ms']['total'] < 15000
    json.dumps(r)                                                 # everything is JSON-serialisable


def test_landmarks_normalised_and_caliper(svc):
    img, P = synthetic_face()
    Pn = P / [1200, 900]
    r = svc.analyze(img, Pn.tolist(), options={'caliper_central_mm': 8.6, 'case_id': 'C001'})
    assert r['scale']['source'] == 'caliper'
    assert math.isclose(r['measurements']['teeth']['11']['wmm'], 8.6, rel_tol=1e-6) or \
        math.isclose(r['measurements']['teeth']['21']['wmm'], 8.6, rel_tol=1e-6)
    assert r['measurements']['teeth']['11']['wmm_err'] < 0.5
    assert r['design']['case'] == 'C001'


def test_gap_becomes_missing_and_is_rebuilt(svc):
    img_full, P = synthetic_face()
    base = svc.analyze(img_full, P.tolist(), with_overlay=False)
    # find tooth 22 in mouth-photo pixels from the base run: crop x -> photo x is roughly linear; erase by label instead
    teeth = cv2.resize(tooth_photo(), (480, 300))
    from dsd.postprocess import teeth_from_probs
    p = svc.tooth_probs(teeth)
    lab = teeth_from_probs(p)['labels']
    xs = np.nonzero(lab == 5)[1]                                   # class 5 = tooth 22
    img_gap, P2 = synthetic_face(erase=(int(xs.min()), int(xs.max()) + 1))
    r = svc.analyze(img_gap, P2.tolist(), with_overlay=False)
    assert r['teeth']['status']['22'] in ('missing', 'hidden', 'unsure')
    assert '22' in r['teeth']['needs_confirmation'] or r['teeth']['status']['22'] == 'hidden'
    assert '22' in r['design']['teeth']                            # the design still rebuilds 22


def test_errors(svc):
    img, P = synthetic_face(mouth_open=False)
    with pytest.raises(AnalysisError) as e:
        svc.analyze(img, P.tolist())
    assert e.value.code == 'MOUTH_CLOSED'
    with pytest.raises(AnalysisError) as e:
        svc.analyze(img, P[:100].tolist())
    assert e.value.code == 'NO_LANDMARKS'


def test_http_api(svc):
    from http.server import ThreadingHTTPServer
    from http_api import make_handler
    srv = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(svc))
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    url = f'http://127.0.0.1:{srv.server_address[1]}'
    try:
        assert json.load(urllib.request.urlopen(url + '/api/health'))['ok']
        img, P = synthetic_face()
        body = json.dumps(dict(image_b64=base64.b64encode(img).decode(), landmarks=P.tolist(), overlay=False)).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(url + '/api/analyze', body, {'Content-Type': 'application/json'})))
        assert set(r['design']['teeth']) == {'13', '12', '11', '21', '22', '23'}
        bad = json.dumps(dict(image_b64=base64.b64encode(img).decode(), landmarks=P[:10].tolist())).encode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url + '/api/analyze', bad, {'Content-Type': 'application/json'}))
        assert e.value.code == 422 and json.load(e.value)['error'] == 'NO_LANDMARKS'
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url + '/api/analyze', b'{"x":1}', {'Content-Type': 'application/json'}))
        assert e.value.code == 400
    finally:
        srv.shutdown()
