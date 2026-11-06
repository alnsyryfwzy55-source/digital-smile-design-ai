"""Minimal HTTP API on the Python standard library (no extra packages), for local use and tests.

    python http_api.py --model ../docs/teeth_unet_v2.onnx --port 8800

POST /api/analyze   JSON {"image_b64": "...", "landmarks": [[x,y],...478], "options": {...}}
                    -> 200 analysis JSON, or 422 {"error": code, "message": text}
GET  /api/health    -> {"ok": true, "version": ...}
Production uses app_fastapi.py (same Service).
"""
import argparse, base64, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsd_service import Service, AnalysisError, VERSION   # noqa: E402

MAX_BODY = 25 * 1024 * 1024


def make_handler(svc, allow_origin='*', max_parallel=2):
    """allow_origin: '*' or a comma-separated list of web origins allowed to call the API from a browser
    (e.g. the clinic app's address). Photos are analysed in memory and never written to disk or logged."""
    allowed = [o.strip().rstrip('/') for o in allow_origin.split(',') if o.strip()]
    gate = threading.BoundedSemaphore(max_parallel)       # a small free server: at most N analyses at a time

    class H(BaseHTTPRequestHandler):
        def _origin(self):
            if '*' in allowed:
                return '*'
            o = (self.headers.get('Origin') or '').rstrip('/')
            return o if o in allowed else None

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            o = self._origin()
            if o:
                self.send_header('Access-Control-Allow-Origin', o)
                self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers(); self.wfile.write(body)

        def do_OPTIONS(self):                            # CORS preflight: headers only, no body
            self.send_response(204)
            o = self._origin()
            if o:
                self.send_header('Access-Control-Allow-Origin', o); self.send_header('Vary', 'Origin')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Access-Control-Max-Age', '600')
            self.send_header('Content-Length', '0'); self.end_headers()

        def do_GET(self):
            if self.path in ('/api/health', '/'):
                return self._send(200, dict(ok=True, version=VERSION, service='smile-design analysis'))
            self._send(404, dict(error='NOT_FOUND'))

        def do_POST(self):
            if self.path != '/api/analyze':
                return self._send(404, dict(error='NOT_FOUND'))
            n = int(self.headers.get('Content-Length') or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, dict(error='BODY_SIZE', message='حجم الطلب غير مقبول'))
            try:
                req = json.loads(self.rfile.read(n))
                img = base64.b64decode(req['image_b64'])
                with gate:
                    res = svc.analyze(img, req['landmarks'], req.get('options') or {}, with_overlay=req.get('overlay', True))
                self._send(200, res)
            except AnalysisError as e:
                self._send(422, dict(error=e.code, message=e.message))
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, dict(error='BAD_REQUEST', message=str(e)))
            except Exception as e:                       # never leak a traceback to the client
                self.log_error('analyze failed: %r', e)
                self._send(500, dict(error='INTERNAL', message='خطأ داخلي في الخادم'))

        def log_message(self, fmt, *args):
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))
    return H


def main():
    ap = argparse.ArgumentParser()
    env = os.environ.get
    ap.add_argument('--model', default=env('DSD_MODEL', os.path.join(os.path.dirname(__file__), '..', 'docs', 'teeth_unet_v2.onnx')))
    ap.add_argument('--port', type=int, default=int(env('PORT', 8800))); ap.add_argument('--host', default=env('HOST', '127.0.0.1'))
    ap.add_argument('--origins', default=env('DSD_ORIGINS', '*'), help="comma-separated web origins allowed (default '*')")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Service(a.model), a.origins))
    print(f'listening on http://{a.host}:{a.port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
