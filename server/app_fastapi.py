"""Production API (FastAPI). Same Service as http_api.py.

    pip install fastapi uvicorn
    DSD_MODEL=../docs/teeth_unet_v2.onnx uvicorn app_fastapi:app --host 0.0.0.0 --port 8800
"""
import base64, os, sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsd_service import Service, AnalysisError, VERSION   # noqa: E402

svc = Service(os.environ.get('DSD_MODEL', os.path.join(os.path.dirname(__file__), '..', 'docs', 'teeth_unet_v2.onnx')))
app = FastAPI(title='Smile design API', version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get('DSD_ORIGINS', '*').split(','), allow_methods=['*'], allow_headers=['*'])


class AnalyzeRequest(BaseModel):
    image_b64: str
    landmarks: List[List[float]]
    options: Optional[Dict[str, Any]] = None
    overlay: bool = True


@app.get('/api/health')
def health():
    return dict(ok=True, version=VERSION)


@app.post('/api/analyze')
def analyze(req: AnalyzeRequest):
    try:
        return svc.analyze(base64.b64decode(req.image_b64), req.landmarks, req.options or {}, with_overlay=req.overlay)
    except AnalysisError as e:
        raise HTTPException(status_code=422, detail=dict(error=e.code, message=e.message))
