"""FastAPI service: image in, restored image or detections out.

Endpoints
---------
GET  /health        liveness / readiness probe (used by Docker HEALTHCHECK and CI smoke test)
POST /preprocess    upload an image -> restored PNG (drop-in layer in front of YOUR detector)
POST /detect        upload an image -> JSON detections, with or without restoration

GET  /viz           interactive dashboard: simulated sequence, tracks and 10-minute forecast
GET  /sim/scenario  simulate a sequence (sharp photo + degraded frames) and track the ships
GET  /sim/frame/N   one image of that sequence (PNG)
POST /track         upload an ordered image sequence -> tracks and forecasts (JSON)

Run locally:
    uvicorn isr_restore.api:app --reload

The models directory is read from the ISR_MODEL_DIR environment variable (default: ./models).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response

from . import __version__
from .serving import decode_upload as _decode_upload
from .serving import get_pipeline
from .tracking_api import router as tracking_router

app = FastAPI(
    title="isr-restore",
    version=__version__,
    description="AI pre-processing layer for degraded ISR imagery (SAR, low light, fog, thermal), "
    "with ship tracking and a 10-minute trajectory forecast.",
)
# Tracking / forecasting / dashboard routes live in their own module.
app.include_router(tracking_router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare URL to the dashboard."""
    return RedirectResponse(url="/viz")


@app.get("/health")
def health() -> JSONResponse:
    """Readiness probe: 200 when the model files are present, 503 otherwise.

    Returning 503 (rather than 200 with a warning) lets Docker / Kubernetes / a load
    balancer take an instance that cannot serve out of rotation automatically.
    """
    model_dir = Path(os.environ.get("ISR_MODEL_DIR", "models"))
    ready = (model_dir / "restoration.onnx").exists() and (model_dir / "detector.onnx").exists()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "degraded",
            "models_found": ready,
            "version": __version__,
        },
    )


@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)) -> Response:
    """Restore a degraded image and return it as an 8-bit PNG."""
    image = _decode_upload(await file.read())
    restored = get_pipeline().restore(image)
    ok, encoded = cv2.imencode(".png", (restored * 255.0 + 0.5).astype(np.uint8))
    if not ok:  # pragma: no cover - encoding a uint8 array does not fail in practice
        raise HTTPException(status_code=500, detail="Could not encode the result.")
    return Response(content=encoded.tobytes(), media_type="image/png")


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    restore: bool = Query(True, description="Run the restoration layer before detection."),
    score_threshold: float = Query(0.3, ge=0.0, le=1.0),
) -> dict[str, object]:
    """Detect objects (ships) in an image."""
    image = _decode_upload(await file.read())
    start = time.perf_counter()
    detections = get_pipeline().detect(image, restore=restore, score_threshold=score_threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "restored": restore,
        "inference_ms": round(elapsed_ms, 2),
        "detections": [
            {
                "x_min": round(float(d[0]), 1),
                "y_min": round(float(d[1]), 1),
                "x_max": round(float(d[2]), 1),
                "y_max": round(float(d[3]), 1),
                "score": round(float(d[4]), 4),
            }
            for d in detections
        ],
    }
