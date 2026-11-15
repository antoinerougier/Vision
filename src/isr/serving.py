"""Helpers shared by every HTTP route: the model pipeline and upload validation.

They live in their own module (not in api.py) so that several routers can use them
without importing each other in a circle.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import HTTPException

from .inference import OnnxPipeline

# Basic input hygiene: a service exposed to a network must bound its resource usage.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS = 4096 * 4096


@lru_cache(maxsize=1)
def get_pipeline() -> OnnxPipeline:
    """Load the ONNX models once per process (lazily, on first use)."""
    return OnnxPipeline(os.environ.get("ISR_MODEL_DIR", "models"))


def decode_upload(data: bytes) -> np.ndarray:
    """Decode an uploaded image to float32 grayscale in [0, 1], validating it on the way."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")
    # ANYDEPTH keeps 16-bit PNG / TIFF (common for SAR and thermal) instead of squashing to 8 bit.
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise HTTPException(status_code=422, detail="Could not decode the image.")
    if raw.size > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image has too many pixels (max 4096x4096).")
    scale = 65535.0 if raw.dtype == np.uint16 else 255.0
    return raw.astype(np.float32) / scale
