"""A simple, non-learned pre-processing baseline.

Any claim that "AI restoration helps" must be compared against what an engineer would
do in an afternoon without machine learning. This module is that reference:

    1. a 3x3 median filter  (removes impulsive / speckle-like noise, keeps edges)
    2. a percentile contrast stretch (1st - 99.5th percentile -> full dynamic range),
       which compensates for dark (low light) and washed-out (fog) images

It is deliberately generic and UNTUNED: it is applied identically to every degradation
kind, exactly like the blind learned restorer. A tuned per-sensor chain (Lee / Frost
filters for SAR, a dedicated dehazing algorithm, ...) would do better, so the numbers
reported against it are a lower bound on what classical methods can achieve.
"""

from __future__ import annotations

import cv2
import numpy as np


def classical_enhance(image: np.ndarray) -> np.ndarray:
    """Median denoising + percentile contrast stretch. float32 (H, W) in [0, 1] in and out."""
    as_uint8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    denoised = cv2.medianBlur(as_uint8, 3).astype(np.float32) / 255.0

    low, high = np.percentile(denoised, (1.0, 99.5))
    if high - low < 1e-3:  # flat image: nothing to stretch
        return denoised
    return np.clip((denoised - low) / (high - low), 0.0, 1.0).astype(np.float32)
