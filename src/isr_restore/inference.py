"""ONNX Runtime inference pipeline: (optional) restoration, then detection.

This module is what runs in production (inside the Docker container). It depends only on
numpy, OpenCV and ONNX Runtime: no PyTorch.

    pipeline = OnnxPipeline("models")
    restored = pipeline.restore(image)               # float32 (H, W) in [0, 1]
    boxes    = pipeline.detect(image, restore=True)  # (N, 5): x1, y1, x2, y2, score
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .postprocess import STRIDE, decode

# Both networks downsample by up to 4, so spatial sizes are padded to a multiple of 8
# (safe for both), then the output is cropped back to the original size.
_PAD_MULTIPLE = 8


def _pad_to_multiple(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image.shape
    pad_h = (-h) % _PAD_MULTIPLE
    pad_w = (-w) % _PAD_MULTIPLE
    if pad_h or pad_w:
        # Reflection avoids creating artificial strong edges at the border.
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return image, h, w


class OnnxPipeline:
    """Loads the two ONNX models once and exposes `restore` and `detect`."""

    def __init__(
        self,
        model_dir: str | Path,
        num_threads: int = 2,
        detector_file: str = "detector.onnx",
    ) -> None:
        # `detector_file` lets the benchmark load the baseline detector from the same folder.
        model_dir = Path(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        providers = ["CPUExecutionProvider"]
        self._restorer = ort.InferenceSession(
            str(model_dir / "restoration.onnx"), options, providers=providers
        )
        self._detector = ort.InferenceSession(
            str(model_dir / detector_file), options, providers=providers
        )

    @staticmethod
    def _run(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
        """Run one (H, W) float32 image through a session; returns (C, h, w)."""
        batch = image[None, None].astype(np.float32)
        (output,) = session.run(None, {"image": batch})
        return output[0]

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore a degraded grayscale image. Input and output: float32 (H, W) in [0, 1]."""
        padded, h, w = _pad_to_multiple(image)
        restored = self._run(self._restorer, padded)[0]
        return np.clip(restored[:h, :w], 0.0, 1.0).astype(np.float32)

    def detect(
        self,
        image: np.ndarray,
        restore: bool = True,
        score_threshold: float = 0.3,
    ) -> np.ndarray:
        """Detect objects; optionally restore the image first. Returns (N, 5) detections."""
        if restore:
            image = self.restore(image)
        padded, h, w = _pad_to_multiple(image)
        output = self._run(self._detector, padded)
        detections = decode(output, score_threshold=score_threshold)
        if len(detections):
            # Discard boxes whose centre falls in the padding, then clip to the image.
            keep = ((detections[:, 0] + detections[:, 2]) / 2 < w) & (
                (detections[:, 1] + detections[:, 3]) / 2 < h
            )
            detections = detections[keep]
            detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, w)
            detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, h)
        return detections


__all__ = ["STRIDE", "OnnxPipeline"]
