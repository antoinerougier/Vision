"""ONNX export parity, the inference pipeline and the HTTP API.

These tests use untrained networks (see conftest.py): they verify the plumbing, the
input validation and the response formats, not the quality of the predictions.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from isr_restore.api import app, get_pipeline  # noqa: E402
from isr_restore.inference import OnnxPipeline  # noqa: E402


@pytest.fixture()
def pipeline(tiny_model_dir: Path) -> OnnxPipeline:
    return OnnxPipeline(tiny_model_dir)


@pytest.fixture()
def client(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ISR_MODEL_DIR", str(tiny_model_dir))
    get_pipeline.cache_clear()  # the pipeline is cached per process: reload for this test
    yield fastapi_testclient.TestClient(app)
    get_pipeline.cache_clear()


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


# ------------------------------------------------------------------ inference pipeline


@pytest.mark.parametrize("shape", [(64, 64), (101, 77), (33, 130)])
def test_restore_preserves_any_image_size(pipeline: OnnxPipeline, shape: tuple[int, int]) -> None:
    """Sizes that are not multiples of 8 must work (padding + cropping)."""
    image = np.random.default_rng(0).random(shape).astype(np.float32)
    restored = pipeline.restore(image)
    assert restored.shape == shape and restored.dtype == np.float32
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_detect_returns_n_by_five_inside_the_image(pipeline: OnnxPipeline) -> None:
    image = np.random.default_rng(1).random((90, 120)).astype(np.float32)
    detections = pipeline.detect(image, restore=True, score_threshold=0.0)
    assert detections.ndim == 2 and detections.shape[1] == 5
    assert np.all(detections[:, [0, 2]] >= 0) and np.all(detections[:, [0, 2]] <= 120)
    assert np.all(detections[:, [1, 3]] >= 0) and np.all(detections[:, [1, 3]] <= 90)


def test_pipeline_is_deterministic(pipeline: OnnxPipeline) -> None:
    image = np.random.default_rng(2).random((64, 64)).astype(np.float32)
    np.testing.assert_array_equal(pipeline.restore(image), pipeline.restore(image))


# ------------------------------------------------------------------------------- API


def test_health_reports_ready_when_models_exist(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models_found"] is True


def test_health_reports_degraded_when_models_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ISR_MODEL_DIR", str(tmp_path))
    response = fastapi_testclient.TestClient(app).get("/health")
    assert response.status_code == 503  # orchestrators rely on the status code
    body = response.json()
    assert body["status"] == "degraded" and body["models_found"] is False


def test_detect_endpoint_returns_expected_json(client) -> None:
    image = (np.random.default_rng(0).random((100, 140)) * 255).astype(np.uint8)
    response = client.post(
        "/detect?restore=true&score_threshold=0.0", files={"file": ("x.png", png_bytes(image))}
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"], body["restored"]) == (140, 100, True)
    assert isinstance(body["detections"], list) and body["inference_ms"] >= 0
    for det in body["detections"]:
        assert set(det) == {"x_min", "y_min", "x_max", "y_max", "score"}


def test_detect_without_restoration(client) -> None:
    image = np.full((64, 64), 128, dtype=np.uint8)
    response = client.post("/detect?restore=false", files={"file": ("x.png", png_bytes(image))})
    assert response.status_code == 200 and response.json()["restored"] is False


def test_preprocess_returns_a_png_of_the_same_size(client) -> None:
    image = (np.random.default_rng(0).random((70, 90)) * 255).astype(np.uint8)
    response = client.post("/preprocess", files={"file": ("x.png", png_bytes(image))})
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded.shape == (70, 90)


def test_16_bit_images_are_accepted(client) -> None:
    """SAR and thermal sensors often deliver 16-bit data."""
    image = (np.random.default_rng(0).random((64, 64)) * 65535).astype(np.uint16)
    response = client.post("/detect", files={"file": ("x.png", png_bytes(image))})
    assert response.status_code == 200


def test_colour_images_are_converted_to_grayscale(client) -> None:
    image = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    response = client.post("/detect", files={"file": ("x.png", png_bytes(image))})
    assert response.status_code == 200


def test_garbage_upload_is_rejected_with_422(client) -> None:
    response = client.post("/detect", files={"file": ("x.png", b"this is not an image")})
    assert response.status_code == 422


def test_oversized_upload_is_rejected_with_413(client) -> None:
    response = client.post(
        "/detect", files={"file": ("x.bin", io.BytesIO(b"0" * (11 * 1024 * 1024)))}
    )
    assert response.status_code == 413


def test_invalid_score_threshold_is_rejected(client) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    response = client.post("/detect?score_threshold=2", files={"file": ("x.png", png_bytes(image))})
    assert response.status_code == 422
