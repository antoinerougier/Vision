"""HTTP routes of the tracking layer: dashboard, simulation, frames and sequence upload.

Untrained networks are used (see conftest.py): these tests check the plumbing, the response
schema and the input validation, not the quality of the detections.
"""

from __future__ import annotations

import io
import typing
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
fastapi_testclient = pytest.importorskip("fastapi.testclient")

from isr_restore import tracking_api  # noqa: E402
from isr_restore.api import app, get_pipeline  # noqa: E402
from isr_restore.degradations import KINDS  # noqa: E402
from isr_restore.sequence import MODES  # noqa: E402


@pytest.fixture()
def client(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ISR_MODEL_DIR", str(tiny_model_dir))
    get_pipeline.cache_clear()
    tracking_api._run_simulation.cache_clear()
    yield fastapi_testclient.TestClient(app)
    get_pipeline.cache_clear()
    tracking_api._run_simulation.cache_clear()


SMALL = {"seed": 3, "n_frames": 5}


def png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", (image * 255).astype(np.uint8))
    assert ok
    return encoded.tobytes()


# ---------------------------------------------------------------------------- dashboard


def test_dashboard_page_is_served(client) -> None:
    response = client.get("/viz")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ISR tracking dashboard" in response.text


def test_root_redirects_to_the_dashboard(client) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307) and response.headers["location"] == "/viz"


def test_dashboard_file_is_declared_as_package_data() -> None:
    """The Docker image does a non-editable install: the HTML must ship with the package."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "static/*.html" in pyproject and tracking_api.DASHBOARD_PATH.exists()


# --------------------------------------------------------------------------- simulation


def test_scenario_response_schema(client) -> None:
    response = client.get("/sim/scenario", params=SMALL)
    assert response.status_code == 200
    body = response.json()
    meta = body["meta"]
    assert meta["n_frames"] == 5 and meta["horizons"] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert meta["image_size"] == 256 and meta["mode"] == "restored"
    assert len(body["frames"]) == 5
    first = body["frames"][0]
    assert first["is_reference"] is True and first["t_minutes"] == 0.0
    assert len(first["ground_truth"]) == meta["n_ships"]
    ship = first["ground_truth"][0]
    assert {"id", "x", "y", "box", "future"} <= set(ship)
    assert len(ship["future"]) == 4  # only 4 future frames exist in a 5-frame sequence
    assert body["frames"][-1]["ground_truth"][0]["future"] == []


def test_scenario_is_deterministic(client) -> None:
    first = client.get("/sim/scenario", params=SMALL).json()
    tracking_api._run_simulation.cache_clear()
    assert client.get("/sim/scenario", params=SMALL).json() == first


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs(client, mode: str) -> None:
    response = client.get("/sim/scenario", params={**SMALL, "mode": mode})
    assert response.status_code == 200 and response.json()["meta"]["mode"] == mode


@pytest.mark.parametrize(
    "params",
    [
        {"kind": "rain"},
        {"severity": 9},
        {"severity": -1},
        {"n_frames": 2},
        {"n_frames": 500},
        {"mode": "magic"},
        {"seed": -1},
    ],
)
def test_invalid_scenario_parameters_are_rejected(client, params: dict) -> None:
    assert client.get("/sim/scenario", params=params).status_code == 422


def test_api_literals_match_the_library() -> None:
    assert typing.get_args(tracking_api.Kind) == KINDS
    assert typing.get_args(tracking_api.Mode) == MODES


@pytest.mark.parametrize("view", ["input", "processed", "clean"])
def test_frame_images_are_png_of_the_scene_size(client, view: str) -> None:
    response = client.get("/sim/frame/2", params={**SMALL, "view": view})
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape == (256, 256)


def test_the_sensor_view_differs_from_the_true_scene_after_the_reference(client) -> None:
    def fetch(index: int, view: str) -> np.ndarray:
        raw = client.get(f"/sim/frame/{index}", params={**SMALL, "kind": "fog", "view": view})
        return cv2.imdecode(np.frombuffer(raw.content, np.uint8), cv2.IMREAD_UNCHANGED)

    assert np.array_equal(fetch(0, "input"), fetch(0, "clean"))  # sharp reference photo
    assert not np.array_equal(fetch(3, "input"), fetch(3, "clean"))  # degraded frame


def test_frame_index_out_of_range_is_404(client) -> None:
    assert client.get("/sim/frame/5", params=SMALL).status_code == 404
    assert client.get("/sim/frame/-1", params=SMALL).status_code == 404


# ------------------------------------------------------------------------------ /track


def sequence_files(n: int = 4, size: int = 64) -> list:
    rng = np.random.default_rng(0)
    return [
        ("files", (f"f{i}.png", io.BytesIO(png(rng.random((size, size)))), "image/png"))
        for i in range(n)
    ]


def test_track_endpoint_returns_one_entry_per_image(client) -> None:
    response = client.post("/track", files=sequence_files(4), params={"dt_minutes": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["n_frames"] == 4 and body["meta"]["image_size"] == [64, 64]
    assert [f["t_minutes"] for f in body["frames"]] == [0.0, 3.0, 6.0, 9.0]
    assert all({"detections", "tracks"} <= set(f) for f in body["frames"])


def test_track_endpoint_forecast_horizons_follow_the_parameters(client) -> None:
    body = client.post(
        "/track", files=sequence_files(6), params={"dt_minutes": 2, "horizon_minutes": 6}
    ).json()
    for frame in body["frames"]:
        for track in frame["tracks"]:
            assert [f["minutes_ahead"] for f in track["forecast"]] == [2, 4, 6]


def test_track_endpoint_validation(client) -> None:
    assert client.post("/track", files=sequence_files(1)).status_code == 422  # need >= 2 images
    mixed = sequence_files(2, 64) + sequence_files(1, 32)
    assert client.post("/track", files=mixed).status_code == 422  # different sizes
    garbage = [("files", ("a.png", io.BytesIO(b"nope"), "image/png"))] * 2
    assert client.post("/track", files=garbage).status_code == 422
    assert (
        client.post("/track", files=sequence_files(3), params={"dt_minutes": 0}).status_code == 422
    )
    assert client.post("/track").status_code == 422  # no file at all


def test_api_defaults_match_the_yaml_file() -> None:
    from isr_restore.common import load_config

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert config["tracking"]["forecast_horizon_minutes"] == tracking_api.HORIZON_MINUTES
    assert config["tracking"]["detection"]["score_threshold"] == tracking_api.SCORE_THRESHOLD
