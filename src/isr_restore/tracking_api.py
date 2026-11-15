"""HTTP routes of the tracking layer.

GET  /viz               the interactive dashboard (one self-contained HTML page)
GET  /sim/scenario      simulate a sequence, run detection + tracking + forecast, return JSON
GET  /sim/frame/{n}     one image of the simulated sequence, as PNG
POST /track             upload YOUR OWN ordered image sequence -> tracks and forecasts

The simulation endpoints exist so that the whole chain can be seen and tested without any
real data. `/track` is the production-style entry point: it runs the same code on uploaded
images.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response

from .scenario import Scenario, ScenarioConfig, build_scenario
from .sequence import TrackSnapshot, detect_sequence, horizons_for, run_tracking
from .serving import decode_upload, get_pipeline
from .tracking import TrackerParams

router = APIRouter()

Kind = Literal["sar_speckle", "low_light", "fog", "thermal"]
Mode = Literal["restored", "raw", "classical"]
View = Literal["input", "processed", "clean"]

# The API uses the same defaults as `configs/default.yaml` (a test checks they never drift
# apart): the Docker image does not need the YAML file at runtime.
HORIZON_MINUTES = 10.0
SCORE_THRESHOLD = 0.3
MAX_SEQUENCE_FILES = 64

DASHBOARD_PATH = Path(__file__).parent / "static" / "dashboard.html"


@dataclass
class SimulationRun:
    """Everything computed for one simulated sequence (kept in a small cache)."""

    scenario: Scenario
    mode: str
    processed: list[np.ndarray]  # what the detector actually looked at, frame by frame
    detections: list[np.ndarray]
    snapshots: list[list[TrackSnapshot]]


@lru_cache(maxsize=8)
def _run_simulation(
    seed: int, kind: str, severity: float, n_frames: int, mode: str, model_dir: str
) -> SimulationRun:
    """Simulate, detect and track. Cached: the dashboard asks for 1 JSON + N images."""
    config = replace(ScenarioConfig(), n_frames=n_frames)
    scenario = build_scenario(seed, config, kind, severity)
    pipeline = get_pipeline()
    detections, processed = detect_sequence(pipeline, scenario.frames, mode, SCORE_THRESHOLD)
    snapshots = run_tracking(detections, scenario.times, TrackerParams(), HORIZON_MINUTES)
    return SimulationRun(scenario, mode, processed, detections, snapshots)


def _simulate(seed: int, kind: str, severity: float, n_frames: int, mode: str) -> SimulationRun:
    # The model folder is part of the cache key: another folder means other models.
    return _run_simulation(
        seed, kind, round(severity, 3), n_frames, mode, os.environ.get("ISR_MODEL_DIR", "models")
    )


def _detections_json(detections: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"box": [round(float(v), 1) for v in d[:4]], "score": round(float(d[4]), 3)}
        for d in detections
    ]


def _png(image: np.ndarray) -> Response:
    ok, encoded = cv2.imencode(".png", (np.clip(image, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    if not ok:  # pragma: no cover - encoding a uint8 array does not fail in practice
        raise HTTPException(status_code=500, detail="Could not encode the image.")
    return Response(content=encoded.tobytes(), media_type="image/png")


# ----------------------------------------------------------------------------- routes


@router.get("/viz", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """The interactive dashboard (plain HTML + JavaScript, no external dependency)."""
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@router.get("/sim/scenario")
def sim_scenario(
    seed: int = Query(7, ge=0, le=1_000_000),
    kind: Kind = "sar_speckle",
    severity: float = Query(1.0, ge=0.0, le=1.5),
    n_frames: int = Query(16, ge=4, le=40, description="frame 0 is the sharp photo"),
    mode: Mode = Query("restored", description="how degraded frames are prepared"),
) -> dict[str, Any]:
    """Simulate a sequence, detect and track the ships, forecast their next 10 minutes.

    Includes the ground truth so that a client (the dashboard) can show the prediction next
    to what really happened.
    """
    run = _simulate(seed, kind, severity, n_frames, mode)
    scenario = run.scenario
    horizons = horizons_for(scenario.config.dt_minutes, HORIZON_MINUTES)
    n_future = len(horizons)

    frames = []
    for t in range(scenario.n_frames):
        ground_truth = []
        for ship in range(scenario.n_ships):
            future = scenario.positions[ship, t + 1 : t + 1 + n_future]
            ground_truth.append(
                {
                    "id": ship + 1,
                    "x": round(float(scenario.positions[ship, t, 0]), 2),
                    "y": round(float(scenario.positions[ship, t, 1]), 2),
                    "box": [round(float(v), 1) for v in scenario.boxes[ship, t]],
                    "future": [[round(float(x), 2), round(float(y), 2)] for x, y in future],
                }
            )
        frames.append(
            {
                "index": t,
                "t_minutes": float(scenario.times[t]),
                "is_reference": t == 0,
                "ground_truth": ground_truth,
                "detections": _detections_json(run.detections[t]),
                "tracks": [s.to_dict() for s in run.snapshots[t]],
            }
        )
    return {
        "meta": {
            "seed": seed,
            "kind": kind,
            "severity": severity,
            "mode": mode,
            "image_size": scenario.config.size,
            "n_frames": scenario.n_frames,
            "n_ships": scenario.n_ships,
            "dt_minutes": scenario.config.dt_minutes,
            "horizon_minutes": HORIZON_MINUTES,
            "horizons": horizons,
        },
        "frames": frames,
    }


@router.get("/sim/frame/{index}")
def sim_frame(
    index: int,
    seed: int = Query(7, ge=0, le=1_000_000),
    kind: Kind = "sar_speckle",
    severity: float = Query(1.0, ge=0.0, le=1.5),
    n_frames: int = Query(16, ge=4, le=40),
    mode: Mode = "restored",
    view: View = Query("input", description="input = sensor, processed = detector input, clean"),
) -> Response:
    """One image of the simulated sequence (same query parameters as /sim/scenario)."""
    run = _simulate(seed, kind, severity, n_frames, mode)
    if not 0 <= index < run.scenario.n_frames:
        raise HTTPException(status_code=404, detail="No such frame.")
    images = {
        "input": run.scenario.frames,
        "processed": run.processed,
        "clean": run.scenario.clean_frames,
    }
    return _png(images[view][index])


@router.post("/track")
def track(
    files: list[UploadFile] = File(..., description="images in chronological order"),
    dt_minutes: float = Query(2.0, gt=0.0, le=1440.0, description="time between two images"),
    horizon_minutes: float = Query(10.0, gt=0.0, le=240.0),
    mode: Mode = Query("restored"),
    reference_first: bool = Query(True, description="the first image is a sharp reference"),
) -> dict[str, Any]:
    """Detect, track and forecast on an uploaded image sequence.

    Positions are in pixels of the uploaded images, time in minutes, velocities in
    pixels per minute (convert with your sensor's ground sampling distance).
    """
    if not 2 <= len(files) <= MAX_SEQUENCE_FILES:
        raise HTTPException(
            status_code=422, detail=f"Send between 2 and {MAX_SEQUENCE_FILES} images."
        )
    frames = [decode_upload(f.file.read()) for f in files]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise HTTPException(status_code=422, detail="All images must have the same size.")

    times = np.arange(len(frames)) * dt_minutes
    detections, _ = detect_sequence(
        get_pipeline(), frames, mode, SCORE_THRESHOLD, reference_first=reference_first
    )
    snapshots = run_tracking(
        detections, times, TrackerParams(), horizon_minutes, reference_first=reference_first
    )
    return {
        "meta": {
            "n_frames": len(frames),
            "image_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
            "dt_minutes": dt_minutes,
            "horizon_minutes": horizon_minutes,
            "mode": mode,
            "reference_first": reference_first,
        },
        "frames": [
            {
                "index": i,
                "t_minutes": float(times[i]),
                "detections": _detections_json(detections[i]),
                "tracks": [s.to_dict() for s in snapshots[i]],
            }
            for i in range(len(frames))
        ],
    }
