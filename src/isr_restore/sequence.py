"""Glue between the detector and the tracker: process a whole image sequence.

    frames (T images) --detect--> detections per frame --track--> snapshots per frame

Used by the evaluation, by the HTTP API (`/sim/scenario`, `/track`) and by the tests, so
that all of them run exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .classical import classical_enhance
from .inference import OnnxPipeline
from .tracking import (
    CHI2_95_2DOF,
    MultiShipTracker,
    TrackerParams,
    ellipse_from_covariance,
    forecast_track,
)

# How degraded frames are prepared before detection.
#   restored   our restoration network, then the detector     (the product)
#   raw        the detector alone on the degraded frame       (the "do nothing" baseline)
#   classical  median + contrast stretch, then the detector   (the "no ML" baseline)
MODES: tuple[str, ...] = ("restored", "raw", "classical")


@dataclass
class TrackSnapshot:
    """State of one confirmed track right after processing one frame."""

    id: int
    x: float
    y: float
    vx: float  # pixels per minute
    vy: float
    misses: int  # consecutive frames without a detection (0 = seen in this frame)
    n_observations: int  # detections associated so far; the speed is unknown below 2
    history: list[tuple[float, float]]  # filtered positions up to now (the trail)
    forecast_horizons: list[float]  # minutes ahead, e.g. [2, 4, 6, 8, 10]
    forecast_means: np.ndarray  # (K, 2) predicted positions
    forecast_covs: np.ndarray  # (K, 2, 2) their covariances
    last_measurements: list[tuple[float, float, float]]  # last 5 raw detections (t, x, y)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.vy, self.vx)))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form, with ready-to-draw uncertainty ellipses for the dashboard."""
        forecast = []
        for tau, mean, cov in zip(
            self.forecast_horizons, self.forecast_means, self.forecast_covs, strict=True
        ):
            a, b, angle = ellipse_from_covariance(cov, CHI2_95_2DOF)
            forecast.append(
                {
                    "minutes_ahead": float(tau),
                    "x": round(float(mean[0]), 2),
                    "y": round(float(mean[1]), 2),
                    "ellipse": {
                        "a": round(a, 2),
                        "b": round(b, 2),
                        "angle": round(angle, 4),
                    },
                }
            )
        return {
            "id": self.id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "speed": round(self.speed, 3),
            "heading_deg": round(self.heading_deg, 1),
            "misses": self.misses,
            "n_observations": self.n_observations,
            "history": [[round(px, 2), round(py, 2)] for px, py in self.history],
            "forecast": forecast,
        }


def box_centers(detections: np.ndarray) -> np.ndarray:
    """(N, 5) boxes -> (N, 2) centres."""
    if len(detections) == 0:
        return np.zeros((0, 2))
    return np.stack(
        [(detections[:, 0] + detections[:, 2]) / 2, (detections[:, 1] + detections[:, 3]) / 2],
        axis=1,
    )


def detect_sequence(
    pipeline: OnnxPipeline,
    frames: np.ndarray | list[np.ndarray],
    mode: str = "restored",
    score_threshold: float = 0.3,
    reference_first: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect ships in every frame. Returns (detections per frame, images given to the detector).

    The sharp reference photo (frame 0 when `reference_first`) never needs restoring, so it
    goes straight to the detector whatever the mode.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {MODES}")
    detections: list[np.ndarray] = []
    processed: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=np.float32)
        is_reference = reference_first and index == 0
        if is_reference or mode == "raw":
            image = frame
        elif mode == "classical":
            image = classical_enhance(frame)
        else:
            image = pipeline.restore(frame)
        processed.append(image)
        # restore=False: the image has already been prepared above.
        detections.append(pipeline.detect(image, restore=False, score_threshold=score_threshold))
    return detections, processed


def horizons_for(step_minutes: float, horizon_minutes: float) -> list[float]:
    """Forecast times, one per frame interval: e.g. step 2, horizon 10 -> [2, 4, 6, 8, 10]."""
    n = max(1, int(round(horizon_minutes / step_minutes)))
    return [step_minutes * k for k in range(1, n + 1)]


def run_tracking(
    detections: list[np.ndarray],
    times: np.ndarray | list[float],
    params: TrackerParams,
    horizon_minutes: float,
    reference_first: bool = True,
) -> list[list[TrackSnapshot]]:
    """Feed the detections of each frame to a tracker. Returns the confirmed tracks per frame."""
    times = [float(t) for t in times]
    step = times[1] - times[0] if len(times) > 1 else 1.0
    horizons = horizons_for(step, horizon_minutes)
    tracker = MultiShipTracker(params)
    history: list[list[TrackSnapshot]] = []

    for index, (t, dets) in enumerate(zip(times, detections, strict=True)):
        tracker.update(t, box_centers(dets), reference=reference_first and index == 0)
        snapshots = []
        for track in tracker.confirmed_tracks():
            means, covs = forecast_track(track, horizons, params.process_noise)
            snapshots.append(
                TrackSnapshot(
                    id=track.id,
                    x=float(track.x[0]),
                    y=float(track.x[1]),
                    vx=float(track.x[2]),
                    vy=float(track.x[3]),
                    misses=track.misses,
                    n_observations=track.hits,
                    history=[(px, py) for _, px, py in track.history],
                    forecast_horizons=horizons,
                    forecast_means=means,
                    forecast_covs=covs,
                    last_measurements=list(track.measurements[-5:]),
                )
            )
        history.append(snapshots)
    return history
