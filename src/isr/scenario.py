"""Simulated surveillance sequences: one sharp reference photo, then degraded frames.

This module reproduces the operational situation the tracking layer is designed for:

    t = 0 min    a SHARP photo (taken e.g. once per hour by a high-quality sensor)
    t = 2 min    a DEGRADED frame (SAR speckle, low light, fog, thermal...)
    t = 4 min    a DEGRADED frame
    ...

Between two sharp photos the ships keep moving, so the degraded frames are the only
information available. The scenario keeps the full ground truth (every ship position at
every time) so that detection, tracking and forecasts can be scored exactly.

Everything is simulated: there is no real data in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .common import sample_rng
from .degradations import degrade
from .synthetic import _draw_ship, _make_background
from .trajectories import sample_fleet


@dataclass(frozen=True)
class ScenarioConfig:
    """Everything that defines how a scenario is generated (see `configs/default.yaml`)."""

    size: int = 256
    dt_minutes: float = 2.0  # time between two frames
    n_frames: int = 16  # frame 0 = sharp reference, frames 1.. = degraded
    n_ships: tuple[int, int] = (2, 4)
    speed_px_per_min: tuple[float, float] = (0.8, 2.5)
    maneuver_probability: float = 0.4
    turn_rate_deg_per_min: tuple[float, float] = (2.0, 6.0)
    n_buoys: tuple[int, int] = (0, 4)  # static false alarms

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ScenarioConfig:
        """Build from the `tracking.scenario` section of the YAML configuration."""
        section = config["tracking"]["scenario"]
        return cls(
            size=int(section["size"]),
            dt_minutes=float(section["dt_minutes"]),
            n_frames=int(section["n_frames"]),
            n_ships=(int(section["n_ships"][0]), int(section["n_ships"][1])),
            speed_px_per_min=(
                float(section["speed_px_per_min"][0]),
                float(section["speed_px_per_min"][1]),
            ),
            maneuver_probability=float(section["maneuver_probability"]),
            turn_rate_deg_per_min=(
                float(section["turn_rate_deg_per_min"][0]),
                float(section["turn_rate_deg_per_min"][1]),
            ),
            n_buoys=(int(section["n_buoys"][0]), int(section["n_buoys"][1])),
        )


@dataclass
class Scenario:
    """A generated sequence plus its ground truth."""

    config: ScenarioConfig
    seed: int
    kind: str
    severity: float
    times: np.ndarray  # (T,) minutes since the sharp reference photo
    clean_frames: np.ndarray  # (T, H, W) the scene without degradation
    frames: np.ndarray  # (T, H, W) what the sensor delivers: frame 0 sharp, others degraded
    positions: np.ndarray  # (S, T, 2) ship centres, pixels
    headings: np.ndarray  # (S, T) radians
    boxes: np.ndarray  # (S, T, 4) x_min, y_min, x_max, y_max
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_ships(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


def _render_static_layers(
    rng: np.random.Generator, size: int, n_buoys: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background, land mask and a static layer of false alarms (buoys, rocks, crests).

    Buoys never move, so a tracker must not confuse them with ships. They are bright blobs
    similar to a small ship, which is exactly what makes them good test cases.
    """
    background, land_mask = _make_background(rng, size)
    static_layer = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(n_buoys[0], n_buoys[1] + 1))):
        px, py = rng.uniform(4, size - 4, size=2)
        if land_mask[int(py), int(px)] > 0.05:
            continue
        cv2.circle(
            static_layer,
            (int(px), int(py)),
            int(rng.integers(1, 3)),
            float(rng.uniform(0.45, 0.8)),
            thickness=-1,
        )
    return background, land_mask, static_layer


def _render_frame(
    background: np.ndarray,
    static_layer: np.ndarray,
    positions: np.ndarray,  # (S, 2) ship centres at this time
    headings: np.ndarray,  # (S,)
    lengths: np.ndarray,
    widths: np.ndarray,
    intensities: np.ndarray,
    ship_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw every ship at its current position. Returns (image, boxes (S, 4))."""
    size = background.shape[0]
    ship_layer = static_layer.copy()
    wake_layer = np.zeros_like(background)
    boxes = np.zeros((len(positions), 4), dtype=np.float32)

    for i, ((cx, cy), heading) in enumerate(zip(positions, headings, strict=True)):
        # The wake trails behind the ship, opposite to its heading.
        tail = (
            cx - np.cos(heading) * lengths[i] * 1.4,
            cy - np.sin(heading) * lengths[i] * 1.4,
        )
        local_mean = float(background[int(np.clip(cy, 0, size - 1)), int(np.clip(cx, 0, size - 1))])
        cv2.line(
            wake_layer,
            (int(cx), int(cy)),
            (int(tail[0]), int(tail[1])),
            local_mean + 0.07,
            thickness=1,
        )
        # A generator seeded per ship makes the bright scatterers identical in every frame,
        # i.e. the ship keeps the same "appearance" while it moves.
        ship_rng = np.random.default_rng(int(ship_seeds[i]))
        boxes[i] = _draw_ship(
            ship_layer,
            ship_rng,
            (float(cx), float(cy)),
            float(lengths[i]),
            float(widths[i]),
            float(heading),
            float(intensities[i]),
        )

    image = np.maximum(background, wake_layer)
    image = np.maximum(image, cv2.GaussianBlur(ship_layer, (0, 0), sigmaX=0.7))
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, size - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, size - 1)
    return np.clip(image, 0.0, 1.0).astype(np.float32), boxes


def build_scenario(
    seed: int,
    config: ScenarioConfig | None = None,
    kind: str = "sar_speckle",
    severity: float = 1.0,
) -> Scenario:
    """Generate one deterministic scenario: the same (seed, config, kind, severity) always
    gives the same frames and the same ground truth."""
    cfg = config or ScenarioConfig()
    rng = sample_rng(seed, 0)  # structure of the scene: background, fleet, appearance
    background, land_mask, static_layer = _render_static_layers(rng, cfg.size, cfg.n_buoys)

    n_requested = int(rng.integers(cfg.n_ships[0], cfg.n_ships[1] + 1))
    positions, headings = sample_fleet(
        rng,
        cfg.size,
        n_requested,
        cfg.n_frames,
        cfg.dt_minutes,
        cfg.speed_px_per_min,
        cfg.maneuver_probability,
        cfg.turn_rate_deg_per_min,
        land_mask=land_mask,
    )
    n_ships = positions.shape[0]

    # Fixed appearance per ship (size and brightness do not change while it sails).
    lengths = rng.uniform(10, 26, size=n_ships)
    widths = rng.uniform(3, 7, size=n_ships)
    intensities = np.array(
        [
            min(
                float(background[int(p[0, 1]), int(p[0, 0])]) + rng.uniform(0.25, 0.65),
                0.98,
            )
            for p in positions
        ]
    )
    ship_seeds = rng.integers(0, 2**31 - 1, size=n_ships)

    clean_frames = np.zeros((cfg.n_frames, cfg.size, cfg.size), dtype=np.float32)
    frames = np.zeros_like(clean_frames)
    boxes = np.zeros((n_ships, cfg.n_frames, 4), dtype=np.float32)
    for t in range(cfg.n_frames):
        image, frame_boxes = _render_frame(
            background,
            static_layer,
            positions[:, t],
            headings[:, t],
            lengths,
            widths,
            intensities,
            ship_seeds,
        )
        clean_frames[t] = image
        boxes[:, t] = frame_boxes
        # Frame 0 is the sharp reference photo; every other frame is degraded, each with
        # its own noise realisation (sensor noise is independent from one frame to the next).
        if t == 0 or severity <= 0:
            frames[t] = image
        else:
            frames[t] = degrade(image, sample_rng(seed + 7919, t), kind, severity)

    return Scenario(
        config=cfg,
        seed=seed,
        kind=kind,
        severity=severity,
        times=np.arange(cfg.n_frames, dtype=np.float64) * cfg.dt_minutes,
        clean_frames=clean_frames,
        frames=frames,
        positions=positions.astype(np.float32),
        headings=headings.astype(np.float32),
        boxes=boxes,
    )
