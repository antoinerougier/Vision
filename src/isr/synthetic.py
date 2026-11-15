"""Procedural maritime scene generator with free ground-truth bounding boxes.

This is the "synthetic data" module of the product idea: when real data is rare,
classified or expensive, a simulator gives unlimited labelled samples, including the
rare hard cases. Here the simulator is deliberately simple (2D, grayscale), so the
whole pipeline can run on a laptop CPU. It generates:

    * a sea background (smooth swell + oriented wave texture + fine clutter)
    * optional land masses (bright textured regions where no ship can be placed)
    * distractors / false alarms: buoys, rocks and wave crests
    * 0-4 ships: rotated hulls with bright scatterers and a faint wake

The result is a float32 image in [0, 1] and an (N, 4) array of boxes in pixel units,
format (x_min, y_min, x_max, y_max).
"""

from __future__ import annotations

import cv2
import numpy as np


def _smooth_noise(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    """Zero-mean, unit-ish-variance spatially correlated noise field."""
    field = rng.normal(size=(size, size)).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)
    return (field - field.mean()) / (field.std() + 1e-6)


def _make_background(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (background image, land mask). The land mask is 1 where land is."""
    base_level = rng.uniform(0.15, 0.32)
    swell = 0.04 * _smooth_noise(rng, size, sigma=rng.uniform(8, 16))

    # Oriented wave texture: a sinusoid in a random direction, modulated in amplitude.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    theta = rng.uniform(0, np.pi)
    period = rng.uniform(6, 14)
    phase = rng.uniform(0, 2 * np.pi)
    waves = np.sin(2 * np.pi * (xx * np.cos(theta) + yy * np.sin(theta)) / period + phase)
    waves = 0.025 * waves * (0.6 + 0.4 * _smooth_noise(rng, size, sigma=10))

    clutter = 0.015 * rng.normal(size=(size, size)).astype(np.float32)
    background = base_level + swell + waves + clutter

    land_mask = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.35:
        # Land = where a very smooth field, biased towards a random corner, exceeds a threshold.
        field = _smooth_noise(rng, size, sigma=size / 6)
        corner_x, corner_y = rng.integers(0, 2, size=2) * (size - 1)
        distance = np.hypot(xx - corner_x, yy - corner_y) / size
        score = field - 3.0 * distance + rng.uniform(0.2, 1.0)
        land_mask = (score > 0.8).astype(np.float32)
        land_mask = cv2.GaussianBlur(land_mask, (0, 0), sigmaX=1.2)
        texture = 0.5 + 0.06 * _smooth_noise(rng, size, sigma=2.0)
        background = background * (1 - land_mask) + texture * land_mask
    return background.astype(np.float32), land_mask


def _draw_ship(
    layer: np.ndarray,
    rng: np.random.Generator,
    center: tuple[float, float],
    length: float,
    width: float,
    angle: float,
    intensity: float,
) -> tuple[float, float, float, float]:
    """Draw one rotated ship on `layer` (in place) and return its axis-aligned box."""
    cx, cy = center
    c, s = np.cos(angle), np.sin(angle)
    half_l, half_w = length / 2, width / 2

    # Pointed hull: a hexagon with a narrower bow.
    local = np.array(
        [
            [-half_l, -half_w],
            [half_l * 0.6, -half_w],
            [half_l, 0.0],
            [half_l * 0.6, half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    points = local @ rotation.T + np.array([cx, cy], dtype=np.float32)
    cv2.fillConvexPoly(layer, np.round(points * 16).astype(np.int32), float(intensity), shift=4)

    # Bright point scatterers (superstructure, corner reflectors).
    for _ in range(int(rng.integers(2, 5))):
        along = rng.uniform(-0.4, 0.4) * length
        across = rng.uniform(-0.3, 0.3) * width
        px = cx + along * c - across * s
        py = cy + along * s + across * c
        cv2.circle(layer, (int(round(px)), int(round(py))), 1, 1.0, thickness=-1)

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def render_scene(
    rng: np.random.Generator,
    size: int = 128,
    max_ships: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one random scene. Returns (image float32 (size, size), boxes float32 (N, 4))."""
    background, land_mask = _make_background(rng, size)

    ship_layer = np.zeros((size, size), dtype=np.float32)
    wake_layer = np.zeros((size, size), dtype=np.float32)

    # --- False alarms: small bright blobs (buoys / rocks) and short wave crests. ---
    for _ in range(int(rng.integers(0, 6))):
        px, py = rng.uniform(2, size - 2, size=2)
        radius = int(rng.integers(1, 3))
        cv2.circle(
            ship_layer,
            (int(px), int(py)),
            radius,
            float(rng.uniform(0.45, 0.8)),
            thickness=-1,
        )
    for _ in range(int(rng.integers(0, 3))):
        x0, y0 = rng.uniform(4, size - 4, size=2)
        angle = rng.uniform(0, np.pi)
        dx, dy = 5 * np.cos(angle), 5 * np.sin(angle)
        cv2.line(
            wake_layer,
            (int(x0 - dx), int(y0 - dy)),
            (int(x0 + dx), int(y0 + dy)),
            float(rng.uniform(0.2, 0.35)),
            thickness=1,
        )

    # --- Ships. ---
    n_ships = 0 if rng.random() < 0.15 else int(rng.integers(1, max_ships + 1))
    boxes: list[tuple[float, float, float, float]] = []
    centers: list[tuple[float, float]] = []
    for _ in range(n_ships):
        for _attempt in range(20):  # rejection sampling to avoid overlaps and land
            length = rng.uniform(10, 26)
            width = rng.uniform(3, 7)
            margin = length / 2 + 2
            cx = rng.uniform(margin, size - margin)
            cy = rng.uniform(margin, size - margin)
            on_land = land_mask[int(cy), int(cx)] > 0.05
            too_close = any(np.hypot(cx - ox, cy - oy) < 22 for ox, oy in centers)
            if on_land or too_close:
                continue

            angle = rng.uniform(0, np.pi)
            local_mean = float(background[int(cy), int(cx)])
            intensity = min(local_mean + rng.uniform(0.25, 0.65), 0.98)

            # Faint wake behind the ship (opposite to the bow direction).
            tail = (cx - np.cos(angle) * length * 1.4, cy - np.sin(angle) * length * 1.4)
            cv2.line(
                wake_layer,
                (int(cx), int(cy)),
                (int(tail[0]), int(tail[1])),
                float(local_mean + 0.07),
                thickness=1,
            )
            box = _draw_ship(ship_layer, rng, (cx, cy), length, width, angle, intensity)
            boxes.append(box)
            centers.append((cx, cy))
            break

    # Compose: background, then wakes, then ships/false alarms, with a soft optical blur.
    image = np.maximum(background, wake_layer)
    image = np.maximum(image, cv2.GaussianBlur(ship_layer, (0, 0), sigmaX=0.7))
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    if boxes:
        boxes_arr = np.array(boxes, dtype=np.float32)
        boxes_arr[:, [0, 2]] = np.clip(boxes_arr[:, [0, 2]], 0, size - 1)
        boxes_arr[:, [1, 3]] = np.clip(boxes_arr[:, [1, 3]], 0, size - 1)
    else:
        boxes_arr = np.zeros((0, 4), dtype=np.float32)
    return image, boxes_arr
