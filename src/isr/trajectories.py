"""Ship motion simulator: realistic-enough trajectories with maneuvers.

The trajectory predictor is only as credible as the motion it is tested on. If every
simulated ship sailed in a perfectly straight line, a constant-velocity extrapolation would
look perfect and we would learn nothing. So ships here:

    * sail at a roughly constant speed (small random speed fluctuations),
    * sometimes start a *turn* (a few consecutive steps with a constant turn rate),
    * have a heading with a bit of random jitter (steering noise, currents).

Conventions used everywhere in the tracking code:

    * positions are in PIXELS, in image coordinates (x to the right, y downwards),
    * time is in MINUTES, speeds are in pixels per minute,
    * a heading of 0 means "moving towards +x"; angles are in radians.
"""

from __future__ import annotations

import numpy as np


def simulate_ship_track(
    rng: np.random.Generator,
    start: tuple[float, float],
    n_steps: int,
    dt_minutes: float,
    speed_range: tuple[float, float],
    maneuver_probability: float,
    turn_rate_deg_per_min: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one ship. Returns (positions (n_steps, 2), headings (n_steps,))."""
    speed = rng.uniform(*speed_range)  # pixels per minute
    heading = rng.uniform(0.0, 2.0 * np.pi)

    # Optional maneuver: one turn episode (constant turn rate for a few steps).
    turn_start, turn_end, turn_rate = n_steps + 1, n_steps + 1, 0.0  # "never" by default
    if rng.random() < maneuver_probability:
        turn_start = int(rng.integers(2, max(3, n_steps - 3)))
        turn_end = turn_start + int(rng.integers(3, 8))
        # Radians per minute, random sign (port or starboard).
        turn_rate = np.deg2rad(rng.uniform(*turn_rate_deg_per_min)) * rng.choice([-1.0, 1.0])

    positions = np.zeros((n_steps, 2), dtype=np.float64)
    headings = np.zeros(n_steps, dtype=np.float64)
    positions[0] = start
    headings[0] = heading

    for k in range(1, n_steps):
        omega = turn_rate if turn_start <= k < turn_end else 0.0
        # Steering jitter: a small random heading change on top of the planned turn.
        new_heading = heading + omega * dt_minutes + rng.normal(0.0, np.deg2rad(1.5))
        speed *= 1.0 + rng.normal(0.0, 0.02)  # slow speed fluctuations
        mean_heading = 0.5 * (heading + new_heading)  # midpoint rule: smoother arcs
        positions[k] = positions[k - 1] + speed * dt_minutes * np.array(
            [np.cos(mean_heading), np.sin(mean_heading)]
        )
        heading = new_heading
        headings[k] = heading
    return positions, headings


def sample_fleet(
    rng: np.random.Generator,
    size: int,
    n_ships: int,
    n_steps: int,
    dt_minutes: float,
    speed_range: tuple[float, float],
    maneuver_probability: float,
    turn_rate_deg_per_min: tuple[float, float],
    land_mask: np.ndarray | None = None,
    margin: float = 16.0,
    min_start_separation: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to `n_ships` trajectories that stay inside the image (and off land).

    Uses rejection sampling: a candidate trajectory is kept only if it remains inside the
    frame (with a margin) for the whole sequence, never crosses land and starts far enough
    from the other ships. Fewer ships than requested are returned if the budget of attempts
    is exhausted (this is why callers must not assume `len(result) == n_ships`).

    Returns (positions (S, n_steps, 2), headings (S, n_steps)).
    """
    all_positions: list[np.ndarray] = []
    all_headings: list[np.ndarray] = []
    for _ in range(n_ships):
        for _attempt in range(200):
            start = rng.uniform(margin, size - margin, size=2)
            if any(np.hypot(*(start - p[0])) < min_start_separation for p in all_positions):
                continue
            positions, headings = simulate_ship_track(
                rng,
                (float(start[0]), float(start[1])),
                n_steps,
                dt_minutes,
                speed_range,
                maneuver_probability,
                turn_rate_deg_per_min,
            )
            inside = np.all((positions >= margin) & (positions <= size - margin))
            if not inside:
                continue
            if land_mask is not None:
                cols = np.clip(positions[:, 0].astype(int), 0, size - 1)
                rows = np.clip(positions[:, 1].astype(int), 0, size - 1)
                if np.any(land_mask[rows, cols] > 0.05):
                    continue
            all_positions.append(positions)
            all_headings.append(headings)
            break

    if not all_positions:
        return np.zeros((0, n_steps, 2)), np.zeros((0, n_steps))
    return np.stack(all_positions), np.stack(all_headings)
