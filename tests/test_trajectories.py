"""Ship motion simulator: determinism, plausible speeds, maneuvers, geometry constraints."""

from __future__ import annotations

import numpy as np

from isr.trajectories import sample_fleet, simulate_ship_track

KWARGS = dict(
    dt_minutes=2.0,
    speed_range=(1.0, 2.0),
    turn_rate_deg_per_min=(4.0, 6.0),
)


def _track(seed: int, maneuver_probability: float, n_steps: int = 16):
    rng = np.random.default_rng(seed)
    return simulate_ship_track(
        rng,
        (100.0, 100.0),
        n_steps,
        maneuver_probability=maneuver_probability,
        **KWARGS,
    )


def test_track_is_deterministic_and_has_the_right_shape() -> None:
    p1, h1 = _track(1, 0.5)
    p2, h2 = _track(1, 0.5)
    assert p1.shape == (16, 2) and h1.shape == (16,)
    assert np.array_equal(p1, p2) and np.array_equal(h1, h2)
    assert np.array_equal(p1[0], [100.0, 100.0])


def test_speed_stays_close_to_the_requested_range() -> None:
    positions, _ = _track(2, 0.0)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / KWARGS["dt_minutes"]
    # Speed fluctuates by about 2 % per step, so it cannot leave the range by much.
    assert speeds.min() > 0.7 and speeds.max() < 2.6


def test_ship_without_maneuver_keeps_a_nearly_constant_heading() -> None:
    drifts = [abs(_track(seed, 0.0)[1][-1] - _track(seed, 0.0)[1][0]) for seed in range(20)]
    assert max(drifts) < np.deg2rad(30)  # only small steering jitter


def test_maneuvering_ships_really_turn() -> None:
    changes = [abs(_track(seed, 1.0)[1][-1] - _track(seed, 1.0)[1][0]) for seed in range(20)]
    # 3-7 steps at 4-6 deg/min over 2 min = at least 24 degrees of heading change.
    assert np.median(changes) > np.deg2rad(20)


def test_fleet_stays_inside_the_frame_and_apart() -> None:
    rng = np.random.default_rng(3)
    positions, headings = sample_fleet(
        rng, size=256, n_ships=4, n_steps=16, maneuver_probability=0.4, **KWARGS
    )
    assert positions.shape[1:] == (16, 2) and headings.shape[1:] == (16,)
    assert positions.min() >= 16 and positions.max() <= 256 - 16
    starts = positions[:, 0]
    for i in range(len(starts)):
        for j in range(i):
            assert np.hypot(*(starts[i] - starts[j])) >= 30


def test_fleet_avoids_land() -> None:
    land = np.zeros((256, 256), dtype=np.float32)
    land[:, :128] = 1.0  # the whole left half is land
    rng = np.random.default_rng(4)
    positions, _ = sample_fleet(rng, 256, 4, 16, maneuver_probability=0.4, land_mask=land, **KWARGS)
    assert len(positions) > 0
    assert positions[:, :, 0].min() >= 128


def test_impossible_fleet_returns_no_ship_instead_of_looping_forever() -> None:
    land = np.ones((64, 64), dtype=np.float32)  # all land
    positions, headings = sample_fleet(
        np.random.default_rng(5),
        64,
        3,
        16,
        maneuver_probability=0.0,
        land_mask=land,
        **KWARGS,
    )
    assert positions.shape == (0, 16, 2) and headings.shape == (0, 16)
