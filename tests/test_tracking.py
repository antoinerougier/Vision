"""Kalman tracker: estimation, association, track life cycle and forecast uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import REPO_ROOT
from isr_restore.common import load_config
from isr_restore.tracking import (
    CHI2_95_2DOF,
    MultiShipTracker,
    TrackerParams,
    ellipse_from_covariance,
    forecast_track,
    inside_ellipse,
    transition_matrices,
)


def straight_line(n: int, start=(50.0, 50.0), velocity=(1.5, 0.5), dt=2.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    xy = np.array(start) + np.outer(t, velocity)
    return t, xy + rng.normal(0, noise, xy.shape) if noise else xy


def run(tracker: MultiShipTracker, times, detections_per_frame, reference_first=True):
    for i, (t, dets) in enumerate(zip(times, detections_per_frame, strict=True)):
        tracker.update(
            t, np.asarray(dets, dtype=float).reshape(-1, 2), reference=reference_first and i == 0
        )


# --------------------------------------------------------------------- model matrices


def test_transition_matrices_are_consistent() -> None:
    F, Q = transition_matrices(dt=2.0, q=0.01)
    assert F[0, 2] == 2.0 and F[1, 3] == 2.0
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-12)  # covariance: positive semi-definite
    _, Q_big = transition_matrices(2.0, 0.1)
    assert np.allclose(Q_big, 10 * Q)  # linear in q


# ------------------------------------------------------------------------ estimation


def test_filter_recovers_the_velocity_of_a_straight_moving_ship() -> None:
    times, xy = straight_line(12, noise=1.0, seed=1)
    tracker = MultiShipTracker(TrackerParams())
    run(tracker, times, xy)
    (track,) = tracker.confirmed_tracks()
    # velocity is in px per minute: the simulated ship moves (1.5, 0.5) px/min
    assert np.allclose(track.x[2:], [1.5, 0.5], atol=0.35)
    assert np.allclose(track.x[:2], xy[-1], atol=2.5)


def test_filtered_position_is_less_noisy_than_the_raw_detections() -> None:
    times, truth = straight_line(30, noise=0.0)
    _, noisy = straight_line(30, noise=2.0, seed=3)
    tracker = MultiShipTracker(TrackerParams())
    errors = []
    for i, (t, z) in enumerate(zip(times, noisy, strict=True)):
        tracker.update(t, z[None], reference=i == 0)
        if i >= 8:
            errors.append(np.linalg.norm(tracker.tracks[0].x[:2] - truth[i]))
    raw = np.linalg.norm(noisy[8:] - truth[8:], axis=1)
    assert np.mean(errors) < np.mean(raw)


# -------------------------------------------------------------------- track life cycle


def test_reference_detections_start_confirmed_tracks_immediately() -> None:
    tracker = MultiShipTracker()
    tracker.update(0.0, np.array([[10.0, 10.0], [80.0, 80.0]]), reference=True)
    assert len(tracker.confirmed_tracks()) == 2


def test_ordinary_detection_needs_a_second_sighting_to_be_confirmed() -> None:
    tracker = MultiShipTracker(TrackerParams(min_hits=2))
    tracker.update(0.0, np.array([[10.0, 10.0]]))
    assert tracker.confirmed_tracks() == [] and len(tracker.tracks) == 1
    tracker.update(2.0, np.array([[11.0, 10.5]]))
    assert len(tracker.confirmed_tracks()) == 1


def test_isolated_false_alarm_is_discarded() -> None:
    tracker = MultiShipTracker()
    tracker.update(0.0, np.array([[100.0, 100.0]]))
    tracker.update(2.0, np.zeros((0, 2)))  # never seen again
    assert tracker.tracks == []


def test_confirmed_track_coasts_through_missed_detections_then_dies() -> None:
    params = TrackerParams(max_misses=3)
    tracker = MultiShipTracker(params)
    tracker.update(0.0, np.array([[50.0, 50.0]]), reference=True)
    for k in range(1, 4):  # three missed frames: still alive, position keeps being predicted
        tracker.update(2.0 * k, np.zeros((0, 2)))
        assert len(tracker.tracks) == 1 and tracker.tracks[0].misses == k
    tracker.update(8.0, np.zeros((0, 2)))
    assert tracker.tracks == []


def test_coasting_track_is_picked_up_again_with_the_same_id() -> None:
    times, xy = straight_line(10, noise=0.5, seed=4)
    tracker = MultiShipTracker()
    for i, (t, z) in enumerate(zip(times, xy, strict=True)):
        detections = np.zeros((0, 2)) if i in (4, 5) else z[None]  # two missed frames
        tracker.update(t, detections, reference=i == 0)
    (track,) = tracker.confirmed_tracks()
    assert track.id == 1 and track.misses == 0
    assert np.linalg.norm(track.x[:2] - xy[-1]) < 3.0


# -------------------------------------------------------------------------- association


def test_two_ships_keep_their_own_identity() -> None:
    times, a = straight_line(10, start=(40.0, 60.0), velocity=(1.0, 0.0), noise=0.5, seed=5)
    _, b = straight_line(10, start=(40.0, 100.0), velocity=(1.0, 0.0), noise=0.5, seed=6)
    tracker = MultiShipTracker()
    ids_a, ids_b = set(), set()
    for i, t in enumerate(times):
        pairs = tracker.update(t, np.stack([a[i], b[i]]), reference=i == 0)
        for track_id, det in pairs:
            (ids_a if det == 0 else ids_b).add(track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_gate_rejects_a_detection_far_from_the_prediction() -> None:
    tracker = MultiShipTracker()
    tracker.update(0.0, np.array([[50.0, 50.0]]), reference=True)
    tracker.update(2.0, np.array([[50.0, 50.0]]))
    pairs = tracker.update(4.0, np.array([[200.0, 200.0]]))  # teleportation: not the same ship
    assert pairs == []
    assert len(tracker.tracks) == 2  # the old track coasts, the far detection starts a new one


# ----------------------------------------------------------------------------- forecast


def test_forecast_of_a_straight_noiseless_track_is_accurate() -> None:
    times, xy = straight_line(10, noise=0.0)
    tracker = MultiShipTracker()
    run(tracker, times, xy)
    (track,) = tracker.confirmed_tracks()
    means, _ = forecast_track(track, [2, 4, 6, 8, 10], q=0.01)
    truth = xy[-1] + np.outer([2, 4, 6, 8, 10], [1.5, 0.5])
    assert np.abs(means - truth).max() < 0.5


def test_forecast_uncertainty_grows_with_the_horizon() -> None:
    times, xy = straight_line(10, noise=1.0, seed=7)
    tracker = MultiShipTracker()
    run(tracker, times, xy)
    (track,) = tracker.confirmed_tracks()
    _, covs = forecast_track(track, [2, 4, 6, 8, 10], q=0.01)
    traces = [np.trace(c) for c in covs]
    assert all(b > a for a, b in zip(traces, traces[1:], strict=False))


def test_ellipse_axes_and_membership() -> None:
    cov = np.diag([9.0, 1.0])
    a, b, angle = ellipse_from_covariance(cov)
    assert a == pytest.approx(np.sqrt(CHI2_95_2DOF * 9.0)) and b == pytest.approx(
        np.sqrt(CHI2_95_2DOF)
    )
    assert abs(np.cos(angle)) == pytest.approx(1.0)  # major axis along x
    mean = np.zeros(2)
    assert inside_ellipse(np.array([a * 0.99, 0.0]), mean, cov, CHI2_95_2DOF)
    assert not inside_ellipse(np.array([0.0, b * 1.05]), mean, cov, CHI2_95_2DOF)


def test_the_95_percent_ellipse_is_calibrated_on_straight_ships() -> None:
    """Monte-Carlo check: the truth falls inside the 95 % ellipse about 95 % of the time."""
    inside = []
    for seed in range(200):
        times, xy = straight_line(14, noise=2.0, seed=seed)
        tracker = MultiShipTracker()
        run(tracker, times[:9], xy[:9])
        (track,) = tracker.confirmed_tracks()
        means, covs = forecast_track(track, [10.0], q=0.01)
        truth = xy[8] + np.array([1.5, 0.5]) * 10.0
        inside.append(inside_ellipse(truth, means[0], covs[0], CHI2_95_2DOF))
    assert 0.90 <= np.mean(inside) <= 1.0


def test_default_parameters_match_the_yaml_file() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert TrackerParams() == TrackerParams.from_config(config)
