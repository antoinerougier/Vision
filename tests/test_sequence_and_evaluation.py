"""Detector-to-tracker glue and the tracking benchmark (logic only, no trained model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from isr.common import load_config
from isr.evaluate_tracking import (
    _aggregate,
    _greedy_match,
    _oracle_detections,
    _predict_baselines,
    _score_scenario,
    check_gate,
    run_tracking_benchmark,
)
from isr.scenario import ScenarioConfig, build_scenario
from isr.sequence import (
    MODES,
    TrackSnapshot,
    box_centers,
    detect_sequence,
    horizons_for,
    run_tracking,
)
from isr.tracking import TrackerParams

CFG = ScenarioConfig(size=128, n_frames=12, n_ships=(2, 3), speed_px_per_min=(0.8, 1.6))


def oracle_run(seed: int = 1):
    scenario = build_scenario(seed, CFG, "sar_speckle", 1.0)
    detections = _oracle_detections(scenario)
    snapshots = run_tracking(detections, scenario.times, TrackerParams(), 10.0)
    return scenario, detections, snapshots


# ----------------------------------------------------------------------------- sequence


def test_horizons_follow_the_frame_interval() -> None:
    assert horizons_for(2.0, 10.0) == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert horizons_for(5.0, 10.0) == [5.0, 10.0]
    assert horizons_for(2.0, 1.0) == [2.0]  # at least one horizon


def test_box_centers() -> None:
    boxes = np.array([[0, 0, 10, 20, 0.9], [4, 4, 6, 8, 0.5]], dtype=np.float32)
    assert np.allclose(box_centers(boxes), [[5, 10], [5, 6]])
    assert box_centers(np.zeros((0, 5))).shape == (0, 2)


def test_snapshots_have_the_expected_content_and_are_json_serialisable() -> None:
    scenario, _, snapshots = oracle_run()
    assert len(snapshots) == scenario.n_frames
    last = snapshots[-1]
    assert len(last) == scenario.n_ships
    snap = last[0]
    assert isinstance(snap, TrackSnapshot) and snap.n_observations >= 2
    payload = snap.to_dict()
    json.dumps(payload)  # must not raise: numpy types would
    assert [f["minutes_ahead"] for f in payload["forecast"]] == [2, 4, 6, 8, 10]
    assert all(f["ellipse"]["a"] >= f["ellipse"]["b"] > 0 for f in payload["forecast"])
    assert len(payload["history"]) == scenario.n_frames


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        detect_sequence(None, np.zeros((2, 16, 16), np.float32), "magic")  # type: ignore[arg-type]
    assert set(MODES) == {"restored", "raw", "classical"}


# --------------------------------------------------------------------------- evaluation


def test_greedy_match_pairs_nearest_points_within_the_distance() -> None:
    a = np.array([[0.0, 0.0], [10.0, 0.0]])
    b = np.array([[10.5, 0.0], [0.4, 0.0], [100.0, 100.0]])
    assert _greedy_match(a, b, max_dist=2.0) == {0: 1, 1: 0}
    assert _greedy_match(a, b[2:], max_dist=2.0) == {}
    assert _greedy_match(a[:0], b, 2.0) == {}


def test_baselines_are_exact_on_a_noiseless_straight_line() -> None:
    scenario, _, snapshots = oracle_run(2)
    # Replace the noisy measurements by exact ones: both linear baselines must be perfect.
    snap = snapshots[6][0]
    t = scenario.times[6]
    exact = [
        (t - 2.0 * k, 10.0 + 1.0 * (t - 2.0 * k), 20.0 + 0.5 * (t - 2.0 * k)) for k in (2, 1, 0)
    ]
    snap = TrackSnapshot(**{**snap.__dict__, "last_measurements": exact})
    horizons = [2.0, 4.0, 6.0, 8.0, 10.0]
    predictions = _predict_baselines(snap, t, horizons)
    truth = np.array([[10 + (t + h), 20 + 0.5 * (t + h)] for h in horizons])
    assert np.allclose(predictions["linear5"], truth, atol=1e-6)
    assert np.allclose(predictions["linear2"], truth, atol=1e-6)
    assert np.allclose(predictions["stationary"], [exact[-1][1:]] * 5)


def test_oracle_detections_give_complete_accurate_tracks() -> None:
    scenario, detections, snapshots = oracle_run(3)
    score = _score_scenario(scenario, detections, snapshots, horizons_for(2.0, 10.0), 8.0)
    metrics = _aggregate([score])
    assert metrics["tracking"]["completeness"] > 0.95
    assert metrics["tracking"]["false_tracks_per_frame"] == 0.0
    forecast = metrics["forecast"]
    assert (
        forecast["fde"]["kalman"] < forecast["fde"]["stationary"]
    )  # forecasting beats "no motion"
    assert forecast["availability"] == pytest.approx(1.0)


def test_no_detection_means_no_forecast_and_no_crash() -> None:
    scenario = build_scenario(4, CFG, "sar_speckle", 1.0)
    empty = [np.zeros((0, 5), np.float32) for _ in range(scenario.n_frames)]
    # Without the reference frame being detected, nothing is ever confirmed.
    snapshots = run_tracking(empty, scenario.times, TrackerParams(), 10.0)
    score = _score_scenario(scenario, empty, snapshots, horizons_for(2.0, 10.0), 8.0)
    metrics = _aggregate([score])
    assert metrics["tracking"]["completeness"] == 0.0
    assert metrics["forecast"]["fde"]["kalman"] is None


def test_gate_logic() -> None:
    report = {
        "sources": {
            "restored": {
                "tracking": {"completeness": 0.7},
                "forecast": {
                    "fde": {"kalman": 5.0, "stationary": 15.0},
                    "ellipse_coverage_by_horizon": [0.97, 0.96],
                },
            }
        }
    }
    gate = {
        "min_track_completeness": 0.55,
        "max_kalman_to_stationary_error_ratio": 0.5,
        "min_ellipse_coverage": 0.9,
    }
    assert check_gate(report, gate) == []
    report["sources"]["restored"]["tracking"]["completeness"] = 0.3
    report["sources"]["restored"]["forecast"]["fde"]["kalman"] = 14.0
    report["sources"]["restored"]["forecast"]["ellipse_coverage_by_horizon"] = [
        0.5,
        0.4,
    ]
    assert len(check_gate(report, gate)) == 3


def test_yaml_gate_section_has_every_key_the_code_reads() -> None:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert {
        "min_track_completeness",
        "max_kalman_to_stationary_error_ratio",
        "min_ellipse_coverage",
    } <= set(config["tracking"]["gate"])


def test_full_benchmark_runs_end_to_end_with_untrained_models(
    tiny_model_dir: Path,
) -> None:
    """Plumbing only (untrained networks): the report must have the documented structure."""
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config["tracking"]["evaluation"]["n_scenarios"] = 2
    config["tracking"]["scenario"]["n_frames"] = 8
    report = run_tracking_benchmark(config, tiny_model_dir)
    assert set(report["sources"]) == {
        "oracle",
        "restored",
        "classical",
        "raw",
    }  # no retrained .onnx
    for entry in report["sources"].values():
        assert {"detection", "tracking", "forecast", "by_kind"} <= set(entry)
    json.dumps(report)
