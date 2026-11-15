"""Simulated sequences: sharp reference photo, degraded frames, exact ground truth."""

from __future__ import annotations

import numpy as np

from conftest import REPO_ROOT  # noqa: E402
from isr_restore.common import load_config
from isr_restore.scenario import ScenarioConfig, build_scenario

SMALL = ScenarioConfig(size=128, n_frames=8, n_ships=(2, 3), speed_px_per_min=(0.5, 1.2))


def test_shapes_ranges_and_time_axis() -> None:
    sc = build_scenario(1, SMALL, "sar_speckle", 1.0)
    assert sc.frames.shape == (8, 128, 128) and sc.clean_frames.shape == sc.frames.shape
    assert sc.positions.shape == (sc.n_ships, 8, 2)
    assert sc.boxes.shape == (sc.n_ships, 8, 4)
    assert sc.frames.min() >= 0.0 and sc.frames.max() <= 1.0
    assert np.allclose(sc.times, np.arange(8) * 2.0)


def test_first_frame_is_the_sharp_reference_and_others_are_degraded() -> None:
    sc = build_scenario(2, SMALL, "low_light", 1.0)
    assert np.array_equal(sc.frames[0], sc.clean_frames[0])
    for t in range(1, sc.n_frames):
        assert not np.allclose(sc.frames[t], sc.clean_frames[t])


def test_severity_zero_means_no_degradation() -> None:
    sc = build_scenario(3, SMALL, "fog", 0.0)
    assert np.array_equal(sc.frames, sc.clean_frames)


def test_same_arguments_give_the_same_scenario_and_other_seeds_differ() -> None:
    a = build_scenario(4, SMALL, "thermal", 1.0)
    b = build_scenario(4, SMALL, "thermal", 1.0)
    c = build_scenario(5, SMALL, "thermal", 1.0)
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.positions, b.positions)
    assert not np.array_equal(a.frames, c.frames)


def test_ground_truth_boxes_surround_the_ship_positions() -> None:
    sc = build_scenario(6, SMALL, "sar_speckle", 1.0)
    for ship in range(sc.n_ships):
        for t in range(sc.n_frames):
            x1, y1, x2, y2 = sc.boxes[ship, t]
            cx, cy = sc.positions[ship, t]
            assert x1 - 1 <= cx <= x2 + 1 and y1 - 1 <= cy <= y2 + 1


def test_ships_really_move_between_frames() -> None:
    sc = build_scenario(7, SMALL, "sar_speckle", 1.0)
    step = np.linalg.norm(np.diff(sc.positions, axis=1), axis=2)
    assert sc.n_ships >= 1 and step.min() > 0.3  # at least ~0.15 px/min over 2 minutes


def test_moving_ship_changes_the_clean_image_but_the_background_is_static() -> None:
    sc = build_scenario(8, SMALL, "sar_speckle", 1.0)
    diff = np.abs(sc.clean_frames[-1] - sc.clean_frames[0])
    assert diff.max() > 0.1  # a ship moved: bright pixels appear / disappear
    assert (diff < 1e-6).mean() > 0.8  # most of the sea did not change


def test_config_defaults_match_the_yaml_file() -> None:
    """The API runs on dataclass defaults (no YAML in the Docker image): they must not drift."""
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert ScenarioConfig() == ScenarioConfig.from_config(config)
