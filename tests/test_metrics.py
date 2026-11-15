"""Metrics are tested against hand-computed values: a wrong metric silently ruins every result."""

from __future__ import annotations

import numpy as np
import pytest

from isr.metrics import average_precision, box_iou, psnr, ssim

GT = [np.array([[10, 10, 30, 30]], dtype=np.float32)]


def test_iou_known_values() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    others = np.array(
        [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32
    )
    np.testing.assert_allclose(box_iou(box, others), [1.0, 1 / 3, 0.0], atol=1e-6)


def test_perfect_detection_gives_ap_one() -> None:
    preds = [np.array([[10, 10, 30, 30, 0.9]], dtype=np.float32)]
    assert average_precision(preds, GT) == pytest.approx(1.0)


def test_no_detection_gives_ap_zero() -> None:
    assert average_precision([np.zeros((0, 5), dtype=np.float32)], GT) == 0.0


def test_only_false_positives_gives_ap_zero() -> None:
    preds = [np.array([[60, 60, 80, 80, 0.9]], dtype=np.float32)]
    assert average_precision(preds, GT) == 0.0


def test_ranking_matters() -> None:
    """A wrong high-score box ranked above the right one halves the precision at recall 1."""
    wrong_first = [
        np.array([[60, 60, 80, 80, 0.9], [10, 10, 30, 30, 0.5]], dtype=np.float32)
    ]
    right_first = [
        np.array([[60, 60, 80, 80, 0.5], [10, 10, 30, 30, 0.9]], dtype=np.float32)
    ]
    assert average_precision(wrong_first, GT) == pytest.approx(0.5)
    assert average_precision(right_first, GT) == pytest.approx(1.0)


def test_duplicate_detection_counts_as_false_positive() -> None:
    """Two boxes on the same object: the second one is a false positive (but AP stays 1.0)."""
    preds = [np.array([[10, 10, 30, 30, 0.9], [11, 11, 31, 31, 0.8]], dtype=np.float32)]
    assert average_precision(preds, GT) == pytest.approx(1.0)
    # ...while a second ground truth that is missed lowers it:
    two_gt = [np.array([[10, 10, 30, 30], [60, 60, 80, 80]], dtype=np.float32)]
    assert average_precision(preds, two_gt) == pytest.approx(0.5)


def test_iou_threshold_is_respected() -> None:
    shifted = [np.array([[20, 10, 40, 30, 0.9]], dtype=np.float32)]  # IoU = 1/3
    assert average_precision(shifted, GT, iou_threshold=0.5) == 0.0
    assert average_precision(shifted, GT, iou_threshold=0.3) == pytest.approx(1.0)


def test_no_ground_truth_gives_zero() -> None:
    assert (
        average_precision([np.zeros((0, 5), dtype=np.float32)], [np.zeros((0, 4))])
        == 0.0
    )


def test_psnr_known_values() -> None:
    a = np.zeros((8, 8), dtype=np.float32)
    assert psnr(a, a) == 99.0
    assert psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-3)  # MSE 0.01 -> 20 dB


def test_ssim_bounds() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((64, 64)).astype(np.float32)
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    assert ssim(image, rng.random((64, 64)).astype(np.float32)) < 0.5
