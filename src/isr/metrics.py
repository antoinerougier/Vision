"""Evaluation metrics (numpy / OpenCV only, so they also run inside the CI gate).

average_precision   single-class AP at a given IoU threshold (= "mAP@0.5" with one class)
psnr                peak signal-to-noise ratio, a pixel-wise restoration metric
ssim                structural similarity (Gaussian-window version)
"""

from __future__ import annotations

import cv2
import numpy as np


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and many boxes (M, 4), all as x1, y1, x2, y2."""
    if len(others) == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    intersection = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def average_precision(
    predictions: list[np.ndarray],
    ground_truths: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Single-class average precision over a whole dataset.

    Parameters
    ----------
    predictions   one (N_i, 5) array per image: x1, y1, x2, y2, score
    ground_truths one (M_i, 4) array per image: x1, y1, x2, y2

    Predictions are ranked by score across the whole dataset; each one is matched
    greedily to the best still-unmatched ground truth of its image (IoU >= threshold),
    otherwise it counts as a false positive. AP is the area under the resulting
    precision/recall curve (all-point interpolation, as in Pascal VOC 2010+).
    """
    n_ground_truth = sum(len(g) for g in ground_truths)
    if n_ground_truth == 0:
        return 0.0

    # Flatten all predictions into (score, image_index, box) records.
    records = [
        (float(p[4]), image_index, p[:4])
        for image_index, preds in enumerate(predictions)
        for p in preds
    ]
    records.sort(key=lambda r: -r[0])

    matched = [np.zeros(len(g), dtype=bool) for g in ground_truths]
    true_positive = np.zeros(len(records))
    false_positive = np.zeros(len(records))

    for rank, (_score, image_index, box) in enumerate(records):
        ious = box_iou(box, ground_truths[image_index])
        if len(ious) > 0:
            # Only still-unmatched ground truths are candidates.
            ious = np.where(matched[image_index], -1.0, ious)
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold:
                matched[image_index][best] = True
                true_positive[rank] = 1
                continue
        false_positive[rank] = 1

    cum_tp = np.cumsum(true_positive)
    cum_fp = np.cumsum(false_positive)
    recall = cum_tp / n_ground_truth
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # Make precision monotonically decreasing, then integrate over recall steps.
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    changes = np.nonzero(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """PSNR in dB for images in [0, 1]. Higher is better."""
    mse = float(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Mean SSIM (Gaussian window, sigma = 1.5) for images in [0, 1]."""
    c1, c2 = 0.01**2, 0.03**2
    x, y = reference.astype(np.float64), estimate.astype(np.float64)

    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a, (11, 11), 1.5)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov_xy = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(ssim_map.mean())
