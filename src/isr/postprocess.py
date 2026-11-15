"""Detector target encoding and output decoding (numpy only).

We use a CenterNet-style, anchor-free detector: instead of predicting boxes directly,
the network predicts, on a grid that is `STRIDE` times coarser than the image:

    channel 0     a heatmap: "is there an object centre in this cell?"  (logits)
    channels 1-2  the object width and height, in units of `SIZE_SCALE` pixels
    channels 3-4  the sub-cell offset of the centre (x, y), in [0, 1) cell units

Both the *encoding* (ground truth boxes -> training targets) and the *decoding*
(network output -> boxes) live here, in plain numpy, so that the exact same decoding
code is used during training, in the evaluation benchmark and in the production API.
That removes a classic source of train / serve skew.
"""

from __future__ import annotations

import numpy as np

STRIDE = 4  # image pixels per output cell
SIZE_SCALE = 32.0  # box sizes are regressed in units of 32 px to keep targets around 0..1


def encode_targets(
    boxes: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn ground-truth boxes into dense CenterNet training targets.

    Returns
    -------
    heat   (1, h, w)  Gaussian bump centred on each object (1.0 exactly at the centre)
    wh     (2, h, w)  width / height / SIZE_SCALE, only meaningful where `mask` is 1
    offset (2, h, w)  sub-cell centre offset, only meaningful where `mask` is 1
    mask   (1, h, w)  1 at the cell holding an object centre, else 0
    """
    out = image_size // STRIDE
    heat = np.zeros((1, out, out), dtype=np.float32)
    wh = np.zeros((2, out, out), dtype=np.float32)
    offset = np.zeros((2, out, out), dtype=np.float32)
    mask = np.zeros((1, out, out), dtype=np.float32)

    ys, xs = np.mgrid[0:out, 0:out].astype(np.float32)
    for x_min, y_min, x_max, y_max in boxes:
        box_w, box_h = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) / 2 / STRIDE, (y_min + y_max) / 2 / STRIDE
        ix, iy = min(int(cx), out - 1), min(int(cy), out - 1)

        # Gaussian radius grows with object size, with a floor so small objects stay visible.
        sigma = max(0.8, (box_w + box_h) / (2 * STRIDE) / 6.0)
        bump = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
        heat[0] = np.maximum(heat[0], bump)

        wh[:, iy, ix] = (box_w / SIZE_SCALE, box_h / SIZE_SCALE)
        offset[:, iy, ix] = (cx - ix, cy - iy)
        mask[0, iy, ix] = 1.0
    return heat, wh, offset, mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode(
    output: np.ndarray,
    score_threshold: float = 0.3,
    max_detections: int = 50,
) -> np.ndarray:
    """Decode one network output (5, h, w) into detections (N, 5): x1, y1, x2, y2, score.

    Non-maximum suppression is done the CenterNet way: a cell is kept only if it is the
    maximum of its 3x3 neighbourhood in the heatmap. No IoU-based NMS is needed.
    """
    heat = _sigmoid(output[0])
    h, w = heat.shape

    # 3x3 local maximum test using padded shifts (pure numpy, no scipy / torch).
    padded = np.pad(heat, 1, mode="constant", constant_values=-1.0)
    neighbourhood_max = np.max(
        [padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)], axis=0
    )
    is_peak = (heat >= neighbourhood_max) & (heat >= score_threshold)

    ys, xs = np.nonzero(is_peak)
    if len(ys) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    scores = heat[ys, xs]
    order = np.argsort(-scores)[:max_detections]
    ys, xs, scores = ys[order], xs[order], scores[order]

    box_w = np.maximum(output[1, ys, xs], 0.0) * SIZE_SCALE
    box_h = np.maximum(output[2, ys, xs], 0.0) * SIZE_SCALE
    cx = (xs + np.clip(output[3, ys, xs], 0.0, 1.0)) * STRIDE
    cy = (ys + np.clip(output[4, ys, xs], 0.0, 1.0)) * STRIDE

    detections = np.stack(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2, scores], axis=1
    )
    return detections.astype(np.float32)
