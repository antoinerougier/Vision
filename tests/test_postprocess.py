"""Encoding ground truth into CenterNet targets and decoding it back must be lossless."""

from __future__ import annotations

import numpy as np

from isr_restore.postprocess import STRIDE, decode, encode_targets


def perfect_output(boxes: np.ndarray, size: int) -> np.ndarray:
    """Build the network output that a *perfect* detector would produce for these boxes."""
    heat, wh, offset, _mask = encode_targets(boxes, size)
    heat = np.clip(heat, 1e-4, 1 - 1e-4)
    logits = np.log(heat / (1 - heat))
    return np.concatenate([logits, wh, offset], axis=0)


def sort_boxes(boxes: np.ndarray) -> np.ndarray:
    return boxes[np.lexsort((boxes[:, 1], boxes[:, 0]))]


def test_encode_decode_roundtrip() -> None:
    boxes = np.array([[10, 12, 30, 22], [70, 80, 85, 100]], dtype=np.float32)
    detections = decode(perfect_output(boxes, 128), score_threshold=0.5)
    assert detections.shape == (2, 5)
    np.testing.assert_allclose(sort_boxes(detections[:, :4]), sort_boxes(boxes), atol=0.01)


def test_encoded_targets_have_expected_structure() -> None:
    boxes = np.array([[16, 16, 32, 24]], dtype=np.float32)
    heat, wh, offset, mask = encode_targets(boxes, 128)
    assert heat.shape == (1, 128 // STRIDE, 128 // STRIDE)
    assert heat.max() == 1.0  # exactly 1 at the object centre
    assert mask.sum() == 1.0
    iy, ix = np.unravel_index(np.argmax(mask[0]), mask[0].shape)
    assert (ix, iy) == (6, 5)  # centre (24, 20) / stride 4
    np.testing.assert_allclose(wh[:, iy, ix] * 32.0, [16, 8], atol=1e-5)
    assert np.all((offset[:, iy, ix] >= 0) & (offset[:, iy, ix] < 1))


def test_no_boxes_gives_empty_targets_and_no_detections() -> None:
    boxes = np.zeros((0, 4), dtype=np.float32)
    heat, _, _, mask = encode_targets(boxes, 64)
    assert heat.sum() == 0 and mask.sum() == 0
    output = np.zeros((5, 16, 16), dtype=np.float32)
    output[0] = -10.0  # very confident "nothing here"
    assert decode(output).shape == (0, 5)


def test_decode_respects_score_threshold_and_max_detections() -> None:
    output = np.zeros((5, 16, 16), dtype=np.float32)
    output[0] = -10.0
    for i, logit in enumerate([4.0, 2.0, 0.5]):  # three well-separated peaks
        output[0, 2 + 5 * i, 2 + 5 * i] = logit
    assert len(decode(output, score_threshold=0.9)) == 1  # sigmoid(4.0) = 0.98 only
    assert len(decode(output, score_threshold=0.5)) == 3
    assert len(decode(output, score_threshold=0.5, max_detections=2)) == 2


def test_decode_keeps_only_local_maxima() -> None:
    """Two adjacent high cells must give ONE detection (CenterNet-style NMS)."""
    output = np.zeros((5, 16, 16), dtype=np.float32)
    output[0] = -10.0
    output[0, 8, 8] = 5.0
    output[0, 8, 9] = 4.0
    assert len(decode(output, score_threshold=0.5)) == 1
