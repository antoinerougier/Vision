"""The scene generator must produce valid images with valid, in-bounds ground truth."""

from __future__ import annotations

import numpy as np

from isr.synthetic import render_scene


def test_scene_shapes_and_range() -> None:
    image, boxes = render_scene(np.random.default_rng(0), 128)
    assert image.shape == (128, 128) and image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert boxes.ndim == 2 and boxes.shape[1] == 4


def test_scene_is_reproducible() -> None:
    a, boxes_a = render_scene(np.random.default_rng(3), 96)
    b, boxes_b = render_scene(np.random.default_rng(3), 96)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(boxes_a, boxes_b)


def test_boxes_are_valid_and_inside_the_image() -> None:
    for seed in range(60):
        _, boxes = render_scene(np.random.default_rng(seed), 128)
        assert len(boxes) <= 4
        assert np.all(boxes[:, 2] > boxes[:, 0]) and np.all(boxes[:, 3] > boxes[:, 1])
        assert np.all(boxes >= 0) and np.all(boxes <= 127)


def test_dataset_mixes_empty_and_populated_scenes() -> None:
    counts = [len(render_scene(np.random.default_rng(s), 128)[1]) for s in range(100)]
    assert min(counts) == 0, "empty scenes teach the detector not to hallucinate ships"
    assert max(counts) >= 2


def test_ships_are_brighter_than_their_surroundings() -> None:
    """Sanity check of the generator: the pixels inside a box are brighter than the image mean."""
    brighter, total = 0, 0
    for seed in range(40):
        image, boxes = render_scene(np.random.default_rng(seed), 128)
        for x1, y1, x2, y2 in boxes.astype(int):
            total += 1
            brighter += image[y1 : y2 + 1, x1 : x2 + 1].mean() > image.mean()
    assert total > 0 and brighter / total > 0.9
