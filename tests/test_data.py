"""Datasets: reproducibility, shapes, label/flip alignment and the YOLO folder reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from isr_restore.data import (
    PairedRestorationDataset,
    SyntheticDetectionDataset,
    YoloFolderDataset,
    load_yolo_folder,
)
from isr_restore.postprocess import decode


def test_synthetic_dataset_is_reproducible_and_seed_dependent() -> None:
    a = SyntheticDetectionDataset(4, 64, seed=1)[2]
    b = SyntheticDetectionDataset(4, 64, seed=1)[2]
    c = SyntheticDetectionDataset(4, 64, seed=2)[2]
    assert np.array_equal(a["image"].numpy(), b["image"].numpy())
    assert not np.array_equal(a["image"].numpy(), c["image"].numpy())


def test_synthetic_dataset_shapes() -> None:
    item = SyntheticDetectionDataset(2, 64, seed=0)[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["heat"].shape == (1, 16, 16)
    assert item["wh"].shape == (2, 16, 16)
    assert item["offset"].shape == (2, 16, 16)
    assert item["mask"].shape == (1, 16, 16)


def test_targets_stay_aligned_with_the_image_after_random_flips() -> None:
    """Decode the *targets* and check that the boxes still sit on bright ships.

    If a flip were applied to the image but not (correctly) to the boxes, the decoded boxes
    would land on dark sea and this test would fail.
    """
    dataset = SyntheticDetectionDataset(40, 128, seed=3)
    brighter, total = 0, 0
    for i in range(len(dataset)):
        item = dataset[i]
        heat = np.clip(item["heat"].numpy(), 1e-4, 1 - 1e-4)
        output = np.concatenate(
            [np.log(heat / (1 - heat)), item["wh"].numpy(), item["offset"].numpy()], axis=0
        )
        image = item["image"].numpy()[0]
        for x1, y1, x2, y2, _ in decode(output, score_threshold=0.9):
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            region = image[y1 : int(y2) + 1, x1 : int(x2) + 1]
            total += 1
            brighter += region.size > 0 and region.mean() > image.mean()
    assert total > 20 and brighter / total > 0.9


def test_paired_dataset_returns_matching_pairs() -> None:
    item = PairedRestorationDataset(3, image_size=64, seed=0)[1]
    assert item["clean"].shape == item["degraded"].shape == (1, 64, 64)
    assert item["degraded"].min() >= 0.0 and item["degraded"].max() <= 1.0
    assert not np.array_equal(item["clean"].numpy(), item["degraded"].numpy())


def test_paired_dataset_can_crop_from_a_bank_of_real_images() -> None:
    bank = [np.random.default_rng(0).random((200, 150)).astype(np.float32)]
    item = PairedRestorationDataset(2, image_size=64, seed=0, clean_images=bank)[0]
    assert item["clean"].shape == (1, 64, 64)


def _make_yolo_folder(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    image = np.zeros((100, 200), dtype=np.uint8)  # height 100, width 200
    cv2.imwrite(str(root / "images" / "train" / "a.png"), image)
    # One object: centre (100, 50) px, size 40 x 20 px  ->  normalised 0.5 0.5 0.2 0.2
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_yolo_folder_loader_converts_boxes_to_pixels(tmp_path: Path) -> None:
    _make_yolo_folder(tmp_path)
    ((image, boxes),) = load_yolo_folder(tmp_path, "train")
    assert image.shape == (100, 200) and image.dtype == np.float32
    np.testing.assert_allclose(boxes, [[80, 40, 120, 60]], atol=1e-4)


def test_yolo_folder_dataset_resizes_and_encodes(tmp_path: Path) -> None:
    _make_yolo_folder(tmp_path)
    item = YoloFolderDataset(tmp_path, "train", image_size=64)[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["mask"].sum() == 1.0
