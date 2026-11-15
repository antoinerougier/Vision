"""PyTorch datasets.

Three datasets are provided:

    SyntheticDetectionDataset  clean synthetic scenes + CenterNet targets (trains the detector)
    PairedRestorationDataset   (degraded, clean) pairs made on the fly (trains the restorer)
    YoloFolderDataset          real images in the standard YOLO folder layout

Reproducibility rule: the sample at index `i` is generated from `sample_rng(seed, i)`
(see common.py), so the same (seed, index) always gives the same sample.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import sample_rng
from .degradations import degrade
from .postprocess import encode_targets
from .synthetic import render_scene


class SyntheticDetectionDataset(Dataset):
    """Synthetic scenes with dense CenterNet targets.

    By default the detector is trained on CLEAN images only. This mimics the real-world
    situation the product addresses: a customer's detector trained in "laboratory"
    conditions that then meets degraded imagery in the field.

    `degrade_probability > 0` builds the main competing approach instead: a detector
    trained WITH degradation augmentation ("just retrain the model on degraded data"),
    used as a baseline in the benchmark.
    """

    def __init__(
        self,
        length: int,
        image_size: int = 128,
        seed: int = 0,
        degrade_probability: float = 0.0,
    ) -> None:
        self.length, self.image_size, self.seed = length, image_size, seed
        self.degrade_probability = degrade_probability

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = sample_rng(self.seed, index)
        image, boxes = render_scene(rng, self.image_size)

        # Random horizontal / vertical flips (boxes are flipped consistently).
        if rng.random() < 0.5:
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = self.image_size - 1 - boxes[:, [2, 0]]
        if rng.random() < 0.5:
            image = image[::-1, :]
            boxes = boxes.copy()
            boxes[:, [1, 3]] = self.image_size - 1 - boxes[:, [3, 1]]

        # Only draw from the RNG when augmenting, so that clean-data runs stay bit-identical.
        image = np.ascontiguousarray(image)
        if self.degrade_probability > 0 and rng.random() < self.degrade_probability:
            image = degrade(image, rng)  # random kind, random severity

        heat, wh, offset, mask = encode_targets(boxes, self.image_size)
        return {
            "image": torch.from_numpy(image)[None],
            "heat": torch.from_numpy(heat),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "mask": torch.from_numpy(mask),
        }


class PairedRestorationDataset(Dataset):
    """(degraded, clean) image pairs, generated on the fly.

    Because degradation parameters are drawn randomly at every access, the network
    effectively sees an unlimited stream of different pairs: a form of synthetic data
    augmentation that makes over-fitting to a fixed set of images very unlikely.
    """

    def __init__(
        self,
        length: int,
        image_size: int = 96,
        seed: int = 0,
        clean_images: list[np.ndarray] | None = None,
    ) -> None:
        self.length, self.image_size, self.seed = length, image_size, seed
        # Optional bank of REAL clean images (float32 (H, W) in [0, 1]) to crop from,
        # instead of rendering synthetic scenes.
        self.clean_images = clean_images

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = sample_rng(self.seed, index)
        if self.clean_images:
            clean = self._random_crop(
                self.clean_images[int(rng.integers(len(self.clean_images)))], rng
            )
        else:
            clean, _ = render_scene(rng, self.image_size)
        degraded = degrade(clean, rng)  # random kind, random severity
        return {
            "degraded": torch.from_numpy(degraded)[None],
            "clean": torch.from_numpy(np.ascontiguousarray(clean))[None],
        }

    def _random_crop(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape
        s = self.image_size
        if h < s or w < s:
            image = cv2.resize(image, (max(w, s), max(h, s)))
            h, w = image.shape
        y, x = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        return np.ascontiguousarray(image[y : y + s, x : x + s])


def load_yolo_folder(root: str | Path, split: str = "val") -> list[tuple[np.ndarray, np.ndarray]]:
    """Read a YOLO-format dataset into memory as a list of (image, boxes).

    Expected layout (the de-facto standard used by Ultralytics, Roboflow, ...):

        root/images/<split>/*.png|jpg     grayscale or colour images
        root/labels/<split>/*.txt         one line per object: `class cx cy w h` (normalised)

    Images are converted to float32 grayscale in [0, 1]; boxes to absolute pixels
    (x1, y1, x2, y2). All classes are merged into one ("object"), since this project
    is a single-class detector.
    """
    root = Path(root)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for image_path in sorted((root / "images" / split).glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        image = gray.astype(np.float32) / 255.0
        h, w = image.shape

        label_path = root / "labels" / split / f"{image_path.stem}.txt"
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append(
                    [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                )
        samples.append((image, np.array(boxes, dtype=np.float32).reshape(-1, 4)))
    return samples


class YoloFolderDataset(Dataset):
    """Real images in YOLO layout, resized to a fixed square size, with CenterNet targets."""

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 128) -> None:
        self.image_size = image_size
        self.samples = load_yolo_folder(root, split)
        if not self.samples:
            raise FileNotFoundError(f"No images found in {Path(root) / 'images' / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, boxes = self.samples[index]
        h, w = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        boxes = boxes * np.array([self.image_size / w, self.image_size / h] * 2, dtype=np.float32)
        heat, wh, offset, mask = encode_targets(boxes, self.image_size)
        return {
            "image": torch.from_numpy(image)[None],
            "heat": torch.from_numpy(heat),
            "wh": torch.from_numpy(wh),
            "offset": torch.from_numpy(offset),
            "mask": torch.from_numpy(mask),
        }
