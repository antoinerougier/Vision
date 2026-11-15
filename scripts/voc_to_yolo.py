"""Convert a Pascal VOC dataset (XML annotations) to the YOLO folder layout.

Most public SAR ship datasets (SSDD, SAR-Ship-Dataset, LS-SSDD, ...) ship their
annotations as Pascal VOC XML files. This script converts them to the layout expected by
`isr.data.load_yolo_folder`:

    <out>/images/<split>/<name>.<ext>
    <out>/labels/<split>/<name>.txt        class cx cy w h   (all normalised to [0, 1])

All classes are merged into class 0 ("ship"), since the project is a single-class detector.

Usage:
    python scripts/voc_to_yolo.py --images path/to/JPEGImages --annotations path/to/Annotations \
        --out data/real --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_voc_xml(
    xml_path: Path,
) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    """Return (width, height, [(xmin, ymin, xmax, ymax), ...]) from one VOC annotation file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size> element")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        boxes.append(
            tuple(
                float(bndbox.findtext(tag)) for tag in ("xmin", "ymin", "xmax", "ymax")
            )
        )
    return width, height, boxes


def to_yolo_line(
    box: tuple[float, float, float, float], width: int, height: int
) -> str:
    """Format one absolute box as a normalised YOLO label line (class 0)."""
    x_min, y_min, x_max, y_max = box
    cx, cy = (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height
    w, h = (x_max - x_min) / width, (y_max - y_min) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert(
    images: Path, annotations: Path, out: Path, val_fraction: float, seed: int
) -> None:
    stems = sorted(
        p.stem for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    random.Random(seed).shuffle(stems)  # seeded shuffle: the split is reproducible
    n_val = int(len(stems) * val_fraction)
    splits = {"val": stems[:n_val], "train": stems[n_val:]}

    image_by_stem = {
        p.stem: p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }
    for split, split_stems in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in split_stems:
            xml_path = annotations / f"{stem}.xml"
            if not xml_path.exists():
                continue
            width, height, boxes = parse_voc_xml(xml_path)
            lines = [to_yolo_line(b, width, height) for b in boxes]
            shutil.copy2(
                image_by_stem[stem], out / "images" / split / image_by_stem[stem].name
            )
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print(
        f"Converted {len(stems)} images -> {out} ({n_val} val / {len(stems) - n_val} train)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/real"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(args.images, args.annotations, args.out, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
