"""Write a degraded synthetic scene to a PNG file (demo input for the API and the CI smoke test).

Usage:
    python scripts/make_sample_image.py --out sample.png --kind sar_speckle --severity 1.0

Only needs the core dependencies (numpy + OpenCV), not PyTorch.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from isr.degradations import KINDS, degrade
from isr.synthetic import render_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="sample.png")
    parser.add_argument("--kind", choices=KINDS, default="sar_speckle")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    clean, boxes = render_scene(rng, args.size)
    degraded = degrade(clean, rng, args.kind, args.severity)
    cv2.imwrite(args.out, (degraded * 255).astype(np.uint8))
    print(
        f"Wrote {args.out}: {args.kind} at severity {args.severity}, {len(boxes)} ship(s)"
    )


if __name__ == "__main__":
    main()
