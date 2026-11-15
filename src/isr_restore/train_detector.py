"""Train the CenterNet-style detector on CLEAN images.

Usage:
    python -m isr_restore.train_detector --config configs/default.yaml

The detector plays the role of "the customer's existing model": it is trained under
clean, laboratory-like conditions only and never sees degraded images, nor the
restoration network.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import load_config, set_seed
from .data import SyntheticDetectionDataset, YoloFolderDataset
from .models import DetectorNet, count_parameters


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss from CenterNet (Zhou et al., 2019).

    Cells exactly at an object centre (target == 1) are positives. All other cells are
    negatives, but cells close to a centre (target close to 1) are penalised less, because
    a slightly shifted prediction is nearly correct. The (1 - p)^2 / p^2 terms
    down-weight the huge number of easy background cells.
    """
    prob = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative_weight = torch.pow(1 - target, 4)

    positive_loss = -torch.log(prob) * torch.pow(1 - prob, 2) * positive
    negative_loss = -torch.log(1 - prob) * torch.pow(prob, 2) * negative_weight * (1 - positive)
    n_positive = positive.sum().clamp(min=1.0)
    return (positive_loss.sum() + negative_loss.sum()) / n_positive


def detection_loss(output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Heatmap focal loss + L1 on box size and centre offset (only at object centres)."""
    heat_loss = focal_loss(output[:, 0:1], batch["heat"])
    mask = batch["mask"]
    n_objects = mask.sum().clamp(min=1.0)
    size_loss = (F.l1_loss(output[:, 1:3], batch["wh"], reduction="none") * mask).sum() / n_objects
    offset_loss = (
        F.l1_loss(output[:, 3:5], batch["offset"], reduction="none") * mask
    ).sum() / n_objects
    return heat_loss + size_loss + offset_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None, help="override config (smoke tests)")
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="train the BASELINE detector that sees degraded images during training "
        "(the 'just retrain your model with augmentation' alternative to our restoration layer)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["detector"]
    steps = args.steps or cfg["steps"]
    degrade_probability = cfg["augmented_degrade_probability"] if args.augmented else 0.0
    checkpoint_path = cfg["augmented_checkpoint"] if args.augmented else cfg["checkpoint"]
    set_seed(config["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if config["data"]["source"] == "yolo":
        dataset = YoloFolderDataset(
            config["data"]["yolo_root"], "train", config["data"]["image_size"]
        )
    else:
        dataset = SyntheticDetectionDataset(
            length=steps * cfg["batch_size"],
            image_size=config["data"]["image_size"],
            seed=cfg["train_seed"],
            degrade_probability=degrade_probability,
        )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    model = DetectorNet(cfg["base_channels"])
    print(f"DetectorNet: {count_parameters(model):,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # Cosine decay to zero: a simple schedule that works well for short runs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    model.train()
    start, step = time.time(), 0
    while step < steps:
        for batch in loader:
            output = model(batch["image"])
            loss = detection_loss(output, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0 or step == 1:
                print(
                    f"[detector] step {step:5d}/{steps}  loss {loss.item():.4f}  "
                    f"{time.time() - start:6.0f}s",
                    flush=True,
                )
            if step >= steps:
                break

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "base_channels": cfg["base_channels"]}, checkpoint
    )
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
