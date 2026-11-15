"""Train the blind restoration network on (degraded, clean) pairs.

Usage:
    python -m isr.train_restoration --config configs/default.yaml

The restorer is trained independently of any detector: its only objective is to
reconstruct the clean image. This is what makes it a drop-in pre-processing layer,
usable in front of ANY downstream model without retraining that model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import load_config, set_seed
from .data import PairedRestorationDataset, load_yolo_folder
from .models import RestorationNet, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None, help="override config (smoke tests)")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["restoration"]
    steps = args.steps or cfg["steps"]
    set_seed(config["seed"])

    clean_bank = None
    if config["data"]["source"] == "yolo":
        # Use real, clean (or at least good-quality) images as the source of "ground truth".
        clean_bank = [img for img, _ in load_yolo_folder(config["data"]["yolo_root"], "train")]

    dataset = PairedRestorationDataset(
        length=steps * cfg["batch_size"],
        image_size=cfg["patch_size"],
        seed=cfg["train_seed"],
        clean_images=clean_bank,
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)

    model = RestorationNet(cfg["base_channels"])
    print(f"RestorationNet: {count_parameters(model):,} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    model.train()
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        restored = model(batch["degraded"])
        # L1 is more robust than L2 to the heavy-tailed errors of speckle and shot noise,
        # and tends to give sharper edges (important to keep small ships visible).
        loss = F.l1_loss(restored, batch["clean"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == 1:
            print(
                f"[restoration] step {step:5d}/{steps}  L1 {loss.item():.4f}  "
                f"{time.time() - start:6.0f}s",
                flush=True,
            )
        if step >= steps:
            break

    checkpoint = Path(cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "base_channels": cfg["base_channels"]},
        checkpoint,
    )
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
