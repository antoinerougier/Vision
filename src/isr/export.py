"""Export the trained PyTorch checkpoints to ONNX and verify numerical parity.

Usage:
    python -m isr.export --config configs/default.yaml

Why ONNX? The production container only needs ONNX Runtime (a few tens of MB) instead of
the full PyTorch stack (hundreds of MB to several GB), and the same .onnx files can later
be compiled by TensorRT for a Jetson, or by OpenVINO for an Intel edge device.

The parity check is a safety net: after export we run the SAME random input through the
PyTorch model and through ONNX Runtime and fail loudly if the outputs differ. An export
that silently changes the model's behaviour is a classic deployment bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .common import load_config
from .models import DetectorNet, RestorationNet


def export_model(
    model: torch.nn.Module,
    path: Path,
    opset: int,
    tolerance: float,
    input_size: int = 128,
) -> float:
    """Export one model with dynamic spatial axes and return the max abs parity error."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 1, input_size, input_size)

    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["output"],
        # Batch, height and width are dynamic: the API can receive images of any size.
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=opset,
        dynamo=False,  # classic TorchScript-based exporter: stable and dependency-free
    )

    # --- Parity check on a different size than the tracing size (tests the dynamic axes). ---
    probe = torch.rand(2, 1, 64, 96)
    with torch.no_grad():
        expected = model(probe).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {"image": probe.numpy()})
    error = float(np.max(np.abs(expected - actual)))
    if error > tolerance:
        raise RuntimeError(
            f"ONNX parity check FAILED for {path.name}: max abs error {error:.2e}"
        )
    print(f"Exported {path} (max abs error vs PyTorch: {error:.2e})")
    return error


def load_checkpoint(
    path: str | Path, model_cls: type[torch.nn.Module]
) -> torch.nn.Module:
    """Rebuild a model from a checkpoint written by the training scripts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = model_cls(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    export_cfg = config["export"]
    out_dir = Path(export_cfg["onnx_dir"])

    for name, model_cls in (("restoration", RestorationNet), ("detector", DetectorNet)):
        model = load_checkpoint(config[name]["checkpoint"], model_cls)
        export_model(
            model,
            out_dir / f"{name}.onnx",
            export_cfg["opset"],
            export_cfg["parity_tolerance"],
            config["data"]["image_size"],
        )

    # The baseline detector (trained with degradation augmentation) is exported too when it
    # exists. It is only used by the benchmark and is NOT copied into the Docker image.
    baseline = Path(config["detector"]["augmented_checkpoint"])
    if baseline.exists():
        export_model(
            load_checkpoint(baseline, DetectorNet),
            out_dir / "detector_augmented.onnx",
            export_cfg["opset"],
            export_cfg["parity_tolerance"],
            config["data"]["image_size"],
        )


if __name__ == "__main__":
    main()
