"""Shared pytest fixtures.

The API / inference tests must not depend on a trained model being committed: they use
freshly initialised, *untrained* networks exported to ONNX in a temporary folder. We
test the plumbing (shapes, formats, error handling, ONNX parity), not model quality.
Model quality is the job of the separate quality gate (`evaluate --gate`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder with untrained restoration.onnx and detector.onnx."""
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from isr_restore.export import export_model
    from isr_restore.models import DetectorNet, RestorationNet

    model_dir = tmp_path_factory.mktemp("models")
    export_model(RestorationNet(8), model_dir / "restoration.onnx", 17, 1e-4, 64)
    export_model(DetectorNet(8), model_dir / "detector.onnx", 17, 1e-4, 64)
    return model_dir


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
