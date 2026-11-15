"""Small helpers shared by the training / export / evaluation entry points."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file.

    All hyper-parameters live in `configs/*.yaml` and never in the code: a training run
    is then fully described by (git commit, config file), which is the basis of
    reproducibility in MLOps.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # serving environment without PyTorch
        pass


def sample_rng(seed: int, index: int) -> np.random.Generator:
    """Deterministic per-sample random generator.

    The sample at index `i` is always generated from the RNG seed `seed * 1_000_003 + i`.
    The same (seed, index) therefore always gives the same sample, whatever the number of
    DataLoader workers, and disjoint seeds give disjoint splits (train / test).
    Lives here (not in data.py) because the evaluation must not import PyTorch.
    """
    return np.random.default_rng(seed * 1_000_003 + index)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file, creating parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
