"""The degradation simulator must be valid, reproducible and monotonic in severity."""

from __future__ import annotations

import numpy as np
import pytest

from isr.degradations import KINDS, degrade
from isr.metrics import psnr
from isr.synthetic import render_scene


@pytest.fixture()
def clean_image() -> np.ndarray:
    image, _ = render_scene(np.random.default_rng(7), 96)
    return image


@pytest.mark.parametrize("kind", KINDS)
def test_output_is_valid_image(
    kind: str, clean_image: np.ndarray, rng: np.random.Generator
) -> None:
    out = degrade(clean_image, rng, kind=kind, severity=1.0)
    assert out.shape == clean_image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_gives_same_result(kind: str, clean_image: np.ndarray) -> None:
    a = degrade(clean_image, np.random.default_rng(5), kind=kind, severity=0.8)
    b = degrade(clean_image, np.random.default_rng(5), kind=kind, severity=0.8)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_degradation_actually_changes_the_image(
    kind: str, clean_image: np.ndarray
) -> None:
    out = degrade(clean_image, np.random.default_rng(5), kind=kind, severity=1.0)
    assert psnr(clean_image, out) < 25.0  # clearly different from the clean image


@pytest.mark.parametrize("kind", KINDS)
def test_higher_severity_is_worse(kind: str, clean_image: np.ndarray) -> None:
    """PSNR against the clean image must drop as severity increases."""
    mild = degrade(clean_image, np.random.default_rng(5), kind=kind, severity=0.2)
    harsh = degrade(clean_image, np.random.default_rng(5), kind=kind, severity=1.2)
    assert psnr(clean_image, harsh) < psnr(clean_image, mild)


def test_random_kind_and_severity_are_drawn_when_omitted(
    clean_image: np.ndarray,
) -> None:
    rng = np.random.default_rng(0)
    outputs = [degrade(clean_image, rng) for _ in range(8)]
    # Eight independent draws should not all be identical.
    assert len({o.tobytes() for o in outputs}) > 1


def test_unknown_kind_raises(clean_image: np.ndarray, rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="Unknown degradation"):
        degrade(clean_image, rng, kind="not_a_kind")
