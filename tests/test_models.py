"""Network shapes, initial behaviour, losses and a tiny end-to-end training smoke test."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from isr.data import SyntheticDetectionDataset  # noqa: E402
from isr.models import DetectorNet, RestorationNet, count_parameters  # noqa: E402
from isr.train_detector import detection_loss, focal_loss  # noqa: E402


def test_restoration_net_is_identity_at_initialisation() -> None:
    model = RestorationNet(8).eval()
    x = torch.rand(2, 1, 32, 32)
    assert torch.allclose(model(x), x)


@pytest.mark.parametrize("size", [(32, 32), (64, 96)])
def test_restoration_net_preserves_spatial_size(size: tuple[int, int]) -> None:
    out = RestorationNet(8).eval()(torch.rand(1, 1, *size))
    assert out.shape == (1, 1, *size)


def test_detector_output_is_stride_four_with_five_channels() -> None:
    out = DetectorNet(8).eval()(torch.rand(2, 1, 64, 96))
    assert out.shape == (2, 5, 16, 24)


def test_models_are_small_enough_for_edge_deployment() -> None:
    assert count_parameters(RestorationNet(16)) < 200_000
    assert count_parameters(DetectorNet(24)) < 200_000


def test_focal_loss_is_lower_for_a_correct_prediction() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    good = torch.full((1, 1, 8, 8), -8.0)
    good[0, 0, 4, 4] = 8.0
    bad = torch.full((1, 1, 8, 8), 8.0)  # predicts objects everywhere
    assert focal_loss(good, target) < focal_loss(bad, target)


def test_focal_loss_handles_empty_images() -> None:
    """No object at all must not divide by zero."""
    loss = focal_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    assert torch.isfinite(loss)


@pytest.mark.slow
def test_detector_loss_decreases_when_overfitting_a_small_batch() -> None:
    """If the loss cannot go down on 8 images, something is broken in the training logic."""
    torch.manual_seed(0)
    dataset = SyntheticDetectionDataset(8, 64, seed=5)
    batch = {k: torch.stack([dataset[i][k] for i in range(8)]) for k in dataset[0]}
    model = DetectorNet(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = detection_loss(model(batch["image"]), batch).item()
    for _ in range(60):
        loss = detection_loss(model(batch["image"]), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.6 * first
