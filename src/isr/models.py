"""PyTorch networks. Both are fully convolutional, so they accept any image size.

    RestorationNet  a small residual U-Net: degraded image -> restored image
    DetectorNet     a small CenterNet-style detector (stride 4, 5 output channels)

They are intentionally tiny (< 150k parameters each) so that they train on a laptop CPU
and run in a few milliseconds, which is the whole point of an "Edge AI" pre-processing
layer. Swap them for bigger backbones when moving to real data.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv(in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """3x3 convolution + BatchNorm + ReLU. Padding keeps the spatial size (before stride)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class RestorationNet(nn.Module):
    """Residual U-Net that learns the *correction* to apply to a degraded image.

    Predicting `degraded + correction` (rather than the clean image directly) makes
    the task easier: the network only has to learn what is wrong, and the identity
    mapping is the starting point at initialisation.

    Input / output: (N, 1, H, W) floats in [0, 1]. H and W must be multiples of 4.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 3

        # Encoder: two downsampling stages. Skip connections keep fine detail (small ships).
        self.enc1 = nn.Sequential(_conv(1, c1), _conv(c1, c1))
        self.enc2 = nn.Sequential(_conv(c1, c2, stride=2), _conv(c2, c2))
        self.enc3 = nn.Sequential(_conv(c2, c3, stride=2), _conv(c3, c3), _conv(c3, c3, dilation=2))

        # Decoder: bilinear upsampling + conv (exports cleanly to ONNX, no checkerboard artefacts).
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(_conv(c3 + c2, c2), _conv(c2, c2))
        self.dec1 = nn.Sequential(_conv(c2 + c1, c1), _conv(c1, c1))
        self.head = nn.Conv2d(c1, 1, 3, padding=1)

        # Start as the identity mapping: the residual branch outputs zero at init.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return x + self.head(d1)


class DetectorNet(nn.Module):
    """CenterNet-style single-class detector with output stride 4.

    Output: (N, 5, H/4, W/4) = [heatmap logit, width, height, offset_x, offset_y].
    See `postprocess.py` for the exact meaning and the decoding.
    """

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _conv(1, base, stride=2),  # stride 2
            _conv(base, base),
            _conv(base, base * 2, stride=2),  # stride 4
            _conv(base * 2, base * 2),
            _conv(base * 2, base * 2, dilation=2),  # larger receptive field, same resolution
            _conv(base * 2, base * 2, dilation=3),
        )
        self.head = nn.Sequential(_conv(base * 2, base * 2), nn.Conv2d(base * 2, 5, 1))

        # Focal-loss friendly initialisation: start with "almost no object" everywhere
        # (prior probability ~ 0.1) so the first gradient steps are not dominated by
        # the huge number of background cells.
        nn.init.constant_(self.head[-1].bias, 0.0)
        with torch.no_grad():
            self.head[-1].bias[0] = -2.19

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (reported in the README and the metrics file)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
