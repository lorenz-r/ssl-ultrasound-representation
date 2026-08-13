"""ResNet-18 with a stem adapted to an arbitrary input channel count
(PLAN.md §5.1). The primary VICReg arm's input is ``(2K, size, size)`` — 2
IQ channels times K receive sectors, e.g. 32 channels at K=16 — so the stock
3-channel stem (built for RGB) cannot be reused as-is.

Kept as a single small module, factored out of what was previously inlined
in :class:`ssl_us.eval.features.RandomResNet18`, so the trained VICReg
encoder and the random-init baseline share exactly one stem-adaptation
implementation rather than two that could silently drift apart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

BACKBONE_DIM = 512  # ResNet-18's penultimate feature width — the RankMe axis (PLAN.md 4.7)


def resnet18_backbone(in_channels: int) -> nn.Module:
    """Random-init ResNet-18, `fc` replaced with `Identity`, stem's `conv1`
    replaced with a ``Conv2d(in_channels, 64, 7, stride=2, padding=3)``.

    Returns a module whose forward pass is ``(N, in_channels, H, W) -> (N, 512)``.
    Never loads ImageNet weights: pretrained RGB filters have no meaning on
    IQ/sector-sum channels, so `weights=None` is not a simplification, it is
    the only correct choice here.
    """
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Identity()
    return net


class ResNetBackbone(nn.Module):
    """Thin wrapper so callers get an explicit `.dim` and `.in_channels`
    without reaching into torchvision's `ResNet` internals."""

    def __init__(self, in_channels: int, seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = resnet18_backbone(in_channels)
        self.in_channels = in_channels
        self.dim = BACKBONE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
