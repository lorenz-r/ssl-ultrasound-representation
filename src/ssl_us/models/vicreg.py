"""VICReg: projector + invariance/variance/covariance loss (PLAN.md §5.1, WI-12).

The three terms are always logged **separately** (PLAN.md §4.7) — never only
their weighted sum — because the sum can fall while the representation
collapses into a low-dimensional subspace (a falling *variance* term with a
falling loss is exactly the dimensional-collapse signature the RankMe
instrumentation in `train/instrument.py` exists to catch).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """3-layer MLP projector, 2048-2048-2048 by default (PLAN.md §5.1).

    BatchNorm + ReLU between the first two layers only, matching the
    standard VICReg projector (the final layer is linear so the loss sees a
    raw embedding, not a post-activation one).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class VICRegLossWeights:
    invariance: float = 25.0
    variance: float = 25.0
    covariance: float = 1.0
    variance_margin: float = 1.0  # the hinge target std (gamma in the VICReg paper)


def invariance_term(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """MSE between the two views' projections — the "same underlying scene" term."""
    return F.mse_loss(z1, z2)


def variance_term(z: torch.Tensor, margin: float, eps: float = 1e-4) -> torch.Tensor:
    """Hinge loss pushing every projector dimension's std up to at least
    ``margin``. This is the term that most directly prevents collapse to a
    constant (or low-rank) embedding.
    """
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(margin - std).mean()


def covariance_term(z: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal covariance — decorrelates projector dimensions
    so they carry independent information instead of duplicating one signal.
    """
    z = z - z.mean(dim=0)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return (off_diag**2).sum() / d


def vicreg_loss(
    z1: torch.Tensor, z2: torch.Tensor, weights: VICRegLossWeights | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full VICReg loss. Returns ``(total, {"invariance":..., "variance":...,
    "covariance":...})`` — callers must log all three keys separately, not
    just ``total`` (PLAN.md §4.7).
    """
    weights = weights or VICRegLossWeights()
    inv = invariance_term(z1, z2)
    var1 = variance_term(z1, weights.variance_margin)
    var2 = variance_term(z2, weights.variance_margin)
    var = (var1 + var2) / 2
    cov = (covariance_term(z1) + covariance_term(z2)) / 2
    total = weights.invariance * inv + weights.variance * var + weights.covariance * cov
    return total, {"invariance": inv, "variance": var, "covariance": cov}


class ViTBackbone(nn.Module):
    """ViT-S emitting one mean-pooled vector, so it drops into `VICReg`
    wherever `ResNetBackbone` fits (WI-18, docs/MEMORY.md §16.7).

    Why this exists: §16 measured I-JEPA against the VICReg arms and could not
    separate the objective from the backbone, because the two moved together —
    ViT-S with a latent-target objective against ResNet-18 with a
    joint-embedding one. This fills the missing cell of that 2x2. Holding the
    objective fixed and swapping only the backbone is the *only* comparison
    that attributes §16.3's result to one or the other.

    Pooling is a plain mean over patch tokens with no CLS, matching
    `IJEPA.encode` and `MAE2D.encode` exactly — the three ViT arms must be
    probed on the same kind of vector or the comparison acquires a second
    difference. `ViTEncoder` is imported from `ijepa` because that is where the
    generic ViT already lives; it is not I-JEPA-specific (`ijepa` itself
    imports `PatchEmbed` from `mae2d` for the same reason).
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        embed_dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        seed: int | None = None,
    ):
        super().__init__()
        from .ijepa import ViTEncoder

        if seed is not None:
            torch.manual_seed(seed)
        self.net = ViTEncoder(in_channels, img_size, patch_size, embed_dim, depth, heads)
        self.dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W) -> (B, embed_dim)``."""
        return self.net(x).mean(1)


def build_backbone(
    name: str, in_channels: int, crop_size: int, patch_size: int = 0, seed: int | None = None
) -> nn.Module:
    """Backbone by name, returning a module with a `.dim` attribute.

    `patch_size` 0 means "derive from the crop", `crop_size // 16`, which is the
    same rule `scripts/train/ijepa.py` applies — a 16x16 = 256-token grid at
    every K. Keeping the token count fixed across arms is what stops sequence
    length becoming a third difference in the comparison.
    """
    if name == "resnet18":
        from .resnet import ResNetBackbone

        return ResNetBackbone(in_channels, seed=seed)
    if name == "vit_s":
        return ViTBackbone(
            in_channels, crop_size, patch_size or max(1, crop_size // 16), seed=seed
        )
    raise ValueError(f"unknown backbone {name!r} (expected 'resnet18' or 'vit_s')")


class VICReg(nn.Module):
    """Backbone + projector, exposing both the 512-d backbone feature (what
    RankMe and the frozen-feature probes read, PLAN.md §4.7) and the
    projector output (what the loss and the per-dimension std histogram read).

    `backbone_dim` is the backbone's output width, 512 for ResNet-18 and 384
    for the ViT-S of `build_backbone("vit_s", ...)`; read it off the backbone's
    `.dim` rather than assuming the ResNet constant.
    """

    def __init__(self, backbone: nn.Module, backbone_dim: int, projector_dim: int = 2048):
        super().__init__()
        self.backbone = backbone
        self.projector = Projector(backbone_dim, projector_dim, projector_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(backbone_features, projected)``."""
        h = self.backbone(x)
        z = self.projector(h)
        return h, z
