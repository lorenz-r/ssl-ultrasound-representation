"""I-JEPA: image joint-embedding predictive architecture (WI-17, Assran et al. 2023).

The third pre-training arm, and the one the project's own measurements argue
for. §1.3 factored a speckle patch into ``(local scale) x (interference
pattern)`` and measured the neighbour correlation of each: **0.87 lateral for
the scale, -0.0001 for the normalised pattern**. §3.4 then swept MAE across
patch 4-16 and mask 25-75 % and found *pixel-space* masked reconstruction
extracts **< 1 % of masked-patch variance** anywhere in that grid. I-JEPA's
premise is exactly the escape route those two numbers point at: predict a
**representation** of the masked region rather than its pixels, so the
objective is never asked to reproduce an interference realisation that is
uncorrelated even between adjacent patches.

Three components, standard (§5.2 WI-17):

- a **context encoder** ``f_theta``, the ViT that is trained and later probed;
- a **target encoder** ``f_theta_bar``, an EMA copy of it, never receiving
  gradient;
- a narrow **predictor** ``g_phi``, which is handed the context tokens plus the
  positional embeddings of one target block and must produce that block's
  target-encoder representation.

Masks follow the reference multi-block sampler: **4 target blocks** at scale
0.15-0.2 with aspect ratio 0.75-1.5, and **one context block** at scale
0.85-1.0 from which every token overlapping a target block is then *removed*.
That removal is the whole design — leave it out and the predictor solves the
task by copying an input token, which is the same shortcut §4.5a mitigates on
the aperture axis.

**Masks are shared across the batch**, sampled once per forward pass, matching
the reference `MaskCollator` (it samples in the collator, so every image in a
batch gets the same configuration). Two consequences worth knowing: the four
target blocks share one sampled ``(h, w)`` so their token counts match and can
be stacked, and token selection is a single `index_select` rather than a
per-sample gather.

Deliberate deviations from `models/mae2d.py`, so the two ViT arms are not
assumed identical:

- **Pre-norm, GELU, no dropout.** `mae2d.py` takes torch's
  `TransformerEncoderLayer` defaults, which are post-norm, ReLU and
  ``dropout=0.1``. I-JEPA has no reconstruction anchor and collapse is its
  characteristic failure, so the from-scratch-stable convention is used here.
  Dropout is off for a second and stronger reason: it is active in `train()`
  mode, and a dropout-perturbed *target* encoder would inject noise straight
  into the regression target. :meth:`IJEPA.train` additionally pins the target
  encoder to eval mode so no future change to that default can leak in.
- **The probe-facing representation is the target (EMA) encoder**, mean-pooled
  — what the paper evaluates and what its released weights are.

`PatchEmbed` and `sincos_2d_pos_embed` are imported from `mae2d` rather than
re-implemented, for the reason `models/resnet.py` exists: two copies of the
tokenisation is how the two ViT arms come to differ for a reason that is not
the objective.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mae2d import PatchEmbed, sincos_2d_pos_embed

MASK_SAMPLE_ATTEMPTS = 20


@dataclass(frozen=True)
class MaskConfig:
    """Reference multi-block masking parameters (I-JEPA §4, `MaskCollator`)."""

    n_targets: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    context_scale: tuple[float, float] = (0.85, 1.0)
    context_aspect: tuple[float, float] = (1.0, 1.0)
    # Floor on how many context tokens may survive the target-block removal. A
    # context of one or two tokens is not a hard task, it is a broken one, and
    # it happens easily on the small grids this project runs (16x16, against
    # the paper's 14x14 on 224 px — but with 4 target blocks that can land
    # anywhere).
    min_keep: int = 4


def _sample_block_size(
    grid_h: int, grid_w: int, scale: tuple[float, float], aspect: tuple[float, float],
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """One ``(h, w)`` in *tokens* for the given area scale and aspect ratio."""
    u = torch.rand(2, generator=generator)
    area = (scale[0] + float(u[0]) * (scale[1] - scale[0])) * grid_h * grid_w
    ratio = aspect[0] + float(u[1]) * (aspect[1] - aspect[0])
    h = max(1, min(grid_h, int(round(math.sqrt(area / ratio)))))
    w = max(1, min(grid_w, int(round(math.sqrt(area * ratio)))))
    return h, w


def block_token_ids(grid_w: int, top: int, left: int, h: int, w: int) -> torch.Tensor:
    """Flattened token ids of the ``h x w`` block whose top-left corner is
    ``(top, left)``, in the row-major order `PatchEmbed` emits."""
    rows = torch.arange(top, top + h).view(-1, 1)
    cols = torch.arange(left, left + w).view(1, -1)
    return (rows * grid_w + cols).reshape(-1)


def _sample_block(
    grid_h: int, grid_w: int, h: int, w: int, generator: torch.Generator | None
) -> torch.Tensor:
    top = int(torch.randint(0, grid_h - h + 1, (1,), generator=generator))
    left = int(torch.randint(0, grid_w - w + 1, (1,), generator=generator))
    return block_token_ids(grid_w, top, left, h, w)


def multiblock_masks(
    grid_h: int, grid_w: int, config: MaskConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample one context mask and ``n_targets`` target masks for a batch.

    Returns:
        ``(context_ids, [target_ids, ...])`` — 1-D `int64` tensors of token
        indices, shared by every image in the batch. Every target block has the
        same length (one ``(h, w)`` is drawn and reused, as in the reference
        collator), so the per-block predictions stack.

    Raises:
        ValueError: if no attempt leaves at least ``config.min_keep`` context
            tokens after the target blocks are removed. That is a geometry
            problem — too few tokens, too many target blocks, or too large a
            target scale — and it is raised rather than silently returning a
            degenerate context.
    """
    config = config or MaskConfig()
    n_tokens = grid_h * grid_w
    for _ in range(MASK_SAMPLE_ATTEMPTS):
        th, tw = _sample_block_size(
            grid_h, grid_w, config.target_scale, config.target_aspect, generator
        )
        targets = [
            _sample_block(grid_h, grid_w, th, tw, generator) for _ in range(config.n_targets)
        ]
        ch, cw = _sample_block_size(
            grid_h, grid_w, config.context_scale, config.context_aspect, generator
        )
        context = _sample_block(grid_h, grid_w, ch, cw, generator)

        # Remove every context token that lands in any target block. Without
        # this the predictor can read the answer off its own input.
        blocked = torch.zeros(n_tokens, dtype=torch.bool)
        for t in targets:
            blocked[t] = True
        context = context[~blocked[context]]
        if context.numel() >= config.min_keep:
            return context, targets

    raise ValueError(
        f"could not sample a context of >= {config.min_keep} tokens on a "
        f"{grid_h}x{grid_w} grid in {MASK_SAMPLE_ATTEMPTS} attempts: "
        f"{config.n_targets} target blocks at scale {config.target_scale} cover "
        "too much of it. Use a finer patch size, fewer target blocks, or a "
        "smaller target scale."
    )


class ViTEncoder(nn.Module):
    """Plain ViT over patch tokens, with optional token subsetting.

    No CLS token: the probe-facing representation is a mean over tokens
    (`IJEPA.encode`), matching how `MAE2D.encode` pools and keeping the two ViT
    arms comparable on that axis at least.
    """

    def __init__(
        self, in_channels: int, img_size: int, patch_size: int, embed_dim: int = 384,
        depth: int = 12, heads: int = 6,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.grid = img_size // patch_size
        self.n_tokens = self.grid * self.grid
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.register_buffer("pos_embed", sincos_2d_pos_embed(self.grid, self.grid, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, dim_feedforward=embed_dim * 4, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, keep_ids: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, C, H, W) -> (B, N_keep, D)``, positional embeddings added
        before subsetting so a kept token carries its *original* position."""
        tokens = self.patch_embed(x) + self.pos_embed
        if keep_ids is not None:
            tokens = tokens.index_select(1, keep_ids)
        return self.norm(self.blocks(tokens))


class Predictor(nn.Module):
    """Narrow ViT mapping (context tokens, target positions) -> target features.

    Deliberately much smaller than the encoder (192-d, 6 blocks against 384-d,
    12): the predictor is discarded after pre-training, and giving it enough
    capacity to model the interference realisation itself would defeat the
    point of predicting in representation space.
    """

    def __init__(
        self, embed_dim: int, grid: int, predictor_dim: int = 192, depth: int = 6, heads: int = 6
    ):
        super().__init__()
        self.embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", sincos_2d_pos_embed(grid, grid, predictor_dim))
        layer = nn.TransformerEncoderLayer(
            predictor_dim, heads, dim_feedforward=predictor_dim * 4, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self, context: torch.Tensor, context_ids: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """``(B, Nc, D), (Nc,), (Nt,) -> (B, Nt, D)``."""
        b, n_context, _ = context.shape
        x = self.embed(context) + self.pos_embed.index_select(0, context_ids)
        queries = (self.mask_token + self.pos_embed.index_select(0, target_ids)).expand(b, -1, -1)
        h = self.blocks(torch.cat([x, queries], dim=1))
        return self.proj(self.norm(h[:, n_context:]))


def momentum_schedule(
    step: int, total_steps: int, base: float = 0.996, final: float = 1.0
) -> float:
    """EMA momentum, linear from ``base`` to ``final`` over the run.

    The paper's schedule. It matters more here than in the paper's setting: an
    EMA that is too fast early makes the target a moving copy of the context
    encoder, which is the configuration that collapses.
    """
    if total_steps <= 0:
        return final
    return base + (final - base) * min(max(step, 0), total_steps) / total_steps


def ijepa_loss(
    predictions: torch.Tensor, targets: torch.Tensor, kind: str = "smooth_l1"
) -> torch.Tensor:
    """Mean prediction error over all target tokens of all target blocks.

    ``smooth_l1`` is what the reference implementation uses (the paper states
    squared L2, and ``l2`` is available for that reading). Both are offered
    because they are not equivalent on this data: the target is a LayerNormed
    representation, but the *input* amplitude distribution is heavy-tailed, and
    a squared penalty lets the few high-amplitude blocks set the gradient — the
    same mechanism that made MAE's raw-target run park at the patch mean (§4.1).
    """
    if kind == "smooth_l1":
        return F.smooth_l1_loss(predictions, targets)
    if kind == "l2":
        return F.mse_loss(predictions, targets)
    raise ValueError(f"unknown loss kind {kind!r}; expected 'smooth_l1' or 'l2'")


class IJEPA(nn.Module):
    """Context encoder + EMA target encoder + predictor (WI-17).

    Defaults are ViT-S-shaped, matching `MAE2D`'s encoder so the two arms
    differ in objective rather than capacity; dims are configurable so a CPU
    test can build a tiny one without changing any mechanics.
    """

    def __init__(
        self, in_channels: int, img_size: int = 128, patch_size: int = 8, embed_dim: int = 384,
        depth: int = 12, heads: int = 6, predictor_dim: int = 192, predictor_depth: int = 6,
        predictor_heads: int = 6, mask_config: MaskConfig | None = None,
        loss_kind: str = "smooth_l1",
    ):
        super().__init__()
        self.context_encoder = ViTEncoder(
            in_channels, img_size, patch_size, embed_dim, depth, heads
        )
        # A copy, not a second random init: the target must start where the
        # context encoder starts, or the first thousand steps regress toward an
        # unrelated random function.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = Predictor(
            embed_dim, self.context_encoder.grid, predictor_dim, predictor_depth, predictor_heads
        )
        # What the optimizer sees. The target encoder is excluded here rather
        # than relying on `requires_grad=False`, so weight decay can never
        # reach it — decaying an EMA target would shrink the regression target
        # toward zero, which is collapse by construction.
        self.trainable = nn.ModuleList([self.context_encoder, self.predictor])
        self.mask_config = mask_config or MaskConfig()
        self.loss_kind = loss_kind
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_size = img_size

    @property
    def dim(self) -> int:
        """Width of the representation `encode` returns, mirroring
        `ResNetBackbone.dim` and `MAE2D.dim` so probes treat all arms alike."""
        return self.context_encoder.embed_dim

    @property
    def grid(self) -> int:
        return self.context_encoder.grid

    def train(self, mode: bool = True):
        """Keep the target encoder in eval mode always.

        It is never trained, and any train-mode-only layer in it (dropout
        today is 0.0, but that is a default someone could change) would inject
        noise into the regression target rather than into the model being
        regularised.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """``theta_bar <- m * theta_bar + (1 - m) * theta``, params and buffers."""
        for target, online in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(), self.context_encoder.buffers(), strict=True
        ):
            target.copy_(online)

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor, encoder: str = "target") -> torch.Tensor:
        """``(B, C, H, W) -> (B, N, dim)`` — unmasked patch tokens, before pooling.

        `encode` is this followed by a mean. Split out for WI-19's segmentation
        probe, which reads the token grid `encode` collapses; keeping one
        implementation means the dense and pooled readings can never disagree
        about *which* encoder (target vs context) they came from.
        """
        net = self.target_encoder if encoder == "target" else self.context_encoder
        return net(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, encoder: str = "target") -> torch.Tensor:
        """``(B, C, H, W) -> (B, dim)`` — the probe-facing representation.

        Unmasked and mean-pooled, from the **target** encoder by default: that
        is the network I-JEPA evaluates and publishes. `encoder="context"`
        gives the online one, which is what the loss actually optimises and is
        worth logging alongside when diagnosing a divergence between them.
        """
        return self.encode_tokens(x, encoder=encoder).mean(dim=1)

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(predictions, targets)``, both ``(B, n_targets, Nt, D)``.

        `generator` is a **CPU** generator — masks are index tensors sampled on
        the host and moved to `x`'s device, so they cost nothing to sample and
        stay reproducible independently of the CUDA RNG the augmentor uses.
        """
        context_ids, target_ids = multiblock_masks(
            self.context_encoder.grid, self.context_encoder.grid,
            self.mask_config, generator=generator,
        )
        context_ids = context_ids.to(x.device)
        target_ids = [t.to(x.device) for t in target_ids]

        with torch.no_grad():
            tokens = self.target_encoder(x)
            # LayerNorm over the feature axis before the loss sees it (paper
            # §3). Without it the target's own scale drift is free loss
            # reduction, and the predictor learns to track that instead.
            tokens = F.layer_norm(tokens, (tokens.shape[-1],))
            targets = torch.stack([tokens.index_select(1, t) for t in target_ids], dim=1)

        context = self.context_encoder(x, keep_ids=context_ids)
        predictions = torch.stack(
            [self.predictor(context, context_ids, t) for t in target_ids], dim=1
        )
        return predictions, targets
