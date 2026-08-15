"""Spatial MAE baseline arm (PLAN.md §5.1, WI-14 — documented baseline only).

ViT-S/16 on 128x128 crops (64 tokens), 75% masking, same crops and eval
protocol as the primary VICReg arm. Included because the project's pivot to
joint-embedding rests on the claim that MAE underperforms VICReg on linear
probing — PLAN.md's WI-14 note is explicit that this should be *measured on
this data*, not cited from the literature.

Standard MAE mechanics (He et al. 2022): patchify -> random per-sample
masking (`random_masking`, the shuffle/unshuffle trick) -> encode only the
visible tokens -> insert learned mask tokens at the masked positions ->
decode -> MSE loss on the **masked patches only**.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_2d_pos_embed(grid_h: int, grid_w: int, embed_dim: int) -> torch.Tensor:
    """Fixed 2-D sinusoidal positional embedding, ``(grid_h*grid_w, embed_dim)``.

    Half the channels encode the row index, half the column index — the
    standard ViT/MAE convention, and simple enough not to need learned
    positions for a from-scratch model this size.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    dim_each = embed_dim // 2

    def _1d(pos: np.ndarray, dim: int) -> np.ndarray:
        omega = 1.0 / (10000 ** (np.arange(dim // 2) / (dim // 2)))
        out = np.outer(pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    pe = np.concatenate([_1d(rows.reshape(-1), dim_each), _1d(cols.reshape(-1), dim_each)], axis=1)
    return torch.from_numpy(pe).float()


class PatchEmbed(nn.Module):
    """Flat ``Linear(in_channels * patch^2, embed_dim)`` — no conv striding,
    for the same reason the old repo settled on it (no aliasing of anything
    near-Nyquist in the patch content, PLAN.md §1.2's ``patch_embed``).
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        n_h, n_w = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, n_h * n_w, c * p * p)
        return self.proj(x)  # (B, N, D)


def random_masking(
    x: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep a random ``1 - mask_ratio`` fraction of tokens per sample.

    Returns:
        ``(x_visible, mask, ids_restore)`` — ``x_visible`` is
        ``(B, n_keep, D)``; ``mask`` is ``(B, N)`` with 1 at masked
        positions, 0 at visible ones, in the *original* token order;
        ``ids_restore`` un-shuffles a shuffled-order tensor back to original
        order (the standard MAE shuffle/argsort trick).
    """
    b, n, d = x.shape
    n_keep = max(1, int(round(n * (1 - mask_ratio))))
    noise = torch.rand(b, n, device=x.device, generator=generator)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))

    mask = torch.ones(b, n, device=x.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_visible, mask, ids_restore


class MAE2D(nn.Module):
    """ViT-S/16-shaped MAE. Defaults match PLAN.md WI-14; dims are
    configurable so a CPU smoke test can use a far smaller model without
    changing any mechanics.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_tokens = self.grid * self.grid
        self.mask_ratio = mask_ratio
        # He et al.'s "normalized pixels" target, on by default because they
        # adopt it specifically to improve *linear probing* — the metric WI-14
        # is measured on. With raw targets the loss is dominated by whichever
        # patches happen to carry large amplitude, and on this zero-mean
        # beamformed data the model parks at the patch mean: measured 11.3 %
        # of masked-patch variance explained, loss flat from epoch 3, and
        # RankMe decaying 331 -> 105 while it sat there.
        self.norm_pix_loss = norm_pix_loss
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.register_buffer("pos_embed", sincos_2d_pos_embed(self.grid, self.grid, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, heads, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        decoder_pos_embed = sincos_2d_pos_embed(self.grid, self.grid, decoder_embed_dim)
        self.register_buffer("decoder_pos_embed", decoder_pos_embed)
        decoder_layer = nn.TransformerEncoderLayer(
            decoder_embed_dim,
            decoder_heads,
            dim_feedforward=decoder_embed_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_depth)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Ground truth for the loss: same tokenization as `PatchEmbed`, no projection."""
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, self.n_tokens, c * p * p)
        return x

    @property
    def dim(self) -> int:
        """Width of the representation `encode` returns, mirroring
        `ResNetBackbone.dim` so probes can treat the two arms alike."""
        return self.patch_embed.proj.out_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, embed_dim)`` representation for probing — **no masking**.

        This is what the linear probes read, and it is deliberately not
        `forward`: `forward` masks 75 % of the tokens because that is the
        pretext task, but a representation scored against another encoder must
        see the whole crop, exactly as the VICReg backbone does. Tokens are
        mean-pooled because this MAE has no CLS token; that is the standard
        alternative and is what `dim` reports the width of.
        """
        tokens = self.patch_embed(x) + self.pos_embed
        return self.encoder(tokens).mean(dim=1)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None):
        """Returns ``(pred, target, mask)`` — ``pred``/``target`` are
        ``(B, N, patch_dim)``; ``mask`` is ``(B, N)``, 1 where masked (i.e.
        where the loss applies).
        """
        tokens = self.patch_embed(x) + self.pos_embed
        visible, mask, ids_restore = random_masking(tokens, self.mask_ratio, generator=generator)
        encoded = self.encoder(visible)

        b, n_keep, _ = encoded.shape
        decoder_tokens = self.decoder_embed(encoded)
        n_mask = self.n_tokens - n_keep
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        full = torch.cat([decoder_tokens, mask_tokens], dim=1)
        full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        full = full + self.decoder_pos_embed
        decoded = self.decoder(full)
        pred = self.decoder_pred(decoded)

        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        return pred, target, mask


def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-patch MSE, averaged over masked patches only (PLAN.md WI-14 /
    the standard MAE convention — the model is never scored on patches it
    could see)."""
    per_patch = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
    return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)
