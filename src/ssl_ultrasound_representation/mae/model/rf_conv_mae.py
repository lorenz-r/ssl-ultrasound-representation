"""Anisotropic dilated conv autoencoder for masked RF/IQ reconstruction.

The transformer MAE in `us_mae.py` reconstructs the *envelope* well while
`complex_corr_coeff` and `spectral_coherence` sit near zero — it never models
the fast-time waveform. This module is the replacement architecture from
`masked_rf_autoencoder_spec.md` §3: MAE-style masked reconstruction without a
transformer, designed around the three things that kill phase.

    1. **Anisotropy.** Fast-time (axial) and lateral are not interchangeable
       axes, so every kernel is `(k_fast, k_lat) = (7, 3)` — tall and thin,
       never square.
    2. **Dilation instead of pooling.** Striding fast-time destroys the phase
       we are trying to predict, so the trunk grows its receptive field with
       exponentially increasing fast-time dilation (`1, 2, 4, 8, 16`, WaveNet
       style) at full axial resolution. `fast_downsample` defaults to 1;
       downsampling happens on the lateral axis only.
    3. **Receptive field ≥ mask length.** The middle of a masked run has to be
       reachable from unmasked signal on both sides, or the carrier inside the
       gap is unrecoverable and the model correctly regresses to the mean.
       :func:`assert_receptive_field` enforces this at construction time.

Skip connections are load-bearing here, not a stylistic choice: the bottleneck
carries envelope / low-frequency context, and the skips are what route
phase-resolved high-frequency detail around it to the output.

Layout
------
The model is a plain `(B, C, H_fast, W_lat)` conv net::

    input : (B, 3, H_fast, W_lat)   [I, Q, mask_flag]
    output: (B, 2, H_fast, W_lat)   [I, Q]

The third input channel is the binary mask flag; masked positions are expected
to already carry the learned mask token (see the masking module, spec §2). This
file only defines the network — masking, losses and the training loop live
elsewhere.

Dataset frames are 5-D, `(B, transducer, scanline, 2, sample)`, i.e. they have
*two* lateral axes. :func:`fold_frame` / :func:`unfold_frame` collapse one of
them into the batch. The default (`lateral="transducer"`) keeps the receive
aperture as `W_lat` and folds scanlines into the batch, because aperture is the
axis the spatial-coherence metrics (SLSC, coherence factor) are computed over.

Acquisition facts this is sized for (`configs/beamform_params.json`):
`fs = 7.8125 MHz` after the build script's ×4 decimation, `fc = 7.813 MHz`,
`H_fast = 215`, 192 elements, 192 scanlines, 3 focal zones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_CTX = ("mamba", "attn", "se", "none")
NORMS = ("group", "none")
LateralAxis = Literal["transducer", "scanline"]


# ── frame <-> model layout ──────────────────────────────────────────────────
def fold_frame(
    frame: torch.Tensor, lateral: LateralAxis = "transducer"
) -> torch.Tensor:
    """`(B, transducer, scanline, 2, sample)` -> `(B*, 2, H_fast, W_lat)`.

    A frame has two lateral axes but the network has one, so the axis that is
    *not* `lateral` is folded into the batch. `lateral="transducer"` (default)
    keeps the receive aperture in the convolution's field of view — the axis
    SLSC and the coherence factor are measured across — and treats each
    scanline as an independent sample.
    """
    if frame.ndim != 5:
        raise ValueError(
            f"expected (B, transducer, scanline, 2, sample); got {tuple(frame.shape)}"
        )
    b, n_t, n_s, iq, n_fast = frame.shape
    if iq != 2:
        raise ValueError(f"IQ axis must have length 2; got {iq}")
    if lateral == "transducer":
        return frame.permute(0, 2, 3, 4, 1).reshape(b * n_s, 2, n_fast, n_t)
    if lateral == "scanline":
        return frame.permute(0, 1, 3, 4, 2).reshape(b * n_t, 2, n_fast, n_s)
    raise ValueError(f"lateral must be 'transducer' or 'scanline'; got {lateral!r}")


def unfold_frame(
    x: torch.Tensor, batch_size: int, lateral: LateralAxis = "transducer"
) -> torch.Tensor:
    """Inverse of :func:`fold_frame` — back to `(B, transducer, scanline, 2, sample)`.

    Restores the layout `eval.metrics` and `utils.beamform` expect (`iq_axis=-2`,
    aperture first), so a reconstruction can be handed straight to
    `compute_all_metrics`.
    """
    folded, c, n_fast, w_lat = x.shape
    if folded % batch_size:
        raise ValueError(f"{folded} folded rows is not a multiple of batch {batch_size}")
    other = folded // batch_size
    y = x.reshape(batch_size, other, c, n_fast, w_lat)
    if lateral == "transducer":  # (B, S, C, N, T) -> (B, T, S, C, N)
        return y.permute(0, 4, 1, 2, 3)
    if lateral == "scanline":  # (B, T, C, N, S) -> (B, T, S, C, N)
        return y.permute(0, 1, 4, 2, 3)
    raise ValueError(f"lateral must be 'transducer' or 'scanline'; got {lateral!r}")


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class RFMaeConfig:
    """Architecture hyperparameters (spec §3).

    `depths` sets the number of residual blocks per stage and therefore the
    number of stages; channel width doubles per stage from `c0`. `dilations` is
    a *global* cycle: blocks are numbered across the whole trunk and block `k`
    takes `dilations[k % len(dilations)]`, so the exponential schedule is
    actually traversed rather than restarting at 1 in every stage.

    `l_max` / `rf_margin` are the masking module's largest fast-time run and the
    safety margin on top of it; they exist here so the receptive-field check can
    run at construction time.
    """

    in_channels: int = 3  # I, Q, mask flag
    out_channels: int = 2  # I, Q
    c0: int = 32
    depths: tuple[int, ...] = (2, 2, 2)
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    k_fast: int = 7
    k_lat: int = 3
    lat_downsample: int = 2
    fast_downsample: int = 1  # keep at 1 to protect phase; 2 is the allowed maximum
    global_ctx: str = "attn"  # "mamba" | "attn" | "se" | "none"
    attn_heads: int = 4
    attn_max_len: int = 512
    se_reduction: int = 8
    norm: str = "group"
    norm_groups: int = 8
    l_max: int = 32  # largest fast-time mask run, in samples
    rf_margin: int = 8

    def __post_init__(self) -> None:
        if self.k_fast % 2 == 0 or self.k_lat % 2 == 0:
            raise ValueError(
                f"kernels must be odd for symmetric padding; got "
                f"k_fast={self.k_fast}, k_lat={self.k_lat}"
            )
        if not self.depths:
            raise ValueError("depths must contain at least one stage")
        if not self.dilations:
            raise ValueError("dilations must contain at least one value")
        if self.fast_downsample not in (1, 2):
            raise ValueError(
                "fast_downsample must be 1 (preferred) or 2 — heavier axial "
                f"striding destroys the carrier phase; got {self.fast_downsample}"
            )
        if self.lat_downsample < 1:
            raise ValueError(f"lat_downsample must be >= 1; got {self.lat_downsample}")
        if self.global_ctx not in GLOBAL_CTX:
            raise ValueError(f"global_ctx must be one of {GLOBAL_CTX}; got {self.global_ctx!r}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}; got {self.norm!r}")

    @property
    def widths(self) -> list[int]:
        return [self.c0 * 2**i for i in range(len(self.depths))]


# ── encoder plan (single source of truth for build + receptive field) ───────
@dataclass(frozen=True)
class LayerSpec:
    """One encoder layer, as both the build recipe and the RF calculator read it."""

    kind: str  # "stem" | "block" | "down"
    stage: int
    c_in: int
    c_out: int
    dilation: int
    stride_fast: int
    stride_lat: int
    n_convs: int  # convs on the fast-time path (the first one carries the stride)


def encoder_plan(cfg: RFMaeConfig) -> list[LayerSpec]:
    """Enumerate the encoder. Both `RFConvMAE.__init__` and
    :func:`fast_time_receptive_field` consume this, so the asserted receptive
    field can never drift from the network that actually gets built.
    """
    widths = cfg.widths
    plan = [LayerSpec("stem", 0, cfg.in_channels, widths[0], 1, 1, 1, 2)]
    block_idx = 0
    for stage, (depth, width) in enumerate(zip(cfg.depths, widths)):
        if stage > 0:
            plan.append(
                LayerSpec(
                    "down",
                    stage,
                    widths[stage - 1],
                    width,
                    dilation=1,
                    # The one permitted gentle axial stride is spent at the first
                    # transition only; every later transition is lateral-only.
                    stride_fast=cfg.fast_downsample if stage == 1 else 1,
                    stride_lat=cfg.lat_downsample,
                    n_convs=1,
                )
            )
        for _ in range(depth):
            plan.append(
                LayerSpec(
                    "block",
                    stage,
                    width,
                    width,
                    cfg.dilations[block_idx % len(cfg.dilations)],
                    1,
                    1,
                    2,
                )
            )
            block_idx += 1
    return plan


def fast_time_receptive_field(cfg: RFMaeConfig) -> int:
    """Fast-time receptive field of the encoder, in samples.

    Standard accumulation: a conv with kernel `k` and dilation `d` adds
    `(k-1)*d*jump`, and a stride `s` multiplies `jump` for everything after it.
    Only the fast-time axis is tracked — lateral striding does not change how
    far along a trace the bottleneck can see.
    """
    rf, jump = 1, 1
    for spec in encoder_plan(cfg):
        for i in range(spec.n_convs):
            rf += (cfg.k_fast - 1) * spec.dilation * jump
            if i == 0:
                jump *= spec.stride_fast
    return rf


def assert_receptive_field(
    cfg: RFMaeConfig, l_max: int | None = None, margin: int | None = None
) -> int:
    """Spec §3 build-time assertion. Returns the receptive field on success.

    If the trunk cannot span the largest fast-time mask, the samples in the
    middle of a gap are not a function of any observed input — the carrier there
    is literally unrecoverable and the model will regress to the conditional
    mean, which is exactly the near-zero `complex_corr_coeff` failure this
    architecture exists to fix.
    """
    l_max = cfg.l_max if l_max is None else l_max
    margin = cfg.rf_margin if margin is None else margin
    rf = fast_time_receptive_field(cfg)
    if rf < l_max + margin:
        raise ValueError(
            f"fast-time receptive field {rf} < L_max + margin ({l_max} + {margin}) "
            "-> carrier inside the largest mask is unrecoverable. Add dilations, "
            "increase k_fast, or add blocks."
        )
    return rf


# ── building blocks ─────────────────────────────────────────────────────────
def _norm(kind: str, channels: int, groups: int) -> nn.Module:
    """GroupNorm by default — batch statistics are unreliable at batch_size=2
    on ~140 frames, and BatchNorm's train/eval mismatch shows up directly in the
    reconstruction.
    """
    if kind == "none":
        return nn.Identity()
    return nn.GroupNorm(math.gcd(groups, channels), channels)


class AnisoDilatedBlock(nn.Module):
    """Residual block of two anisotropic convs, dilated on fast-time only.

    `kernel=(k_fast, k_lat)`, `dilation=(d, 1)`: the receptive field grows down
    the trace while the lateral footprint stays at `k_lat`. Padding keeps both
    spatial sizes exactly, so masked positions never shift.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        k_fast: int = 7,
        k_lat: int = 3,
        dilation: int = 1,
        norm: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        pad = ((k_fast - 1) * dilation // 2, (k_lat - 1) // 2)
        kernel = (k_fast, k_lat)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel, padding=pad, dilation=(dilation, 1))
        self.norm1 = _norm(norm, c_out, norm_groups)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel, padding=pad, dilation=(dilation, 1))
        self.norm2 = _norm(norm, c_out, norm_groups)
        self.act = nn.GELU()
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class AnisoDownsample(nn.Module):
    """Strided anisotropic conv between stages (lateral by default)."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        k_fast: int,
        k_lat: int,
        stride_fast: int,
        stride_lat: int,
        norm: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in,
            c_out,
            (k_fast, k_lat),
            stride=(stride_fast, stride_lat),
            padding=((k_fast - 1) // 2, (k_lat - 1) // 2),
        )
        self.norm = _norm(norm, c_out, norm_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AnisoUpsample(nn.Module):
    """Nearest-neighbour resize to an explicit size, then an anisotropic conv.

    Deliberately not `ConvTranspose2d`: transposed convolution lays down a
    periodic checkerboard, and a periodic artefact along fast-time is precisely
    a spurious carrier. Resizing to the recorded encoder size also makes odd
    extents (`H_fast = 215`) exact without output-padding arithmetic.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        k_fast: int,
        k_lat: int,
        norm: str = "group",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, (k_fast, k_lat), padding=((k_fast - 1) // 2, (k_lat - 1) // 2)
        )
        self.norm = _norm(norm, c_out, norm_groups)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="nearest")
        return self.act(self.norm(self.conv(x)))


# ── bottleneck global context ───────────────────────────────────────────────
class SqueezeExcite(nn.Module):
    """Cheapest global context: a global summary gating the channels."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(s))))


class FastTimeAttention(nn.Module):
    """One pre-norm self-attention layer along fast-time at the bottleneck.

    Lateral positions are folded into the batch, so the sequence is `H_fast`
    long (215 here, or 108 with `fast_downsample=2`) — short enough that full
    attention is cheap, and it gives every axial position a global view of the
    trace, which stacked convs only approximate.
    """

    def __init__(self, channels: int, n_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.pos = nn.Parameter(torch.zeros(1, max_len, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h > self.pos.shape[1]:
            raise ValueError(
                f"bottleneck fast-time length {h} exceeds attn_max_len {self.pos.shape[1]}"
            )
        seq = x.permute(0, 3, 2, 1).reshape(b * w, h, c)  # (B*W, H, C)
        q = self.norm(seq) + self.pos[:, :h]
        a, _ = self.attn(q, q, q, need_weights=False)
        out = seq + a
        return out.reshape(b, w, h, c).permute(0, 3, 2, 1)


class FastTimeMamba(nn.Module):
    """Selective-SSM layer along fast-time — global context at linear cost.

    Preferred over attention in the spec, but `mamba-ssm` ships fused CUDA
    kernels that need a recent toolchain; this project is pinned to
    torch 2.4.1+cu11x for Pascal support, so the import may well fail. The error
    points at `global_ctx="attn"`, which is the same idea at quadratic (but here
    negligible) cost.
    """

    def __init__(self, channels: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "global_ctx='mamba' needs the `mamba-ssm` package (not a project "
                "dependency, and its CUDA kernels may not build against the "
                "pinned torch 2.4.1+cu11x). Use global_ctx='attn' instead."
            ) from exc
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        seq = seq + self.mamba(self.norm(seq))
        return seq.reshape(b, w, h, c).permute(0, 3, 2, 1)


def build_global_ctx(cfg: RFMaeConfig, channels: int) -> nn.Module:
    if cfg.global_ctx == "none":
        return nn.Identity()
    if cfg.global_ctx == "se":
        return SqueezeExcite(channels, cfg.se_reduction)
    if cfg.global_ctx == "attn":
        return FastTimeAttention(channels, cfg.attn_heads, cfg.attn_max_len)
    return FastTimeMamba(channels)


# ── model ───────────────────────────────────────────────────────────────────
class RFConvMAE(nn.Module):
    """U-Net-shaped anisotropic dilated autoencoder over `(B, 3, H_fast, W_lat)`.

    Encoder: stem -> dilated blocks -> lateral downsample -> ... -> bottleneck,
    with the bottleneck's global-context module selected by `cfg.global_ctx`.
    Decoder mirrors it, concatenating the encoder feature map at each scale
    before its blocks, and a final 1x1 conv emits `(I, Q)`.

    Masking is not performed here — the input is expected to already carry the
    learned mask token in the masked positions and the binary flag in channel 2.
    """

    def __init__(self, cfg: RFMaeConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or RFMaeConfig()
        self.receptive_field = assert_receptive_field(cfg)

        widths = cfg.widths
        block_kw = dict(
            k_fast=cfg.k_fast, k_lat=cfg.k_lat, norm=cfg.norm, norm_groups=cfg.norm_groups
        )

        # --- encoder, built straight off the plan the RF calculator reads ---
        plan = encoder_plan(cfg)
        self.stem = AnisoDilatedBlock(
            plan[0].c_in, plan[0].c_out, dilation=plan[0].dilation, **block_kw
        )
        stages: list[nn.Sequential] = [nn.Sequential() for _ in cfg.depths]
        downs: dict[int, nn.Module] = {}
        for spec in plan[1:]:
            if spec.kind == "down":
                downs[spec.stage] = AnisoDownsample(
                    spec.c_in,
                    spec.c_out,
                    k_fast=cfg.k_fast,
                    k_lat=cfg.k_lat,
                    stride_fast=spec.stride_fast,
                    stride_lat=spec.stride_lat,
                    norm=cfg.norm,
                    norm_groups=cfg.norm_groups,
                )
            else:
                stages[spec.stage].append(
                    AnisoDilatedBlock(
                        spec.c_in, spec.c_out, dilation=spec.dilation, **block_kw
                    )
                )
        self.enc_stages = nn.ModuleList(stages)
        self.downs = nn.ModuleList([downs[i] for i in range(1, len(cfg.depths))])

        self.global_ctx = build_global_ctx(cfg, widths[-1])

        # --- decoder: one step per encoder transition, deepest first ---
        n_up = len(cfg.depths) - 1
        # Continue the dilation cycle rather than restarting, so the decoder does
        # not re-run the small-dilation blocks the encoder already applied.
        block_idx = sum(cfg.depths)
        ups, fuses, dec_stages = [], [], []
        for i in reversed(range(n_up)):
            ups.append(
                AnisoUpsample(
                    widths[i + 1],
                    widths[i],
                    k_fast=cfg.k_fast,
                    k_lat=cfg.k_lat,
                    norm=cfg.norm,
                    norm_groups=cfg.norm_groups,
                )
            )
            fuses.append(nn.Conv2d(2 * widths[i], widths[i], 1))
            blocks = nn.Sequential()
            for _ in range(cfg.depths[i]):
                blocks.append(
                    AnisoDilatedBlock(
                        widths[i],
                        widths[i],
                        dilation=cfg.dilations[block_idx % len(cfg.dilations)],
                        **block_kw,
                    )
                )
                block_idx += 1
            dec_stages.append(blocks)
        self.ups = nn.ModuleList(ups)
        self.fuses = nn.ModuleList(fuses)
        self.dec_stages = nn.ModuleList(dec_stages)

        self.head = nn.Conv2d(widths[0], cfg.out_channels, 1)

    # -- forward ------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder. Returns the bottleneck and the per-scale skips."""
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"expected {self.cfg.in_channels} input channels (I, Q, mask_flag); "
                f"got {x.shape[1]}"
            )
        h = self.stem(x)
        skips: list[torch.Tensor] = []
        for i, stage in enumerate(self.enc_stages):
            if i > 0:
                h = self.downs[i - 1](h)
            h = stage(h)
            if i < len(self.enc_stages) - 1:
                skips.append(h)
        return self.global_ctx(h), skips

    def decode(self, h: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        for up, fuse, blocks, skip in zip(
            self.ups, self.fuses, self.dec_stages, reversed(skips)
        ):
            h = up(h, skip.shape[-2:])
            h = fuse(torch.cat([h, skip], dim=1))
            h = blocks(h)
        return self.head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H_fast, W_lat)` -> `(B, 2, H_fast, W_lat)` reconstruction."""
        h, skips = self.encode(x)
        return self.decode(h, skips)

    @torch.no_grad()
    def latent(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Bottleneck representation, for downstream probing (`eval/latent.py`).

        `pool="mean"` -> `(B, C)`; `pool="none"` -> the raw `(B, C, H, W)` map.
        """
        h, _ = self.encode(x)
        if pool == "mean":
            return h.mean(dim=(2, 3))
        if pool == "none":
            return h
        raise ValueError(f"unknown pool: {pool!r} (use 'mean' or 'none')")

    # -- introspection ------------------------------------------------------
    def describe(self) -> dict:
        """Build summary worth logging next to the run config."""
        return {
            "params": sum(p.numel() for p in self.parameters()),
            "fast_time_receptive_field": self.receptive_field,
            "l_max": self.cfg.l_max,
            "widths": self.cfg.widths,
            "global_ctx": self.cfg.global_ctx,
            "fast_downsample": self.cfg.fast_downsample,
            "lat_downsample": self.cfg.lat_downsample,
        }
