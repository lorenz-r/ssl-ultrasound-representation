"""Augmentation policies A (anatomy-focused) and B (geometry-only), PLAN.md
§3.2b / WI-11 — **batched, on whatever device the tensors live on**.

This is torch rather than NumPy, and batched rather than per-crop, for a
measured reason. With the loader doing augmentation on the CPU, profiling on
the cluster's 4-core instances put ~59% of loader time in drawing Gaussian
noise and ~20% in the K-ladder reduction, capping throughput at ~157 crop
pairs/s against the ~650 PLAN.md §5.4's training budget needs. Every operation
here is elementwise or a grouped reduction — exactly what the (otherwise idle)
GPU does well — so the policies moved to the GPU and the loader was left doing
only decompress-and-slice. There is deliberately **one** implementation: a
NumPy copy kept "for tests" would be a second definition of the semantics,
free to drift from the one that actually trains the model.

Everything runs on CPU tensors too (torch is a core dependency and needs no
GPU for that), so tests and figures do not require a GPU.

**Axis convention.** Fixed and canonical, unlike the earlier per-call
``z_axis``/``x_axis`` arguments: raw sector sums are ``(B, 2, Z, X, K)`` and
occupancy is ``(B, Z, X, K)``, matching the store's own ``(2, Z, X, K)`` frame
layout (PLAN.md §3.7) with a batch axis in front. The old explicit-axis
signatures existed to serve a second, rank-3 call site; that site is gone, and
a fixed layout removes a class of silent transposition bugs.

**Randomness is per sample, not per batch.** Every draw below — arc span,
dropout mask, TGC curve, SNR, K level — is independent for each item in the
batch. A per-batch draw would correlate the augmentation across a batch, and
VICReg's variance/covariance terms are computed *across* the batch, so that
correlation would land directly in the loss.

**Never axial flip** (Policy B): depth is physically asymmetric here —
attenuation, the transmit f-number geometry, and per-depth aperture coverage
all vary with z — so flipping z teaches a false invariance. Only a lateral
(x-axis) flip is offered; there is no axial-flip function to call by mistake.

**Policy A priority order** (PLAN.md §3.2b, §11 WI-11): (1) receive
angular-sector subsetting — primary, always available; (2) transmit-angle
subsetting — secondary, from the PW(5) stack; (3) sector dropout, TGC/gain
jitter, additive complex noise; (4) depth-invariance augmentations — ablation
only, never default (kept out of this module entirely; PLAN.md §4.5c).

**Transmit-angle subsetting is unavailable.** The PW(5) stack was not built:
§10.3(b) measured the reciprocity residual and judged it non-physical/mostly
redundant with the wider receive-sector span, so the build stayed at
FSA×K=16 alone (27.3 GB, not 61.5 GB). :func:`transmit_angle_pair` exists as
the documented seam and raises unconditionally until that decision is
revisited.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .transforms import reduce_and_normalize_torch

# ── Policy B: geometry-only ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyBConfig:
    jitter_px: int = 4
    flip_prob: float = 0.5


def _rand(shape, generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device)


def _randint(low: int, high: int, shape, generator, device) -> torch.Tensor:
    """Uniform integers in ``[low, high)`` — torch's own semantics."""
    return torch.randint(low, high, shape, generator=generator, device=device)


def lateral_flip(
    crop: torch.Tensor, generator: torch.Generator | None = None, prob: float = 0.5
) -> torch.Tensor:
    """Flip along x only, per sample, with probability ``prob``. Never touches z.

    Args:
        crop: ``(B, C, Z, X, ...)`` — x is axis 3.
    """
    flip = _rand((crop.shape[0],), generator, crop.device) < prob
    flipped = torch.flip(crop, dims=(3,))
    return torch.where(flip.view(-1, *([1] * (crop.ndim - 1))), flipped, crop)


def spatial_jitter(
    crop: torch.Tensor, generator: torch.Generator | None = None, max_px: int = 4
) -> torch.Tensor:
    """Shift by up to ``max_px`` px in z and independently in x, per sample.

    Reflect-padded then sliced, so no buffer beyond the crop is needed and the
    far edge is never stitched onto the near edge — wrap-around padding would
    fabricate a discontinuity no real acquisition produces.

    The per-sample shift is applied with a gather rather than a Python loop
    over the batch, so this stays one kernel regardless of batch size.
    """
    if max_px <= 0:
        return crop
    b, _, z, x = crop.shape[:4]
    dev = crop.device
    # Reflect padding cannot exceed the extent it reflects about, so the shift
    # is clamped per axis. For real 128 px crops with max_px=4 this is a no-op;
    # it only bites on degenerate axes, where no shift is meaningful anyway.
    pz, px = min(max_px, z - 1), min(max_px, x - 1)
    dz = _randint(-pz, pz + 1, (b,), generator, dev)
    dx = _randint(-px, px + 1, (b,), generator, dev)

    # Reflect padding needs a 4-D (N,C,H,W) view; fold the trailing sector axis
    # (if any) into channels, pad, then unfold.
    flat = crop.reshape(b, -1, z, x)
    padded = torch.nn.functional.pad(flat, (px, px, pz, pz), mode="reflect")

    zi = (torch.arange(z, device=dev).view(1, z) + pz - dz.view(b, 1))  # (B, Z)
    xi = (torch.arange(x, device=dev).view(1, x) + px - dx.view(b, 1))  # (B, X)
    out = padded[torch.arange(b, device=dev).view(b, 1, 1), :, zi.unsqueeze(-1), xi.unsqueeze(1)]
    # Advanced indexing above yields (B, Z, X, C); restore the original layout.
    return out.permute(0, 3, 1, 2).reshape(crop.shape)


def policy_b(
    crop: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: PolicyBConfig | None = None,
) -> torch.Tensor:
    """Random lateral flip + mild spatial jitter. Geometry-only: nothing here
    touches sector/aperture content, so z_centre and x_centre survive
    (PLAN.md 4.1's "Policy B preserves geometry" row)."""
    config = config or PolicyBConfig()
    out = lateral_flip(crop, generator, prob=config.flip_prob)
    return spatial_jitter(out, generator, max_px=config.jitter_px)


# ── Policy A: anatomy-focused ────────────────────────────────────────────────


def random_sector_subset(
    sums: torch.Tensor,
    occupancy: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    min_span: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero all but a random contiguous arc of >= ``min_span`` sectors, per sample.

    Two independent draws of this from the same crop are two different
    (overlapping or not) receive sub-apertures of the same scatterers — the
    primary Policy A positive-pair generator (PLAN.md 3.2b priority 1).
    Contiguous, not an arbitrary random subset, because sectors are angularly
    ordered and a non-contiguous "sub-aperture" is not a physical aperture.
    """
    b, k = sums.shape[0], sums.shape[-1]
    if not (0 < min_span <= k):
        raise ValueError(f"min_span must be in (0, {k}], got {min_span}")
    dev = sums.device
    span = _randint(min_span, k + 1, (b, 1), generator, dev)
    start = _randint(0, k, (b, 1), generator, dev)
    # offset of each sector from the arc start, modulo k -> inside iff < span
    offset = (torch.arange(k, device=dev).view(1, k) - start) % k
    keep = (offset < span).to(sums.dtype)  # (B, K)
    return sums * keep.view(b, 1, 1, 1, k), occupancy * keep.view(b, 1, 1, k)


def transmit_angle_pair(*_args, **_kwargs):
    """Unavailable — see the module docstring and PLAN.md §10.3(b)."""
    raise NotImplementedError(
        "transmit-angle subsetting has no data source: the PW(5) stack was not "
        "built (PLAN.md 10.3(b) judged the measured reciprocity residual "
        "non-physical and mostly redundant with the receive-sector axis). "
        "Build WI-4b first if this decision is revisited."
    )


def sector_dropout(
    sums: torch.Tensor, generator: torch.Generator | None = None, *, p: float
) -> torch.Tensor:
    """Independently zero each sector with probability ``p``; never all of them."""
    b, k = sums.shape[0], sums.shape[-1]
    keep = _rand((b, k), generator, sums.device) >= p
    # Any sample that lost every sector keeps one, chosen uniformly.
    dead = ~keep.any(dim=1)
    if bool(dead.any()):
        rescue = _randint(0, k, (b,), generator, sums.device)
        keep[dead, rescue[dead]] = True
    return sums * keep.to(sums.dtype).view(b, 1, 1, 1, k)


def tgc_gain_jitter(
    crop: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    max_db: float,
    n_control: int = 4,
) -> torch.Tensor:
    """Multiply by a smooth random depth-dependent gain within ±``max_db`` dB.

    A smooth (few-control-point, linearly-interpolated) curve rather than
    per-depth-independent noise, since a real TGC error is a slowly-varying
    knob setting, not sample-to-sample jitter.
    """
    b, nz = crop.shape[0], crop.shape[2]
    dev = crop.device
    control = (_rand((b, 1, n_control), generator, dev) * 2 - 1) * max_db
    # Linear interpolation between control points == 1-D resize.
    gain_db = torch.nn.functional.interpolate(
        control, size=nz, mode="linear", align_corners=True
    ).view(b, 1, nz, 1, 1)
    return crop * torch.pow(10.0, gain_db / 20.0).to(crop.dtype)


def additive_complex_noise(
    crop: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    snr_db_range: tuple[float, float],
) -> torch.Tensor:
    """Add i.i.d. Gaussian noise to every real channel, at a per-sample SNR
    (relative to that sample's own RMS) drawn uniformly from ``snr_db_range``.

    Adding equal, independent noise to the I and Q channels directly (rather
    than to a magnitude/envelope) keeps the noise circularly symmetric in the
    complex plane, which is what real receiver noise looks like at this stage.
    """
    b = crop.shape[0]
    lo, hi = snr_db_range
    snr_db = _rand((b,), generator, crop.device) * (hi - lo) + lo
    sig_rms = crop.flatten(1).pow(2).mean(dim=1).sqrt()
    noise_rms = (sig_rms / torch.pow(10.0, snr_db / 20.0)).view(b, *([1] * (crop.ndim - 1)))
    noise = torch.randn(crop.shape, generator=generator, device=crop.device, dtype=crop.dtype)
    return crop + noise * noise_rms


def random_k_reduction(
    sums: torch.Tensor,
    occupancy: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    choices: tuple[int, ...],
    mode: str = "mean",
    expand_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce+normalize (PLAN.md WI-5's one correct order) to a per-sample K.

    Each sample draws its own level from ``choices``; samples sharing a level
    are processed together, so this costs one pass per *distinct* level in the
    batch (at most 5) rather than one per sample.

    ``expand_to`` repeats surviving sectors back up to a fixed width so the
    batch can be stacked for an encoder whose input is a fixed ``2K`` channels
    — see :func:`ssl_us.data.dataset.match_sector_channels` for why repeating
    beats zero-padding.

    Returns ``(views, k_per_sample)``.
    """
    b = sums.shape[0]
    dev = sums.device
    if expand_to is not None:
        # An arm of width K cannot be handed a view with *more* than K sectors,
        # so levels coarser than the arm are simply not on its ladder. Filtering
        # here rather than at each call site keeps the arm's K the single source
        # of truth for what the augmentation may draw.
        choices = tuple(c for c in choices if c <= expand_to and expand_to % c == 0)
        if not choices:
            raise ValueError(f"no K level divides expand_to={expand_to}")
    picks = torch.tensor(choices, device=dev)[
        _randint(0, len(choices), (b,), generator, dev)
    ]
    width = expand_to if expand_to is not None else int(picks.min())
    out = torch.empty(
        (*sums.shape[:-1], width), dtype=sums.dtype, device=dev
    )
    for k in picks.unique().tolist():
        sel = picks == k
        view = reduce_and_normalize_torch(sums[sel], occupancy[sel], target_k=int(k), mode=mode)
        if view.shape[-1] != width:
            if width % view.shape[-1]:
                raise ValueError(f"cannot expand k={view.shape[-1]} to {width}")
            view = view.repeat_interleave(width // view.shape[-1], dim=-1)
        out[sel] = view
    return out, picks


@dataclass(frozen=True)
class PolicyAConfig:
    sector_subset_min_span: int = 8
    dropout_p: float = 0.15
    tgc_max_db: float = 3.0
    snr_db_range: tuple[float, float] = (10.0, 30.0)
    k_choices: tuple[int, ...] = (16, 8, 4, 2, 1)
    normalize_mode: str = "mean"


def policy_a(
    sums16: torch.Tensor,
    occupancy16: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    config: PolicyAConfig | None = None,
    expand_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full Policy A chain on raw K=16 sector sums: sub-aperture subset ->
    sector dropout -> TGC jitter -> additive noise -> random K-reduction.

    Order matters: everything that touches raw sums happens *before*
    :func:`random_k_reduction`, so the K-reduction step is always the last
    thing applied to raw data, per WI-5's reduce-then-normalize rule.

    Args:
        sums16: ``(B, 2, Z, X, 16)`` raw sector sums, straight from the store.
        occupancy16: ``(B, Z, X, 16)`` matching element counts.
        expand_to: fixed output sector width (the arm's K).
    """
    config = config or PolicyAConfig()
    s, occ = random_sector_subset(
        sums16, occupancy16, generator, min_span=config.sector_subset_min_span
    )
    s = sector_dropout(s, generator, p=config.dropout_p)
    s = tgc_gain_jitter(s, generator, max_db=config.tgc_max_db)
    s = additive_complex_noise(s, generator, snr_db_range=config.snr_db_range)
    return random_k_reduction(
        s, occ, generator, choices=config.k_choices, mode=config.normalize_mode,
        expand_to=expand_to,
    )
