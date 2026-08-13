"""Augmentation policies A (anatomy-focused) and B (geometry-only), PLAN.md
§3.2b / WI-11.

**Axis convention.** Every function here takes explicit ``z_axis``/``x_axis``
(and, where relevant, a trailing sector axis) rather than assuming a fixed
array rank, because the two call sites use different layouts: the bootstrap
harness's plain ``(size, size)`` bmode crops (PLAN.md 4.6) have no channel or
sector axis, while the real sector-sum store is ``(2, Z, X, K)`` per frame
(PLAN.md §3.7). Hardcoding one layout would silently do the wrong thing on
the other.

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

import numpy as np

from . import transforms as tf

# ── Policy B: geometry-only ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PolicyBConfig:
    jitter_px: int = 4
    flip_prob: float = 0.5


def lateral_flip(
    crop: np.ndarray, rng: np.random.Generator, x_axis: int, prob: float = 0.5
) -> np.ndarray:
    """Flip along ``x_axis`` only, with probability ``prob``. Never touches z."""
    if rng.random() < prob:
        return np.flip(crop, axis=x_axis).copy()
    return crop


def spatial_jitter(
    crop: np.ndarray, rng: np.random.Generator, z_axis: int, x_axis: int, max_px: int
) -> np.ndarray:
    """Shift by up to ``max_px`` pixels in z and independently in x.

    Implemented as reflect-pad + shift + crop-back-to-size, so it needs no
    buffer beyond the crop itself. Reflect padding (not wrap) avoids stitching
    the far edge of the array onto the near edge, which would fabricate a
    discontinuity no real acquisition produces.
    """
    if max_px <= 0:
        return crop
    dz = int(rng.integers(-max_px, max_px + 1))
    dx = int(rng.integers(-max_px, max_px + 1))
    if dz == 0 and dx == 0:
        return crop

    pad = [(0, 0)] * crop.ndim
    pad[z_axis] = (max_px, max_px)
    pad[x_axis] = (max_px, max_px)
    padded = np.pad(crop, pad, mode="reflect")

    sl = [slice(None)] * crop.ndim
    sl[z_axis] = slice(max_px - dz, max_px - dz + crop.shape[z_axis])
    sl[x_axis] = slice(max_px - dx, max_px - dx + crop.shape[x_axis])
    return padded[tuple(sl)]


def policy_b(
    crop: np.ndarray,
    rng: np.random.Generator,
    *,
    z_axis: int,
    x_axis: int,
    config: PolicyBConfig | None = None,
) -> np.ndarray:
    """Random lateral flip + mild spatial jitter. Geometry-only: nothing here
    touches sector/aperture content, so z_centre and x_centre survive
    (PLAN.md 4.1's "Policy B preserves geometry" row)."""
    config = config or PolicyBConfig()
    out = lateral_flip(crop, rng, x_axis=x_axis, prob=config.flip_prob)
    out = spatial_jitter(out, rng, z_axis=z_axis, x_axis=x_axis, max_px=config.jitter_px)
    return out


# ── Policy A: anatomy-focused ────────────────────────────────────────────────


def random_sector_subset(
    sums: np.ndarray,
    occupancy: np.ndarray,
    rng: np.random.Generator,
    *,
    min_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out all but a random contiguous arc of >= ``min_span`` sectors.

    Two independent draws of this from the same crop are two different
    (overlapping or not) receive sub-apertures of the same scatterers — the
    primary Policy A positive-pair generator (PLAN.md 3.2b priority 1).
    Contiguous, not an arbitrary random subset, because sectors are angularly
    ordered and a non-contiguous "sub-aperture" is not a physical aperture.
    """
    k = sums.shape[-1]
    if not (0 < min_span <= k):
        raise ValueError(f"min_span must be in (0, {k}], got {min_span}")
    span = int(rng.integers(min_span, k + 1))
    start = int(rng.integers(0, k))
    keep = np.zeros(k, dtype=bool)
    keep[(start + np.arange(span)) % k] = True
    mask_sums = keep.reshape((1,) * (sums.ndim - 1) + (k,))
    mask_occ = keep.reshape((1,) * (occupancy.ndim - 1) + (k,))
    return sums * mask_sums, occupancy * mask_occ


def transmit_angle_pair(*_args, **_kwargs):
    """Unavailable — see the module docstring and PLAN.md §10.3(b)."""
    raise NotImplementedError(
        "transmit-angle subsetting has no data source: the PW(5) stack was not "
        "built (PLAN.md 10.3(b) judged the measured reciprocity residual "
        "non-physical and mostly redundant with the receive-sector axis). "
        "Build WI-4b first if this decision is revisited."
    )


def sector_dropout(sums: np.ndarray, rng: np.random.Generator, *, p: float) -> np.ndarray:
    """Independently zero each sector with probability ``p``; never all of them."""
    k = sums.shape[-1]
    keep = rng.random(k) >= p
    if not keep.any():
        keep[rng.integers(k)] = True
    mask = keep.reshape((1,) * (sums.ndim - 1) + (k,))
    return sums * mask


def tgc_gain_jitter(
    crop: np.ndarray, rng: np.random.Generator, *, z_axis: int, max_db: float, n_control: int = 4
) -> np.ndarray:
    """Multiply by a smooth random depth-dependent gain within +-``max_db`` dB.

    A smooth (few-control-point, linearly-interpolated) curve rather than
    per-depth-independent noise, since a real TGC error is a slowly-varying
    knob setting, not sample-to-sample jitter.
    """
    nz = crop.shape[z_axis]
    control = rng.uniform(-max_db, max_db, size=n_control)
    xp = np.linspace(0, nz - 1, n_control)
    gain_db = np.interp(np.arange(nz), xp, control)
    gain = (10.0 ** (gain_db / 20.0)).astype(crop.dtype)
    shape = [1] * crop.ndim
    shape[z_axis] = nz
    return crop * gain.reshape(shape)


def additive_complex_noise(
    crop: np.ndarray, rng: np.random.Generator, *, snr_db_range: tuple[float, float]
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to every real channel, at a randomly sampled
    SNR (relative to the crop's own RMS) drawn uniformly from ``snr_db_range``.

    Adding equal, independent noise to the I and Q channels directly (rather
    than to a magnitude/envelope) keeps the noise circularly symmetric in the
    complex plane, which is what real receiver noise looks like at this stage.
    """
    snr_db = rng.uniform(*snr_db_range)
    sig_rms = float(np.sqrt(np.mean(crop.astype(np.float64) ** 2)))
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(scale=noise_rms, size=crop.shape).astype(crop.dtype)
    return crop + noise


def random_k_reduction(
    sums: np.ndarray,
    occupancy: np.ndarray,
    rng: np.random.Generator,
    *,
    choices: tuple[int, ...],
    mode: str = "mean",
) -> tuple[np.ndarray, int]:
    """Reduce+normalize (PLAN.md WI-5's one correct order) to a randomly
    chosen K on the ladder."""
    k = int(rng.choice(choices))
    return tf.reduce_and_normalize(sums, occupancy, target_k=k, mode=mode), k


@dataclass(frozen=True)
class PolicyAConfig:
    sector_subset_min_span: int = 8
    dropout_p: float = 0.15
    tgc_max_db: float = 3.0
    snr_db_range: tuple[float, float] = (10.0, 30.0)
    k_choices: tuple[int, ...] = (16, 8, 4, 2, 1)
    normalize_mode: str = "mean"


def policy_a(
    sums16: np.ndarray,
    occupancy16: np.ndarray,
    rng: np.random.Generator,
    *,
    z_axis: int,
    config: PolicyAConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Full Policy A chain on raw K=16 sector sums: sub-aperture subset ->
    sector dropout -> TGC jitter -> additive noise -> random K-reduction.

    Order matters: everything that touches raw sums happens *before*
    :func:`random_k_reduction`, so the K-reduction step is always the last
    thing applied to raw data, per WI-5's reduce-then-normalize rule.
    """
    config = config or PolicyAConfig()
    s, occ = random_sector_subset(sums16, occupancy16, rng, min_span=config.sector_subset_min_span)
    s = sector_dropout(s, rng, p=config.dropout_p)
    s = tgc_gain_jitter(s, rng, z_axis=z_axis, max_db=config.tgc_max_db)
    s = additive_complex_noise(s, rng, snr_db_range=config.snr_db_range)
    return random_k_reduction(s, occ, rng, choices=config.k_choices, mode=config.normalize_mode)
