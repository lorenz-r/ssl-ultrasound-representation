"""Speckle-invariant image-quality metrics for beamformer validation (WI-4).

**Why this module exists.** WI-4's original done-condition compared our
reconstruction against NVIDIA's shipped ``bmode`` with a pixel-wise
normalised cross-correlation, gated at NCC >= 0.98. That gate is not
reachable — not by us, and not by NVIDIA. Measured on the real corpus
(``results/gates/nvidia_reference_selfconsistency.json``): NVIDIA's *own* two
shipped reconstructions of the *same frame*, ``bmode`` and ``bmode_focused``,
score NCC ~0.19 against each other, while our reconstruction scores ~0.26
against ``bmode`` — i.e. we already match the reference better than the
reference's own sibling does. Two independent beamformers (our
``AngularSectorSum`` and ``zea``'s stock ``DelayAndSum``, which is what
OpenH-RF's reference ``reconstruct.py`` uses) agree with each other to within
0.001 NCC, and beamforming with the ground-truth sound-speed map makes the
score *worse* (0.257 -> 0.134), confirming ``bmode`` really is the constant-c
DAS its converter documents.

The reason is physical, not a bug. B-mode speckle is fully-developed: it is a
random interference pattern, not a picture of scatterers. Two reconstructions
that differ by any sub-wavelength detail — interpolation kernel, apodisation,
a fraction of an RF cycle in the delay model — reshuffle which speckle grains
land bright vs dark while leaving every macroscopic feature (lesion position,
contrast, shadowing, resolution) identical. Pixel-wise correlation therefore
measures *speckle realisation agreement*, which is not a property a correct
beamformer is required to reproduce, and it saturates far below 1.0 between
any two legitimate reconstructions.

So the metrics here deliberately do **not** compare speckle realisations.
They measure the things a beamformer must get right and that survive a
different speckle draw:

- :func:`gcnr` — generalised contrast-to-noise ratio (Rodriguez-Molares et
  al. 2020), the overlap between the lesion and background *intensity
  distributions*. Bounded in [0, 1], invariant to any monotonic transform of
  the image (so dB vs linear, and any gain/TGC, do not move it) and to the
  speckle realisation. This is the primary metric: it needs the shipped
  ``segmentation_map`` and answers "is the inclusion actually resolved?".
- :func:`cnr` — classical contrast-to-noise ratio, in dB. Reported alongside
  gCNR because it is the number most ultrasound papers quote, but it is *not*
  gated: it is unbounded and moves under log-compression.
- :func:`speckle_snr` — mean/std of the *envelope* in a background region.
  For fully-developed speckle the envelope is Rayleigh-distributed and this
  ratio is a known constant, ~1.913, independent of everything else. It is
  the sharpest available check that the reconstruction produces real speckle
  statistics rather than a smoothed or aliased image.
- :func:`structural_ncc` — correlation of *locally averaged* log-envelopes.
  Averaging over a window several speckle grains wide removes the speckle
  realisation and leaves the macroscopic structure, so this does compare
  against the reference, but at the scale where agreement is actually
  meaningful.

All functions take plain NumPy arrays and are backend-agnostic; nothing here
imports keras or zea.
"""

from __future__ import annotations

import numpy as np

# Envelope mean/std for fully-developed (Rayleigh) speckle:
#   mean = sigma*sqrt(pi/2), std = sigma*sqrt(2 - pi/2)  =>  ratio is constant.
RAYLEIGH_SNR = float(np.sqrt(np.pi / 2) / np.sqrt(2 - np.pi / 2))  # ~1.9130


def gcnr(lesion: np.ndarray, background: np.ndarray, bins: int = 256) -> float:
    """Generalised contrast-to-noise ratio between two pixel populations.

    ``gCNR = 1 - OVL``, where ``OVL`` is the overlap area of the two
    normalised intensity histograms. 0 means the distributions are
    indistinguishable (lesion invisible), 1 means they are perfectly separable.

    Invariant to any strictly monotonic transform of the pixel values, so it
    may be passed linear envelope or log-compressed dB interchangeably, and
    invariant to the speckle realisation — which is exactly why it is usable
    as a validation gate where pixel-wise correlation is not.

    Args:
        lesion: Pixel values inside the lesion/inclusion region (any shape).
        background: Pixel values in the background region (any shape).
        bins: Histogram bins used to estimate the overlap.

    Returns:
        gCNR in [0, 1].
    """
    a = np.asarray(lesion, dtype=np.float64).ravel()
    b = np.asarray(background, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    edges = np.linspace(lo, hi, bins + 1)
    # density=False then normalise by count, so each histogram sums to 1 and
    # the overlap is a probability mass regardless of differing region sizes.
    ha = np.histogram(a, bins=edges)[0].astype(np.float64) / a.size
    hb = np.histogram(b, bins=edges)[0].astype(np.float64) / b.size
    return float(1.0 - np.minimum(ha, hb).sum())


def cnr(lesion: np.ndarray, background: np.ndarray) -> float:
    """Classical contrast-to-noise ratio, in dB.

    ``20 log10(|mu_l - mu_b| / sqrt(var_l + var_b))``. Reported for
    comparability with the literature; unlike :func:`gcnr` it is unbounded and
    changes under log-compression, so it is not used as a pass/fail gate.
    """
    a = np.asarray(lesion, dtype=np.float64).ravel()
    b = np.asarray(background, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.sqrt(a.var() + b.var())
    if denom <= 0:
        return float("nan")
    return float(20.0 * np.log10(np.abs(a.mean() - b.mean()) / denom))


def speckle_snr(envelope: np.ndarray) -> float:
    """Mean/std of a **linear** envelope region — ~1.913 for Rayleigh speckle.

    Pass linear envelope amplitude, *not* dB: the constant only holds for the
    Rayleigh-distributed envelope, and log-compression changes it. A value far
    below ~1.9 indicates the region is not fully-developed speckle (e.g. it
    contains structure); far above indicates over-smoothing.
    """
    x = np.asarray(envelope, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    sd = x.std()
    if sd <= 0:
        return float("nan")
    return float(x.mean() / sd)


def speckle_snr_patched(
    envelope: np.ndarray,
    mask: np.ndarray,
    patch: int = 32,
    min_valid_frac: float = 0.9,
) -> float:
    """Median per-patch speckle SNR — the depth-robust form of :func:`speckle_snr`.

    The Rayleigh mean/std constant holds only where the *underlying* mean
    amplitude is constant. Over a deep region it is not: attenuation and TGC
    make the mean vary with depth, so a single pooled mean/std mixes
    populations of differing mean, inflating the standard deviation and
    pushing the ratio well below 1.913 even for textbook speckle. (Measured on
    this corpus: an 18 mm-deep band scores 1.41-1.69 where small patches of
    the same image score ~1.9.)

    Splitting into patches small enough that the mean is locally constant, and
    taking the median across them, removes that bias and is robust to a few
    patches containing structure the segmentation does not label.

    Args:
        envelope: Linear envelope image, 2-D.
        mask: Boolean region to measure in, same shape.
        patch: Patch side length in pixels.
        min_valid_frac: Fraction of a patch that must be inside ``mask`` for
            that patch to count.

    Returns:
        Median of per-patch SNR, or NaN if no patch qualifies.
    """
    env = np.asarray(envelope, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if env.shape != m.shape:
        raise ValueError(f"shape mismatch: {env.shape} vs {m.shape}")

    vals = []
    nz, nx = env.shape
    for z0 in range(0, nz - patch + 1, patch):
        for x0 in range(0, nx - patch + 1, patch):
            sub_m = m[z0 : z0 + patch, x0 : x0 + patch]
            if sub_m.mean() < min_valid_frac:
                continue
            sub = env[z0 : z0 + patch, x0 : x0 + patch][sub_m]
            snr = speckle_snr(sub)
            if np.isfinite(snr):
                vals.append(snr)
    return float(np.median(vals)) if vals else float("nan")


def boxcar_mean(img: np.ndarray, k: int) -> np.ndarray:
    """``k x k`` moving average with edge padding, via a summed-area table."""
    if k <= 1:
        return np.asarray(img, dtype=np.float64)
    a = np.asarray(img, dtype=np.float64)
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    out = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (out / (k * k))[: a.shape[0], : a.shape[1]]


def structural_ncc(
    ours_db: np.ndarray,
    ref_db: np.ndarray,
    window: int = 15,
    dynamic_range_db: float = 40.0,
) -> float:
    """Correlation of locally-averaged log-envelopes (speckle removed).

    Averaging over a window several speckle grains wide suppresses the speckle
    realisation, leaving macroscopic structure. Unlike raw pixel-wise NCC this
    is a fair comparison against a reference produced by a different
    reconstruction, but it is still a *relative* measure and is reported
    rather than hard-gated (see the module docstring for why any
    reference-comparison threshold on this corpus has a low ceiling).

    Args:
        ours_db: Our log-envelope image in dB, max-normalised to 0.
        ref_db: Reference log-envelope in dB, same shape and normalisation.
        window: Boxcar width in pixels.
        dynamic_range_db: Pixels below this (in both images) are excluded.
    """
    a = np.asarray(ours_db, dtype=np.float64)
    b = np.asarray(ref_db, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = (a > -dynamic_range_db) | (b > -dynamic_range_db)
    sa, sb = boxcar_mean(a, window)[mask], boxcar_mean(b, window)[mask]
    if sa.size < 2:
        return float("nan")
    return float(np.corrcoef(sa, sb)[0, 1])
