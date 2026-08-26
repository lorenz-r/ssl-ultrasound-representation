"""Dataset-level amplitude normalization (PLAN.md WI-7).

Exactly one scalar, computed once over a sample of the train split and
applied as a single divide at load time — never a per-depth correction.

This is the operation the old repo's `normalize_frame` got backwards
(PLAN.md 1.3): it divided every fast-time sample by its own RMS over
`(transducer, scanline, iq)`, which *is* a per-depth correction and silently
deletes the attenuation signal (raw IQ decays ~2 orders of magnitude from
shallow to deep) — precisely the axis §4.5c exists to probe. The per-depth
RMS profile is still computed here, but only ever stored as metadata for
that probe; it is never divided out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def complex_magnitude(x: np.ndarray, channel_axis: int = -1) -> np.ndarray:
    """|I + jQ| from the store's native 2-channel real layout."""
    if x.shape[channel_axis] != 2:
        raise ValueError(f"expected size-2 channel axis at {channel_axis}, got shape {x.shape}")
    i, q = np.moveaxis(x, channel_axis, 0)
    return np.sqrt(i.astype(np.float64) ** 2 + q.astype(np.float64) ** 2)


def dataset_rms(
    frames: np.ndarray,
    channel_axis: int = -1,
    sample_size: int | None = None,
    seed: int = 0,
) -> float:
    """sqrt(mean(|z|^2)) — the single scalar the whole dataset is divided by.

    Args:
        frames: raw (un-normalized) frames, real dtype, a size-2 axis at
            ``channel_axis`` holding (I, Q).
        channel_axis: which axis is the 2-length I/Q channel axis.
        sample_size: if given, estimate the RMS from this many frames off
            axis 0 rather than the whole array (PLAN.md WI-7: "over a sample
            of the train split") — the whole point being that this never
            needs to touch the full 27 GB store to compute one number.
        seed: sampling seed.

    Returns:
        A single positive float.
    """
    if sample_size is not None and sample_size < frames.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(frames.shape[0], size=sample_size, replace=False)
        frames = frames[idx]
    mag = complex_magnitude(frames, channel_axis=channel_axis)
    return float(np.sqrt(np.mean(mag**2)))


def per_depth_rms(frames: np.ndarray, depth_axis: int, channel_axis: int = -1) -> np.ndarray:
    """RMS of |I+jQ| at each depth index, pooled over every other axis.

    Metadata only (PLAN.md WI-7) — this is the attenuation signal itself, so
    it must never be divided out; it is computed here purely to be recorded
    and to let a test assert it survives normalization undistorted.
    """
    mag = complex_magnitude(frames, channel_axis=channel_axis)
    # channel_axis has been consumed by complex_magnitude; if it was a lower
    # index than depth_axis, depth_axis shifts left by one.
    ch, dep = channel_axis % frames.ndim, depth_axis % frames.ndim
    d_axis = depth_axis - 1 if ch < dep else depth_axis
    other_axes = tuple(a for a in range(mag.ndim) if a != d_axis)
    return np.sqrt(np.mean(mag**2, axis=other_axes))


def apply_normalization(frames: np.ndarray, scalar_rms: float) -> np.ndarray:
    """The one scalar divide applied at load — nothing depth-dependent here."""
    return (frames / scalar_rms).astype(frames.dtype)


@dataclass(frozen=True)
class NormalizationStats:
    """What WI-7 writes into the store manifest."""

    scalar_rms: float
    per_depth_rms_profile: np.ndarray  # metadata only — never applied
    n_frames_sampled: int

    def to_manifest_dict(self) -> dict:
        return {
            "scalar_rms": self.scalar_rms,
            "per_depth_rms_profile": self.per_depth_rms_profile.tolist(),
            "n_frames_sampled": self.n_frames_sampled,
        }


def compute_normalization_stats(
    frames: np.ndarray,
    depth_axis: int,
    channel_axis: int = -1,
    sample_size: int | None = None,
    seed: int = 0,
) -> NormalizationStats:
    """Both WI-7 artefacts in one pass: the scalar to divide by, and the
    per-depth profile to record but never apply."""
    if sample_size is not None and sample_size < frames.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(frames.shape[0], size=sample_size, replace=False)
        sampled = frames[idx]
    else:
        sampled = frames
    profile = per_depth_rms(sampled, depth_axis=depth_axis, channel_axis=channel_axis)
    return NormalizationStats(
        scalar_rms=dataset_rms(sampled, channel_axis=channel_axis),
        per_depth_rms_profile=profile,
        n_frames_sampled=int(sampled.shape[0]),
    )
