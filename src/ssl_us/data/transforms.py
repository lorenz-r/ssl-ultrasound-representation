"""K-ladder reduction and occupancy normalization, applied at runtime
(PLAN.md §3.2, WI-5).

The store holds **raw** sector sums ``S_k`` (PLAN.md §3.2): sectors contain
different numbers of elements, so a raw amplitude difference between sectors
is a trivial shortcut a model could key on, but normalizing at *build* time
would break the ladder — if normalized means ``N_k = S_k / n_k`` were what
got stored, then ``N_{2j} + N_{2j+1} != (S_{2j} + S_{2j+1}) / (n_{2j} + n_{2j+1})``
in general, so merging sectors after normalization does not equal merging the
underlying raw sums and normalizing once.

So the two operations are kept strictly separate, and strictly ordered:

1. :func:`reduce_ladder` — pairwise sum of **raw** sector sums *and* their
   occupancy counts, in lockstep, all the way from any K down to 1. Exact:
   this is just addition, so it never introduces floating-point drift beyond
   the two inputs' own precision.
2. :func:`normalize` — divide by occupancy, only at the K the caller actually
   wants, never before reducing further.

Reversing that order (normalize each sector, *then* sum sectors together) is
the bug this module exists to prevent; :func:`normalize` is only ever called
last, and a test constructs the wrong order deliberately to show it diverges.
"""

from __future__ import annotations

import numpy as np


def reduce_ladder(values: np.ndarray, target_k: int, axis: int = -1) -> np.ndarray:
    """Pairwise-sum ``values`` along ``axis`` from its current K down to ``target_k``.

    Works identically on raw sector sums and on occupancy counts — call it on
    both, in lockstep, before ever normalizing. ``K`` must be a power of two
    and ``target_k`` must be a power-of-two divisor of it (PLAN.md's ladder:
    16 -> 8 -> 4 -> 2 -> 1).
    """
    values = np.moveaxis(values, axis, -1)
    k = values.shape[-1]
    if k & (k - 1) != 0:
        raise ValueError(f"K must be a power of two, got {k}")
    if target_k & (target_k - 1) != 0 or target_k <= 0:
        raise ValueError(f"target_k must be a positive power of two, got {target_k}")
    if target_k > k:
        raise ValueError(f"target_k ({target_k}) cannot exceed the current K ({k})")

    # One grouped sum rather than a halving loop. Uniform binning means the
    # ladder just partitions the sector axis into `target_k` contiguous groups
    # of `k // target_k` — the same grouping the loop produces — so the whole
    # descent is a single reshape-and-sum that touches each input element once
    # instead of once per rung. This runs per crop in the WI-12 loader, where
    # it was measured as ~49% of loader time before the change.
    out = values
    if k != target_k:
        out = values.reshape(*values.shape[:-1], target_k, k // target_k).sum(axis=-1)
    return np.moveaxis(out, -1, axis)


def normalize(
    sector_sums: np.ndarray, occupancy: np.ndarray, mode: str = "mean"
) -> np.ndarray:
    """Divide raw sector sums by their occupancy.

    Args:
        sector_sums: raw sums, any shape ending in the sector axis.
        occupancy: element counts, broadcastable against ``sector_sums``
            (typically the same shape, or missing a leading frame/channel axis).
        mode: ``"mean"`` (``S_k / n_k``, equalizes signal level — the default)
            or ``"rms"`` (``S_k / sqrt(n_k)``, equalizes incoherent-noise
            variance instead — PLAN.md 3.2's flagged alternative).

    Returns:
        Normalized array, same shape as ``sector_sums``. Empty sectors
        (``n_k == 0``) come back as exactly 0, never NaN or inf — an empty
        sector has nothing summed into it, so 0/0 is the correct value, not
        an error.
    """
    if mode == "mean":
        denom = occupancy.astype(np.float32)
    elif mode == "rms":
        denom = np.sqrt(occupancy.astype(np.float32))
    else:
        raise ValueError(f"unknown normalization mode {mode!r}; expected 'mean' or 'rms'")

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, sector_sums / np.where(denom > 0, denom, 1.0), 0.0)
    return out.astype(sector_sums.dtype)


def reduce_and_normalize(
    sector_sums: np.ndarray,
    occupancy: np.ndarray,
    target_k: int,
    mode: str = "mean",
    axis: int = -1,
) -> np.ndarray:
    """The one correct runtime path: reduce raw sums and occupancy together,
    then normalize once at ``target_k``. Never normalize-then-reduce.
    """
    reduced_sums = reduce_ladder(sector_sums, target_k, axis=axis)
    reduced_occ = reduce_ladder(occupancy, target_k, axis=axis)
    return normalize(reduced_sums, reduced_occ, mode=mode)


# ── torch mirror, for the GPU training path ─────────────────────────────────
#
# The training loader augments on the GPU (see `ssl_us.data.augment`), so the
# ladder is needed on tensors as well as arrays. It is kept in this file,
# directly beside the NumPy version, precisely because it is a second
# implementation of a rule that must not drift — `test_transforms.py` asserts
# the two agree numerically. The NumPy version remains the one the store's
# read path uses (WI-8's verify/profile scripts, the eval matrix); the torch
# one exists only where a device round-trip would otherwise be required.


def reduce_and_normalize_torch(
    sector_sums, occupancy, target_k: int, mode: str = "mean"
):
    """Torch mirror of :func:`reduce_and_normalize`, sector axis last.

    Args:
        sector_sums: ``(..., K)`` raw sums.
        occupancy: ``(..., K)`` counts, broadcastable against ``sector_sums``
            once the sector axis is reduced (typically missing the channel axis).
    """
    import torch

    k = sector_sums.shape[-1]
    if k % target_k:
        raise ValueError(f"target_k ({target_k}) must divide K ({k})")

    def _reduce(t):
        if t.shape[-1] == target_k:
            return t
        return t.reshape(*t.shape[:-1], target_k, t.shape[-1] // target_k).sum(dim=-1)

    sums = _reduce(sector_sums)
    occ = _reduce(occupancy.to(sums.dtype))
    if occupancy.ndim == sector_sums.ndim - 1:
        occ = occ.unsqueeze(1)  # (B, Z, X, k) -> (B, 1, Z, X, k)
    denom = occ if mode == "mean" else torch.sqrt(occ)
    if mode not in ("mean", "rms"):
        raise ValueError(f"unknown normalization mode {mode!r}; expected 'mean' or 'rms'")
    # Empty sectors are exactly 0, never NaN/inf: nothing was summed into them.
    return torch.where(denom > 0, sums / torch.where(denom > 0, denom, 1.0), 0.0)
