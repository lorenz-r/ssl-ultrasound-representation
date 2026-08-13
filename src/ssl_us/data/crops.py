"""Depth-stratified crop sampling with label attachment.

Sampling is **uniform in depth, not uniform in image area** (PLAN.md 4.5c measure 1):
deep regions hold more pixels, so area-uniform sampling over-represents them and
every depth-conditioned measurement downstream becomes hard to read.

Labels come for free at crop level because ``segmentation_map`` is already on the
same (507, 456) grid as ``bmode`` — that is why PLAN.md 3.3 adopts the dataset's own
grid verbatim. The sound-speed map is resampled onto that grid once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo


@dataclass(frozen=True)
class CropSpec:
    """Crop-sampling configuration for one experiment arm.

    Crop size is an *arm-defining* parameter, not a global constant: the K>=8 arm
    trains and transfers at 64, the K<=4 arm at 128 (PLAN.md 3.2d). Mixing sizes
    across the train/transfer boundary reintroduces the feature-statistics mismatch
    the single-size choice exists to avoid.
    """

    size: int = 128
    z_min_mm: float = 0.0  # 8.0 for K>=8 arms (angular-sector occupancy floor)
    z_max_mm: float = 40.0
    n_per_frame: int = 8
    n_depth_bands: int = 8
    seed: int = 0


@dataclass
class CropBatch:
    """Sampled crops plus every label axis in PLAN.md 4.1."""

    images: np.ndarray  # (N, size, size) float32 — the signal-stage crop
    frame_id: np.ndarray  # (N,) str — split unit; every crop inherits its frame's id
    labels: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_id)


def _band_edges(spec: CropSpec) -> np.ndarray:
    """Row indices bounding each equal-width depth band the crop *top* may start in."""
    top_min = geo.z_mm_to_row(spec.z_min_mm)
    # the deepest a crop top may start so the crop still fits inside z_max
    top_max = geo.z_mm_to_row(spec.z_max_mm) - spec.size
    if top_max < top_min:
        raise ValueError(
            f"crop size {spec.size} does not fit in depth band "
            f"[{spec.z_min_mm}, {spec.z_max_mm}] mm "
            f"({top_max - top_min + spec.size} rows available)"
        )
    return np.linspace(top_min, top_max, spec.n_depth_bands + 1)


def sample_frame(
    image: np.ndarray,
    seg: np.ndarray,
    sos_on_grid: np.ndarray,
    spec: CropSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Draw ``spec.n_per_frame`` depth-stratified crops from one frame.

    Args:
        image: (507, 456) signal-stage image.
        seg: (507, 456) uint8 cyst mask.
        sos_on_grid: (507, 456) sound speed resampled onto the B-mode grid.
        spec: sampling configuration.
        rng: seeded generator.

    Returns:
        ``(crops, labels)`` — crops ``(n, size, size)`` float32, labels keyed per
        PLAN.md 4.1.
    """
    s = spec.size
    edges = _band_edges(spec)
    n_band = max(1, spec.n_per_frame // spec.n_depth_bands)
    extra = spec.n_per_frame - n_band * spec.n_depth_bands

    tops: list[int] = []
    for b in range(spec.n_depth_bands):
        k = n_band + (1 if b < extra else 0)
        if k:
            lo, hi = int(edges[b]), max(int(edges[b]) + 1, int(edges[b + 1]))
            tops.extend(rng.integers(lo, hi, size=k))
    lefts = rng.integers(0, geo.BMODE_NX - s + 1, size=len(tops))

    crops = np.empty((len(tops), s, s), dtype=np.float32)
    cyst_frac = np.empty(len(tops), dtype=np.float32)
    sos_mean = np.empty(len(tops), dtype=np.float32)
    sos_std = np.empty(len(tops), dtype=np.float32)
    z_centre = np.empty(len(tops), dtype=np.float32)
    x_centre = np.empty(len(tops), dtype=np.float32)

    for i, (t, left) in enumerate(zip(tops, lefts, strict=True)):
        crops[i] = image[t : t + s, left : left + s]
        seg_c = seg[t : t + s, left : left + s]
        sos_c = sos_on_grid[t : t + s, left : left + s]
        cyst_frac[i] = float(seg_c.mean())
        sos_mean[i] = float(sos_c.mean())
        sos_std[i] = float(sos_c.std())
        z_centre[i] = float(geo.row_to_z_mm(t + s / 2))
        x_centre[i] = float(geo.col_to_x_mm(left + s / 2))

    labels = {
        "cyst_fraction": cyst_frac,
        "contains_cyst": (cyst_frac > 0.01).astype(np.int64),
        "sos_mean": sos_mean,
        "sos_std": sos_std,
        "z_centre": z_centre,
        "x_centre": x_centre,
    }
    return crops, labels


def sample_corpus(
    images: np.ndarray,
    segs: np.ndarray,
    sos_maps: np.ndarray,
    frame_ids: np.ndarray,
    frame_labels: dict[str, np.ndarray],
    spec: CropSpec,
) -> CropBatch:
    """Sample crops across a corpus and attach both crop- and frame-level labels.

    ``frame_labels`` (``phase_error``, ``t0_group``, ``sos_mean_frame``, ``split``, …)
    are broadcast to every crop of their frame. Frame-level targets must still be
    *probed* at frame level by pooling crop features — 98.2 % of crop-mean
    sound-speed variance is between frames, so the effective sample size is the
    frame count, not the crop count (PLAN.md 6.12).
    """
    rng = np.random.default_rng(spec.seed)
    all_crops, all_ids = [], []
    acc: dict[str, list[np.ndarray]] = {}

    for i in range(len(images)):
        sos_grid = geo.resample_sos_to_bmode_grid(sos_maps[i])
        crops, labels = sample_frame(images[i], segs[i], sos_grid, spec, rng)
        all_crops.append(crops)
        all_ids.append(np.full(len(crops), frame_ids[i], dtype=object))
        for k, v in labels.items():
            acc.setdefault(k, []).append(v)
        for k, v in frame_labels.items():
            acc.setdefault(k, []).append(np.full(len(crops), v[i]))

    return CropBatch(
        images=np.concatenate(all_crops),
        frame_id=np.concatenate(all_ids),
        labels={k: np.concatenate(v) for k, v in acc.items()},
    )
