"""Crop sampling: depth-uniform, never axially flipped, depth floor respected."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.data import geometry as geo
from ssl_us.data.crops import CropSpec, sample_corpus


@pytest.fixture
def toy_corpus():
    rng = np.random.default_rng(0)
    n = 6
    images = rng.normal(size=(n, geo.BMODE_NZ, geo.BMODE_NX)).astype(np.float32)
    segs = (rng.random((n, geo.BMODE_NZ, geo.BMODE_NX)) < 0.05).astype(np.uint8)
    sos = rng.normal(1450, 20, size=(n, 32, 32)).astype(np.float32)
    fids = np.array([f"train/{i:04d}" for i in range(n)])
    flat = sos.reshape(n, -1)
    labels = {
        "phase_error": rng.normal(1.8, 0.3, n).astype(np.float32),
        "t0_group": rng.integers(0, 3, n),
        "sos_mean_frame": flat.mean(1),
        "sos_std_frame": flat.std(1),
    }
    return images, segs, sos, fids, labels


def test_crop_shape_and_count(toy_corpus):
    images, segs, sos, fids, labels = toy_corpus
    spec = CropSpec(size=128, n_per_frame=8, seed=0)
    b = sample_corpus(images, segs, sos, fids, labels, spec)
    assert b.images.shape == (6 * 8, 128, 128)
    assert len(b.frame_id) == 48
    assert set(b.frame_id) == set(fids)


def test_depth_sampling_is_uniform_in_depth_not_area(toy_corpus):
    """Deep regions hold more pixels; area-uniform sampling would skew deep."""
    images, segs, sos, fids, labels = toy_corpus
    spec = CropSpec(size=128, n_per_frame=64, n_depth_bands=8, seed=0)
    b = sample_corpus(images, segs, sos, fids, labels, spec)
    z = b.labels["z_centre"]
    counts, _ = np.histogram(z, bins=8)
    # every band populated, and none more than 2x the least populated
    assert counts.min() > 0
    assert counts.max() / counts.min() < 2.0


def test_depth_floor_respected_for_high_k_arm(toy_corpus):
    """K>=8 arms sample only below 8 mm (angular-sector occupancy floor)."""
    images, segs, sos, fids, labels = toy_corpus
    spec = CropSpec(size=64, z_min_mm=8.0, n_per_frame=32, seed=0)
    b = sample_corpus(images, segs, sos, fids, labels, spec)
    top_edge_mm = b.labels["z_centre"] - (spec.size / 2) * geo.bmode_dz_mm()
    assert top_edge_mm.min() >= 8.0 - 1e-6


def test_k_le_4_arm_uses_full_depth_range(toy_corpus):
    """The 8 mm floor is ours and K-dependent; K<=4 must not inherit it."""
    images, segs, sos, fids, labels = toy_corpus
    spec = CropSpec(size=128, z_min_mm=0.0, n_per_frame=64, seed=0)
    b = sample_corpus(images, segs, sos, fids, labels, spec)
    top_edge_mm = b.labels["z_centre"] - 64 * geo.bmode_dz_mm()
    assert top_edge_mm.min() < 1.0


def test_labels_present_and_finite(toy_corpus):
    images, segs, sos, fids, labels = toy_corpus
    b = sample_corpus(images, segs, sos, fids, labels, CropSpec(seed=0))
    for key in ("cyst_fraction", "contains_cyst", "sos_mean", "sos_std",
                "z_centre", "x_centre", "phase_error", "t0_group",
                "sos_mean_frame", "sos_std_frame"):
        assert key in b.labels, key
        assert len(b.labels[key]) == len(b)
        assert np.isfinite(np.asarray(b.labels[key], dtype=float)).all(), key


def test_crop_too_large_for_band_raises():
    images = np.zeros((1, geo.BMODE_NZ, geo.BMODE_NX), dtype=np.float32)
    segs = np.zeros_like(images, dtype=np.uint8)
    sos = np.full((1, 32, 32), 1500.0, dtype=np.float32)
    spec = CropSpec(size=128, z_min_mm=35.0, z_max_mm=40.0)
    with pytest.raises(ValueError, match="does not fit"):
        sample_corpus(images, segs, sos, np.array(["f0"]), {}, spec)


def test_sos_resample_preserves_range_and_shape():
    rng = np.random.default_rng(0)
    sos = rng.normal(1450, 25, size=(32, 32)).astype(np.float32)
    out = geo.resample_sos_to_bmode_grid(sos)
    assert out.shape == (geo.BMODE_NZ, geo.BMODE_NX)
    # bilinear interpolation with edge clamping cannot leave the source range
    assert out.min() >= sos.min() - 1e-3
    assert out.max() <= sos.max() + 1e-3


def test_t0_group_maps_measured_levels():
    for i, level in enumerate(geo.T0_LEVELS_S):
        assert geo.t0_group(level) == i
    with pytest.raises(ValueError):
        geo.t0_group(-1.0e-7)
