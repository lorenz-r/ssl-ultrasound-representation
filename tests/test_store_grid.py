"""WI-16 — crop geometry follows the store's own grid, not NVIDIA's.

`sample_store_crops` reported crop centres through `data/geometry`'s bmode
helpers, which are constants for the (507, 456) 0-40 mm NVIDIA grid. Stanford's
two WI-16 variants are 253x484 over 0-19.9 mm and 304x582 over the same range,
so read through the bmode axis a crop centre comes out on a scale no pixel in
the store occupies — and `z_centre` is a *probe target*, so the damage is a
plausible-looking R^2 against a fabricated axis rather than a crash.

These tests pin the mapping to the store, and pin the NVIDIA path to being
unchanged by the generalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.beamform.store import SectorStore, StoreSpec, open_store
from ssl_us.data import geometry as geo
from ssl_us.eval.store_features import (
    StoreCropSpec,
    StoreGrid,
    _origin_bounds,
    crop_echogenicity_db,
    sample_store_crops,
)

CROP = 4
# A deliberately un-NVIDIA grid: shallower, finer, and laterally narrower than
# the bmode axis, so any leftover bmode assumption shows up as a wrong number.
Z_MM = np.linspace(0.0, 19.9, 24)
X_MM = np.linspace(-9.6, 9.6, 20)


def _labels(frame_ids):
    return {fid: {"session": "s0", "date": "20190318", "kind": "FSA",
                  "source_file": f"{fid}.mat"} for fid in frame_ids}


@pytest.fixture
def store_path(tmp_path):
    spec = StoreSpec(n_frames=3, nz=len(Z_MM), nx=len(X_MM), k=16)
    frame_ids = [str(i) for i in range(spec.n_frames)]
    geometry = sec.build_sector_geometry(k=spec.k, f_number=1.5, z_mm=Z_MM, x_mm=X_MM)
    store = SectorStore.create(
        tmp_path / "s.zarr", spec, frame_ids=frame_ids,
        beamform_params={"k": 16, "f_number": 1.5}, geometry=geometry,
    )
    rng = np.random.default_rng(0)
    for i in range(spec.n_frames):
        store.write_frame(i, rng.normal(size=spec.frame_shape).astype(np.float32))
    store.write_frame_metadata(_labels(frame_ids))
    return tmp_path / "s.zarr", frame_ids


def test_store_records_and_returns_its_own_axes(store_path):
    path, _ = store_path
    store = open_store(path)
    np.testing.assert_allclose(store.z_mm, Z_MM)
    np.testing.assert_allclose(store.x_mm, X_MM)


def test_crop_centres_come_from_the_store_grid_not_the_bmode_grid(store_path):
    from ssl_us.eval.stanford_targets import stanford_frame_labels

    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=4, seed=0)
    batch = sample_store_crops(
        path, frame_ids, spec, frame_label_fn=stanford_frame_labels
    )

    grid = StoreGrid(z_mm=Z_MM, x_mm=X_MM)
    for i, (z0, x0) in enumerate(batch.origins):
        assert batch.labels["z_centre"][i] == pytest.approx(
            grid.row_to_z_mm(z0 + CROP / 2), rel=1e-5
        )
        assert batch.labels["x_centre"][i] == pytest.approx(
            grid.col_to_x_mm(x0 + CROP / 2), rel=1e-5
        )
    # Every centre is inside the store's physical extent. Read through the
    # bmode axis these would run out past 40 mm, which is the bug.
    assert batch.labels["z_centre"].max() <= Z_MM[-1] + 1e-6
    assert batch.labels["x_centre"].min() >= X_MM[0] - 1e-6


def test_bmode_grid_is_unchanged_by_the_generalization():
    grid = StoreGrid.bmode()
    for row in (0, 1, 123, 506):
        assert grid.row_to_z_mm(row) == pytest.approx(geo.row_to_z_mm(row))
    for col in (0, 7, 455):
        assert grid.col_to_x_mm(col) == pytest.approx(geo.col_to_x_mm(col))
    assert grid.z_mm_to_row(8.0) == geo.z_mm_to_row(8.0)
    # And the default when no store is in hand is still that grid.
    assert _origin_bounds(StoreCropSpec(crop_size=64, z_min_mm=8.0), 507, 456)[0] == (
        geo.z_mm_to_row(8.0)
    )


def test_z_min_floor_is_applied_on_the_stores_own_depth_scale(store_path):
    """8 mm is row 101 on NVIDIA's 0-40 mm grid and row ~9 on this 0-19.9 mm
    one. Applying the NVIDIA row to this store would put the floor at 16.7 mm —
    still a legal crop, just at the wrong depth, and silently so."""
    grid = StoreGrid(z_mm=Z_MM, x_mm=X_MM)
    spec = StoreCropSpec(k=16, crop_size=CROP, z_min_mm=8.0)
    z_lo, _, _, _ = _origin_bounds(spec, len(Z_MM), len(X_MM), grid)
    assert grid.row_to_z_mm(z_lo) == pytest.approx(8.0, abs=grid.z_mm[1] - grid.z_mm[0])
    assert z_lo != geo.z_mm_to_row(8.0)


def test_echogenicity_is_off_by_default_and_tracks_brightness(store_path):
    from ssl_us.eval.stanford_targets import stanford_frame_labels

    path, frame_ids = store_path
    off = sample_store_crops(
        path, frame_ids, StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2),
        frame_label_fn=stanford_frame_labels,
    )
    assert "echogenicity" not in off.labels

    on = sample_store_crops(
        path, frame_ids,
        StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, derive_echogenicity=True),
        frame_label_fn=stanford_frame_labels,
    )
    assert on.labels["echogenicity"].shape == (len(on),)
    assert np.isfinite(on.labels["echogenicity"]).all()

    # It is a dB measure, so scaling a crop by a constant must shift it by
    # exactly 20*log10(that constant) — not by some normalization-dependent
    # amount. This is what makes it comparable across crops.
    crop = np.abs(np.random.default_rng(1).normal(size=(2, CROP, CROP, 4))) + 0.5
    assert crop_echogenicity_db(10.0 * crop) == pytest.approx(
        crop_echogenicity_db(crop) + 20.0, abs=1e-6
    )


# ── cross-corpus amplitude scale (WI-16) ────────────────────────────────────


def test_intensity_scale_is_a_pure_change_of_units(store_path):
    """One scalar for the corpus must move every crop together — it may not
    change any *relative* structure. If it did, the matched and raw rows of the
    transfer table would differ for a reason other than the encoder's response
    to scale, and neither would be interpretable."""
    from ssl_us.eval.stanford_targets import stanford_frame_labels

    path, frame_ids = store_path
    base = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=3,
                         derive_echogenicity=True)
    raw = sample_store_crops(path, frame_ids, base, frame_label_fn=stanford_frame_labels)
    scaled = sample_store_crops(
        path, frame_ids,
        StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=3,
                      derive_echogenicity=True, intensity_scale=7.5),
        frame_label_fn=stanford_frame_labels,
    )
    np.testing.assert_allclose(scaled.images, raw.images * 7.5, rtol=1e-5, atol=1e-6)
    # Same crops, same geometry — only the units moved.
    np.testing.assert_array_equal(scaled.origins, raw.origins)
    np.testing.assert_allclose(scaled.labels["z_centre"], raw.labels["z_centre"])
    # Echogenicity is a dB measure, so it shifts by a constant and its spread —
    # the thing the probe actually fits — is unchanged.
    delta = scaled.labels["echogenicity"] - raw.labels["echogenicity"]
    np.testing.assert_allclose(delta, np.full_like(delta, 20 * np.log10(7.5)), atol=1e-3)


def test_default_scale_is_a_no_op(store_path):
    from ssl_us.eval.stanford_targets import stanford_frame_labels

    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0)
    a = sample_store_crops(path, frame_ids, spec, frame_label_fn=stanford_frame_labels)
    b = sample_store_crops(
        path, frame_ids,
        StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0, intensity_scale=1.0),
        frame_label_fn=stanford_frame_labels,
    )
    np.testing.assert_array_equal(a.images, b.images)


def test_corpus_intensity_scale_maps_one_corpus_onto_the_other(store_path, tmp_path):
    """The factor must actually equalise the two corpora's median amplitude —
    it is the whole point, and an inverted ratio would silently make the
    mismatch 360000x worse instead of fixing it."""
    from ssl_us.eval.stanford_targets import stanford_frame_labels
    from ssl_us.eval.store_features import corpus_intensity_scale, corpus_median_abs

    path, frame_ids = store_path
    # A second store of the same shape, 50x brighter — stands in for the real
    # 600x NVIDIA/Stanford gap.
    spec = StoreSpec(n_frames=3, nz=len(Z_MM), nx=len(X_MM), k=16)
    geometry = sec.build_sector_geometry(k=spec.k, f_number=1.5, z_mm=Z_MM, x_mm=X_MM)
    bright_path = tmp_path / "bright.zarr"
    bright = SectorStore.create(
        bright_path, spec, frame_ids=frame_ids,
        beamform_params={"k": 16, "f_number": 1.5}, geometry=geometry,
    )
    rng = np.random.default_rng(0)
    for i in range(spec.n_frames):
        bright.write_frame(i, (50.0 * rng.normal(size=spec.frame_shape)).astype(np.float32))
    bright.write_frame_metadata(_labels(frame_ids))

    crop_spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=4, seed=0)
    scale = corpus_intensity_scale(
        bright_path, path, crop_spec, n_frames=3,
        frame_label_fn=stanford_frame_labels,
        reference_frame_label_fn=stanford_frame_labels,
    )
    ref = corpus_median_abs(path, crop_spec, 3, frame_label_fn=stanford_frame_labels)
    got = corpus_median_abs(bright_path, crop_spec, 3,
                            frame_label_fn=stanford_frame_labels) * scale
    assert got == pytest.approx(ref, rel=1e-6)
    assert scale < 1.0, "the brighter corpus must be scaled down, not up"
