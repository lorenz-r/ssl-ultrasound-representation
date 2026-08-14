"""WI-13 — store-backed eval inputs (`ssl_us.eval.store_features`).

The load-bearing property here is that eval pre-processing matches what WI-12
trained on. If it drifts, the frozen-feature probes measure the mismatch rather
than the representation, and nothing downstream would flag it — so the layout
and the reduce-then-normalize order are both asserted directly against the
training path in `data/transforms.py` rather than against a hardcoded expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.beamform.store import SectorStore, StoreSpec
from ssl_us.data import geometry as geo
from ssl_us.data import transforms as tf
from ssl_us.eval.store_features import (
    StoreCropSpec,
    _origin_bounds,
    frame_labels_from_metadata,
    sample_store_crops,
)

CROP = 4


@pytest.fixture
def spec() -> StoreSpec:
    return StoreSpec(n_frames=3, nz=12, nx=10, k=16)


@pytest.fixture
def store_path(tmp_path, spec):
    frame_ids = [f"train/{i}" for i in range(spec.n_frames)]
    # Real geometry, not a stub: occupancy is what `reduce_and_normalize`
    # divides by, so a store without it cannot exercise the path under test.
    geometry = sec.build_sector_geometry(
        k=spec.k,
        f_number=1.5,
        z_mm=np.linspace(5.0, 40.0, spec.nz),
        x_mm=np.linspace(-5.0, 5.0, spec.nx),
    )
    store = SectorStore.create(
        tmp_path / "store.zarr",
        spec,
        frame_ids=frame_ids,
        beamform_params={"f_number": 1.5, "k": 16, "t_peak": 0.0},
        geometry=geometry,
    )
    rng = np.random.default_rng(0)
    for i in range(spec.n_frames):
        store.write_frame(i, rng.normal(size=spec.frame_shape).astype(np.float32))
    store.write_frame_metadata({
        fid: {
            "t0": geo.T0_LEVELS_S[i % 3],
            "phase_error": 1.0 + i,
            "sos_mean": 1500.0 + i,
            "sos_std": 10.0 + i,
            "cyst_fraction": 0.1 * i,
            "segmentation_empty": False,
        }
        for i, fid in enumerate(frame_ids)
    })
    return tmp_path / "store.zarr", frame_ids


def test_crops_are_in_the_encoder_layout(store_path):
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0)
    batch = sample_store_crops(path, frame_ids, spec)

    # (N, 2K, size, size) — 2K channels, not 2 and not K.
    assert batch.images.shape == (len(frame_ids) * 2, 8, CROP, CROP)
    assert batch.images.dtype == np.float32
    assert np.isfinite(batch.images).all()
    assert len(batch) == len(batch.frame_id) == len(frame_ids) * 2


def test_preprocessing_matches_the_training_path(store_path):
    """The crop must equal reduce_and_normalize + the channel-fold, exactly.

    Normalizing before reducing is the documented wrong order (PLAN.md §3.2)
    and would still produce finite, plausible-looking crops — so it is checked
    against the real transform rather than a smoke test.
    """
    from ssl_us.beamform.store import open_store

    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=1, seed=7)
    batch = sample_store_crops(path, frame_ids, spec)

    store = open_store(path)
    occupancy = np.asarray(store.occupancy)
    for n, fid in enumerate(batch.frame_id):
        frame = store.read_frame(store.index_of(str(fid)))
        z0, x0 = batch.origins[n]
        zs, xs = slice(z0, z0 + CROP), slice(x0, x0 + CROP)
        expected = tf.reduce_and_normalize(
            frame[:, zs, xs, :], occupancy[zs, xs, :], target_k=4
        )
        expected = np.moveaxis(expected, -1, 1).reshape(8, CROP, CROP)
        np.testing.assert_allclose(batch.images[n], expected, rtol=1e-5, atol=1e-6)


def test_frame_labels_broadcast_to_every_crop_of_their_frame(store_path):
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=0)
    batch = sample_store_crops(path, frame_ids, spec)

    for fid in frame_ids:
        mask = batch.frame_id == fid
        assert mask.sum() == 3
        # One frame, one value — a frame-level label that varied across its own
        # crops would silently inflate the frame-level probe's sample size.
        assert len(np.unique(batch.labels["phase_error"][mask])) == 1
        assert len(np.unique(batch.labels["sos_mean_frame"][mask])) == 1
    # Geometry is per-crop, so it must *not* be constant within a frame.
    assert len(np.unique(batch.labels["z_centre"])) > 1


def test_metadata_maps_t0_to_group_and_renames_sos(store_path):
    _path, frame_ids = store_path
    metadata = {
        fid: {"t0": geo.T0_LEVELS_S[i % 3], "phase_error": 1.0, "sos_mean": 1500.0,
              "sos_std": 10.0, "cyst_fraction": 0.2}
        for i, fid in enumerate(frame_ids)
    }
    labels = frame_labels_from_metadata(metadata, frame_ids)

    assert "t0_group" in labels and labels["t0_group"].dtype == np.int64
    assert labels["t0_group"].tolist() == [0, 1, 2]
    # An unrecognised t0 is a data error and must not be silently bucketed.
    with pytest.raises(ValueError, match="matches none"):
        frame_labels_from_metadata(
            {frame_ids[0]: {**metadata[frame_ids[0]], "t0": -1.76e-07}}, [frame_ids[0]]
        )
    # §4.1 names the frame-level variants with the suffix; the unsuffixed form
    # is the crop-level target, which the store cannot supply.
    assert "sos_mean_frame" in labels and "sos_mean" not in labels
    assert "sos_std_frame" in labels and "sos_std" not in labels


def test_z_min_mm_and_impossible_crops_are_enforced():
    shallow = StoreCropSpec(k=4, crop_size=CROP, z_min_mm=0.0)
    z_lo, _z_hi, _x_lo, _x_hi = _origin_bounds(shallow, nz=12, nx=10)
    assert z_lo == 0

    deep = StoreCropSpec(k=16, crop_size=CROP, z_min_mm=8.0)
    z_lo_deep, _, _, _ = _origin_bounds(deep, nz=507, nx=456)
    assert z_lo_deep == geo.z_mm_to_row(8.0) > 0

    # A crop larger than the frame is a config error, not a silent empty batch.
    with pytest.raises(ValueError, match="does not fit"):
        _origin_bounds(StoreCropSpec(crop_size=128), nz=12, nx=10)
