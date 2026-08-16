"""WI-13 — store-backed eval inputs (`ssl_us.eval.store_features`).

The load-bearing property here is that eval pre-processing matches what WI-12
trained on. If it drifts, the frozen-feature probes measure the mismatch rather
than the representation, and nothing downstream would flag it — so the layout
and the reduce-then-normalize order are both asserted directly against the
training path in `data/transforms.py` rather than against a hardcoded expectation.
"""

from __future__ import annotations

from pathlib import Path

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


# ── crop-level anatomy labels (WI-13's previously blocked cells) ─────────────


def _label_maps(frame_ids, cyst_frames=(1,)):
    """A `LabelMaps` over the real (507, 456) grid, cysts in chosen frames."""
    from ssl_us.data.label_maps import LabelMaps

    n = len(frame_ids)
    seg = np.zeros((n, geo.BMODE_NZ, geo.BMODE_NX), dtype=np.uint8)
    for i in cyst_frames:
        seg[i] = 1  # entire frame is cyst -> every crop's fraction is exactly 1
    sos = np.full((n, geo.SOS_NZ, geo.SOS_NX), 1500.0, dtype=np.float32)
    for i in range(n):
        sos[i] += i
    return LabelMaps(
        root=Path("in-memory"),
        frame_ids=list(frame_ids),
        arrays={"seg": seg, "sos": sos},
        scalars={},
    )


def test_label_maps_attach_the_crop_level_anatomy_targets(store_path):
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0)
    batch = sample_store_crops(path, frame_ids, spec, label_maps=_label_maps(frame_ids))

    assert {
        "cyst_fraction_crop",
        "contains_cyst",
        "sos_mean_crop",
        "sos_std_crop",
        "segmentation_empty",
    } <= set(batch.labels)
    # frame 1 is entirely cyst, the others entirely tissue
    is_frame_1 = batch.frame_id == frame_ids[1]
    assert (batch.labels["cyst_fraction_crop"][is_frame_1] == 1.0).all()
    assert (batch.labels["contains_cyst"][is_frame_1] == 1).all()
    assert (batch.labels["cyst_fraction_crop"][~is_frame_1] == 0.0).all()
    assert (batch.labels["contains_cyst"][~is_frame_1] == 0).all()
    # sos is constant per frame here, so the crop mean is the frame's value
    for i, fid in enumerate(frame_ids):
        sel = batch.frame_id == fid
        np.testing.assert_allclose(batch.labels["sos_mean_crop"][sel], 1500.0 + i, rtol=1e-5)


def test_crop_anatomy_labels_use_the_origins_the_batch_records(store_path):
    """The mm centres are a lossy view of the origin; labels must key on the
    recorded `(row, col)` or a half-pixel slip shifts every mask."""
    from ssl_us.data.label_maps import LabelMaps

    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=1)
    n = len(frame_ids)
    seg = np.zeros((n, geo.BMODE_NZ, geo.BMODE_NX), dtype=np.uint8)
    seg[:, ::2, :] = 1  # every other row -> crop fraction depends on the origin's parity
    maps = LabelMaps(
        root=Path("in-memory"),
        frame_ids=list(frame_ids),
        arrays={
            "seg": seg,
            "sos": np.zeros((n, geo.SOS_NZ, geo.SOS_NX), dtype=np.float32),
        },
        scalars={},
    )
    batch = sample_store_crops(path, frame_ids, spec, label_maps=maps)
    for i, (top, left) in enumerate(batch.origins):
        expected = seg[
            frame_ids.index(batch.frame_id[i]), top : top + CROP, left : left + CROP
        ].mean()
        assert batch.labels["cyst_fraction_crop"][i] == pytest.approx(expected)


def test_without_label_maps_nothing_extra_is_attached(store_path):
    """The maps stay optional — the store-only path must be unchanged."""
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0)
    labels = sample_store_crops(path, frame_ids, spec).labels
    assert "cyst_fraction_crop" not in labels
    assert "contains_cyst" not in labels
    assert "segmentation_empty" not in labels
    # and the frame-level cyst_fraction keeps its meaning
    assert "cyst_fraction" in labels


# ── chunked materialization (the 4 GB cgroup on the `data` instance) ─────────


def test_only_frames_gives_exactly_the_whole_corpus_crops(store_path):
    """Walking the corpus in chunks must not change which crops are drawn.

    Re-seeding per chunk would — and silently: every crop would still be legal,
    just different, so a chunked run and a whole-corpus run would disagree with
    no error anywhere.
    """
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=7)
    whole = sample_store_crops(path, frame_ids, spec)

    chunks = [
        sample_store_crops(path, frame_ids, spec, only_frames=[fid]) for fid in frame_ids
    ]
    images = np.concatenate([c.images for c in chunks])
    ids = np.concatenate([c.frame_id for c in chunks])
    origins = np.concatenate([c.origins for c in chunks])

    np.testing.assert_array_equal(ids, whole.frame_id)
    np.testing.assert_array_equal(origins, whole.origins)
    np.testing.assert_array_equal(images, whole.images)
    for key in whole.labels:
        np.testing.assert_array_equal(
            np.concatenate([c.labels[key] for c in chunks]), whole.labels[key], err_msg=key
        )


def test_origins_are_drawn_before_any_pixel_is_read(store_path):
    """`sample_store_origins` is the contract chunking relies on: same spec,
    same frame list -> same origins, independent of what gets materialized."""
    from ssl_us.eval.store_features import sample_store_origins

    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=3, seed=7)
    ids_a, origins_a = sample_store_origins(frame_ids, spec, 12, 10)
    ids_b, origins_b = sample_store_origins(frame_ids, spec, 12, 10)
    assert ids_a == ids_b
    np.testing.assert_array_equal(origins_a, origins_b)
    np.testing.assert_array_equal(
        origins_a, sample_store_crops(path, frame_ids, spec).origins
    )
    assert len(ids_a) == len(frame_ids) * spec.crops_per_frame


def test_batch_length_survives_dropping_the_images(store_path):
    """`wi13_matrix` frees the image array after feature extraction and keeps
    the batch for its labels. If `__len__` read `images`, every per-target mask
    built from it would come out empty and every probe would be skipped."""
    path, frame_ids = store_path
    spec = StoreCropSpec(k=4, crop_size=CROP, crops_per_frame=2, seed=0)
    batch = sample_store_crops(path, frame_ids, spec)
    n = len(batch)
    batch.images = np.empty((0,), dtype=np.float32)
    assert len(batch) == n == len(frame_ids) * 2
