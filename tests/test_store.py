"""Zarr K-sector store layout and round-trip (PLAN.md §3.7, WI-8).

Small grids throughout: the layout, the raw-sums contract and the reader are
what is under test, not the 27.3 GB the real build produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.beamform.store import SectorStore, StoreSpec, open_store
from ssl_us.data import transforms


@pytest.fixture
def spec() -> StoreSpec:
    return StoreSpec(n_frames=3, nz=12, nx=10, k=16)


@pytest.fixture
def store(tmp_path, spec) -> SectorStore:
    return SectorStore.create(
        tmp_path / "store.zarr",
        spec,
        frame_ids=[f"frame{i}" for i in range(spec.n_frames)],
        beamform_params={"f_number": 1.5, "k": 16, "t_peak": 0.0},
    )


def test_frame_shape_and_size_match_plan(spec):
    assert spec.frame_shape == (2, 12, 10, 16)
    # PLAN.md 3.2/3.7: 2 x 507 x 456 x 16 x 4 B = 29.59 MB/frame -> 27.3 GB.
    full = StoreSpec(n_frames=923)
    assert full.frame_shape == (2, 507, 456, 16)
    assert full.frame_nbytes / 1e6 == pytest.approx(29.59, abs=0.01)
    assert full.frame_nbytes * 923 / 1e9 == pytest.approx(27.3, abs=0.1)


def test_write_read_roundtrip_is_exact(store, spec):
    rng = np.random.default_rng(0)
    frame = rng.normal(size=spec.frame_shape).astype(np.float32)
    store.write_frame(1, frame)
    assert store.read_frame(1) == pytest.approx(frame)


def test_frames_are_independent(store, spec):
    rng = np.random.default_rng(1)
    a = rng.normal(size=spec.frame_shape).astype(np.float32)
    store.write_frame(0, a)
    assert not np.any(store.read_frame(1)), "unwritten frame should still be zeros"
    assert store.read_frame(0) == pytest.approx(a)


def test_wrong_frame_shape_is_rejected(store):
    with pytest.raises(ValueError, match="frame shape"):
        store.write_frame(0, np.zeros((2, 12, 10, 8), dtype=np.float32))


def test_frame_ids_length_must_match_spec(tmp_path, spec):
    with pytest.raises(ValueError, match="frame_ids"):
        SectorStore.create(tmp_path / "bad.zarr", spec, frame_ids=["only-one"], beamform_params={})


def test_reopened_store_sees_everything(tmp_path, spec):
    rng = np.random.default_rng(2)
    frame = rng.normal(size=spec.frame_shape).astype(np.float32)
    params = {"f_number": 1.5, "k": 16, "t_peak": 0.0}
    written = SectorStore.create(
        tmp_path / "s.zarr", spec, [f"f{i}" for i in range(3)], params
    )
    written.write_frame(2, frame)

    reopened = open_store(tmp_path / "s.zarr")
    assert len(reopened) == 3
    assert reopened.spec.frame_shape == spec.frame_shape
    assert reopened.frame_ids == ["f0", "f1", "f2"]
    assert reopened.beamform_params == params
    assert reopened.index_of("f2") == 2
    assert reopened.read_frame(2) == pytest.approx(frame)


def test_store_declares_itself_unnormalized(store):
    """The store holds raw sums; a consumer must not have to infer that."""
    assert store.root.attrs["normalized"] is False
    assert "reduce_and_normalize" in store.root.attrs["normalization_note"]


def test_geometry_is_stored_once_and_matches_sectors_module(tmp_path):
    z_mm = np.linspace(5.0, 40.0, 9)
    x_mm = np.linspace(-5.0, 5.0, 7)
    g = sec.build_sector_geometry(k=16, f_number=1.5, z_mm=z_mm, x_mm=x_mm)
    spec = StoreSpec(n_frames=1, nz=len(z_mm), nx=len(x_mm), k=16)
    s = SectorStore.create(tmp_path / "g.zarr", spec, ["f0"], {}, geometry=g)

    reopened = open_store(tmp_path / "g.zarr")
    assert reopened.occupancy.shape == (len(z_mm), len(x_mm), 16)
    assert np.array_equal(reopened.occupancy, g.occupancy)
    assert np.array_equal(reopened.sector_id, g.sector_id)
    # Stored once, not per frame.
    assert reopened.root["occupancy"].shape[0] == len(z_mm)
    assert s.root.attrs["sector_geometry"]["f_number"] == 1.5


def test_z_mm_x_mm_attrs_reflect_the_geometry_actually_used(tmp_path):
    """A store built on a non-NVIDIA grid (e.g. WI-16's Stanford variants) must
    not silently record NVIDIA's own (507, 456) bmode axes in its attrs."""
    z_mm = np.linspace(0.0, 19.9, 9)
    x_mm = np.linspace(-19.1, 19.1, 7)
    g = sec.build_sector_geometry(k=16, f_number=1.5, z_mm=z_mm, x_mm=x_mm)
    spec = StoreSpec(n_frames=1, nz=len(z_mm), nx=len(x_mm), k=16)
    SectorStore.create(tmp_path / "g.zarr", spec, ["f0"], {}, geometry=g)

    reopened = open_store(tmp_path / "g.zarr")
    assert reopened.root.attrs["z_mm"] == pytest.approx(z_mm.tolist())
    assert reopened.root.attrs["x_mm"] == pytest.approx(x_mm.tolist())


def test_geometry_grid_mismatch_with_spec_is_rejected(tmp_path):
    """A geometry built on one grid but declared under a different spec would
    otherwise silently mislabel the store's z_mm/x_mm — catch it at creation."""
    z_mm, x_mm = np.linspace(0.0, 19.9, 9), np.linspace(-19.1, 19.1, 7)
    g = sec.build_sector_geometry(k=16, f_number=1.5, z_mm=z_mm, x_mm=x_mm)
    wrong_spec = StoreSpec(n_frames=1, nz=len(z_mm) + 1, nx=len(x_mm), k=16)
    with pytest.raises(ValueError, match="geometry grid"):
        SectorStore.create(tmp_path / "bad.zarr", wrong_spec, ["f0"], {}, geometry=g)


def test_stored_raw_sums_support_the_k_ladder(tmp_path):
    """The reason the store keeps raw sums: reducing K then normalizing must
    equal normalizing the merged sectors — which fails if the store had
    normalized at build time (PLAN.md 3.2 / WI-5)."""
    z_mm, x_mm = np.linspace(8.0, 40.0, 6), np.linspace(-4.0, 4.0, 5)
    g = sec.build_sector_geometry(k=16, f_number=1.5, z_mm=z_mm, x_mm=x_mm)
    spec = StoreSpec(n_frames=1, nz=len(z_mm), nx=len(x_mm), k=16)
    s = SectorStore.create(tmp_path / "l.zarr", spec, ["f0"], {}, geometry=g)

    rng = np.random.default_rng(3)
    frame = rng.normal(size=spec.frame_shape).astype(np.float32)
    s.write_frame(0, frame)

    raw = s.read_frame(0)  # (2, Z, X, K)
    occ = s.occupancy  # (Z, X, K)
    reduced = transforms.reduce_and_normalize(raw, occ[None], target_k=4, axis=-1)
    assert reduced.shape == (2, len(z_mm), len(x_mm), 4)

    # Equivalent to summing raw sectors in groups of four, then dividing once.
    manual_sums = raw.reshape(2, len(z_mm), len(x_mm), 4, 4).sum(-1)
    manual_occ = occ.reshape(len(z_mm), len(x_mm), 4, 4).sum(-1)[None]
    expected = np.where(manual_occ > 0, manual_sums / np.where(manual_occ > 0, manual_occ, 1), 0.0)
    assert reduced == pytest.approx(expected, abs=1e-5)


def test_frame_metadata_sidecar_roundtrip(store):
    meta = {"frame0": {"sos_mean": 1480.0, "segmentation_empty": False}}
    store.write_frame_metadata(meta)
    assert store.read_frame_metadata() == meta


def test_missing_metadata_sidecar_reads_as_empty(store):
    assert store.read_frame_metadata() == {}
