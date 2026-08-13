"""Store-backed crop loader (PLAN.md WI-12 / §5.4's frame-cached shuffle buffer).

The store is built by WI-8 and is 23 GB, so these build a tiny synthetic store
instead: the loader's contract — arm geometry, channel layout, depth
stratification, frame mixing, determinism — does not depend on corpus size.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.beamform.store import SectorStore, StoreSpec
from ssl_us.data import geometry as geo
from ssl_us.data.dataset import (
    FrameCache,
    LoaderConfig,
    SectorCropDataset,
    match_sector_channels,
    to_model_input,
)


@pytest.fixture(scope="module")
def tiny_store(tmp_path_factory):
    """A real zarr store on the dataset's native grid, with few frames."""
    path = tmp_path_factory.mktemp("store") / "tiny.zarr"
    n = 6
    spec = StoreSpec(n_frames=n, k=16)  # native 507x456 so crop geometry is real
    geom = sec.build_sector_geometry(k=16, f_number=1.5)
    store = SectorStore.create(
        path, spec, [f"train/{i}" for i in range(n)], {"k": 16}, geometry=geom
    )
    rng = np.random.default_rng(0)
    for i in range(n):
        # Frame index encoded in the values, so tests can tell frames apart.
        store.write_frame(i, (rng.normal(size=spec.frame_shape) + 10 * i).astype(np.float32))
    return path


# ── arm configuration (PLAN.md §3.2c/d) ──────────────────────────────────────


def test_arm_for_low_k_is_128px_full_depth():
    cfg = LoaderConfig.for_arm("s.zarr", k=4)
    assert (cfg.crop_size, cfg.z_min_mm) == (128, 0.0)
    assert cfg.in_channels == 8


def test_arm_for_high_k_is_64px_and_respects_the_8mm_floor():
    """K >= 8's shallow sectors are too sparsely occupied to be meaningful."""
    cfg = LoaderConfig.for_arm("s.zarr", k=16)
    assert (cfg.crop_size, cfg.z_min_mm) == (64, 8.0)
    assert cfg.in_channels == 32


# ── channel layout ───────────────────────────────────────────────────────────


def test_to_model_input_packs_iq_then_sectors():
    view = np.zeros((2, 4, 5, 3), dtype=np.float32)
    view[0, ..., 2] = 1.0  # I, sector 2
    view[1, ..., 0] = 2.0  # Q, sector 0
    out = to_model_input(view)
    assert out.shape == (6, 4, 5)
    assert out[2].max() == 1.0  # I-block: channels 0..2
    assert out[3].max() == 2.0  # Q-block starts at 3


def test_match_sector_channels_repeats_rather_than_zero_pads():
    """Zero-padding would let the model read off `k` from which channels are
    zero — a shortcut instead of invariance."""
    view = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2)
    out = match_sector_channels(view, 8)
    assert out.shape == (2, 2, 2, 8)
    assert np.all(out != 0) or np.count_nonzero(out) >= np.count_nonzero(view)
    np.testing.assert_array_equal(out[..., :4], np.repeat(view[..., :1], 4, axis=-1))


def test_match_sector_channels_rejects_non_multiple():
    with pytest.raises(ValueError, match="not a multiple"):
        match_sector_channels(np.zeros((2, 2, 2, 3)), 8)


# ── the frame cache: batches must mix frames ─────────────────────────────────


def test_frame_cache_interleaves_many_frames(tiny_store):
    """VICReg's variance/covariance terms are computed across the batch, so
    consecutive crops must not all come from one frame."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=8, cache_frames=4)
    ds = SectorCropDataset(cfg)
    rng = np.random.default_rng(0)
    cache = FrameCache(ds.store, ds.indices, cfg, rng)
    drawn = [cache.draw()[0] for _ in range(16)]
    assert len(set(drawn)) >= 3, f"crops came from too few frames: {set(drawn)}"


def test_frame_cache_respects_the_per_frame_quota(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=3, cache_frames=2)
    ds = SectorCropDataset(cfg)
    cache = FrameCache(ds.store, ds.indices, cfg, np.random.default_rng(0))
    counts: dict[int, int] = {}
    while not cache.empty():
        idx, _ = cache.draw()
        counts[idx] = counts.get(idx, 0) + 1
    assert set(counts.values()) == {3}
    assert len(counts) == len(ds.indices)


# ── iteration contract ───────────────────────────────────────────────────────


def test_yields_pairs_of_the_right_shape(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=4, crop_size=128, crops_per_frame=2, cache_frames=2)
    ds = SectorCropDataset(cfg)
    v1, v2 = next(iter(ds))
    assert v1.shape == v2.shape == (cfg.in_channels, 128, 128)
    assert v1.dtype == np.float32
    assert np.isfinite(v1).all() and np.isfinite(v2).all()


def test_two_views_of_the_same_crop_differ(tiny_store):
    """Positive pair = two augmentations of one region, not two regions."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2, policy="A")
    v1, v2 = next(iter(SectorCropDataset(cfg)))
    assert not np.allclose(v1, v2)


def test_policy_b_also_produces_arm_width_channels(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=2, crops_per_frame=2, cache_frames=2, policy="B")
    v1, v2 = next(iter(SectorCropDataset(cfg)))
    assert v1.shape[0] == cfg.in_channels == 4


def test_epoch_is_deterministic_and_epochs_differ(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2, seed=3)
    ds = SectorCropDataset(cfg)
    a = next(ds.iter_epoch(epoch=0))[0]
    b = next(ds.iter_epoch(epoch=0))[0]
    c = next(ds.iter_epoch(epoch=1))[0]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c), "epoch 1 replayed epoch 0"


def test_shards_do_not_share_frames(tiny_store):
    """Workers must not decompress the same frames or emit the same crops."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2)
    ds = SectorCropDataset(cfg)
    a = list(ds.iter_epoch(shard=0, num_shards=2))
    b = list(ds.iter_epoch(shard=1, num_shards=2))
    assert len(a) + len(b) == len(ds.indices) * cfg.crops_per_frame
    assert not np.allclose(a[0][0], b[0][0])


def test_crops_stay_inside_the_arm_depth_range(tiny_store):
    """K >= 8 must not sample above 8 mm (§3.2c)."""
    cfg = LoaderConfig.for_arm(tiny_store, k=16, crops_per_frame=4, cache_frames=2)
    ds = SectorCropDataset(cfg)
    rng = np.random.default_rng(0)
    dz = geo.bmode_dz_mm()
    for _ in range(200):
        z0, x0 = ds._sample_origin(rng)
        assert z0 * dz >= cfg.z_min_mm - 1e-9
        assert z0 + cfg.crop_size <= geo.BMODE_NZ
        assert 0 <= x0 <= geo.BMODE_NX - cfg.crop_size


def test_restricting_to_a_split_only_reads_those_frames(tiny_store):
    cfg = LoaderConfig(
        store_path=tiny_store, k=4, crops_per_frame=1, cache_frames=2,
        frame_ids=["train/1", "train/4"],
    )
    ds = SectorCropDataset(cfg)
    assert ds.indices == [1, 4]
    assert len(list(ds.iter_epoch())) == 2


def test_unknown_frame_id_is_rejected(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=4, frame_ids=["train/0", "nope/9"])
    with pytest.raises(KeyError, match="not in the store"):
        SectorCropDataset(cfg)


def test_crop_too_large_for_the_depth_range_raises(tiny_store):
    cfg = LoaderConfig(store_path=tiny_store, k=4, crop_size=128, z_min_mm=38.0)
    ds = SectorCropDataset(cfg)
    with pytest.raises(ValueError, match="no valid depth range"):
        ds._sample_origin(np.random.default_rng(0))
