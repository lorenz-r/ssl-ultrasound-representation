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
from ssl_us.data.dataset import LoaderConfig, SectorCropDataset


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


# ── raw-crop contract ────────────────────────────────────────────────────────


def test_ships_whole_frames_not_crops(tiny_store):
    """The host side is deliberately just decompress-and-hand-over: cropping,
    occupancy lookup and augmentation are all device-side. Shipping the 29.6 MB
    frame rather than the 134 MB of crops drawn from it is the whole reason the
    GPU can be kept fed on a 4-core instance."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crop_size=128, crops_per_frame=2, cache_frames=2)
    frame = next(iter(SectorCropDataset(cfg)))
    assert frame.shape == (2, geo.BMODE_NZ, geo.BMODE_NX, 16)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()


def test_every_frame_is_shipped_exactly_once_per_epoch(tiny_store):
    """A frame is transferred once and then yields its whole quota from device
    memory; re-shipping it would throw away the entire point of the cache."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2)
    ds = SectorCropDataset(cfg)
    assert len(list(ds.iter_epoch())) == len(ds.indices)


# ── the GPU frame cache: batches must mix frames ─────────────────────────────


def _cache(ds, capacity=None, seed=0):
    """A CPU-device GpuFrameCache — nothing in it is CUDA-specific."""
    import torch

    from ssl_us.data.dataset import GpuFrameCache

    return GpuFrameCache(ds, "cpu", torch.Generator().manual_seed(seed), capacity)


def test_gpu_cache_interleaves_many_frames(tiny_store):
    """VICReg's variance/covariance terms are computed across the batch, so a
    batch's crops must not all come from one frame."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=8, cache_frames=4)
    ds = SectorCropDataset(cfg)
    cache = _cache(ds)
    frames = iter(ds.iter_epoch())
    slots = cache._assign_slots(frames, 16)
    assert len(set(slots.tolist())) >= 3, f"batch drew from too few frames: {set(slots.tolist())}"


def test_gpu_cache_respects_the_per_frame_quota(tiny_store):
    """Each frame contributes exactly `crops_per_frame` crops per epoch, which
    is what makes the step count — and so the LR schedule — deterministic."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=3, cache_frames=2)
    ds = SectorCropDataset(cfg)
    cache = _cache(ds)
    batches = list(cache.iter_epoch(ds.iter_epoch(), batch_size=2))
    total = len(ds.indices) * cfg.crops_per_frame
    assert len(batches) == total // 2
    assert all(b[0].shape[0] == 2 for b in batches)


def test_gpu_cache_fills_batches_larger_than_one_frames_quota(tiny_store):
    """A batch is assembled across evictions, so `cache_frames x
    crops_per_frame` need not exceed `batch_size` for the epoch to complete —
    requiring that would have silently truncated epochs."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=1)
    ds = SectorCropDataset(cfg)
    cache = _cache(ds)
    batches = list(cache.iter_epoch(ds.iter_epoch(), batch_size=4))
    assert len(batches) == (len(ds.indices) * 2) // 4
    assert all(b[0].shape[0] == 4 for b in batches)


def test_gpu_cache_crops_match_slicing_the_frame_directly(tiny_store):
    """The device-side gather is the only thing standing between the store and
    the encoder's input, so it must reproduce a plain NumPy window exactly."""
    import torch

    cfg = LoaderConfig(store_path=tiny_store, k=4, crop_size=64, crops_per_frame=4, cache_frames=2)
    ds = SectorCropDataset(cfg)
    cache = _cache(ds)
    sums, z0, x0 = next(iter(cache.iter_epoch(ds.iter_epoch(), batch_size=8)))
    assert sums.shape == (8, 2, 64, 64, 16)

    # Every crop must equal *some* resident frame's window at that origin.
    resident = [ds.store.read_frame(i) for i in ds.indices]
    for b in range(sums.shape[0]):
        z, x = int(z0[b]), int(x0[b])
        want = [f[:, z : z + 64, x : x + 64, :] for f in resident]
        got = sums[b].numpy()
        assert any(np.array_equal(got, w) for w in want), f"crop {b} matches no frame window"
        assert torch.isfinite(sums[b]).all()


# ── iteration contract ───────────────────────────────────────────────────────


def test_epoch_is_deterministic_and_epochs_differ(tiny_store):
    """Frame *order* must reshuffle per epoch: it sets which frames are
    co-resident, and so which frames get mixed together in a batch."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2, seed=3)
    ds = SectorCropDataset(cfg)
    a = next(ds.iter_epoch(epoch=0))
    b = next(ds.iter_epoch(epoch=0))
    order0 = [f[0, 0, 0, 0] for f in ds.iter_epoch(epoch=0)]
    order1 = [f[0, 0, 0, 0] for f in ds.iter_epoch(epoch=1)]
    np.testing.assert_array_equal(a, b)
    assert order0 != order1, "epoch 1 replayed epoch 0's frame order"
    assert sorted(order0) == sorted(order1), "an epoch dropped or duplicated a frame"


def test_shards_do_not_share_frames(tiny_store):
    """Workers must not decompress the same frames."""
    cfg = LoaderConfig(store_path=tiny_store, k=4, crops_per_frame=2, cache_frames=2)
    ds = SectorCropDataset(cfg)
    a = list(ds.iter_epoch(shard=0, num_shards=2))
    b = list(ds.iter_epoch(shard=1, num_shards=2))
    assert len(a) + len(b) == len(ds.indices)
    assert not np.allclose(a[0], b[0])


def test_device_drawn_origins_obey_the_arm_depth_range(tiny_store):
    """Crop origins are now drawn device-side, so the arm's 8 mm floor (§3.2c)
    has to hold there too — `origin_bounds` is the single definition both the
    NumPy reference draw and the device draw read, and this checks the device
    path actually uses it."""
    cfg = LoaderConfig.for_arm(tiny_store, k=16, crops_per_frame=16, cache_frames=2)
    ds = SectorCropDataset(cfg)
    dz = geo.bmode_dz_mm()
    seen_z = []
    for _sums, z0, x0 in _cache(ds).iter_epoch(ds.iter_epoch(), batch_size=16):
        seen_z.extend(z0.tolist())
        assert int(z0.min()) * dz >= cfg.z_min_mm - 1e-9
        assert int(z0.max()) + cfg.crop_size <= geo.BMODE_NZ
        assert int(x0.min()) >= 0
        assert int(x0.max()) + cfg.crop_size <= geo.BMODE_NX
    # Uniform in z, not collapsed onto one depth.
    assert len(set(seen_z)) > 10, "device draw is not spreading over depth"


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


def test_gathered_occupancy_matches_slicing_the_map_directly(tiny_store):
    """The loader ships ``(z0, x0)`` instead of a 1.05 MB occupancy slice per
    crop, so `occupancy_for` must reproduce that slice *exactly* — this is the
    whole correctness burden of not sending it. Batched, with distinct origins
    per item, since the gather broadcasts two 1-D index vectors and a bug there
    would show up only when the origins differ.
    """
    import torch

    from ssl_us.data.dataset import GpuAugmentor

    cfg = LoaderConfig.for_arm(tiny_store, k=4)
    ds = SectorCropDataset(cfg)
    augmentor = GpuAugmentor(cfg, device="cpu")

    rng = np.random.default_rng(0)
    origins = [ds._sample_origin(rng) for _ in range(8)]
    z0 = torch.tensor([z for z, _ in origins])
    x0 = torch.tensor([x for _, x in origins])

    got = augmentor.occupancy_for(z0, x0)
    s = cfg.crop_size
    assert got.shape == (8, s, s, 16)
    for i, (z, x) in enumerate(origins):
        expected = ds.occupancy[z : z + s, x : x + s, :].astype(np.float32)
        np.testing.assert_array_equal(got[i].numpy(), expected)


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


# ── K-jitter is an ablation, not a default ───────────────────────────────────


def test_k_jitter_is_off_by_default():
    """Enabling it confounds WI-13's K-ablation: it trains the K=4 arm to be
    invariant to coarsening toward K=1, which discards exactly what the
    ablation is trying to measure. On is a documented ablation, not the
    default — see the LoaderConfig comment before flipping this."""
    assert LoaderConfig(store_path="s.zarr", k=4).k_jitter is False
    assert LoaderConfig.for_arm("s.zarr", k=16).k_jitter is False


def test_k_jitter_off_pins_the_ladder_to_the_arm(tiny_store):
    """With jitter off every sample must sit at the arm's K, so the encoder
    never sees a coarsened view."""

    from ssl_us.data.dataset import GpuAugmentor

    cfg = LoaderConfig(store_path=tiny_store, k=4, policy="A", k_jitter=False)
    augmentor = GpuAugmentor(cfg, device="cpu")
    assert augmentor._acfg.k_choices == (4,)

    ds = SectorCropDataset(cfg)
    sums, z0, x0 = next(iter(_cache(ds).iter_epoch(ds.iter_epoch(), batch_size=2)))
    v1, _v2 = augmentor(sums, augmentor.occupancy_for(z0, x0))
    assert v1.shape[1] == cfg.in_channels == 8


def test_k_jitter_on_offers_only_levels_at_or_below_the_arm(tiny_store):
    """An arm of width K cannot be handed a view with more than K sectors."""
    from ssl_us.data.dataset import GpuAugmentor

    cfg = LoaderConfig(store_path=tiny_store, k=4, policy="A", k_jitter=True)
    assert set(GpuAugmentor(cfg, device="cpu")._acfg.k_choices) == {4, 2, 1}
