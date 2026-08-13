"""Store-backed crop loader for WI-12 (PLAN.md §5.4's frame-cached shuffle buffer).

**Why this is not a plain `Dataset[i] -> crop`.** The store chunks one *frame*
per zarr chunk (§3.7), so reading any single 128×128 crop costs a full-frame
decompress — measured ~30 ms for a `(2, 507, 456, 16)` frame. A naive
crop-indexed `Dataset` would therefore pay 30 ms per crop: at 64 crops/frame ×
923 frames that is ~30 min *per epoch* of pure decompression, for ~400 epochs.
PLAN.md §5.4 flags exactly this ("the binding constraint is host I/O, not GPU")
and prescribes a frame-cached shuffle buffer. That is what this implements.

Two distinct jobs, and the second is the one that is easy to miss:

1. **Amortise decompression** — decompress a frame once, draw many crops from
   it. Turns 30 ms/crop into ~30 ms/`crops_per_frame`.
2. **Keep batches mixed across frames** — this is a *correctness* requirement,
   not a nicety. VICReg's variance and covariance terms are computed across the
   batch, so a batch whose crops all come from one frame has collapsed batch
   statistics and the loss stops meaning what it should. So the buffer holds
   `cache_frames` frames resident *simultaneously* and draws each successive
   crop from a randomly chosen resident frame; a frame is evicted once it has
   yielded its quota and replaced by a fresh one. Crops therefore arrive
   interleaved across ~`cache_frames` different frames at all times.

Memory is `cache_frames × 29.6 MB` per worker (≈1.9 GB at the default 64), so
worker count and cache size trade off against host RAM — see `LoaderConfig`.

**Read order is load-bearing.** Everything downstream goes through
:func:`ssl_us.data.transforms.reduce_and_normalize`: reduce the K-ladder by
pairwise sums *then* divide by occupancy, never the reverse (PLAN.md §3.2 /
WI-5). Policy A augments the **raw** sums before that reduction, which is why
the loader hands it raw crops and lets the policy do the reduction itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..beamform.store import open_store
from . import augment as aug
from . import geometry as geo
from . import transforms as tf


@dataclass
class LoaderConfig:
    """One training arm's data configuration.

    K and crop size are **arm-defining**, not global constants (PLAN.md §3.2d):
    the plan runs `K ≤ 4 @ 128×128` over the full 0–40 mm depth range and
    `K ≥ 8 @ 64×64` restricted to z ≥ 8 mm, because at K ≥ 8 the shallow
    sectors are too sparsely occupied to be meaningful (§3.2c).
    :meth:`for_arm` builds the matching config so the two never drift apart.
    """

    store_path: str | Path
    k: int = 4
    crop_size: int = 128
    z_min_mm: float = 0.0
    z_max_mm: float = geo.BMODE_Z_MAX_MM
    policy: str = "A"  # "A" (anatomy) or "B" (geometry-only)
    crops_per_frame: int = 64
    cache_frames: int = 64
    seed: int = 0
    frame_ids: list[str] | None = None  # restrict to a split (WI-10)
    # How Policy A's random K-reduction is reconciled with the encoder's fixed
    # 2K input channels — see `sector_channel_mode` in the module notes.
    k_jitter: bool = True
    policy_a_config: aug.PolicyAConfig | None = None
    policy_b_config: aug.PolicyBConfig | None = None
    label_keys: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def for_arm(cls, store_path: str | Path, k: int, **kw) -> LoaderConfig:
        """Config for PLAN.md §3.2c/d's two internally-consistent arms."""
        if k >= 8:
            kw.setdefault("crop_size", 64)
            kw.setdefault("z_min_mm", 8.0)
        else:
            kw.setdefault("crop_size", 128)
            kw.setdefault("z_min_mm", 0.0)
        return cls(store_path=store_path, k=k, **kw)

    @property
    def in_channels(self) -> int:
        """Encoder input channels: I/Q × K sectors (`Conv2d(2K, 64, 7, 2, 3)`)."""
        return 2 * self.k


def to_model_input(view: np.ndarray) -> np.ndarray:
    """``(2, H, W, k) -> (2k, H, W)``, I-block then Q-block.

    Channel order is `[I_0..I_{k-1}, Q_0..Q_{k-1}]`. Stated because the stem is
    a plain `Conv2d(2K, ...)` that cannot tell you what its channels mean, and
    any transfer or probe code slicing channels has to agree with this.
    """
    return np.ascontiguousarray(np.moveaxis(view, -1, 1).reshape(-1, *view.shape[1:3]))


def match_sector_channels(view: np.ndarray, k_target: int) -> np.ndarray:
    """Expand a k-sector view back to ``k_target`` channels by repeating sectors.

    Policy A's *primary* augmentation is a random K-reduction (PLAN.md WI-11),
    but the encoder's input width is fixed at 2K for the arm. Repeating each
    surviving sector `k_target/k` times keeps the tensor shape constant while
    preserving what the augmentation actually did — destroy angular resolution.

    Zero-padding was rejected: which channels are zero would encode `k`
    exactly, handing the model a trivial shortcut to identify the augmentation
    rather than be invariant to it.
    """
    k = view.shape[-1]
    if k == k_target:
        return view
    if k_target % k:
        raise ValueError(f"cannot expand k={k} to k_target={k_target}: not a multiple")
    return np.repeat(view, k_target // k, axis=-1)


class FrameCache:
    """`cache_frames` decompressed frames held resident, crops drawn interleaved.

    Kept separate from the dataset so the mixing behaviour can be tested
    directly — that crops really do arrive spread across many frames is the
    property VICReg's batch statistics depend on.
    """

    def __init__(self, store, indices: list[int], cfg: LoaderConfig, rng: np.random.Generator):
        self._store = store
        self._pending = list(indices)
        rng.shuffle(self._pending)
        self._cfg = cfg
        self._rng = rng
        self._slots: list[dict] = []
        while len(self._slots) < min(cfg.cache_frames, len(self._pending) + len(self._slots)):
            if not self._admit():
                break

    def _admit(self) -> bool:
        if not self._pending:
            return False
        idx = self._pending.pop()
        self._slots.append(
            {"idx": idx, "frame": self._store.read_frame(idx), "left": self._cfg.crops_per_frame}
        )
        return True

    def empty(self) -> bool:
        return not self._slots

    def draw(self) -> tuple[int, np.ndarray]:
        """One (frame_index, frame) drawn from a random resident slot."""
        pos = int(self._rng.integers(len(self._slots)))
        slot = self._slots[pos]
        slot["left"] -= 1
        out = (slot["idx"], slot["frame"])
        if slot["left"] <= 0:
            self._slots.pop(pos)
            self._admit()
        return out

    @property
    def resident(self) -> list[int]:
        return [s["idx"] for s in self._slots]


class SectorCropDataset:
    """Iterable of ``(view1, view2)`` crop pairs for VICReg (WI-12).

    Yields NumPy arrays of shape ``(2K, crop, crop)``; the training script wraps
    them in tensors. Implemented as a plain iterable rather than a torch
    `IterableDataset` so it stays importable and testable without torch, and so
    the worker-sharding logic is explicit (`shard`/`num_shards`).
    """

    def __init__(self, config: LoaderConfig):
        self.cfg = config
        self.store = open_store(config.store_path)
        self.occupancy = self.store.occupancy  # (Z, X, 16), frame-invariant
        ids = config.frame_ids if config.frame_ids is not None else self.store.frame_ids
        known = {f: i for i, f in enumerate(self.store.frame_ids)}
        missing = [f for f in ids if f not in known]
        if missing:
            raise KeyError(f"{len(missing)} frame ids are not in the store, e.g. {missing[:3]}")
        self.indices = [known[f] for f in ids]
        self._z_axis, self._x_axis = 1, 2  # within a (2, H, W, k) crop

    # ── crop geometry ───────────────────────────────────────────────────────

    def _sample_origin(self, rng: np.random.Generator) -> tuple[int, int]:
        """Depth-stratified: uniform in **z**, not in image area (§4.5c)."""
        cfg = self.cfg
        dz = geo.bmode_dz_mm()
        size = cfg.crop_size
        z_lo_px = int(np.ceil(cfg.z_min_mm / dz))
        z_hi_px = min(geo.BMODE_NZ, int(np.floor(cfg.z_max_mm / dz))) - size
        if z_hi_px <= z_lo_px:
            raise ValueError(
                f"crop_size={size} leaves no valid depth range in "
                f"[{cfg.z_min_mm}, {cfg.z_max_mm}] mm"
            )
        z0 = int(rng.integers(z_lo_px, z_hi_px + 1))
        x0 = int(rng.integers(0, geo.BMODE_NX - size + 1))
        return z0, x0

    def _view(self, frame: np.ndarray, z0: int, x0: int, rng: np.random.Generator) -> np.ndarray:
        """One augmented view from a raw frame, as ``(2K, H, W)``."""
        cfg = self.cfg
        s = cfg.crop_size
        # Crop FIRST, augment second: Policy A's chain is O(crop), not O(frame).
        sums = frame[:, z0 : z0 + s, x0 : x0 + s, :]
        occ = self.occupancy[z0 : z0 + s, x0 : x0 + s, :]

        if cfg.policy.upper() == "A":
            acfg = cfg.policy_a_config or aug.PolicyAConfig(
                # Without jitter the K-reduction is pinned to the arm's K, so
                # the channel count is constant by construction.
                k_choices=tuple(c for c in (16, 8, 4, 2, 1) if c <= cfg.k) if cfg.k_jitter
                else (cfg.k,)
            )
            view, _k = aug.policy_a(sums, occ, rng, z_axis=self._z_axis, config=acfg)
            view = match_sector_channels(view, cfg.k)
        else:
            # Policy B is geometry-only: reduce/normalize to the arm's K first,
            # then apply flips/jitter to the finished crop.
            view = tf.reduce_and_normalize(sums, occ[None], target_k=cfg.k, axis=-1)
            view = aug.policy_b(
                view,
                rng,
                z_axis=self._z_axis,
                x_axis=self._x_axis,
                config=cfg.policy_b_config,
            )
        return to_model_input(view.astype(np.float32, copy=False))

    # ── iteration ───────────────────────────────────────────────────────────

    def iter_epoch(self, epoch: int = 0, shard: int = 0, num_shards: int = 1):
        """Yield ``(view1, view2)`` for one epoch of this shard's frames."""
        cfg = self.cfg
        # Distinct stream per (seed, epoch, shard): workers must not draw the
        # same crops, and epochs must not repeat the same ones.
        rng = np.random.default_rng([cfg.seed, epoch, shard])
        mine = self.indices[shard::num_shards]
        if not mine:
            return
        cache = FrameCache(self.store, mine, cfg, rng)
        while not cache.empty():
            _idx, frame = cache.draw()
            z0, x0 = self._sample_origin(rng)
            # Both views share the crop location — VICReg's positive pair is
            # two augmentations of the SAME region, not two regions.
            yield self._view(frame, z0, x0, rng), self._view(frame, z0, x0, rng)

    def __iter__(self):
        return self.iter_epoch()

    def __len__(self) -> int:
        """Crops per epoch (each is a view pair)."""
        return len(self.indices) * self.cfg.crops_per_frame


# ── torch integration ───────────────────────────────────────────────────────
#
# Kept at the bottom and imported lazily so the module above stays usable (and
# testable) without torch — the beamforming venv has no torch installed.


def make_dataloader(
    config: LoaderConfig,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
):
    """A torch ``DataLoader`` of ``(view1, view2)`` batches, sharded across workers.

    Each worker gets a disjoint slice of frames and its own RNG stream, so no
    two workers decompress the same frame or emit the same crops. The epoch
    counter is bumped in the *parent* before workers are forked, so every epoch
    reshuffles frame order and crop positions rather than replaying epoch 0 —
    with an `IterableDataset` that does not happen for free.

    `num_workers` trades throughput against host RAM: each worker holds
    `cache_frames` decompressed frames (~29.6 MB each).
    """
    import torch
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info

    dataset = SectorCropDataset(config)

    class _Iterable(IterableDataset):
        def __init__(self):
            super().__init__()
            self.epoch = 0

        def __iter__(self):
            info = get_worker_info()
            shard, num_shards = (info.id, info.num_workers) if info else (0, 1)
            for v1, v2 in dataset.iter_epoch(self.epoch, shard, num_shards):
                yield torch.from_numpy(v1), torch.from_numpy(v2)

    class _EpochDataLoader(DataLoader):
        def __iter__(self):
            # Runs in the parent, before workers are spawned and the dataset is
            # pickled to them, so the bump actually reaches every worker.
            self.dataset.epoch += 1
            return super().__iter__()

    return _EpochDataLoader(
        _Iterable(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
    )
