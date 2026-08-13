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

**Augmentation is not done here — it runs on the GPU.** The loader yields
*raw* crops and :class:`GpuAugmentor` turns each batch into a VICReg view
pair. Profiling the earlier CPU-side version on the cluster's 4-core instances
put ~59% of loader time in Gaussian noise generation and ~20% in the K-ladder,
capping throughput at ~157 pairs/s against the ~650 the training budget needs;
the same policies on the (otherwise idle) GPU reach ~895 pairs/s. See
:mod:`ssl_us.data.augment`, which is the single implementation of both
policies.

**Read order is load-bearing.** Everything downstream reduces the K-ladder by
pairwise sums *then* divides by occupancy, never the reverse (PLAN.md §3.2 /
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
    # Random K-reduction: **off by default, an ablation rather than a default.**
    #
    # PLAN.md WI-11 lists it under Policy A's priority-1 augmentation, but
    # enabling it confounds the very experiment the store exists for. WI-13's
    # K-ablation trains a separate model per arm (K ∈ {4,2,1} @128, {16,8} @64)
    # to ask whether finer receive angular resolution buys anything. Jitter
    # trains the K=4 arm on views coarsened toward K=1 — i.e. explicitly teaches
    # it to be *invariant to angular resolution*. If that works, the K=4 arm
    # discards exactly what distinguishes it from K=1, the ablation reports "K
    # doesn't matter", and that finding is an artifact of the augmentation
    # rather than physics. The arms are not even augmented comparably: at K=1
    # jitter is a no-op, since only one level exists.
    #
    # Policy A does not need it. Its priority-1 augmentation is the random
    # contiguous *sector subset*, which gives the intended positive pair (two
    # receive sub-apertures of the same scatterers) while leaving the surviving
    # sectors at full angular resolution, so channel ↔ angle stays stable.
    #
    # Turning it on is a legitimate ablation ("does angular-resolution
    # invariance help?"), which is how PLAN.md §4.5c already treats the
    # depth-invariance augmentations: measured, never assumed. When on, each
    # sample draws a level ≤ the arm's K and surviving sectors are repeated back
    # up to K — repeating, not zero-padding, so `k` cannot be read off empty
    # channels. Note the repeat produces duplicate channels (a blur plus a
    # half-bin angular shift), and at k=1 a view carries 1/K the angular
    # information of its partner — an aggressive invariance demand for an
    # objective as collapse-prone as VICReg (§4.7).
    k_jitter: bool = False
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
    """Iterable of raw ``(sums, occupancy)`` crops from the store (WI-12).

    Yields NumPy arrays — ``(2, crop, crop, 16)`` raw sector sums and the
    matching ``(crop, crop, 16)`` occupancy — for :class:`GpuAugmentor` to turn
    into view pairs. Implemented as a plain iterable rather than a torch
    `IterableDataset` so the worker-sharding logic stays explicit
    (`shard`/`num_shards`) and testable on its own.
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

    # ── iteration ───────────────────────────────────────────────────────────

    def iter_epoch(self, epoch: int = 0, shard: int = 0, num_shards: int = 1):
        """Yield ``(raw_crop, occupancy_crop)`` for one epoch of this shard.

        **Raw, not augmented.** Augmentation is batched onto the GPU
        (:class:`GpuAugmentor`), so the CPU side does only the part that has to
        happen here — decompress a frame and slice a window out of it. That is
        the whole point of the split: on the cluster's 4-core instances,
        CPU-side augmentation capped throughput at ~157 pairs/s against the
        ~650 the training budget needs.

        The occupancy crop rides along rather than being re-derived on the GPU
        because it depends only on the crop's ``(z0, x0)``, and shipping it is
        simpler than gathering 923 frames' worth of geometry device-side.
        """
        cfg = self.cfg
        # Distinct stream per (seed, epoch, shard): workers must not draw the
        # same crops, and epochs must not repeat the same ones.
        rng = np.random.default_rng([cfg.seed, epoch, shard])
        mine = self.indices[shard::num_shards]
        if not mine:
            return
        s = cfg.crop_size
        cache = FrameCache(self.store, mine, cfg, rng)
        while not cache.empty():
            _idx, frame = cache.draw()
            z0, x0 = self._sample_origin(rng)
            # np.ascontiguousarray so the worker ships a compact buffer rather
            # than a strided view of the whole 29.6 MB frame.
            yield (
                np.ascontiguousarray(frame[:, z0 : z0 + s, x0 : x0 + s, :]),
                np.ascontiguousarray(self.occupancy[z0 : z0 + s, x0 : x0 + s, :].astype(
                    np.float32, copy=False
                )),
            )

    def __iter__(self):
        return self.iter_epoch()

    def __len__(self) -> int:
        """Crops per epoch (each is a view pair)."""
        return len(self.indices) * self.cfg.crops_per_frame


# ── torch integration ───────────────────────────────────────────────────────


class GpuAugmentor:
    """Turns a batch of raw crops into a VICReg view pair, on the GPU.

    Both views come from the *same* raw crop with independent draws — a VICReg
    positive pair is two augmentations of one region, not two regions — and
    the whole batch is augmented in a handful of kernels rather than per crop
    on the CPU.

    Usage wraps a raw loader and preserves `train_vicreg`'s contract (any
    iterable of ``(view1, view2)`` tensor batches)::

        loader = make_dataloader(cfg, batch_size=256)
        for x1, x2 in GpuAugmentor(cfg, device="cuda").wrap(loader):
            ...
    """

    def __init__(self, config: LoaderConfig, device: str = "cuda", seed: int | None = None):
        import torch

        self.cfg = config
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed if seed is None else seed)
        if config.policy.upper() == "A":
            # Without jitter the K-reduction is pinned to the arm's K, so the
            # channel count is constant by construction.
            self._acfg = config.policy_a_config or aug.PolicyAConfig(
                k_choices=(
                    tuple(c for c in (16, 8, 4, 2, 1) if c <= config.k)
                    if config.k_jitter
                    else (config.k,)
                )
            )

    def __call__(self, sums, occ) -> tuple:
        """``(B,2,Z,X,16), (B,Z,X,16) -> two ``(B, 2K, Z, X)`` view batches."""
        import torch

        sums = sums.to(self.device, non_blocking=True)
        occ = occ.to(self.device, non_blocking=True)
        views = []
        for _ in range(2):
            if self.cfg.policy.upper() == "A":
                v, _k = aug.policy_a(
                    sums, occ, self.generator, config=self._acfg, expand_to=self.cfg.k
                )
            else:
                v = tf.reduce_and_normalize_torch(sums, occ, target_k=self.cfg.k)
                v = aug.policy_b(v, self.generator, config=self.cfg.policy_b_config)
            # (B, 2, Z, X, k) -> (B, 2k, Z, X), I-block then Q-block.
            views.append(torch.movedim(v, -1, 2).flatten(1, 2).contiguous())
        return views[0], views[1]

    def wrap(self, loader):
        """Yield augmented ``(view1, view2)`` batches from a raw loader."""
        for sums, occ in loader:
            yield self(sums, occ)


def make_dataloader(
    config: LoaderConfig,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
):
    """A torch ``DataLoader`` of **raw** ``(sums, occupancy)`` crop batches.

    Feed it through :class:`GpuAugmentor` to get view pairs. Each worker gets a
    disjoint slice of frames and its own RNG stream, so no two workers
    decompress the same frame or emit the same crops. The epoch counter is
    bumped in the *parent* before workers are forked, so every epoch reshuffles
    frame order and crop positions rather than replaying epoch 0 — with an
    `IterableDataset` that does not happen for free.

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
            for sums, occ in dataset.iter_epoch(self.epoch, shard, num_shards):
                yield torch.from_numpy(sums), torch.from_numpy(occ)

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
