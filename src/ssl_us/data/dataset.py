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

**The cache lives on the GPU, and the host ships whole frames.** This is what
keeps the GPU fed, and it is worth being precise about why, because the obvious
version of the idea is *slower* rather than faster.

A 128×128×16 crop is 2.1 MB and a frame is 29.6 MB, so the 64 crops drawn from
one frame are 134 MB — 4.5× the frame they came from. Shipping frames instead
of crops therefore moves 4.5× fewer bytes, and those bytes are the binding
constraint: on the cluster's 4-core `24g` instance each crop is copied ~4× on
its way to the device (contiguous-ise, worker shm, collate, pin), which capped
the crop-shipping loader at ~800 crops/s while the GPU could absorb ~1260.

The trap is that this only pays off with a cache that *persists across
batches*. Assembling one 256-crop batch out of freshly-shipped frames needs
~32 distinct frames for the mixing in (2) — 947 MB, against the 537 MB the
crops themselves would have cost. Worse, not better. The win comes from
shipping each frame **once** and then drawing its whole 64-crop quota from
device memory over the following ~64 batches: steady-state that is ~4 new
frames (118 MB) per batch instead of 256 crops (537 MB). :class:`GpuFrameCache`
is that resident buffer; the host side degenerates to "decompress a frame and
hand it over", which is the one thing 4 cores can do fast enough.

Memory splits accordingly: `cache_frames × 29.6 MB` of **VRAM** for the
resident buffer (≈950 MB at the default 32), and only the frames in flight
between the workers and the device on the host — `num_workers ×
prefetch_factor × 29.6 MB`, a few hundred MB rather than the many GB the
crop-shipping loader kept in flight.

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


class SectorCropDataset:
    """Iterable of raw ``(2, Z, X, 16)`` **frames** from the store (WI-12).

    Named for what it ultimately serves — the crops — but it ships frames, and
    :class:`GpuFrameCache` does the cropping on the device; see the module
    docstring for why that is the fast direction. This class therefore owns two
    things: the frame stream, and :meth:`origin_bounds`, the single definition
    of where in a frame a crop may start.

    Implemented as a plain iterable rather than a torch `IterableDataset` so
    the worker-sharding logic stays explicit (`shard`/`num_shards`) and
    testable on its own.
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

    def origin_bounds(self) -> tuple[int, int, int, int]:
        """Inclusive ``(z_lo, z_hi, x_lo, x_hi)`` bounds for a crop's origin.

        The arm's depth restriction (§3.2c — K ≥ 8 starts at 8 mm) lives here
        and only here, so the NumPy reference draw and the device-side draw in
        :meth:`GpuFrameCache.draw` cannot disagree about which crops are legal.
        """
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
        return z_lo_px, z_hi_px, 0, geo.BMODE_NX - size

    def _sample_origin(self, rng: np.random.Generator) -> tuple[int, int]:
        """Depth-stratified: uniform in **z**, not in image area (§4.5c)."""
        z_lo, z_hi, x_lo, x_hi = self.origin_bounds()
        return int(rng.integers(z_lo, z_hi + 1)), int(rng.integers(x_lo, x_hi + 1))

    # ── iteration ───────────────────────────────────────────────────────────

    def iter_epoch(self, epoch: int = 0, shard: int = 0, num_shards: int = 1):
        """Yield this shard's decompressed ``(2, Z, X, 16)`` frames, shuffled.

        Deliberately *all* the host does. Cropping, occupancy lookup and
        augmentation are all device-side; the CPU decompresses and hands over,
        which is the only part that cannot move (the store is zarr-chunked one
        frame per chunk, §3.7). See the module docstring for the byte
        accounting that makes shipping frames rather than crops the fast
        direction.

        Each frame is yielded **once**; :class:`GpuFrameCache` draws its
        `crops_per_frame` quota from device memory. Shuffled per epoch so the
        residency window — and therefore which frames are mixed together in a
        batch — differs every epoch.
        """
        # Distinct stream per (seed, epoch, shard): workers must not read the
        # same frames, and epochs must not repeat the same order.
        rng = np.random.default_rng([self.cfg.seed, epoch, shard])
        mine = self.indices[shard::num_shards]
        rng.shuffle(mine)
        for idx in mine:
            yield self.store.read_frame(idx)

    def __iter__(self):
        return self.iter_epoch()

    def __len__(self) -> int:
        """Crops per epoch (each is a view pair)."""
        return len(self.indices) * self.cfg.crops_per_frame


# ── torch integration ───────────────────────────────────────────────────────


class GpuFrameCache:
    """`cache_frames` frames resident **in VRAM**, crops sliced device-side.

    The shuffle buffer of PLAN.md §5.4, moved onto the GPU. It owns the two
    properties the module docstring calls out: a frame is decompressed and
    transferred once and then yields its whole `crops_per_frame` quota (job 1),
    and every batch is drawn across all resident frames rather than from one
    (job 2 — a *correctness* requirement, since VICReg's variance and
    covariance terms are computed across the batch).

    Runs on a CPU device too, which is how it is tested: nothing here is
    CUDA-specific beyond where the buffer happens to live.

    **Layout.** The buffer is ``(cache, Z, X, 2, K)`` — sector axis last, IQ
    second-to-last — rather than the store's ``(2, Z, X, K)``. That puts the
    three *indexed* axes (slot, z, x) first and contiguous, so drawing a batch
    is one advanced-index gather. With the store's layout the IQ axis sits
    between the indexed axes, which forces NumPy/torch's "advanced indices
    separated by a slice" rule and an extra transposing copy of every batch.
    The permute happens once per frame on admission instead.
    """

    def __init__(
        self,
        dataset: SectorCropDataset,
        device,
        generator,
        capacity: int | None = None,
    ):
        import torch

        self.ds = dataset
        cfg = dataset.cfg
        self.cfg = cfg
        self.device = torch.device(device)
        self.generator = generator
        self.capacity = int(capacity or cfg.cache_frames)
        z, x, k = self.ds.occupancy.shape
        self.buffer = torch.empty(
            (self.capacity, z, x, 2, k), dtype=torch.float32, device=self.device
        )
        # Quotas are tiny bookkeeping over <=cache_frames slots, so they stay on
        # the CPU regardless of where the buffer lives — a device round-trip per
        # batch to read them back would cost more than the arithmetic.
        self.quota = torch.zeros(self.capacity, dtype=torch.long)
        self.free: list[int] = list(range(self.capacity))
        self.active: list[int] = []
        self._cpu_gen = torch.Generator().manual_seed(int(cfg.seed) + 977)

    # ── residency ───────────────────────────────────────────────────────────

    def _admit(self, frames) -> bool:
        """Pull one frame from `frames` into a free slot. False if either ran out."""
        import torch

        if not self.free:
            return False
        frame = next(frames, None)
        if frame is None:
            return False
        slot = self.free.pop()
        if not torch.is_tensor(frame):
            frame = torch.from_numpy(frame)
        # (2, Z, X, K) -> (Z, X, 2, K); the copy into the preallocated slot is
        # what makes this a fixed-footprint cache rather than a churn of
        # 29.6 MB device allocations.
        self.buffer[slot].copy_(
            frame.to(self.device, non_blocking=True).permute(1, 2, 0, 3)
        )
        self.quota[slot] = self.cfg.crops_per_frame
        self.active.append(slot)
        return True

    def _evict_exhausted(self) -> None:
        still = [s for s in self.active if self.quota[s] > 0]
        self.free.extend(s for s in self.active if self.quota[s] <= 0)
        self.active = still

    def _allocate(self, n: int):
        """How many crops each active slot contributes to a batch of `n`.

        Uniform over resident frames but capped by each one's remaining quota,
        so a frame yields exactly `crops_per_frame` crops per epoch and the
        step count is deterministic. The loop re-draws only the overflow, which
        settles in one or two passes because overflow is rare until a slot is
        nearly spent.
        """
        import torch

        avail = self.quota[self.active]
        counts = torch.zeros_like(avail)
        need = n
        while need > 0:
            room = avail - counts
            open_slots = torch.nonzero(room > 0, as_tuple=False).flatten()
            if open_slots.numel() == 0:
                break
            pick = open_slots[
                torch.randint(0, open_slots.numel(), (need,), generator=self._cpu_gen)
            ]
            counts = torch.minimum(
                counts + torch.bincount(pick, minlength=avail.numel()), avail
            )
            need = n - int(counts.sum())
        return counts

    # ── drawing ─────────────────────────────────────────────────────────────

    def _assign_slots(self, frames, batch_size: int):
        """Pick which resident frame each of `batch_size` crops comes from.

        Assembles the batch across evictions rather than requiring a full
        batch's worth of quota to be resident at once: draw what the current
        residents can give, retire the spent ones, admit their replacements,
        and keep going. Requiring it up front would deadlock whenever
        `cache_frames × crops_per_frame` is not comfortably above
        `batch_size` — which is exactly the regime the tests run in, and a
        silent truncation of the epoch in production.

        Returns the ``(B,)`` slot index tensor, or None if the frame stream
        ended before the batch could be filled (that partial batch is dropped).
        """
        import torch

        parts, total = [], 0
        while total < batch_size:
            while self.free and self._admit(frames):
                pass
            if not self.active:
                return None
            counts = self._allocate(batch_size - total)
            if int(counts.sum()) == 0:
                return None
            parts.append(
                torch.repeat_interleave(torch.tensor(self.active, dtype=torch.long), counts)
            )
            for slot, taken in zip(self.active, counts.tolist(), strict=True):
                self.quota[slot] -= taken
            total += int(counts.sum())
            self._evict_exhausted()
        return torch.cat(parts)

    def draw(self, frames, batch_size: int):
        """One ``(sums, z0, x0)`` batch: ``(B, 2, crop, crop, K)`` and origins."""
        import torch

        slots = self._assign_slots(frames, batch_size)
        if slots is None:
            return None

        s = self.cfg.crop_size
        z_lo, z_hi, x_lo, x_hi = self.ds.origin_bounds()
        dev, gen = self.device, self.generator
        z0 = torch.randint(z_lo, z_hi + 1, (batch_size,), generator=gen, device=dev)
        x0 = torch.randint(x_lo, x_hi + 1, (batch_size,), generator=gen, device=dev)

        span = torch.arange(s, device=dev)
        zi = (z0.view(-1, 1) + span).unsqueeze(-1)  # (B, crop, 1)
        xi = (x0.view(-1, 1) + span).unsqueeze(1)  # (B, 1, crop)
        crops = self.buffer[slots.to(dev).view(-1, 1, 1), zi, xi]
        # (B, crop, crop, 2, K) -> (B, 2, crop, crop, K), the augmentors' layout.
        return crops.permute(0, 3, 1, 2, 4), z0, x0

    def iter_epoch(self, frames, batch_size: int):
        """Yield ``(sums, z0, x0)`` batches until the frame stream is spent.

        Partial final batches are dropped, matching the old `drop_last=True`
        loader, so every step sees the same batch size — VICReg's variance and
        covariance terms are batch statistics and a short tail batch would
        quietly change them.
        """
        frames = iter(frames)
        while True:
            batch = self.draw(frames, batch_size)
            if batch is None:
                return
            yield batch


class GpuAugmentor:
    """Turns a batch of raw crops into a VICReg view pair, on the GPU.

    Both views come from the *same* raw crop with independent draws — a VICReg
    positive pair is two augmentations of one region, not two regions — and
    the whole batch is augmented in a handful of kernels rather than per crop
    on the CPU.

    Usage wraps a frame loader and preserves `train_vicreg`'s contract (any
    iterable of ``(view1, view2)`` tensor batches)::

        loader = make_dataloader(cfg)
        for x1, x2 in GpuAugmentor(cfg, device="cuda").wrap(loader, batch_size=256):
            ...
    """

    def __init__(self, config: LoaderConfig, device: str = "cuda", seed: int | None = None):
        import torch

        self.cfg = config
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed if seed is None else seed)
        self._occupancy = None  # (Z, X, 16) on `device`; loaded on first use
        self._cache: GpuFrameCache | None = None  # built on first wrap()
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

    def occupancy_for(self, z0, x0):
        """Gather each crop's ``(crop, crop, 16)`` occupancy from its origin.

        The store's occupancy map is frame-invariant, so one ``(Z, X, 16)``
        upload (14.8 MB) serves every crop of every frame for the whole run and
        the loader never has to ship a per-crop copy — see
        :meth:`SectorCropDataset.iter_epoch` for the measured reason that
        matters. Two 1-D index vectors are broadcast into a 2-D window rather
        than looping the batch, so this is a single gather kernel.

        Args:
            z0, x0: ``(B,)`` integer tensors of crop origins.

        Returns:
            ``(B, crop, crop, 16)`` float32 on this augmentor's device.
        """
        import torch

        if self._occupancy is None:
            occ = open_store(self.cfg.store_path).occupancy
            self._occupancy = torch.as_tensor(
                np.ascontiguousarray(occ, dtype=np.float32), device=self.device
            )
        s = self.cfg.crop_size
        z0 = z0.to(self.device, non_blocking=True).view(-1, 1)
        x0 = x0.to(self.device, non_blocking=True).view(-1, 1)
        span = torch.arange(s, device=self.device).view(1, s)
        zi = (z0 + span).unsqueeze(-1)  # (B, crop, 1)
        xi = (x0 + span).unsqueeze(1)  # (B, 1, crop)
        return self._occupancy[zi, xi]  # (B, crop, crop, 16)

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

    def wrap(self, loader, batch_size: int = 256, cache: GpuFrameCache | None = None):
        """Yield augmented ``(view1, view2)`` batches from a **frame** loader.

        `loader` streams whole frames; a :class:`GpuFrameCache` holds them
        resident and cuts the batch's crops out device-side, and occupancy is
        gathered from the crop origins. ``__call__`` still takes an explicit
        occupancy tensor so the augmentation stays callable — and testable —
        without a store or a cache.

        The cache is built once and reused across epochs (it is ~950 MB of
        VRAM; reallocating it every epoch would churn the allocator for
        nothing), so pass `cache` only to supply a pre-built one.
        """
        if cache is None:
            if self._cache is None:
                self._cache = GpuFrameCache(
                    SectorCropDataset(self.cfg), self.device, self.generator
                )
            cache = self._cache
        for sums, z0, x0 in cache.iter_epoch(loader, batch_size):
            yield self(sums, self.occupancy_for(z0, x0))


def make_dataloader(
    config: LoaderConfig,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
):
    """A torch ``DataLoader`` streaming one decompressed **frame** per item.

    Feed it through :meth:`GpuAugmentor.wrap`, which puts a
    :class:`GpuFrameCache` in front of it to cut the crops and batch them
    device-side. Each worker gets a disjoint slice of frames and its own RNG
    stream, so no two workers decompress the same frame. The epoch counter is
    bumped in the *parent* before workers are forked, so every epoch reshuffles
    frame order rather than replaying epoch 0 — with an `IterableDataset` that
    does not happen for free.

    `batch_size=None` disables the DataLoader's own batching: a batch is drawn
    from the *device* cache across many frames, so collating frames here would
    only re-introduce the host-side copy this design exists to avoid. Host RAM
    is therefore just the frames in flight, `num_workers × prefetch_factor ×
    29.6 MB`.
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
            for frame in dataset.iter_epoch(self.epoch, shard, num_shards):
                yield torch.from_numpy(frame)

    class _EpochDataLoader(DataLoader):
        def __iter__(self):
            # Runs in the parent, before workers are spawned and the dataset is
            # pickled to them, so the bump actually reaches every worker.
            self.dataset.epoch += 1
            return super().__iter__()

    return _EpochDataLoader(
        _Iterable(),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 else {}),
    )
