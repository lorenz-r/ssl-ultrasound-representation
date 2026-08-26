"""Train one VICReg arm (Policy A or B) against the beamformed K-sector store (WI-12).

The primary arm of PLAN.md §5.2 WI-12: ResNet-18 with a `Conv2d(2K, 64, 7, 2, 3)`
stem, 2048-2048-2048 projector, VICReg loss, depth-stratified crops from the
K ≤ 4 @ 128x128 full-depth arm (§3.2c/d). Two runs are specified — Policy A and
Policy B — which is two invocations of this script differing only in `--policy`.

Usage (cluster `24g`, `.venv`):

    python scripts/train/vicreg.py --policy A --epochs 224 --out checkpoints/wi12_A

Re-running the same command **resumes** from `<out>/last.pt` rather than
restarting, which is what makes a 6-hour unattended run survive a killed job.
Pass `--no-resume` to start clean.

Sizing note: PLAN.md §5.4 budgets each run against ~11.8M sampled pairs
("400 epochs x 115 steps x 256"). That step count predates the verified split —
the real WI-10 train split is 830 frames, which at `crops_per_frame=64` and
batch 256 gives ~206 steps/epoch, not 115. Epochs are therefore set from the
*pair budget*, not carried over as a literal 400, so the run matches the
budgeted 5-7 h rather than doubling it. `--pair-budget` does that arithmetic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import GpuAugmentor, LoaderConfig, make_dataloader
from ssl_us.data.splits import outer_split
from ssl_us.models.vicreg import VICReg, build_backbone
from ssl_us.train.common import clamp_num_workers
from ssl_us.train.vicreg import DimensionalCollapse, VICRegTrainConfig, train_vicreg
from ssl_us.utils.monitoring import TrainingMonitor

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"


class AugmentedLoader:
    """Re-iterable, length-aware view of `GpuAugmentor.wrap(loader)`.

    Two reasons this adapter has to exist rather than passing the wrapped
    generator straight to `train_vicreg`:

    1. `wrap()` is a **generator** — iterating it a second time yields nothing,
       so epoch 2 of a multi-epoch run would silently train on zero batches.
       `__iter__` rebuilds it per epoch instead.
    2. `train_vicreg` needs `len()` to size the LR schedule, and a `DataLoader`
       over an `IterableDataset` has no length at all (torch raises TypeError).

    Batches are cut from the *device-side* frame cache, which pools every
    worker's frames, so the length is one floor over the whole epoch's crops —
    unlike the crop-shipping loader it replaced, where each worker batched its
    own shard and `drop_last` cost a partial batch per worker.
    """

    def __init__(self, augmentor: GpuAugmentor, loader, n_frames: int, cfg: LoaderConfig,
                 batch_size: int, num_workers: int):
        self._augmentor = augmentor
        self._loader = loader
        self._batch_size = batch_size
        self._len = (n_frames * cfg.crops_per_frame) // batch_size

    def __iter__(self):
        return self._augmentor.wrap(self._loader, batch_size=self._batch_size)

    def __len__(self) -> int:
        return self._len


def build_arm(args) -> tuple[AugmentedLoader, LoaderConfig, int]:
    store = open_store(args.store)
    frame_ids = np.array(store.frame_ids)
    train_ids = outer_split(frame_ids)["train"]
    if args.limit_frames:
        train_ids = train_ids[: args.limit_frames]

    cfg = LoaderConfig.for_arm(
        args.store,
        k=args.k,
        policy=args.policy,
        frame_ids=train_ids,
        seed=args.seed,
        crops_per_frame=args.crops_per_frame,
        cache_frames=args.cache_frames,
    )
    workers = clamp_num_workers(args.num_workers)
    loader = make_dataloader(
        cfg,
        num_workers=workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=not args.no_pin_memory,
    )
    augmentor = GpuAugmentor(cfg, device=args.device, seed=args.seed)
    return (
        AugmentedLoader(augmentor, loader, len(train_ids), cfg, args.batch_size, workers),
        cfg,
        len(train_ids),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["A", "B"], required=True)
    # WI-18: the missing cell of the (objective x backbone) 2x2 that docs/MEMORY.md
    # §16.7 names. Default is unchanged, so every existing WI-12/WI-13 command
    # reproduces bitwise.
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "vit_s"])
    p.add_argument(
        "--patch-size", type=int, default=0, dest="patch_size",
        help="vit_s only; 0 => crop_size // 16, i.e. a 256-token grid at every K",
    )
    p.add_argument("--k", type=int, default=4, help="receive sectors; K<=4 => 128px crops")
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--out", required=True, help="checkpoint/history directory")
    p.add_argument("--epochs", type=int, default=0, help="0 => derive from --pair-budget")
    p.add_argument("--pair-budget", type=float, default=11.8e6,
                   help="PLAN.md §5.4's per-run sampled-pair budget")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    # Host RAM here is a 24 GB *cgroup* cap, not the 62 GB `free` reports. Now
    # that the loader ships frames rather than crops an in-flight item is
    # 29.6 MB, not 3.15 MB x batch, so the footprint is `workers x prefetch x
    # 29.6 MB` — a few hundred MB, against the ~12.9 GB the crop-shipping
    # loader held at prefetch 4 (which OOM-killed the job before step 1). A
    # deeper queue is now cheap and buys tolerance for slow frames.
    p.add_argument("--prefetch-factor", type=int, default=6)
    # Pinning costs a copy in the main process, which on a 4-core box competes
    # with the workers. At 29.6 MB/frame that is now a small enough copy to be
    # worth the faster H2D; it was not when the item was a 537 MB batch.
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--crops-per-frame", type=int, default=64)
    p.add_argument("--cache-frames", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    # Exposed for WI-18. The 1e-6 default is the ResNet recipe and is kept, so
    # every existing command reproduces; a ViT-S wants ~0.05 and running one at
    # 1e-6 would handicap the arm for reasons that have nothing to do with the
    # backbone question it is being run to answer.
    p.add_argument("--weight-decay", type=float, default=1e-6, dest="weight_decay")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--collapse-patience", type=int, default=10)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # Every step of a run sees the same shapes, so cudnn may as well autotune
    # its algorithm choice once on step 1 and reuse it for the next ~46k.
    torch.backends.cudnn.benchmark = True
    train_loader, cfg, n_frames = build_arm(args)

    steps = len(train_loader)
    epochs = args.epochs or max(1, round(args.pair_budget / (steps * args.batch_size)))
    pairs = epochs * steps * args.batch_size

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[wi12] policy={args.policy} K={args.k} crop={cfg.crop_size} "
        f"z>={cfg.z_min_mm}mm frames={n_frames}\n"
        f"[wi12] {steps} steps/epoch x {epochs} epochs x batch {args.batch_size} "
        f"= {pairs/1e6:.2f}M pairs -> {out}",
        flush=True,
    )

    backbone = build_backbone(
        args.backbone, cfg.in_channels, cfg.crop_size, args.patch_size, seed=args.seed
    )
    model = VICReg(backbone, backbone_dim=backbone.dim, projector_dim=2048)
    print(
        f"[wi12] backbone={args.backbone} dim={backbone.dim} "
        f"{sum(q.numel() for q in model.parameters())/1e6:.1f}M params",
        flush=True,
    )

    run_config = {
        "wi": 12, "policy": args.policy, "k": args.k, "crop_size": cfg.crop_size,
        "z_min_mm": cfg.z_min_mm, "batch_size": args.batch_size, "epochs": epochs,
        "steps_per_epoch": steps, "pairs": pairs, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "crops_per_frame": args.crops_per_frame, "n_train_frames": n_frames,
        "k_jitter": cfg.k_jitter,
        # Read back by `VICRegCheckpointEncoder` to rebuild the right backbone;
        # absent in the pre-WI-18 checkpoints, which default to resnet18.
        "backbone": args.backbone,
        "backbone_dim": backbone.dim,
        "projector_dim": 2048,
        "patch_size": args.patch_size or (cfg.crop_size // 16 if args.backbone == "vit_s" else 0),
    }
    # Written now as well as at the end: a 2-hour unattended run is monitored
    # while it is still going, and the epoch target it is working toward is
    # derived here from the pair budget rather than being a constant anyone
    # watching could look up.
    (out / "run_config.json").write_text(json.dumps({**run_config, "status": "running"}, indent=2))
    monitor = TrainingMonitor(
        run_config,
        enabled=not args.no_wandb,
        project="ssl-us",
        run_name=f"wi12-vicreg-policy{args.policy}-k{args.k}-{args.backbone}",
    )

    train_config = VICRegTrainConfig(
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        checkpoint_dir=out,
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
        collapse_patience=args.collapse_patience,
        log_every_steps=args.log_every_steps,
    )

    started = time.monotonic()
    status = "completed"
    try:
        history = train_vicreg(model, train_loader, train_config, monitor=monitor)
    except DimensionalCollapse as exc:
        history, status = exc.history, "halted_dimensional_collapse"
        print(f"[wi12] HALT: {exc}", flush=True)
    finally:
        hours = (time.monotonic() - started) / 3600.0

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "run_config.json").write_text(
        json.dumps({**run_config, "status": status, "hours": hours}, indent=2)
    )
    if history.get("rankme"):
        print(
            f"[wi12] {status} in {hours:.2f} h | "
            f"loss {history['loss'][0]:.2f} -> {history['loss'][-1]:.2f} | "
            f"RankMe {history['rankme'][0]:.1f} -> {history['rankme'][-1]:.1f}",
            flush=True,
        )
    monitor.finish({"status": status, "hours": hours})
    # Exit 3, not 1, for the deliberate collapse halt. An uncaught exception
    # also exits 1, so the old code made a genuine crash indistinguishable from
    # a scientific finding — and train_vicreg_policies.sh's "rc=1 => don't retry" rule
    # then skipped the arm instead of resuming it. Keep these codes distinct.
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
