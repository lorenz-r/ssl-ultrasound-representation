"""WI-14 — train the spatial MAE baseline against the WI-8 store.

PLAN.md WI-14: ViT-S/16 on 128x128 (64 tokens), mask 75 %, **same crops, same
eval** as the WI-12 VICReg arms. It exists because the project's pivot rests on
the claim that MAE underperforms joint embedding on linear probing, and that
claim should be measured on this data rather than cited.

    python scripts/wi14_train.py --out checkpoints/wi14_mae_k4

Re-running the same command **resumes** from `<out>/last.pt`. Pass `--no-resume`
to start clean.

"Same crops" is meant literally: this feeds the *un-augmented* crop that
`GpuAugmentor` would otherwise turn into a view pair. MAE's pretext task is
reconstruction under masking, so a Policy-A/B augmentation on top would change
what is being reconstructed and make the comparison against VICReg a comparison
of two different input distributions. There is therefore **one MAE run per K**,
not one per policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssl_us.beamform.store import open_store
from ssl_us.data import transforms as tf
from ssl_us.data.dataset import (
    GpuAugmentor,
    GpuFrameCache,
    LoaderConfig,
    SectorCropDataset,
    make_dataloader,
)
from ssl_us.data.splits import outer_split
from ssl_us.models.mae2d import MAE2D
from ssl_us.train.common import clamp_num_workers
from ssl_us.train.mae import MAETrainConfig, train_mae
from ssl_us.utils.monitoring import TrainingMonitor

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"


class RawCropLoader:
    """Re-iterable loader of single, un-augmented crops in the model's layout.

    Reuses the WI-12 frame-shipping path — `GpuFrameCache` cuts crops device-
    side from resident frames — and stops short of `GpuAugmentor.__call__`,
    which is the part that makes the two augmented views. What is left is
    exactly `reduce_and_normalize` plus the K-to-channel fold, i.e. the same
    tensor `eval/store_features.py` builds for probing.

    Re-iterable and length-aware for the same two reasons `AggregatedLoader` is
    in wi12_train.py: `iter_epoch` is a generator, so epoch 2 of a multi-epoch
    run would otherwise train on nothing, and the LR schedule needs `len()`.
    """

    def __init__(self, cfg: LoaderConfig, loader, n_frames: int, batch_size: int,
                 device: str, seed: int):
        self._cfg = cfg
        self._loader = loader
        self._batch_size = batch_size
        self._len = (n_frames * cfg.crops_per_frame) // batch_size
        # Borrowed only for its occupancy gather and frame cache; the
        # augmentation entry point is deliberately never called.
        self._aug = GpuAugmentor(cfg, device=device, seed=seed)
        self._cache: GpuFrameCache | None = None

    def __iter__(self):
        if self._cache is None:
            self._cache = GpuFrameCache(
                SectorCropDataset(self._cfg), self._aug.device, self._aug.generator
            )
        for sums, z0, x0 in self._cache.iter_epoch(self._loader, self._batch_size):
            occ = self._aug.occupancy_for(z0, x0)
            v = tf.reduce_and_normalize_torch(sums, occ, target_k=self._cfg.k)
            # (B, 2, Z, X, k) -> (B, 2k, Z, X), I-block then Q-block.
            yield torch.movedim(v, -1, 2).flatten(1, 2).contiguous()

    def __len__(self) -> int:
        return self._len


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=0, help="0 => derive from --crop-budget")
    # VICReg's budget is in *pairs*; MAE sees one crop per sample, so the
    # comparable budget is the same number of encoder forward passes — 11.8M
    # pairs is 23.6M view-encodes, but matching *samples seen* (11.8M) is the
    # convention MAE papers use and keeps wall-clock in PLAN's 2-3 h band.
    p.add_argument("--crop-budget", type=float, default=11.8e6, dest="crop_budget")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mask-ratio", type=float, default=0.75, dest="mask_ratio")
    p.add_argument("--patch-size", type=int, default=16, dest="patch_size")
    # He et al.'s normalized-pixel target, on by default — see mae2d.py for the
    # measured reason (raw targets plateau at 11% variance explained here).
    p.add_argument("--no-norm-pix-loss", action="store_true", dest="no_norm_pix_loss")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--crops-per-frame", type=int, default=64, dest="crops_per_frame")
    p.add_argument("--cache-frames", type=int, default=24, dest="cache_frames")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    store = open_store(args.store)
    train_ids = outer_split(np.array(store.frame_ids))["train"]
    if args.limit_frames:
        train_ids = train_ids[: args.limit_frames]

    # Geometry from for_arm, exactly as WI-12 and WI-13 do, so the MAE arm sees
    # the crops the VICReg arms saw. Hardcoding 128 here would silently break
    # the comparison for K >= 8.
    cfg = LoaderConfig.for_arm(
        args.store, k=args.k, frame_ids=train_ids, seed=args.seed,
        crops_per_frame=args.crops_per_frame, cache_frames=args.cache_frames,
    )
    workers = clamp_num_workers(args.num_workers)
    loader = make_dataloader(cfg, num_workers=workers, prefetch_factor=args.prefetch_factor)
    train_loader = RawCropLoader(
        cfg, loader, len(train_ids), args.batch_size, args.device, args.seed
    )

    steps = len(train_loader)
    epochs = args.epochs or max(1, round(args.crop_budget / (steps * args.batch_size)))
    crops = epochs * steps * args.batch_size

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[wi14] MAE K={args.k} crop={cfg.crop_size} in_ch={cfg.in_channels} "
        f"patch={args.patch_size} mask={args.mask_ratio} frames={len(train_ids)}\n"
        f"[wi14] {steps} steps/epoch x {epochs} epochs x batch {args.batch_size} "
        f"= {crops/1e6:.2f}M crops -> {out}",
        flush=True,
    )

    norm_pix_loss = not args.no_norm_pix_loss
    model = MAE2D(
        in_channels=cfg.in_channels,
        img_size=cfg.crop_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=norm_pix_loss,
    )
    print(f"[wi14] {model.n_tokens} tokens, encode dim {model.dim}", flush=True)

    run_config = {
        "wi": 14, "arch": "mae2d", "k": args.k, "crop_size": cfg.crop_size,
        "in_channels": cfg.in_channels, "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio, "n_tokens": model.n_tokens,
        "norm_pix_loss": norm_pix_loss,
        "batch_size": args.batch_size, "epochs": epochs, "steps_per_epoch": steps,
        "crops": crops, "lr": args.lr, "seed": args.seed,
        "n_train_frames": len(train_ids),
    }
    (out / "run_config.json").write_text(json.dumps({**run_config, "status": "running"}, indent=2))
    monitor = TrainingMonitor(
        run_config, enabled=not args.no_wandb, project="ssl-us",
        run_name=f"wi14-mae-k{args.k}",
    )

    started = time.monotonic()
    history = train_mae(
        model,
        train_loader,
        MAETrainConfig(
            epochs=epochs, lr=args.lr, warmup_epochs=args.warmup_epochs,
            device=args.device, checkpoint_dir=out,
            checkpoint_every=args.checkpoint_every, resume=not args.no_resume,
            log_every_steps=args.log_every_steps,
        ),
        monitor=monitor,
    )
    hours = (time.monotonic() - started) / 3600.0

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "run_config.json").write_text(
        json.dumps({**run_config, "status": "completed", "hours": hours}, indent=2)
    )
    print(
        f"[wi14] completed in {hours:.2f} h | loss {history['loss'][0]:.4f} -> "
        f"{history['loss'][-1]:.4f} | RankMe {history['rankme'][0]:.1f} -> "
        f"{history['rankme'][-1]:.1f}",
        flush=True,
    )
    monitor.finish({"status": "completed", "hours": hours})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
