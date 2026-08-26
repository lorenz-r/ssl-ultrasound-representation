"""Train the I-JEPA arm against the beamformed K-sector store (WI-17).

Same crops, same budget, same eval as WI-14's MAE arm; the objective is the
only thing that moves. The point of the arm is stated in `models/ijepa.py`:
§1.3 and §3.4 measured that *pixel-space* masked reconstruction has almost
nothing to predict on speckle, and I-JEPA is the architecture that follows
from that measurement — it predicts a **representation** of the masked block
instead. It is also the project's only **augmentation-free** trained arm, so it
is the one cell that can say what survives without a hand-designed invariance.

    python scripts/train/ijepa.py --out /mnt/home/benbarkow/checkpoints/wi17_ijepa_k4

Re-running the same command **resumes** from `<out>/last.pt`. Pass `--no-resume`
to start clean.

**Patch size is derived from the crop, not fixed.** The default is
``crop_size // 16``, i.e. a 16x16 = 256-token grid for every K arm: patch 8 at
the K<=4 crop of 128, patch 4 at the K>=8 crop of 64. Two reasons, and they
pull the same way. Block masking on the 8x8 grid `--patch-size 16` would give
is degenerate — a 0.15-scale target block is 2x2 patches and the four of them
plus a 0.85-scale context leave nothing to keep. And §3.4 measured the speckle
ACF at 2 px axial / 4 px lateral half-width on this grid, so patch 8 is ~2-4
correlation lengths and patch 16 is ~4-8; MAE found signal only at patch 4.
Holding the *token count* fixed across K also keeps the sequence length out of
the K comparison.

**Batch 64, not 256, and it is not free.** Measured on a `12g` card (RTX
2080 Ti, 10.6 GiB usable) at K=4, patch 8: **batch 64 peaks at 6.2 GB and runs
~300 crops/s; batch 128 OOMs.** The matched 11.8M-crop budget is therefore
~11 h there — against 1.95 h for the MAE arm, because 256 tokens against a
ViT-S is 4x the attention of `--patch-size 16`'s 64. **Train this on `24g`**,
where the batch can grow. Batch 64 also happens to be the right comparator:
§3.4's patch-4 MAE is the project's only other 256-token run and it used 64
too. The cost is the usual one and is recorded in `run_config.json` — 4x the
optimizer steps of a batch-256 arm for the same number of crops seen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig, RawCropLoader, make_dataloader
from ssl_us.data.splits import outer_split
from ssl_us.models.ijepa import IJEPA, MaskConfig
from ssl_us.train.common import clamp_num_workers
from ssl_us.train.ijepa import IJEPATrainConfig, train_ijepa
from ssl_us.train.vicreg import DimensionalCollapse
from ssl_us.utils.monitoring import TrainingMonitor

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
TOKEN_GRID = 16  # patch size is crop_size // TOKEN_GRID unless --patch-size is given


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=0, help="0 => derive from --crop-budget")
    # Matched to WI-12/WI-14: same number of crops seen, so the three arms
    # differ in objective rather than in how much data they got.
    p.add_argument("--crop-budget", type=float, default=11.8e6, dest="crop_budget")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--patch-size", type=int, default=0, dest="patch_size",
        help="0 => crop_size // 16, i.e. a 256-token grid at every K",
    )
    p.add_argument("--embed-dim", type=int, default=384, dest="embed_dim")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--predictor-dim", type=int, default=192, dest="predictor_dim")
    p.add_argument("--predictor-depth", type=int, default=6, dest="predictor_depth")
    p.add_argument("--predictor-heads", type=int, default=6, dest="predictor_heads")
    p.add_argument("--n-targets", type=int, default=4, dest="n_targets")
    p.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "l2"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--crops-per-frame", type=int, default=64, dest="crops_per_frame")
    p.add_argument("--cache-frames", type=int, default=24, dest="cache_frames")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05, dest="weight_decay")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--ema-base", type=float, default=0.996, dest="ema_base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--log-every-steps", type=int, default=50)
    # Both collapse instruments on by default here — unlike the MAE arm, this
    # objective is minimised by a constant target and nothing forbids it.
    p.add_argument("--collapse-patience", type=int, default=5, dest="collapse_patience")
    p.add_argument("--collapse-grace-epochs", type=int, default=50, dest="collapse_grace_epochs")
    p.add_argument("--min-target-std", type=float, default=1e-3, dest="min_target_std")
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

    # Geometry from for_arm, exactly as WI-12/13/14 do, so this arm sees the
    # crops the other arms saw.
    cfg = LoaderConfig.for_arm(
        args.store, k=args.k, frame_ids=train_ids, seed=args.seed,
        crops_per_frame=args.crops_per_frame, cache_frames=args.cache_frames,
    )
    patch_size = args.patch_size or max(1, cfg.crop_size // TOKEN_GRID)
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

    mask_config = MaskConfig(n_targets=args.n_targets)
    model = IJEPA(
        in_channels=cfg.in_channels,
        img_size=cfg.crop_size,
        patch_size=patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        mask_config=mask_config,
        loss_kind=args.loss,
    )
    trainable = sum(p.numel() for p in model.trainable.parameters())
    print(
        f"[wi17] I-JEPA K={args.k} crop={cfg.crop_size} in_ch={cfg.in_channels} "
        f"patch={patch_size} grid={model.grid}x{model.grid} "
        f"({model.grid ** 2} tokens) targets={args.n_targets} loss={args.loss}\n"
        f"[wi17] {trainable/1e6:.1f}M trainable params | {steps} steps/epoch x {epochs} "
        f"epochs x batch {args.batch_size} = {crops/1e6:.2f}M crops -> {out}",
        flush=True,
    )

    run_config = {
        "wi": 17, "arch": "ijepa", "k": args.k, "crop_size": cfg.crop_size,
        "in_channels": cfg.in_channels, "patch_size": patch_size,
        "embed_dim": args.embed_dim, "depth": args.depth, "heads": args.heads,
        "predictor_dim": args.predictor_dim, "predictor_depth": args.predictor_depth,
        "predictor_heads": args.predictor_heads,
        "n_tokens": model.grid ** 2, "n_targets": args.n_targets, "loss": args.loss,
        "batch_size": args.batch_size, "epochs": epochs, "steps_per_epoch": steps,
        "crops": crops, "lr": args.lr, "weight_decay": args.weight_decay,
        "ema_base": args.ema_base, "seed": args.seed, "n_train_frames": len(train_ids),
        "augmentation": None,  # by design — see the module docstring
    }
    (out / "run_config.json").write_text(json.dumps({**run_config, "status": "running"}, indent=2))
    monitor = TrainingMonitor(
        run_config, enabled=not args.no_wandb, project="ssl-us",
        run_name=f"wi17-ijepa-k{args.k}",
    )

    started = time.monotonic()
    try:
        history = train_ijepa(
            model,
            train_loader,
            IJEPATrainConfig(
                epochs=epochs, lr=args.lr, weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs, device=args.device,
                ema_base=args.ema_base, checkpoint_dir=out,
                checkpoint_every=args.checkpoint_every, resume=not args.no_resume,
                log_every_steps=args.log_every_steps,
                collapse_patience=args.collapse_patience,
                collapse_grace_epochs=args.collapse_grace_epochs,
                min_target_std=args.min_target_std,
            ),
            monitor=monitor,
        )
    except DimensionalCollapse as exc:
        # §5's lesson: a scientific halt and a crash must not be
        # indistinguishable. The history is written out and the run is recorded
        # as collapsed, with exit code 2 rather than Python's uncaught-exception 1.
        hours = (time.monotonic() - started) / 3600.0
        (out / "history.json").write_text(json.dumps(exc.history, indent=2))
        (out / "run_config.json").write_text(json.dumps(
            {**run_config, "status": "collapsed", "hours": hours, "reason": str(exc)}, indent=2
        ))
        print(f"[wi17] HALTED — {exc}", flush=True)
        monitor.finish({"status": "collapsed", "hours": hours})
        return 2
    hours = (time.monotonic() - started) / 3600.0

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "run_config.json").write_text(
        json.dumps({**run_config, "status": "completed", "hours": hours}, indent=2)
    )
    print(
        f"[wi17] completed in {hours:.2f} h | loss {history['loss'][0]:.4f} -> "
        f"{history['loss'][-1]:.4f} | RankMe {history['rankme'][0]:.1f} -> "
        f"{history['rankme'][-1]:.1f} | target_std {history['target_std'][0]:.3f} -> "
        f"{history['target_std'][-1]:.3f}",
        flush=True,
    )
    monitor.finish({"status": "completed", "hours": hours})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
