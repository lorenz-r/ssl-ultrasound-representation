"""Training loop for the anisotropic dilated conv MAE (`rf_conv_mae.RFConvMAE`).

Separate entry point from `train.py`, which is wired to `SignalMAE`'s patch API
(`model(x, pos_embed) -> (pred, target, mask)`, `unpatchify`). This model is a
dense conv net over `(B, 2, n_fast, n_lat)`, so the two loops share the dataset,
the physics metrics and the wandb monitor but not the step itself.

    python -m ssl_ultrasound_representation.train_rf_mae --hf-repo <user/repo> \
        --epochs 200 --rows-per-frame 8

**Lateral rows, not whole frames.** One frame is
`(192 transducer, 192 scanline, 2, 215 sample)`; folding it for the model yields
192 independent rows of `(2, 215, 192)`, and pushing all of them through a
32-channel stem at full axial resolution is ~1 GB of activations per frame. Each
step therefore samples `rows_per_frame` scanlines at random. That is not a
compromise — it is spec §5's "extract many crops per acquisition", and with ~140
frames it turns one epoch over frames into a much larger effective sample count.

Model selection is gated on the **masked complex correlation** (spec §5), not on
total loss: that is the metric that exposes carrier-reconstruction failure, and
total loss can fall while it stays pinned at zero.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_ultrasound_representation.dataset import build_split_datasets
from ssl_ultrasound_representation.mae.model.rf_conv_mae import (
    RFConvMAE,
    RFMaeConfig,
    fold_frame,
    unfold_frame,
)
from ssl_ultrasound_representation.mae.model.rf_losses import (
    RF_METRICS,
    LossWeights,
    RFMaeLoss,
    compute_metrics,
)
from ssl_ultrasound_representation.mae.model.rf_masking import RFMasker
from ssl_ultrasound_representation.train import (
    default_device,
    normalize_frame,
    param_groups,
    _amp_beneficial,
)
from ssl_ultrasound_representation.utils.monitoring import TrainingMonitor


def sample_rows(
    frames: torch.Tensor, rows: int, lateral: str, generator=None
) -> torch.Tensor:
    """Take `rows` lateral slices out of a frame batch and fold them for the model.

    Slicing *before* folding matters: `fold_frame` on a whole batch copies every
    frame (63 MB each) only to discard most of it.

    Args:
        frames: (B, transducer, scanline, 2, sample)
    Returns:
        (B * rows, 2, n_fast, n_lat)
    """
    # lateral="transducer" keeps the receive aperture as the model's lateral
    # axis, so the axis we subsample is scanlines (and vice versa).
    axis = 2 if lateral == "transducer" else 1
    n = frames.shape[axis]
    idx = torch.randperm(n, generator=generator)[: min(rows, n)]
    sub = frames[:, :, idx] if axis == 2 else frames[:, idx]
    return fold_frame(sub, lateral)


@torch.no_grad()
def reconstruct_frames(
    model, masker, frames: torch.Tensor, lateral: str, device, chunk: int = 16
) -> torch.Tensor:
    """Reconstruct whole frames row-by-row, for the physics metrics.

    Args:
        frames: (N, transducer, scanline, 2, sample) on CPU.
    Returns:
        Same shape and layout, ready for `eval.metrics.compute_all_metrics`.
    """
    model.eval()
    out = []
    for i in range(frames.shape[0]):
        rows = fold_frame(frames[i : i + 1], lateral)
        preds = []
        for j in range(0, rows.shape[0], chunk):
            xb = rows[j : j + chunk].to(device)
            masked, _ = masker(xb)
            preds.append(model(masked).float().cpu())
        out.append(unfold_frame(torch.cat(preds, dim=0), 1, lateral))
    return torch.cat(out, dim=0)


def train(config: dict | None = None) -> dict:
    config = dict(config or {})

    # ── hyperparameters ────────────────────────────────────────────────────
    lr = config.get("lr", 1e-3)  # spec §5: ~1e-3 scale for this conv stack
    lr_schedule = config.get("lr_schedule", "cosine")
    weight_decay = config.get("weight_decay", 0.05)
    epochs = config.get("epochs", 200)
    warmup_epochs = config.get("warmup_epochs", 10)
    batch_size = config.get("batch_size", 1)  # frames per step
    rows_per_frame = config.get("rows_per_frame", 8)  # lateral slices per frame
    grad_clip = config.get("grad_clip", 1.0)
    lateral = config.get("lateral", "transducer")
    device = config.get("device", default_device())
    device_type = torch.device(device).type
    use_amp = config.get("amp", _amp_beneficial(device))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    preload = config.get("preload", True)
    num_workers = config.get("num_workers", 0 if preload else 4)
    try:
        num_workers = min(num_workers, len(os.sched_getaffinity(0)))
    except AttributeError:
        num_workers = min(num_workers, os.cpu_count() or 1)
    vis_every = config.get("vis_every", 10)
    save_checkpoints = config.get("save_checkpoints", True)
    # Loss-family diagnostics logged every epoch beside the optimised loss, each
    # reduced over the masked positions and over the full signal (spec §4 asks
    # for both). Distinct from the `physics/*` metrics, which run on whole frames
    # every `vis_every` epochs.
    metrics = tuple(config.get("metrics", ("mse", "envelope", "correlation")))
    unknown = [m for m in metrics if m not in RF_METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; valid: {list(RF_METRICS)}")
    # Model selection reads `correlation` (spec §5), so compute it even when the
    # caller did not ask for it — otherwise the gate has nothing to read.
    if "correlation" not in metrics:
        metrics = metrics + ("correlation",)

    # ── model / masker / loss ──────────────────────────────────────────────
    model_cfg = RFMaeConfig(
        c0=config.get("c0", 32),
        depths=tuple(config.get("depths", (2, 2, 2))),
        dilations=tuple(config.get("dilations", (1, 2, 4, 8, 16))),
        k_fast=config.get("k_fast", 7),
        k_lat=config.get("k_lat", 3),
        lat_downsample=config.get("lat_downsample", 2),
        fast_downsample=config.get("fast_downsample", 1),
        global_ctx=config.get("global_ctx", "none"),
        l_max=config.get("l_max", 32),
    )
    model = RFConvMAE(model_cfg).to(device)

    masker = RFMasker(
        mask_ratio=config.get("mask_ratio", 0.55),
        l_min=config.get("l_min", 8),
        l_max=model_cfg.l_max,
        p_chan=config.get("p_chan", 0.1),
        share_lateral=config.get("share_lateral", True),
        seed=config.get("mask_seed"),  # set for a frozen pattern (debug/overfit)
    ).to(device)
    # Validation must not move under the model: a fresh mask every epoch makes
    # val curves jitter for reasons that have nothing to do with learning.
    val_masker = RFMasker(
        mask_ratio=config.get("mask_ratio", 0.55),
        l_min=config.get("l_min", 8),
        l_max=model_cfg.l_max,
        p_chan=config.get("p_chan", 0.1),
        share_lateral=config.get("share_lateral", True),
        seed=config.get("val_mask_seed", 1234),
    ).to(device)
    val_masker.mask_token = masker.mask_token  # share the learned token

    weights = LossWeights(
        w_cpx=config.get("w_cpx", 1.0),
        w_env=config.get("w_env", 0.1),
        w_phase=config.get("w_phase", 1.0),
        w_spec=config.get("w_spec", 0.3),
        w_full=config.get("w_full", 0.1),
    )
    criterion = RFMaeLoss(weights)

    print(f"Model: {model.describe()}")
    print(f"Device: {device}  |  AMP: {use_amp}  |  rows/frame: {rows_per_frame}")

    # ── data ───────────────────────────────────────────────────────────────
    hf_repo = config.get("hf_repo")
    data_source = (
        {
            "repo_id": hf_repo,
            "max_frames": config.get("max_frames"),
            "token": config.get("hf_token"),
            "download_workers": config.get("download_workers", 8),
        }
        if hf_repo
        else {
            "local_zip": config.get(
                "data_zip", "data/processed/ultrasonic_dataset.zarr.zip"
            ),
            "local_metadata": config.get(
                "data_metadata", "data/processed/ultrasonic_dataset.metadata.json"
            ),
        }
    )
    overfit = config.get("overfit", False)
    splits, _ = build_split_datasets(
        **data_source,
        split_strategy=config.get("split_strategy", "subject"),
        ratios=config.get("split_ratios", (1.0, 0.0, 0.0) if overfit else (0.85, 0.15, 0.0)),
        transform=normalize_frame,
        focal_zone=config.get("focal_zone", 0),
        preload=preload,
        share_memory=preload and num_workers > 0,
    )
    if overfit:
        splits["val"] = splits["train"]

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", False),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(splits["train"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(splits["val"], shuffle=False, **loader_kwargs)
    print(f"Train frames: {len(splits['train'])}  |  Val frames: {len(splits['val'])}")

    # ── monitoring ─────────────────────────────────────────────────────────
    monitor = TrainingMonitor(
        config={
            "arch": "rf_conv_mae",
            "lr": lr,
            "lr_schedule": lr_schedule,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "grad_clip": grad_clip,
            "batch_size": batch_size,
            "rows_per_frame": rows_per_frame,
            "lateral": lateral,
            "mask_ratio": config.get("mask_ratio", 0.55),
            "l_min": config.get("l_min", 8),
            "l_max": model_cfg.l_max,
            "p_chan": config.get("p_chan", 0.1),
            "metrics": list(metrics),
            **criterion.config(),
            **model.describe(),
        },
        enabled=config.get("use_wandb", True),
        project=config.get("wandb_project", "ssl-ultrasound-mae"),
        run_name=config.get("run_name"),
        beamform_params=config.get("beamform_params", "configs/beamform_params.json"),
        focal_zone=config.get("focal_zone", 0),
    )
    vis_source = splits["val"] if len(splits["val"]) else splits["train"]
    vis_idx = sorted({0, len(vis_source) // 2})[: config.get("n_vis_frames", 2)]
    monitor.set_vis_frames(
        torch.stack([vis_source[i] for i in vis_idx]), [f"vis[{i}]" for i in vis_idx]
    )

    # ── optimiser ──────────────────────────────────────────────────────────
    params = param_groups(model, weight_decay) + [
        {"params": [masker.mask_token], "weight_decay": 0.0}
    ]
    optimizer = AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps, warmup_steps = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if lr_schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    if save_checkpoints:
        Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # ── loop ───────────────────────────────────────────────────────────────
    best_corr = -float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        n_steps = 0
        grad_total = 0.0
        for frames in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = sample_rows(frames, rows_per_frame, lateral).to(device, non_blocking=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                masked, mask = masker(x)
                pred = model(masked)
            loss, parts = criterion(pred.float(), x.float(), mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_total += float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
            for k, v in parts.items():
                key = f"loss_terms/{k}"
                sums[key] = sums.get(key, 0.0) + float(v)
            for k, v in compute_metrics(pred.float(), x.float(), mask, metrics).items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n_steps += 1

        train_log = {f"train/{k}": v / max(n_steps, 1) for k, v in sums.items()}
        train_log["train/grad_norm"] = grad_total / max(n_steps, 1)
        train_log["train/lr"] = scheduler.get_last_lr()[0]

        # ── validation (frozen mask) ───────────────────────────────────────
        model.eval()
        vsums: dict[str, float] = {}
        v_steps = 0
        with torch.no_grad():
            for frames in val_loader:
                x = sample_rows(frames, rows_per_frame, lateral).to(device)
                masked, mask = val_masker(x)
                pred = model(masked)
                loss, parts = criterion(pred.float(), x.float(), mask)
                vsums["loss"] = vsums.get("loss", 0.0) + float(loss)
                for k, v in parts.items():
                    key = f"loss_terms/{k}"
                    vsums[key] = vsums.get(key, 0.0) + float(v)
                for k, v in compute_metrics(pred, x.float(), mask, metrics).items():
                    vsums[k] = vsums.get(k, 0.0) + float(v)
                v_steps += 1
        val_log = {f"val/{k}": v / max(v_steps, 1) for k, v in vsums.items()}
        # Re(rho) on masked positions is exactly `1 - correlation`. Reading it
        # off the logged metric keeps the gated quantity and the plotted one the
        # same number.
        val_corr = 1.0 - val_log.get("val/correlation_masked", float("nan"))

        metric_str = "".join(
            f"  |  {k}: {val_log[f'val/{k}_masked']:.5f}"
            for k in metrics
            if f"val/{k}_masked" in val_log
        )
        print(
            f"Epoch {epoch:3d}  |  "
            f"train loss: {train_log['train/loss']:.6f}  |  "
            f"val loss:   {val_log.get('val/loss', float('nan')):.6f}  |  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
            f"{metric_str}"
        )
        monitor.log_epoch(epoch, {**train_log, **val_log})

        # ── periodic whole-frame physics pass ──────────────────────────────
        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            vis = monitor.vis_frames
            if vis is not None:
                recon = reconstruct_frames(model, val_masker, vis, lateral, device)
                monitor.log_physics_frames(epoch, recon.numpy())
                monitor.log_reconstruction_frames(epoch, recon.numpy())

        # ── checkpoint on complex correlation, not on loss (spec §5) ───────
        if v_steps and val_corr > best_corr:
            best_corr = val_corr
            if save_checkpoints:
                path = "checkpoints/best_rf_conv_mae.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "mask_token": masker.mask_token.detach().cpu(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "loss_weights": criterion.config(),
                        "val_complex_corr_masked": val_corr,
                        "config": config,
                    },
                    path,
                )
                monitor.log_checkpoint(
                    path,
                    name="rf_conv_mae",
                    aliases=["best", f"epoch-{epoch}"],
                    metadata={"epoch": epoch, "val_complex_corr_masked": val_corr},
                    keep_last=1,
                )

    monitor.finish(summary={"best_val_complex_corr_masked": best_corr})
    print(f"Training complete. Best masked complex correlation: {best_corr:.4f}")
    return {"best_val_complex_corr_masked": best_corr}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train the RF conv MAE.")
    p.add_argument("--epochs", type=int)
    p.add_argument("--warmup-epochs", type=int, dest="warmup_epochs")
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int, dest="batch_size", help="frames per step")
    p.add_argument("--rows-per-frame", type=int, dest="rows_per_frame")
    p.add_argument("--lateral", choices=["transducer", "scanline"])
    p.add_argument("--mask-ratio", type=float, dest="mask_ratio")
    p.add_argument("--l-min", type=int, dest="l_min")
    p.add_argument("--l-max", type=int, dest="l_max")
    p.add_argument("--p-chan", type=float, dest="p_chan")
    p.add_argument("--mask-seed", type=int, dest="mask_seed")
    p.add_argument("--global-ctx", choices=["mamba", "attn", "se", "none"], dest="global_ctx")
    p.add_argument("--c0", type=int)
    p.add_argument("--w-cpx", type=float, dest="w_cpx")
    p.add_argument("--w-env", type=float, dest="w_env")
    p.add_argument("--w-phase", type=float, dest="w_phase")
    p.add_argument("--w-spec", type=float, dest="w_spec")
    p.add_argument("--w-full", type=float, dest="w_full")
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=RF_METRICS,
        help="Extra reconstruction metrics logged every epoch (not optimised). "
        "Each is logged over the masked positions and over the full signal: "
        "train/<name>_masked, val/<name>_all, etc. Names and formulas match "
        "mae/model/loss.py, so they compare directly against train.py runs. "
        "Default: mse envelope correlation.",
    )
    p.add_argument("--hf-repo", dest="hf_repo")
    p.add_argument(
        "--hf-token",
        dest="hf_token",
        help="HuggingFace token for the dataset download — authenticated "
        "requests get much higher rate limits, so the per-chunk fetch stops "
        "stalling on 429 backoff. Falls back to $HF_TOKEN / "
        "$HUGGING_FACE_HUB_TOKEN / a cached `huggingface-cli login` when "
        "omitted; prefer those over the flag to keep the token out of shell "
        "history and `ps` output.",
    )
    p.add_argument("--max-frames", type=int, dest="max_frames")
    p.add_argument("--data-zip", dest="data_zip")
    p.add_argument("--data-metadata", dest="data_metadata")
    p.add_argument("--focal-zone", type=int, dest="focal_zone")
    p.add_argument("--device")
    p.add_argument("--run-name", dest="run_name")
    p.add_argument("--vis-every", type=int, dest="vis_every")
    p.add_argument("--overfit", action="store_true", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-preload", action="store_true")

    args = vars(p.parse_args())
    if args.pop("no_wandb", None):
        args["use_wandb"] = False
    if args.pop("no_preload", None):
        args["preload"] = False
    train({k: v for k, v in args.items() if v is not None})


if __name__ == "__main__":
    main()
