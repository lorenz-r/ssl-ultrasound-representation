from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssl_ultrasound_representation.mae.model.us_mae import SignalMAE
from ssl_ultrasound_representation.mae.model.loss import calculate_loss
from ssl_ultrasound_representation.mae.model.pos_encoding import (
    build_grid_positional_encoding,
)
from ssl_ultrasound_representation.dataset import build_split_datasets
from ssl_ultrasound_representation.utils.monitoring import TrainingMonitor


def normalize_frame(t: torch.Tensor) -> torch.Tensor:
    """Per-depth RMS normalization of one frame `(transducer, scanline, iq, sample)`.

    Raw IQ decays ~2 orders of magnitude from shallow to deep, so a single
    global std lets the shallow echoes own the reconstruction loss and the deep
    field collapses toward zero (black beamform below the near field). Instead
    we divide each depth sample by its RMS computed over (transducer, scanline,
    iq) — one positive scalar per depth, applied to I and Q alike, so the
    across-aperture phase is preserved exactly while every depth contributes
    equally to the loss.
    """
    rms = t.pow(2).mean(dim=(0, 1, 2), keepdim=True).sqrt()  # (1, 1, 1, sample)
    return t / (rms + 1e-8)


# Focal-zone selection happens at the zarr level (focal_zone=0 in
# build_split_datasets); the transform only standardizes.


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _amp_beneficial(device: str) -> bool:
    """
    Whether mixed precision actually speeds things up on this device.

    fp16 is only a win on GPUs with Tensor Cores (Volta sm_70+), which also
    have the flash-attention kernels. On older cards (e.g. Pascal TITAN Xp,
    sm_61) fp16 runs many times slower than fp32, so autocast would be a large
    slowdown. Evaluated at runtime so a heterogeneous cluster picks the right
    setting per job — AMP off on old GPUs, on automatically on newer ones.
    """
    if device != "cuda":
        return False
    return torch.cuda.get_device_capability()[0] >= 7


def train(config: dict | None = None) -> dict:
    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    lr = config.get("lr", 1.5e-4)
    weight_decay = config.get("weight_decay", 0.05)
    epochs = config.get("epochs", 200)
    warmup_epochs = config.get("warmup_epochs", 20)
    batch_size = config.get("batch_size", 2)
    # Preloaded data lives in shared RAM — workers would only add respawn
    # latency at every epoch boundary, so default to 0 in that case.
    preload = config.get("preload", True)
    num_workers = config.get("num_workers", 0 if preload else 4)
    # Don't over-subscribe: more workers than CPUs only adds spawn/contention
    # overhead and triggers DataLoader's "excessive worker" warning. Use the
    # cgroup's CPU allotment (sched_getaffinity) rather than os.cpu_count(),
    # which reports the host's CPUs and over-counts inside a container (e.g. 12
    # host CPUs vs a 4-CPU cgroup) — over-counting also under-clamps workers and
    # invites the cgroup OOM killer, since each forked worker duplicates the
    # preloaded shared cache's bookkeeping on top of a capped memory budget.
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms (e.g. macOS)
        cpu_count = os.cpu_count() or 1
    if num_workers > cpu_count:
        num_workers = cpu_count
    grad_clip = config.get("grad_clip", 1.0)
    device = config.get("device", default_device())
    device_type = torch.device(device).type
    if device == "cuda":
        # Fixed token/patch shapes — let cuDNN lock in the fastest conv algos.
        torch.backends.cudnn.benchmark = True
    # Mixed precision only when the GPU benefits (Tensor Cores, sm_70+);
    # auto-detected per run so the right setting is picked on whatever cluster
    # GPU this job landed on. Override with config["amp"] / --no-amp.
    use_amp = config.get("amp", _amp_beneficial(device))
    if device == "cuda":
        name = torch.cuda.get_device_name()
        cap = ".".join(map(str, torch.cuda.get_device_capability()))
        print(f"GPU: {name} (sm_{cap})  |  AMP: {use_amp}")
    log_every = config.get("log_every", 10)  # progress-bar sync cadence (steps)
    mask_ratio = config.get("mask_ratio", 0.0)  # 0.0 = plain autoencoder
    loss_on = config.get("loss_on", "all")  # "all" keeps every patch supervised
    loss_type = config.get("loss_type", "mse")  # "mse" or "iq" (IQ-aware)
    save_checkpoints = config.get("save_checkpoints", False)

    # Monitoring
    use_wandb = config.get("use_wandb", True)
    wandb_project = config.get("wandb_project", "ssl-ultrasound-mae")
    run_name = config.get("run_name")
    vis_every = config.get("vis_every", 10)  # beamformed snapshot cadence (epochs)
    n_vis_frames = config.get("n_vis_frames", 2)
    beamform_params = config.get("beamform_params", "configs/beamform_params.json")

    # Data source: HF dataset repo (cluster) or local zarr.zip (default)
    hf_repo = config.get("hf_repo")
    max_frames = config.get("max_frames")  # HF only: fetch just the first N frames
    data_zip = config.get("data_zip", "data/processed/ultrasonic_dataset.zarr.zip")
    data_metadata = config.get(
        "data_metadata", "data/processed/ultrasonic_dataset.metadata.json"
    )

    # Data dimensions: zarr layout (frame, transducer, scanline, focal_zone, iq, sample);
    # only the first focal zone is used (selected in preprocess_frame).
    n_transducers = 192
    n_scanlines = 192
    n_samples = 215
    depth_window = config.get("depth_window", 43)
    transducer_window = config.get("transducer_window", 64)
    n_depth_windows = n_samples // depth_window
    n_transducer_groups = n_transducers // transducer_window

    embed_dim = config.get("embed_dim", 256)

    # ------------------------------------------------------------------
    # Model — ViT-S-ish encoder; sized for the 24 GB GPU rather than the
    # laptop. Watch val loss: with ~140 training frames a larger model can
    # overfit, so capacity is config-overridable.
    # ------------------------------------------------------------------
    # Explicit, self-contained model spec — saved into every checkpoint so the
    # eval suite can rebuild the exact architecture (and its positional grid)
    # from the checkpoint alone, without the training config.
    model_config = {
        "n_transducers": n_transducers,
        "transducer_window": transducer_window,
        "n_scanlines": n_scanlines,
        "n_samples": n_samples,
        "depth_window": depth_window,
        "embed_dim": embed_dim,
        "encoder_depth": config.get("encoder_depth", 8),
        "encoder_heads": config.get("encoder_heads", 8),
        "decoder_embed_dim": config.get("decoder_embed_dim", 128),
        "decoder_depth": config.get("decoder_depth", 4),
        "decoder_heads": config.get("decoder_heads", 8),
        "dim_feedforward": config.get("dim_feedforward", 1024),
        "mask_ratio": mask_ratio,
        # Positional encoding only in the decoder (encoder learns content-only
        # embeddings) when False — see --no-pos-in-encoder.
        "pos_in_encoder": config.get("pos_in_encoder", True),
        # "resizeconv" (interpolate+conv, no checkerboard) or legacy "deconv".
        "decoder_head": config.get("decoder_head", "resizeconv"),
    }
    model = SignalMAE(**model_config).to(device)

    # Encoder positional encoding over the (scanline, transducer-group, depth-window) grid
    pos_embed = build_grid_positional_encoding(
        (n_scanlines, n_transducer_groups, n_depth_windows), embed_dim
    ).to(device)

    # ------------------------------------------------------------------
    # Data — file-level split (subject split is degenerate with 2 subjects),
    # no test set for now. Focal zone 0 is sliced at the zarr level and frames
    # are preloaded into shared RAM (see UltrasoundMAEDataset); the transform
    # only standardizes each frame.
    # ------------------------------------------------------------------
    data_source = (
        {
            "repo_id": hf_repo,
            "max_frames": max_frames,
            "token": config.get("hf_token"),
            "download_workers": config.get("download_workers", 8),
        }
        if hf_repo
        else {"local_zip": data_zip, "local_metadata": data_metadata}
    )
    splits, meta = build_split_datasets(
        **data_source,
        split_strategy="file",
        ratios=config.get("split_ratios", (0.85, 0.15, 0.0)),
        transform=normalize_frame,
        focal_zone=config.get("focal_zone", 0),
        preload=preload,
        # Shared memory is only needed when workers fork; it routes through
        # /dev/shm (often capped < RAM), so avoid it in the common num_workers=0
        # preload path to keep peak memory low.
        share_memory=preload and num_workers > 0,
    )

    # Persistent workers avoid respawning (and re-reading the first frame) at
    # every epoch boundary; pointless when num_workers == 0.
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", False),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        splits["train"], shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(splits["val"], shuffle=False, **loader_kwargs)

    print(
        f"Train samples: {len(splits['train'])}  |  Val samples: {len(splits['val'])}"
    )
    print(f"Tokens/frame: {model.num_tokens}  |  patch dim: {model.patch_dim}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Monitoring (wandb + beamformed reconstruction snapshots)
    # ------------------------------------------------------------------
    monitor = TrainingMonitor(
        config={
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "batch_size": batch_size,
            "grad_clip": grad_clip,
            "mask_ratio": mask_ratio,
            "loss_on": loss_on,
            "loss_type": loss_type,
            "depth_window": depth_window,
            "transducer_window": transducer_window,
            "embed_dim": embed_dim,
            "encoder_depth": config.get("encoder_depth", 8),
            "decoder_depth": config.get("decoder_depth", 4),
            "amp": use_amp,
            "num_tokens": model.num_tokens,
            "patch_dim": model.patch_dim,
            "model_params": sum(p.numel() for p in model.parameters()),
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "device": device,
        },
        enabled=use_wandb,
        project=wandb_project,
        run_name=run_name,
        beamform_params=beamform_params,
    )

    # Fixed snapshot frames from the val split (fall back to train if empty)
    vis_source = splits["val"] if len(splits["val"]) > 0 else splits["train"]
    vis_idx = sorted({0, len(vis_source) // 2})[:n_vis_frames]
    monitor.set_vis_frames(
        torch.stack([vis_source[i] for i in vis_idx]),
        [f"val[{i}]" if len(splits["val"]) else f"train[{i}]" for i in vis_idx],
    )

    # ------------------------------------------------------------------
    # Optimiser + LR schedule (linear warmup -> cosine decay, per step)
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    if save_checkpoints:
        Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # Loss configuration shared by the train/val passes. spatial_shape lets the
    # "iq" loss build its optional spectral (envelope-FFT) term per patch.
    loss_kwargs = dict(
        loss_on=loss_on,
        loss_type=loss_type,
        spatial_shape=(transducer_window, depth_window),
        env_weight=config.get("env_weight", 1.0),
        phase_weight=config.get("phase_weight", 0.1),
        spectral_weight=config.get("spectral_weight", 0.0),
    )
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        # Accumulate on-device and sync once per epoch — a per-step .item()
        # would stall the pipeline on a small, launch-bound model.
        train_loss = torch.zeros((), device=device)
        grad_norm_sum = torch.zeros((), device=device)
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            # batch: (B, transducer, scanline, iq, sample) — preprocessed
            x = batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred, target, mask = model(x, pos_embed)
                loss = calculate_loss(pred, target, mask, **loss_kwargs)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # unscale so grad clipping sees true norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.detach()
            grad_norm_sum += grad_norm.detach()
            train_batches += 1
            if train_batches % log_every == 0:
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_train_loss = (train_loss / max(train_batches, 1)).item()
        avg_grad_norm = (grad_norm_sum / max(train_batches, 1)).item()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_loss = torch.zeros((), device=device)
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    pred, target, mask = model(x, pos_embed)
                    loss = calculate_loss(pred, target, mask, **loss_kwargs)
                val_loss += loss.detach()
                val_batches += 1

        avg_val_loss = (val_loss / val_batches).item() if val_batches else float("nan")

        print(
            f"Epoch {epoch:3d}  |  "
            f"train loss: {avg_train_loss:.6f}  |  "
            f"val loss:   {avg_val_loss:.6f}  |  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        monitor.log_epoch(
            epoch,
            {
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": avg_grad_norm,
            },
        )
        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            monitor.log_reconstructions(epoch, model, pos_embed, device)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if val_batches and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoints:
                ckpt_path = "checkpoints/best_mae_checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "config": config,
                        # Self-contained spec for standalone reload (eval suite).
                        "model_config": model_config,
                    },
                    ckpt_path,
                )
                # Version the best checkpoint in wandb so the eval suite can pull
                # it by alias ("best" / "<run>:latest") instead of a local path.
                monitor.log_checkpoint(
                    ckpt_path,
                    aliases=["best", f"epoch-{epoch}"],
                    metadata={
                        "epoch": epoch,
                        "val_loss": avg_val_loss,
                        "train_loss": avg_train_loss,
                        "model_config": model_config,
                    },
                )

    monitor.finish(summary={"best_val_loss": best_val_loss})
    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return {"best_val_loss": best_val_loss}


def _cli() -> dict:
    """Build a config dict from CLI flags (only explicitly passed flags)."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the ultrasound MAE.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup-epochs", type=int, dest="warmup_epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--mask-ratio", type=float, dest="mask_ratio")
    parser.add_argument("--decoder-head",
                        choices=["resizeconv", "pixelshuffle", "deconv"],
                        dest="decoder_head",
                        help="'resizeconv' (interpolate+conv, smooth), "
                             "'pixelshuffle' (sub-pixel conv, sharp + no "
                             "checkerboard), or legacy 'deconv' (transpose conv).")
    parser.add_argument("--loss-on", choices=["all", "masked"], dest="loss_on")
    parser.add_argument("--loss-type", choices=["mse", "iq"], dest="loss_type",
                        help="'mse' (squared complex error) or 'iq' (IQ-aware: "
                             "log-envelope + amplitude-weighted phase coherence).")
    parser.add_argument("--env-weight", type=float, dest="env_weight",
                        help="iq loss: weight of the log-envelope term (default 1).")
    parser.add_argument("--phase-weight", type=float, dest="phase_weight",
                        help="iq loss: weight of the phase-coherence term (default 1).")
    parser.add_argument("--spectral-weight", type=float, dest="spectral_weight",
                        help="iq loss: weight of the envelope-FFT term (default 0=off).")
    parser.add_argument("--depth-window", type=int, dest="depth_window")
    parser.add_argument("--transducer-window", type=int, dest="transducer_window")
    parser.add_argument("--embed-dim", type=int, dest="embed_dim")
    parser.add_argument("--encoder-depth", type=int, dest="encoder_depth")
    parser.add_argument("--decoder-depth", type=int, dest="decoder_depth")
    parser.add_argument("--decoder-embed-dim", type=int, dest="decoder_embed_dim")
    parser.add_argument("--dim-feedforward", type=int, dest="dim_feedforward")
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--device")
    parser.add_argument(
        "--hf-repo",
        dest="hf_repo",
        help="HF dataset repo id (e.g. user/ssl-ultrasound-rats); "
        "falls back to local data/processed/ files if omitted.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        dest="max_frames",
        help="With --hf-repo: download/use only the first N frames "
        "(quick smoke tests without the full ~30 GB fetch).",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        help="HF auth token for the download (higher rate limits). "
        "Falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN / "
        "cached login if omitted.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        dest="download_workers",
        help="Concurrent chunk downloads (default 8). Lower it "
        "(e.g. 4) if the transfer stalls periodically.",
    )
    parser.add_argument("--data-zip", dest="data_zip")
    parser.add_argument("--data-metadata", dest="data_metadata")
    parser.add_argument("--beamform-params", dest="beamform_params")
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--vis-every", type=int, dest="vis_every")
    parser.add_argument("--focal-zone", type=int, dest="focal_zone")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Stream frames from zarr per access instead of "
        "preloading into RAM (use only if RAM-constrained).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (on by default on CUDA).",
    )
    parser.add_argument(
        "--save-checkpoints", action="store_true", dest="save_checkpoints"
    )
    parser.add_argument(
        "--no-pos-in-encoder",
        action="store_true",
        help="Add positional encoding only in the decoder, forcing the encoder "
        "to learn content-only embeddings (reduces position-dominated latents).",
    )
    args = vars(parser.parse_args())

    config = {k: v for k, v in args.items() if v not in (None, False)}
    if config.pop("no_wandb", None):
        config["use_wandb"] = False
    if config.pop("no_preload", None):
        config["preload"] = False
    if config.pop("no_amp", None):
        config["amp"] = False
    if config.pop("no_pos_in_encoder", None):
        config["pos_in_encoder"] = False
    return config


if __name__ == "__main__":
    train(_cli())
