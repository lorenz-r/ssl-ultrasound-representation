from __future__ import annotations

import math
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
    """Standardize one frame to zero mean / unit std (raw RF dynamic range is huge)."""
    return (t - t.mean()) / (t.std() + 1e-8)


def preprocess_frame(t: torch.Tensor) -> torch.Tensor:
    """(transducer, scanline, focal_zone, iq, sample) -> first focal zone, standardized."""
    return normalize_frame(t[:, :, 0])


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    num_workers = config.get("num_workers", 2)
    grad_clip = config.get("grad_clip", 1.0)
    device = config.get("device", default_device())
    mask_ratio = config.get("mask_ratio", 0.0)  # 0.0 = plain autoencoder
    loss_on = config.get("loss_on", "all")  # "all" keeps every patch supervised
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

    embed_dim = config.get("embed_dim", 128)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SignalMAE(
        n_transducers=n_transducers,
        transducer_window=transducer_window,
        n_scanlines=n_scanlines,
        n_samples=n_samples,
        depth_window=depth_window,
        embed_dim=embed_dim,
        encoder_depth=config.get("encoder_depth", 4),
        encoder_heads=config.get("encoder_heads", 8),
        decoder_embed_dim=config.get("decoder_embed_dim", 64),
        decoder_depth=config.get("decoder_depth", 2),
        decoder_heads=config.get("decoder_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        mask_ratio=mask_ratio,
    ).to(device)

    # Encoder positional encoding over the (scanline, transducer-group, depth-window) grid
    pos_embed = build_grid_positional_encoding(
        (n_scanlines, n_transducer_groups, n_depth_windows), embed_dim
    ).to(device)

    # ------------------------------------------------------------------
    # Data — file-level split (subject split is degenerate with 2 subjects),
    # no test set for now; transform keeps focal zone 0 and standardizes.
    # ------------------------------------------------------------------
    data_source = (
        {"repo_id": hf_repo} if hf_repo
        else {"local_zip": data_zip, "local_metadata": data_metadata}
    )
    splits, meta = build_split_datasets(
        **data_source,
        split_strategy="file",
        ratios=config.get("split_ratios", (0.85, 0.15, 0.0)),
        transform=preprocess_frame,
    )

    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", False),
        drop_last=False,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", False),
    )

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
            "depth_window": depth_window,
            "transducer_window": transducer_window,
            "embed_dim": embed_dim,
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
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norm_sum = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            # batch: (B, transducer, scanline, iq, sample) — preprocessed
            x = batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred, target, mask = model(x, pos_embed)
            loss = calculate_loss(pred, target, mask, loss_on=loss_on)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            grad_norm_sum += grad_norm.item()
            train_batches += 1
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_train_loss = train_loss / max(train_batches, 1)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device, non_blocking=True)
                pred, target, mask = model(x, pos_embed)
                loss = calculate_loss(pred, target, mask, loss_on=loss_on)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else float("nan")

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
                "train/grad_norm": grad_norm_sum / max(train_batches, 1),
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
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "config": config,
                    },
                    "checkpoints/best_mae_checkpoint.pt",
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
    parser.add_argument("--loss-on", choices=["all", "masked"], dest="loss_on")
    parser.add_argument("--depth-window", type=int, dest="depth_window")
    parser.add_argument("--transducer-window", type=int, dest="transducer_window")
    parser.add_argument("--embed-dim", type=int, dest="embed_dim")
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--device")
    parser.add_argument("--hf-repo", dest="hf_repo",
                        help="HF dataset repo id (e.g. user/ssl-ultrasound-rats); "
                             "falls back to local data/processed/ files if omitted.")
    parser.add_argument("--data-zip", dest="data_zip")
    parser.add_argument("--data-metadata", dest="data_metadata")
    parser.add_argument("--beamform-params", dest="beamform_params")
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--vis-every", type=int, dest="vis_every")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true",
                        dest="save_checkpoints")
    args = vars(parser.parse_args())

    config = {k: v for k, v in args.items() if v not in (None, False)}
    if config.pop("no_wandb", None):
        config["use_wandb"] = False
    return config


if __name__ == "__main__":
    train(_cli())
