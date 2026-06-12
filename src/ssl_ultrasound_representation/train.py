from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ssl_ultrasound_representation.mae.model.us_mae import SignalMAE
from src.ssl_ultrasound_representation.mae.model.loss import calculate_loss
from src.ssl_ultrasound_representation.mae.model.pos_encoding import get_sinusoidal_positional_encoding
from src.ssl_ultrasound_representation.dataset import build_split_datasets


def train(config: dict | None = None) -> dict:
    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    lr = config.get("lr", 1.5e-4)
    weight_decay = config.get("weight_decay", 0.05)
    epochs = config.get("epochs", 200)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    mask_ratio = config.get("mask_ratio", 0.75)

    # Data dimensions (from the US dataset)
    depth = 215 * 2        # S = sample × IQ
    n_channels = 192       # C = transducer elements
    n_temporal = 192 * 3   # N = scanline × focal zones
    n_patches = 192        # P = transducer elements (patch_size[1]=1)

    embed_dim = config.get("embed_dim", 128)
    decoder_embed_dim = config.get("decoder_embed_dim", 64)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = SignalMAE(
        original_channels=depth,
        embed_dim=embed_dim,
        encoder_depth=config.get("encoder_depth", 4),
        encoder_heads=config.get("encoder_heads", 8),
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=config.get("decoder_depth", 2),
        decoder_heads=config.get("decoder_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        patch_size=(depth, 1),
        mask_ratio=mask_ratio,
        grid_spatial=n_patches,
        grid_temporal=n_temporal,
    ).to(device)

    # ------------------------------------------------------------------
    # Positional encodings (sinusoidal, precomputed once)
    # ------------------------------------------------------------------
    pos_embed_spatial = get_sinusoidal_positional_encoding(n_patches, embed_dim).to(device)
    pos_embed_temporal = get_sinusoidal_positional_encoding(n_temporal, embed_dim).to(device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    splits, meta = build_split_datasets(
        config.get("hf_path", "benbarkow/test-us-ssl"),
        split_strategy=config.get("split_strategy", "file"),
        cache_dir=config.get("cache_dir", "/tmp/hf_cache"),
    )

    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        splits["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(splits['train'])}  |  Val samples: {len(splits['val'])}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            # dataset returns (B, transducers, scanlines, focal_zones, 2, samples)
            x = batch.to(device, non_blocking=True)
            x = x.permute(0, 5, 4, 1, 2, 3).reshape(x.size(0), depth, n_channels, n_temporal)

            optimizer.zero_grad()

            x_pred, ids_restore, len_keep = model(x, pos_embed_spatial, pos_embed_temporal)

            loss = calculate_loss(x_pred, x, ids_restore, mask_ratio)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix(loss=loss.item())

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
                x = x.permute(0, 5, 4, 1, 2, 3).reshape(x.size(0), depth, n_channels, n_temporal)
                x_pred, ids_restore, len_keep = model(x, pos_embed_spatial, pos_embed_temporal)
                loss = calculate_loss(x_pred, x, ids_restore, mask_ratio)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch {epoch:3d}  |  "
            f"train loss: {avg_train_loss:.6f}  |  "
            f"val loss:   {avg_val_loss:.6f}"
        )

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "config": config,
                },
                "best_mae_checkpoint.pt",
            )

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return {"best_val_loss": best_val_loss}


if __name__ == "__main__":
    train()
