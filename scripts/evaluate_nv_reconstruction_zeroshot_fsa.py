"""Zero-shot NV autoencoder evaluation on the 192-element rat FSA dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ssl_ultrasound_representation.mae.model.pos_encoding import (
    build_grid_positional_encoding,
)
from ssl_ultrasound_representation.mae.model.us_mae import SignalMAE
from ssl_ultrasound_representation.dataset import build_split_datasets
from ssl_ultrasound_representation.utils.beamform import ZoneBeamformer


def normalize_frame(frame: torch.Tensor) -> torch.Tensor:
    """Apply the same per-depth RMS normalization used for NV training."""
    rms = frame.pow(2).mean(dim=(0, 1, 2), keepdim=True).sqrt()
    return frame / (rms + 1e-8)


def build_model(
    checkpoint: dict, device: torch.device
) -> tuple[SignalMAE, torch.Tensor]:
    """Recreate the NV reconstruction model from checkpoint metadata."""
    config = checkpoint.get("config", {})
    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = {
            "n_transducers": 180,
            "transducer_window": int(config.get("transducer_window", 60)),
            "n_scanlines": 180,
            "n_samples": int(config.get("n_samples", 2056)),
            "depth_window": int(config.get("depth_window", 257)),
            "embed_dim": int(config.get("embed_dim", 256)),
            "encoder_depth": int(config.get("encoder_depth", 8)),
            "encoder_heads": int(config.get("encoder_heads", 8)),
            "decoder_embed_dim": int(config.get("decoder_embed_dim", 128)),
            "decoder_depth": int(config.get("decoder_depth", 4)),
            "decoder_heads": int(config.get("decoder_heads", 8)),
            "dim_feedforward": int(config.get("dim_feedforward", 1024)),
            "mask_ratio": float(config.get("mask_ratio", 0.0)),
        }
    else:
        model_config = dict(model_config)
        attention_mode = model_config.pop("attention_mode", "global")
        if attention_mode != "global":
            raise ValueError(
                "zero-shot evaluation only supports global-attention checkpoints"
            )

    model = SignalMAE(**model_config).to(device)
    position_encoding = checkpoint.get(
        "position_encoding", config.get("position_encoding", "legacy_sum")
    )
    pos_embed = build_grid_positional_encoding(
        (
            model.n_scanlines,
            model.n_transducer_groups,
            model.n_depth_windows,
        ),
        model.embed_dim,
        mode=position_encoding,
    ).to(device)
    return model, pos_embed


def frame_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = reconstructed.astype(np.float64) - original.astype(np.float64)
    mse = float(np.mean(diff * diff))
    power = float(np.mean(original.astype(np.float64) ** 2))
    return {
        "iq_mse": mse,
        "iq_mae": float(np.mean(np.abs(diff))),
        "iq_rel_mse": mse / power if power > 0 else float("nan"),
    }


def image_metrics(original_db: np.ndarray, reconstructed_db: np.ndarray) -> dict[str, float]:
    diff = reconstructed_db.astype(np.float64) - original_db.astype(np.float64)
    return {
        "bf_db_mae": float(np.mean(np.abs(diff))),
        "bf_db_rmse": float(np.sqrt(np.mean(diff * diff))),
        "bf_db_p95_abs": float(np.percentile(np.abs(diff), 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-zarr", type=Path, required=True)
    parser.add_argument("--local-metadata", type=Path, required=True)
    parser.add_argument("--beamform-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        choices=("crop_pad", "resample"),
        default="crop_pad",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--fsa-tx-step", type=int, default=8)
    parser.add_argument("--db-range", type=float, default=50.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ssl-ultrasound-mae")
    parser.add_argument("--wandb-group", default="nv-reconstruction-zeroshot-fsa")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def _center_bounds(source: int, target: int) -> tuple[int, int]:
    if source < target:
        raise ValueError(f"cannot crop source size {source} to {target}")
    start = (source - target) // 2
    return start, start + target


def adapt_frame(
    frame: torch.Tensor,
    *,
    target_shape: tuple[int, int, int],
    mode: str,
) -> tuple[torch.Tensor, dict[str, int | tuple[int, ...] | str]]:
    """Map native `(RX, TX, IQ, depth)` data to an NV model input."""
    rx, tx, iq, depth = frame.shape
    target_rx, target_tx, target_depth = target_shape
    if iq != 2:
        raise ValueError(f"expected two IQ channels, got {iq}")

    if mode == "crop_pad":
        rx0, rx1 = _center_bounds(rx, target_rx)
        tx0, tx1 = _center_bounds(tx, target_tx)
        cropped = frame[rx0:rx1, tx0:tx1]
        if depth >= target_depth:
            adapted = cropped[..., :target_depth]
        else:
            adapted = F.pad(cropped, (0, target_depth - depth))
        info = {
            "mode": mode,
            "native_shape": tuple(frame.shape),
            "rx0": rx0,
            "rx1": rx1,
            "tx0": tx0,
            "tx1": tx1,
            "valid_depth": min(depth, target_depth),
        }
        return adapted.contiguous(), info

    # Treat I/Q as independent channels. This is deliberately an ablation:
    # axial interpolation changes carrier phase, while crop_pad preserves every
    # retained complex sample exactly.
    volume = frame.permute(2, 3, 0, 1).unsqueeze(0)
    adapted = F.interpolate(
        volume,
        size=(target_depth, target_rx, target_tx),
        mode="trilinear",
        align_corners=True,
    )[0].permute(2, 3, 0, 1)
    return adapted.contiguous(), {
        "mode": mode,
        "native_shape": tuple(frame.shape),
    }


def restore_frame(
    adapted: torch.Tensor,
    info: dict[str, int | tuple[int, ...] | str],
) -> torch.Tensor:
    """Map an adapted/model output back to the native FSA acquisition grid."""
    native_shape = tuple(int(v) for v in info["native_shape"])
    rx, tx, iq, depth = native_shape
    if info["mode"] == "crop_pad":
        restored = adapted.new_zeros(native_shape)
        rx0, rx1 = int(info["rx0"]), int(info["rx1"])
        tx0, tx1 = int(info["tx0"]), int(info["tx1"])
        valid_depth = int(info["valid_depth"])
        restored[rx0:rx1, tx0:tx1, :, :valid_depth] = adapted[..., :valid_depth]
        return restored

    volume = adapted.permute(2, 3, 0, 1).unsqueeze(0)
    return F.interpolate(
        volume,
        size=(depth, rx, tx),
        mode="trilinear",
        align_corners=True,
    )[0].permute(2, 3, 0, 1).contiguous()


def plot_comparison(
    *,
    original_db: np.ndarray,
    adapter_db: np.ndarray,
    reconstructed_db: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    title: str,
    out_path: Path,
    db_range: float,
    dpi: int,
) -> None:
    extent = [x_grid[0] * 1e3, x_grid[-1] * 1e3, z_grid[-1] * 1e3, z_grid[0] * 1e3]
    error = np.abs(reconstructed_db - original_db)
    panels = (
        (original_db, "Original rat FSA", "gray", -db_range, 0.0),
        (adapter_db, "Adapter round-trip", "gray", -db_range, 0.0),
        (reconstructed_db, "NV reconstruction zero-shot", "gray", -db_range, 0.0),
        (error, "Absolute dB error", "viridis", 0.0, db_range),
    )
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4), constrained_layout=True)
    for ax, (image, label, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            image,
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )
        ax.set_title(label)
        ax.set_xlabel("lateral [mm]")
        ax.set_ylabel("depth [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, pos_embed = build_model(checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint

    splits, metadata = build_split_datasets(
        local_zip=args.local_zarr,
        local_metadata=args.local_metadata,
        split_strategy="file",
        ratios=(0.7, 0.15, 0.15),
        seed=42,
        focal_zone=0,
        transform=normalize_frame,
    )
    dataset = splits[args.split]
    bad = [i for i in args.sample_indices if i < 0 or i >= len(dataset)]
    if bad:
        raise ValueError(f"sample indices outside {args.split} split: {bad}")

    beamformer = ZoneBeamformer(
        args.beamform_params,
        nx=args.nx,
        nz=args.nz,
        tx_step=args.fsa_tx_step,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    image_paths: list[Path] = []
    run = None
    if args.log_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name or f"nv-reconstruction-zeroshot-fsa-{args.adapter}",
            job_type="zero-shot-eval",
            config={
                "checkpoint": str(args.checkpoint),
                "adapter": args.adapter,
                "split": args.split,
                "sample_indices": args.sample_indices,
            },
        )

    print(
        f"NV model input: RX={model.n_transducers}, TX={model.n_scanlines}, "
        f"depth={model.n_samples}; adapter={args.adapter}",
        flush=True,
    )
    with torch.inference_mode():
        for local_idx in args.sample_indices:
            global_idx = int(dataset.indices[local_idx])
            native = dataset[local_idx]
            adapted, info = adapt_frame(
                native,
                target_shape=(
                    model.n_transducers,
                    model.n_scanlines,
                    model.n_samples,
                ),
                mode=args.adapter,
            )
            adapter_roundtrip = restore_frame(adapted, info)

            batch = adapted.unsqueeze(0).to(device)
            predicted, target, mask = model(batch, pos_embed)
            if bool(mask.any()):
                predicted = torch.where(mask.unsqueeze(-1).bool(), predicted, target)
            model_adapted = model.unpatchify(predicted.float().cpu())[0]
            reconstructed = restore_frame(model_adapted, info)

            native_np = native.numpy().astype(np.float32, copy=False)
            adapter_np = adapter_roundtrip.numpy().astype(np.float32, copy=False)
            reconstructed_np = reconstructed.numpy().astype(np.float32, copy=False)
            print(
                f"Beamforming {args.split}[{local_idx}] global={global_idx}",
                flush=True,
            )
            original_db, adapter_db = beamformer.bmode_pair(
                native_np, adapter_np, db_range=args.db_range
            )
            _, reconstructed_db = beamformer.bmode_pair(
                native_np, reconstructed_np, db_range=args.db_range
            )

            adapter_metrics = {
                **frame_metrics(native_np, adapter_np),
                **image_metrics(original_db, adapter_db),
            }
            model_metrics = {
                **frame_metrics(native_np, reconstructed_np),
                **image_metrics(original_db, reconstructed_db),
            }
            frame_meta = metadata["frames"][global_idx]
            out_path = args.out_dir / (
                f"{args.adapter}_{args.split}_{local_idx:04d}_global_{global_idx:04d}.png"
            )
            plot_comparison(
                original_db=original_db,
                adapter_db=adapter_db,
                reconstructed_db=reconstructed_db,
                x_grid=beamformer.x_grid,
                z_grid=beamformer.z_grid,
                title=(
                    f"{frame_meta.get('subject', 'rat')} | "
                    f"{Path(frame_meta.get('relative_path', '')).name} | "
                    f"adapter={args.adapter}"
                ),
                out_path=out_path,
                db_range=args.db_range,
                dpi=args.dpi,
            )
            image_paths.append(out_path)
            row: dict[str, object] = {
                "adapter": args.adapter,
                "split": args.split,
                "local_idx": local_idx,
                "global_idx": global_idx,
                "subject": frame_meta.get("subject", ""),
                "relative_path": frame_meta.get("relative_path", ""),
            }
            row.update({f"adapter_{k}": v for k, v in adapter_metrics.items()})
            row.update({f"model_{k}": v for k, v in model_metrics.items()})
            metric_rows.append(row)
            print(
                f"  adapter relMSE={adapter_metrics['iq_rel_mse']:.4f}, "
                f"BF MAE={adapter_metrics['bf_db_mae']:.3f}; "
                f"model relMSE={model_metrics['iq_rel_mse']:.4f}, "
                f"BF MAE={model_metrics['bf_db_mae']:.3f}",
                flush=True,
            )

    metrics_path = args.out_dir / f"metrics_{args.adapter}.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Wrote {metrics_path}", flush=True)

    if run is not None:
        table = wandb.Table(columns=list(metric_rows[0]))
        for row in metric_rows:
            table.add_data(*(row[key] for key in metric_rows[0]))
        run.log(
            {
                "zero_shot/metrics": table,
                "zero_shot/figures": [wandb.Image(str(path)) for path in image_paths],
                "zero_shot/model_iq_rel_mse_mean": float(
                    np.mean([row["model_iq_rel_mse"] for row in metric_rows])
                ),
                "zero_shot/model_bf_db_mae_mean": float(
                    np.mean([row["model_bf_db_mae"] for row in metric_rows])
                ),
            }
        )
        run.save(str(metrics_path), base_path=str(args.out_dir))
        run.finish()


if __name__ == "__main__":
    main()
