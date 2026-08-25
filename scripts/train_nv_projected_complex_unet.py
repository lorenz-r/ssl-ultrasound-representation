"""Train a full-depth complex U-Net on NV FSA acquisitions."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from scripts.train_nv_complex_unet import (
    as_spatial_mask,
    load_frame,
    masked_metrics,
    read_indices,
    reciprocal_fill,
    to_raw_layout,
)
from ssl_ultrasound_representation.mae.model.complex_unet import (
    ComplexUNet3d,
    ProjectedComplexUNet2d,
)
from ssl_ultrasound_representation.utils.beamform import ZoneBeamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zarr", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beamform-params")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-project", default="ssl-ultrasound-mae")
    parser.add_argument("--wandb-group", default="nv-projected-complex-unet")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--steps", type=int, default=16600)
    parser.add_argument("--warmup-steps", type=int, default=830)
    parser.add_argument("--frames-per-step", type=int, default=1)
    parser.add_argument(
        "--architecture",
        choices=("projected_complex_unet_2d", "complex_unet_3d"),
        default="projected_complex_unet_2d",
    )
    parser.add_argument("--mask-count", type=int, default=60)
    parser.add_argument(
        "--masking-mode",
        choices=("scanline", "rx_tube"),
        default="scanline",
        help="Mask complete TX rows or complete RX columns across all depths.",
    )
    parser.add_argument(
        "--loss-region",
        choices=("masked", "all"),
        default="masked",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("uniform", "per_depth_relative"),
        default="uniform",
        help="Optionally balance MSE independently at every fast-time sample.",
    )
    parser.add_argument(
        "--normalization",
        choices=("per_frame_rms", "per_depth_rms"),
        help="Defaults to per-frame for 3D and per-depth for projected 2D.",
    )
    parser.add_argument("--depth-embedding-dim", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--validate-every", type=int, default=830)
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb-log-every", type=int, default=20)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("both", "best", "final"),
        default="best",
    )
    parser.add_argument(
        "--validation-frames", type=int, nargs="+", default=[830, 831]
    )
    parser.add_argument("--validation-mask-count", type=int, default=4)
    parser.add_argument("--beamform-tx-step", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_mask(
    masking_mode: str,
    mask_count: int,
    *,
    generator: torch.Generator,
    width: int = 180,
) -> torch.Tensor:
    """Return a TX/RX mask spanning the complete fast-time axis."""
    selected = torch.randperm(width, generator=generator)[:mask_count]
    mask = torch.zeros(width, width, dtype=torch.bool)
    if masking_mode == "scanline":
        mask[selected, :] = True
    elif masking_mode == "rx_tube":
        mask[:, selected] = True
    else:
        raise ValueError(f"unknown masking mode: {masking_mode}")
    return mask


def make_validation_masks(args: argparse.Namespace) -> list[torch.Tensor]:
    return [
        make_mask(
            args.masking_mode,
            args.mask_count,
            generator=torch.Generator().manual_seed(args.seed + 100_000 + 7919 * i),
        )
        for i in range(args.validation_mask_count)
    ]


def load_volume_frame(
    array: zarr.Array, index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a frame with one scale factor, preserving axial amplitude structure."""
    raw = np.asarray(array[index, :, :, 0, :, :], dtype=np.float32)
    frame = torch.from_numpy(raw).permute(3, 2, 1, 0).contiguous()
    rms_scalar = frame.square().mean().sqrt().clamp_min(1e-8)
    rms = rms_scalar.expand(frame.shape[0]).clone()
    return frame / rms_scalar, rms


def model_inputs(
    frame: torch.Tensor,
    masked: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build full-depth [batch, depth, TX, RX] complex model inputs."""
    if frame.ndim == 4:
        frame = frame[None]
    if frame.ndim != 5:
        raise ValueError("frame must be [depth, IQ, TX, RX] or batched")
    spatial_mask = as_spatial_mask(masked, frame.shape[-1]).to(device)
    visibility = (~spatial_mask).to(frame.dtype)[None, None].expand(
        frame.shape[0], -1, -1, -1
    )
    target = frame.to(device, non_blocking=True)
    real = target[:, :, 0] * visibility
    imag = target[:, :, 1] * visibility
    return real, imag, visibility


def prediction_metrics(
    predicted_real: torch.Tensor,
    predicted_imag: torch.Tensor,
    frame: torch.Tensor,
    masked: torch.Tensor,
) -> dict[str, float]:
    device = predicted_real.device
    if frame.ndim == 4:
        frame = frame[None]
    frame = frame.to(device, non_blocking=True)
    spatial_mask = as_spatial_mask(masked, frame.shape[-1]).to(device)
    metric_mask = spatial_mask if spatial_mask.any() else ~spatial_mask
    pred_real = predicted_real[:, :, metric_mask].float()
    pred_imag = predicted_imag[:, :, metric_mask].float()
    target_real = frame[:, :, 0][:, :, metric_mask].float()
    target_imag = frame[:, :, 1][:, :, metric_mask].float()
    error = (pred_real - target_real).square() + (pred_imag - target_imag).square()
    depth_error = error.sum(dim=-1)
    depth_power = (target_real.square() + target_imag.square()).sum(dim=-1)
    target_power = target_real.square().sum() + target_imag.square().sum()
    pred_power = pred_real.square().sum() + pred_imag.square().sum()
    cross_real = (pred_real * target_real + pred_imag * target_imag).sum()
    cross_imag = (pred_imag * target_real - pred_real * target_imag).sum()
    correlation = torch.sqrt(cross_real.square() + cross_imag.square()) / torch.sqrt(
        pred_power * target_power
    ).clamp_min(1e-12)
    return {
        "mse": float(error.mean().detach().cpu()),
        "energy": float((pred_power / target_power.clamp_min(1e-12)).detach().cpu()),
        "correlation": float(correlation.detach().cpu()),
        "phase_error": float(torch.atan2(cross_imag.abs(), cross_real).detach().cpu()),
        "depth_relative_mse": float(
            (depth_error / depth_power.clamp_min(1e-8)).mean().detach().cpu()
        ),
    }


def forward_model(
    model: ProjectedComplexUNet2d | ComplexUNet3d,
    real: torch.Tensor,
    imag: torch.Tensor,
    visibility: torch.Tensor,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if architecture == "projected_complex_unet_2d":
        return model(real, imag, visibility)
    depth = real.shape[1]
    visibility_volume = visibility[:, :, None].expand(-1, -1, depth, -1, -1)
    input_real = torch.cat((real[:, None], visibility_volume), dim=1)
    input_imag = torch.cat((imag[:, None], torch.zeros_like(visibility_volume)), dim=1)
    predicted_real, predicted_imag = model(input_real, input_imag)
    return predicted_real[:, 0], predicted_imag[:, 0]


@torch.inference_mode()
def evaluate(
    model: ProjectedComplexUNet2d | ComplexUNet3d,
    frames: dict[int, torch.Tensor],
    masks: list[torch.Tensor],
    *,
    device: torch.device,
    architecture: str,
) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {
        "mse": [],
        "energy": [],
        "correlation": [],
        "phase_error": [],
        "depth_relative_mse": [],
        "reciprocal_mse": [],
        "reciprocal_energy": [],
        "reciprocal_correlation": [],
    }
    for frame in frames.values():
        for masked in masks:
            real, imag, visibility = model_inputs(frame, masked, device=device)
            predicted_real, predicted_imag = forward_model(
                model, real, imag, visibility, architecture
            )
            metrics = prediction_metrics(predicted_real, predicted_imag, frame, masked)
            for key, value in metrics.items():
                collected[key].append(value)

            spatial_mask = as_spatial_mask(masked, frame.shape[-1])
            metric_mask = spatial_mask if spatial_mask.any() else ~spatial_mask
            reciprocal_real, reciprocal_imag = reciprocal_fill(frame, masked)
            reciprocal = masked_metrics(
                reciprocal_real, reciprocal_imag, frame, metric_mask
            )
            collected["reciprocal_mse"].append(reciprocal["mse"])
            collected["reciprocal_energy"].append(reciprocal["energy"])
            collected["reciprocal_correlation"].append(reciprocal["correlation"])
    return {key: float(np.mean(values)) for key, values in collected.items()}


def checkpoint_payload(
    model: ProjectedComplexUNet2d | ComplexUNet3d,
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, float],
) -> dict:
    return {
        "model": model.state_dict(),
        "step": step,
        "metrics": metrics,
        "architecture": args.architecture,
        "model_config": model_config(args),
        "experiment_config": vars(args),
    }


def model_config(args: argparse.Namespace) -> dict:
    config = {
        "base_channels": args.base_channels,
        "levels": args.levels,
    }
    if args.architecture == "projected_complex_unet_2d":
        config.update(
            {
                "depth_size": 256,
                "embedding_dim": args.depth_embedding_dim,
                "coefficient_wise": True,
            }
        )
    else:
        config.update({"in_channels": 2, "out_channels": 1})
    return config


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def beamform_summary(
    model: ProjectedComplexUNet2d | ComplexUNet3d,
    frame: torch.Tensor,
    rms: torch.Tensor,
    masked: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, float], Path]:
    model.eval()
    spatial_mask = as_spatial_mask(masked, frame.shape[-1])
    metric_mask = spatial_mask if spatial_mask.any() else ~spatial_mask
    real, imag, visibility = model_inputs(frame, masked, device=device)
    predicted_real, predicted_imag = forward_model(
        model, real, imag, visibility, args.architecture
    )
    predicted = torch.stack(
        (predicted_real[0].cpu(), predicted_imag[0].cpu()), dim=1
    )

    visible = frame.clone()
    visible.masked_fill_(metric_mask[None, None], 0)
    predicted_only = torch.zeros_like(frame)
    predicted_only[:, :, metric_mask] = predicted[:, :, metric_mask]
    hybrid = frame.clone()
    hybrid[:, :, metric_mask] = predicted[:, :, metric_mask]

    reciprocal_real, reciprocal_imag = reciprocal_fill(frame, metric_mask)
    reciprocal = frame.clone()
    reciprocal[:, 0][:, metric_mask] = reciprocal_real[:, 0][:, metric_mask]
    reciprocal[:, 1][:, metric_mask] = reciprocal_imag[:, 0][:, metric_mask]

    raw_gt = to_raw_layout(frame, rms)
    decoder_full = torch.stack(
        (predicted_real[0].cpu(), predicted_imag[0].cpu()), dim=1
    )
    raw_views = {
        "visible only": to_raw_layout(visible, rms),
        "predicted masked contribution": to_raw_layout(predicted_only, rms),
        "decoder full": to_raw_layout(decoder_full, rms),
        "U-Net hybrid": to_raw_layout(hybrid, rms),
        "reciprocal hybrid": to_raw_layout(reciprocal, rms),
    }
    beamformer = ZoneBeamformer(
        args.beamform_params,
        nx=96,
        nz=128,
        tx_step=args.beamform_tx_step,
    )
    gt_db, views_db = beamformer.bmode_many(raw_gt, raw_views)
    metrics = {
        f"{name}/bf_mae_db": float(np.mean(np.abs(image - gt_db)))
        for name, image in views_db.items()
    }
    figure, axes = plt.subplots(1, 6, figsize=(25, 4), constrained_layout=True)
    for axis, (name, image) in zip(
        axes, {"ground truth": gt_db, **views_db}.items(), strict=True
    ):
        axis.imshow(image, cmap="gray", vmin=-50, vmax=0, aspect="auto")
        axis.set_title(name)
        axis.set_xlabel("lateral")
        axis.set_ylabel("depth")
    figure.suptitle(
        f"{args.architecture} | frame {args.validation_frames[0]} | "
        f"{args.masking_mode} k={args.mask_count}"
    )
    output_path = output_dir / "beamforming_frame0.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return metrics, output_path


def main() -> None:
    args = parse_args()
    if args.normalization is None:
        args.normalization = (
            "per_depth_rms"
            if args.architecture == "projected_complex_unet_2d"
            else "per_frame_rms"
        )
    if not 0 <= args.mask_count < 180:
        raise ValueError("mask-count must be between 0 and 179")
    if args.loss_region == "masked" and args.mask_count == 0:
        raise ValueError("masked loss requires mask-count greater than zero")
    if args.loss_region == "all" and args.mask_count != 0:
        raise ValueError("all-region reconstruction requires mask-count zero")
    if args.frames_per_step < 1:
        raise ValueError("frames-per-step must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array = zarr.open(args.data_zarr, mode="r", path="rf_iq")
    if tuple(array.shape[1:]) != (180, 180, 1, 2, 256):
        raise ValueError(f"unexpected NV array shape: {array.shape}")
    metadata = json.loads(Path(args.metadata).read_text())
    train_indices = read_indices(args.train_manifest)
    frame_loader = load_frame if args.normalization == "per_depth_rms" else load_volume_frame
    publisher_train = {
        int(frame["global_frame_index"])
        for frame in metadata["frames"]
        if frame.get("split") == "train"
    }
    if not set(train_indices).issubset(publisher_train):
        raise ValueError("training manifest includes publisher-validation frames")

    validation_frames: dict[int, torch.Tensor] = {}
    validation_rms: dict[int, torch.Tensor] = {}
    for index in args.validation_frames:
        validation_frames[index], validation_rms[index] = frame_loader(array, index)
    validation_masks = make_validation_masks(args)
    if args.mask_count == 0:
        validation_masks = [torch.zeros(180, 180, dtype=torch.bool)]
    (output_dir / "validation_masks.json").write_text(
        json.dumps([mask.tolist() for mask in validation_masks]) + "\n"
    )
    (output_dir / "train_global_indices.txt").write_text(
        "\n".join(map(str, train_indices)) + "\n"
    )

    if args.architecture == "projected_complex_unet_2d":
        model = ProjectedComplexUNet2d(
            depth_size=256,
            embedding_dim=args.depth_embedding_dim,
            base_channels=args.base_channels,
            levels=args.levels,
            coefficient_wise=True,
        ).to(device)
    else:
        model = ComplexUNet3d(
            in_channels=2,
            out_channels=1,
            base_channels=args.base_channels,
            levels=args.levels,
        ).to(device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if initial.get("architecture") != args.architecture:
            raise ValueError("initial checkpoint architecture does not match")
        expected_config = model_config(args)
        if initial.get("model_config") != expected_config:
            raise ValueError("initial checkpoint architecture does not match")
        model.load_state_dict(initial["model"])
        print(f"Initialized model from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        f"Model parameters: {model.num_parameters:,} | "
        f"architecture: {args.architecture} | "
        f"GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else device}",
        flush=True,
    )

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.run_name,
            id=args.run_name,
            resume="allow",
            mode=args.wandb_mode,
            dir=str(output_dir / "wandb"),
            config=vars(args),
        )

    best_validation = float("inf")
    selection_metric = (
        "depth_relative_mse"
        if args.loss_weighting == "per_depth_relative"
        else "mse"
    )
    last_metrics: dict[str, float] = {}
    if args.validate_at_start:
        last_metrics = evaluate(
            model,
            validation_frames,
            validation_masks,
            device=device,
            architecture=args.architecture,
        )
        best_validation = last_metrics[selection_metric]
        print(f"validation step=0: {last_metrics}", flush=True)
        if args.checkpoint_policy in ("both", "best"):
            torch.save(
                checkpoint_payload(model, args, 0, last_metrics),
                output_dir / "best.pt",
            )
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in last_metrics.items()}, step=0)

    frame_order: list[int] = []
    running_loss = 0.0
    train_start = time.time()
    for step in range(args.steps):
        if not frame_order:
            frame_order = torch.randperm(
                len(train_indices), generator=generator
            ).tolist()
        batch_frames = []
        for _ in range(args.frames_per_step):
            if not frame_order:
                frame_order = torch.randperm(
                    len(train_indices), generator=generator
                ).tolist()
            frame_index = train_indices[frame_order.pop()]
            frame, _ = frame_loader(array, frame_index)
            batch_frames.append(frame)
        frame = torch.stack(batch_frames)
        masked = make_mask(
            args.masking_mode,
            args.mask_count,
            generator=generator,
            width=180,
        )
        real, imag, visibility = model_inputs(frame, masked, device=device)
        target = frame.to(device, non_blocking=True)
        spatial_mask = as_spatial_mask(masked, 180).to(device)
        loss_mask = (
            torch.ones_like(spatial_mask)
            if args.loss_region == "all"
            else spatial_mask
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted_real, predicted_imag = forward_model(
            model, real, imag, visibility, args.architecture
        )
        error = (
            (predicted_real - target[:, :, 0]).square()
            + (predicted_imag - target[:, :, 1]).square()
        )
        selected_error = error[:, :, loss_mask]
        if args.loss_weighting == "per_depth_relative":
            selected_power = (
                target[:, :, 0][:, :, loss_mask].square()
                + target[:, :, 1][:, :, loss_mask].square()
            )
            loss = (
                selected_error.mean(dim=-1)
                / selected_power.mean(dim=-1).detach().clamp_min(1e-8)
            ).mean()
        else:
            loss = selected_error.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        running_loss += float(loss.detach().cpu())

        train_metrics = prediction_metrics(
            predicted_real.detach(), predicted_imag.detach(), target, masked
        )
        log_data = {
            "train/masked_mse": train_metrics["mse"],
            "train/masked_energy": train_metrics["energy"],
            "train/masked_correlation": train_metrics["correlation"],
            "train/phase_error_rad": train_metrics["phase_error"],
            "train/grad_norm": float(grad_norm.detach().cpu()),
            "train/lr": lr,
            "train/step": step + 1,
            "train/frames_seen": (step + 1) * args.frames_per_step,
            "train/frame_epochs": (
                (step + 1) * args.frames_per_step / len(train_indices)
            ),
            "train/mask_count": args.mask_count,
        }
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - train_start
            print(
                f"step {step + 1:5d}/{args.steps} "
                f"loss={running_loss / args.log_every:.4f} "
                f"corr={train_metrics['correlation']:.3f} "
                f"energy={train_metrics['energy']:.3f} "
                f"{elapsed / (step + 1):.3f}s/step",
                flush=True,
            )
            running_loss = 0.0

        should_validate = (step + 1) % args.validate_every == 0 or step + 1 == args.steps
        if should_validate:
            last_metrics = evaluate(
                model,
                validation_frames,
                validation_masks,
                device=device,
                architecture=args.architecture,
            )
            log_data.update({f"val/{k}": v for k, v in last_metrics.items()})
            print(f"validation step={step + 1}: {last_metrics}", flush=True)
            if last_metrics[selection_metric] < best_validation:
                best_validation = last_metrics[selection_metric]
                if args.checkpoint_policy in ("both", "best"):
                    torch.save(
                        checkpoint_payload(model, args, step + 1, last_metrics),
                        output_dir / "best.pt",
                    )
        if wandb_run is not None and (
            (step + 1) % args.wandb_log_every == 0 or should_validate
        ):
            wandb_run.log(log_data, step=step + 1)

    if args.checkpoint_policy in ("both", "final"):
        torch.save(
            checkpoint_payload(model, args, args.steps, last_metrics),
            output_dir / "final.pt",
        )
    if (output_dir / "best.pt").is_file():
        best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])

    summary = {
        "run_name": args.run_name,
        "architecture": args.architecture,
        "normalization": args.normalization,
        "loss_weighting": args.loss_weighting,
        "masking_mode": args.masking_mode,
        "mask_count": args.mask_count,
        "depth_embedding_dim": (
            args.depth_embedding_dim
            if args.architecture == "projected_complex_unet_2d"
            else None
        ),
        "frames_per_step": args.frames_per_step,
        "frames_seen": args.steps * args.frames_per_step,
        "frame_epochs": args.steps * args.frames_per_step / len(train_indices),
        "steps": args.steps,
        "parameters": model.num_parameters,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        ),
        "elapsed_seconds": time.time() - train_start,
        "best_validation_mse": best_validation,
        "best_validation_metric": selection_metric,
        **{f"validation/{key}": value for key, value in last_metrics.items()},
    }
    if args.beamform_params:
        first_index = args.validation_frames[0]
        beam_metrics, image_path = beamform_summary(
            model,
            validation_frames[first_index],
            validation_rms[first_index],
            validation_masks[0],
            args,
            output_dir,
            device,
        )
        summary.update(beam_metrics)
        if wandb_run is not None:
            import wandb

            wandb_run.log(
                {**beam_metrics, "beamforming/comparison": wandb.Image(str(image_path))},
                step=args.steps,
            )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if wandb_run is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                wandb_run.summary[key] = value
        wandb_run.finish()


if __name__ == "__main__":
    main()
