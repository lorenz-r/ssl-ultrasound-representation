"""Train a complex U-Net on masked NV FSA TX/RX response matrices."""

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

from ssl_ultrasound_representation.mae.model.complex_unet import ComplexUNet2d
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
    parser.add_argument("--wandb-group", default="nv-complex-unet")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--warmup-steps", type=int, default=160)
    parser.add_argument("--slices-per-step", type=int, default=8)
    parser.add_argument("--mask-count", type=int, default=60)
    parser.add_argument(
        "--masking-mode",
        choices=("scanline", "receive_tube"),
        default="scanline",
        help=(
            "scanline masks complete TX rows; receive_tube masks exactly "
            "mask-count TX positions independently for every RX sensor."
        ),
    )
    parser.add_argument(
        "--loss-region",
        choices=("masked", "all"),
        default="masked",
        help="Use all TX rows for zero-mask reconstruction or only masked rows for MAE.",
    )
    parser.add_argument(
        "--fixed-mask-file",
        help="Optional comma-separated TX mask. When set, use it for every step.",
    )
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument(
        "--validate-at-start",
        action="store_true",
        help="Evaluate and log the initialized model at step zero.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--wandb-log-every",
        type=int,
        default=1,
        help="Log training batches to W&B at this interval; validation is always logged.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("both", "best", "final"),
        default="both",
        help="Control which checkpoint files are retained.",
    )
    parser.add_argument("--validation-frames", type=int, nargs="+", default=[830, 831])
    parser.add_argument("--validation-mask-count", type=int, default=4)
    parser.add_argument("--validation-depth-count", type=int, default=32)
    parser.add_argument(
        "--max-preload-frames",
        type=int,
        default=256,
        help="Preload at most this many training frames; larger manifests stream from Zarr.",
    )
    parser.add_argument(
        "--stream-depth-windows",
        action="store_true",
        help="Read one contiguous axial window instead of a full frame when streaming.",
    )
    parser.add_argument("--beamform-tx-step", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_indices(path: str | Path) -> list[int]:
    text = Path(path).read_text()
    indices = [int(value) for value in text.replace(",", " ").split()]
    if not indices:
        raise ValueError(f"empty training manifest: {path}")
    if len(indices) != len(set(indices)):
        raise ValueError("training manifest contains duplicate frame indices")
    return indices


def load_frame(array: zarr.Array, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized [depth, IQ, TX, RX] and its per-depth raw RMS."""
    raw = np.asarray(array[index, :, :, 0, :, :], dtype=np.float32)
    frame = torch.from_numpy(raw).permute(3, 2, 1, 0).contiguous()
    rms = frame.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
    return frame / rms[:, None, None, None], rms


def load_depth_window(
    array: zarr.Array,
    index: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Read a small window contained within one 64-depth Zarr chunk."""
    if not 0 < count <= 64:
        raise ValueError(f"streamed depth window must be in [1, 64], got {count}")
    chunk_index = int(torch.randint(0, 4, (), generator=generator))
    offset = int(torch.randint(0, 65 - count, (), generator=generator))
    start = chunk_index * 64 + offset
    raw = np.asarray(
        array[index, :, :, 0, :, start : start + count], dtype=np.float32
    )
    frame = torch.from_numpy(raw).permute(3, 2, 1, 0).contiguous()
    rms = frame.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
    return frame / rms[:, None, None, None]


def make_masks(
    count: int, masked: int, *, seed: int, width: int
) -> list[torch.Tensor]:
    masks = []
    for offset in range(count):
        generator = torch.Generator().manual_seed(seed + offset * 7919)
        masks.append(torch.randperm(width, generator=generator)[:masked].sort().values)
    return masks


def make_receive_tube_masks(
    count: int, masked_per_receiver: int, *, seed: int, width: int
) -> list[torch.Tensor]:
    """Return masks with exactly K TX/depth tubes hidden per RX sensor."""
    masks = []
    for offset in range(count):
        generator = torch.Generator().manual_seed(seed + offset * 7919)
        noise = torch.rand(width, width, generator=generator)
        masked_tx = noise.topk(masked_per_receiver, dim=0, largest=False).indices
        mask = torch.zeros(width, width, dtype=torch.bool)
        mask.scatter_(0, masked_tx, True)
        masks.append(mask)
    return masks


def as_spatial_mask(masked: torch.Tensor, width: int) -> torch.Tensor:
    """Convert TX row indices or an explicit TX/RX mask to a boolean grid."""
    if masked.ndim == 2:
        if masked.shape != (width, width):
            raise ValueError(f"expected a {width}x{width} spatial mask")
        return masked.bool()
    if masked.ndim != 1:
        raise ValueError("mask must contain TX indices or a TX/RX boolean grid")
    spatial = torch.zeros(width, width, dtype=torch.bool, device=masked.device)
    spatial[masked.long(), :] = True
    return spatial


def build_disjoint_spatial_mask(masked: torch.Tensor) -> torch.Tensor:
    """Build an equal-size tube mask disjoint from the supplied spatial mask."""
    if masked.ndim != 2 or masked.shape[0] != masked.shape[1]:
        raise ValueError("disjoint spatial masking requires a square mask")
    width = masked.shape[0]
    result = torch.zeros_like(masked, dtype=torch.bool)
    for rx in range(width):
        source = torch.where(masked[:, rx])[0]
        unavailable = set(int(value) for value in source.tolist())
        selected = []
        for base in source.tolist():
            for delta in range(1, width):
                candidate = (int(base) + delta) % width
                if candidate not in unavailable:
                    selected.append(candidate)
                    unavailable.add(candidate)
                    break
            else:
                raise RuntimeError("could not construct a disjoint tube mask")
        result[selected, rx] = True
    return result


def build_disjoint_mask(training_mask: torch.Tensor, width: int) -> torch.Tensor:
    """Match the deterministic nearest-higher mask used by the MAE analysis."""
    unavailable = set(int(index) for index in training_mask.tolist())
    selected: list[int] = []
    for base in training_mask.tolist():
        for delta in range(1, width):
            candidate = (int(base) + delta) % width
            if candidate not in unavailable:
                selected.append(candidate)
                unavailable.add(candidate)
                break
        else:
            raise RuntimeError("could not construct a disjoint evaluation mask")
    return torch.tensor(selected, dtype=torch.long)


def model_inputs(
    target: torch.Tensor, masked: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build complex [IQ, visibility] input from [batch, IQ, TX, RX]."""
    spatial_mask = as_spatial_mask(masked, target.shape[-1])
    visible = torch.ones(
        target.shape[0], 1, target.shape[2], target.shape[3],
        device=target.device,
        dtype=target.dtype,
    )
    visible.masked_fill_(spatial_mask[None, None], 0)
    input_real = torch.cat((target[:, 0:1] * visible, visible), dim=1)
    input_imag = torch.cat((target[:, 1:2] * visible, torch.zeros_like(visible)), dim=1)
    return input_real, input_imag, visible


def masked_metrics(
    pred_real: torch.Tensor,
    pred_imag: torch.Tensor,
    target: torch.Tensor,
    masked: torch.Tensor,
) -> dict[str, float]:
    spatial_mask = as_spatial_mask(masked, target.shape[-1])
    pred_real = pred_real[:, 0][:, spatial_mask].float()
    pred_imag = pred_imag[:, 0][:, spatial_mask].float()
    target_real = target[:, 0][:, spatial_mask].float()
    target_imag = target[:, 1][:, spatial_mask].float()
    error = (pred_real - target_real).square() + (pred_imag - target_imag).square()
    target_power = target_real.square().sum() + target_imag.square().sum()
    pred_power = pred_real.square().sum() + pred_imag.square().sum()
    cross_real = (pred_real * target_real + pred_imag * target_imag).sum()
    cross_imag = (pred_imag * target_real - pred_real * target_imag).sum()
    cross_abs = torch.sqrt(cross_real.square() + cross_imag.square())
    scale = torch.sqrt(pred_power * target_power).clamp_min(1e-12)
    return {
        "mse": float(error.mean().detach().cpu()),
        "energy": float((pred_power / target_power.clamp_min(1e-12)).detach().cpu()),
        "correlation": float((cross_abs / scale).detach().cpu()),
        "phase_error": float(torch.atan2(cross_imag.abs(), cross_real).detach().cpu()),
    }


def reciprocal_fill(
    target: torch.Tensor, masked: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy only reciprocal samples whose source TX remains visible."""
    spatial_mask = as_spatial_mask(masked, target.shape[-1])
    visible_source = (~spatial_mask.transpose(0, 1)).to(target.dtype)
    real = target[:, 0].transpose(-1, -2) * visible_source[None]
    imag = target[:, 1].transpose(-1, -2) * visible_source[None]
    return real[:, None], imag[:, None]


@torch.inference_mode()
def evaluate(
    model: ComplexUNet2d,
    frames: dict[int, torch.Tensor],
    masks: list[torch.Tensor],
    depth_indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {
        "mse": [], "energy": [], "correlation": [], "phase_error": [],
        "reciprocal_mse": [], "reciprocal_energy": [], "reciprocal_correlation": [],
    }
    for frame in frames.values():
        for masked_cpu in masks:
            masked = masked_cpu.to(device)
            spatial_mask = as_spatial_mask(masked, 180)
            metric_mask = spatial_mask if spatial_mask.any() else ~spatial_mask
            for chunk in depth_indices.split(batch_size):
                target = frame[chunk].to(device, non_blocking=True)
                input_real, input_imag, _ = model_inputs(target, masked)
                pred_real, pred_imag = model(input_real, input_imag)
                metrics = masked_metrics(
                    pred_real, pred_imag, target, metric_mask
                )
                for key, value in metrics.items():
                    collected[key].append(value)
                reciprocal_real, reciprocal_imag = reciprocal_fill(target, masked)
                reciprocal = masked_metrics(
                    reciprocal_real, reciprocal_imag, target, metric_mask
                )
                collected["reciprocal_mse"].append(reciprocal["mse"])
                collected["reciprocal_energy"].append(reciprocal["energy"])
                collected["reciprocal_correlation"].append(reciprocal["correlation"])
    return {key: float(np.mean(values)) for key, values in collected.items()}


@torch.inference_mode()
def evaluate_physical(
    model: ComplexUNet2d,
    frames: dict[int, torch.Tensor],
    rms_by_frame: dict[int, torch.Tensor],
    masks: list[torch.Tensor],
    depth_indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Aggregate normalized and raw metrics without averaging batch ratios."""
    model.eval()
    totals = {
        "norm_error": 0.0,
        "norm_target": 0.0,
        "norm_pred": 0.0,
        "raw_error": 0.0,
        "raw_target": 0.0,
        "raw_pred": 0.0,
        "raw_cross_real": 0.0,
        "raw_cross_imag": 0.0,
        "complex_count": 0,
    }
    for global_index, frame in frames.items():
        rms = rms_by_frame[global_index]
        for masked_cpu in masks:
            masked = masked_cpu.to(device)
            spatial_mask = as_spatial_mask(masked, frame.shape[-1])
            for chunk in depth_indices.split(batch_size):
                target = frame[chunk].to(device, non_blocking=True)
                input_real, input_imag, _ = model_inputs(target, masked)
                pred_real, pred_imag = model(input_real, input_imag)
                pred_real = pred_real[:, 0][:, spatial_mask].float()
                pred_imag = pred_imag[:, 0][:, spatial_mask].float()
                target_real = target[:, 0][:, spatial_mask].float()
                target_imag = target[:, 1][:, spatial_mask].float()
                error_real = pred_real - target_real
                error_imag = pred_imag - target_imag
                totals["norm_error"] += float(
                    (error_real.square() + error_imag.square()).sum().cpu()
                )
                totals["norm_target"] += float(
                    (target_real.square() + target_imag.square()).sum().cpu()
                )
                totals["norm_pred"] += float(
                    (pred_real.square() + pred_imag.square()).sum().cpu()
                )

                scale = rms[chunk].to(device)[:, None]
                raw_pred_real = pred_real * scale
                raw_pred_imag = pred_imag * scale
                raw_target_real = target_real * scale
                raw_target_imag = target_imag * scale
                raw_error_real = raw_pred_real - raw_target_real
                raw_error_imag = raw_pred_imag - raw_target_imag
                totals["raw_error"] += float(
                    (raw_error_real.square() + raw_error_imag.square()).sum().cpu()
                )
                totals["raw_target"] += float(
                    (raw_target_real.square() + raw_target_imag.square()).sum().cpu()
                )
                totals["raw_pred"] += float(
                    (raw_pred_real.square() + raw_pred_imag.square()).sum().cpu()
                )
                totals["raw_cross_real"] += float(
                    (
                        raw_pred_real * raw_target_real
                        + raw_pred_imag * raw_target_imag
                    ).sum().cpu()
                )
                totals["raw_cross_imag"] += float(
                    (
                        raw_pred_imag * raw_target_real
                        - raw_pred_real * raw_target_imag
                    ).sum().cpu()
                )
                totals["complex_count"] += target_real.numel()

    norm_target = max(totals["norm_target"], 1e-12)
    raw_target = max(totals["raw_target"], 1e-12)
    raw_pred = max(totals["raw_pred"], 1e-12)
    cross_abs = math.hypot(totals["raw_cross_real"], totals["raw_cross_imag"])
    return {
        "normalized_complex_mse": totals["norm_error"] / totals["complex_count"],
        "normalized_rel_mse": totals["norm_error"] / norm_target,
        "normalized_energy": totals["norm_pred"] / norm_target,
        "raw_rel_mse": totals["raw_error"] / raw_target,
        "raw_energy": totals["raw_pred"] / raw_target,
        "raw_correlation": cross_abs / math.sqrt(raw_pred * raw_target),
        "raw_phase_error": abs(
            math.atan2(totals["raw_cross_imag"], totals["raw_cross_real"])
        ),
    }


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_payload(
    model: ComplexUNet2d, args: argparse.Namespace, step: int, metrics: dict[str, float]
) -> dict:
    return {
        "model": model.state_dict(),
        "step": step,
        "metrics": metrics,
        "model_config": {
            "in_channels": 2,
            "out_channels": 1,
            "base_channels": args.base_channels,
            "levels": args.levels,
        },
        "experiment_config": vars(args),
    }


@torch.inference_mode()
def infer_full_frame(
    model: ComplexUNet2d,
    frame: torch.Tensor,
    masked_cpu: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    masked = masked_cpu.to(device)
    pred_real_parts = []
    pred_imag_parts = []
    for target in frame.split(batch_size):
        target = target.to(device, non_blocking=True)
        input_real, input_imag, _ = model_inputs(target, masked)
        pred_real, pred_imag = model(input_real, input_imag)
        pred_real_parts.append(pred_real[:, 0].cpu())
        pred_imag_parts.append(pred_imag[:, 0].cpu())
    return torch.cat(pred_real_parts), torch.cat(pred_imag_parts)


def to_raw_layout(frame: torch.Tensor, rms: torch.Tensor) -> np.ndarray:
    raw = frame * rms[:, None, None, None]
    return raw.permute(3, 2, 1, 0).contiguous().numpy()


def beamform_summary(
    model: ComplexUNet2d,
    frame: torch.Tensor,
    rms: torch.Tensor,
    masked: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, float], Path]:
    spatial_mask = as_spatial_mask(masked, frame.shape[-1])
    pred_real, pred_imag = infer_full_frame(
        model, frame, masked, device=device, batch_size=args.slices_per_step
    )
    predicted = torch.stack((pred_real, pred_imag), dim=1)
    visible = frame.clone()
    visible.masked_fill_(spatial_mask[None, None], 0)
    predicted_only = torch.zeros_like(frame)
    predicted_only[:, 0][:, spatial_mask] = pred_real[:, spatial_mask]
    predicted_only[:, 1][:, spatial_mask] = pred_imag[:, spatial_mask]
    hybrid = frame.clone()
    hybrid[:, :, spatial_mask] = predicted[:, :, spatial_mask]

    reciprocal_real, reciprocal_imag = reciprocal_fill(frame, masked)
    reciprocal = frame.clone()
    reciprocal[:, 0][:, spatial_mask] = reciprocal_real[:, 0][:, spatial_mask]
    reciprocal[:, 1][:, spatial_mask] = reciprocal_imag[:, 0][:, spatial_mask]

    raw_gt = to_raw_layout(frame, rms)
    raw_views = {
        "visible only": to_raw_layout(visible, rms),
        "predicted masked": to_raw_layout(predicted_only, rms),
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
    figure, axes = plt.subplots(1, 5, figsize=(21, 4), constrained_layout=True)
    panels = {"ground truth": gt_db, **views_db}
    for axis, (name, image) in zip(axes, panels.items(), strict=True):
        axis.imshow(image, cmap="gray", vmin=-50, vmax=0, aspect="auto")
        axis.set_title(name)
        axis.set_xlabel("lateral")
        axis.set_ylabel("depth")
    complete_rows = int(spatial_mask.all(dim=1).sum())
    mask_description = (
        f"{complete_rows}/180 TX"
        if int(spatial_mask.sum()) == complete_rows * spatial_mask.shape[1]
        else f"{int(spatial_mask.sum())}/{spatial_mask.numel()} TX/RX tubes"
    )
    figure.suptitle(
        f"Complex U-Net | frame {args.validation_frames[0]} | "
        f"masked {mask_description}"
    )
    output_path = output_dir / "beamforming_frame0.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return metrics, output_path


def main() -> None:
    args = parse_args()
    if not 0 <= args.mask_count < 180:
        raise ValueError("mask-count must be between 0 and 179")
    if args.loss_region == "masked" and args.mask_count == 0:
        raise ValueError("masked loss requires mask-count greater than zero")
    if args.loss_region == "all" and args.mask_count != 0:
        raise ValueError("all-region reconstruction requires mask-count zero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array = zarr.open(args.data_zarr, mode="r", path="rf_iq")
    metadata = json.loads(Path(args.metadata).read_text())
    if tuple(array.shape[1:]) != (180, 180, 1, 2, 256):
        raise ValueError(f"unexpected NV array shape: {array.shape}")
    train_indices = read_indices(args.train_manifest)
    fixed_training_mask = None
    disjoint_mask = None
    if args.fixed_mask_file:
        if args.masking_mode != "scanline":
            raise ValueError("fixed-mask-file is currently scanline-only")
        fixed_values = read_indices(args.fixed_mask_file)
        if len(fixed_values) != args.mask_count:
            raise ValueError(
                f"fixed mask has {len(fixed_values)} TX indices, expected "
                f"{args.mask_count}"
            )
        if min(fixed_values) < 0 or max(fixed_values) >= 180:
            raise ValueError("fixed mask TX indices must be in [0, 179]")
        fixed_training_mask = torch.tensor(sorted(fixed_values), dtype=torch.long)
        disjoint_mask = build_disjoint_mask(fixed_training_mask, 180)
    publisher_train = {
        int(frame_meta["global_frame_index"])
        for frame_meta in metadata["frames"]
        if frame_meta.get("split") == "train"
    }
    if not set(train_indices).issubset(publisher_train):
        raise ValueError("training manifest includes publisher-validation frames")

    start = time.time()
    preload_training = len(train_indices) <= args.max_preload_frames
    train_frames: list[torch.Tensor] | None = [] if preload_training else None
    train_rms: list[torch.Tensor] | None = [] if preload_training else None
    if preload_training:
        print(f"Preloading {len(train_indices)} training frames", flush=True)
        assert train_frames is not None and train_rms is not None
        for position, index in enumerate(train_indices, start=1):
            frame, rms = load_frame(array, index)
            train_frames.append(frame)
            train_rms.append(rms)
            print(
                f"  loaded {position}/{len(train_indices)}: global frame {index}",
                flush=True,
            )
    else:
        print(
            f"Streaming {len(train_indices)} training frames from node-local Zarr "
            f"(preload limit {args.max_preload_frames})",
            flush=True,
        )
    validation_frames: dict[int, torch.Tensor] = {}
    validation_rms: dict[int, torch.Tensor] = {}
    for index in args.validation_frames:
        frame, rms = load_frame(array, index)
        validation_frames[index] = frame
        validation_rms[index] = rms
    print(f"Data ready in {time.time() - start:.1f}s", flush=True)

    if fixed_training_mask is None:
        mask_factory = (
            make_masks
            if args.masking_mode == "scanline"
            else make_receive_tube_masks
        )
        validation_masks = mask_factory(
            args.validation_mask_count,
            args.mask_count,
            seed=args.seed + 100_000,
            width=180,
        )
        beamforming_mask = validation_masks[0]
        beamforming_mask_label = "unseen"
    else:
        assert disjoint_mask is not None
        validation_masks = [disjoint_mask]
        beamforming_mask = disjoint_mask
        beamforming_mask_label = "disjoint"
    validation_depths = torch.linspace(
        0, 255, args.validation_depth_count, dtype=torch.float64
    ).round().long().unique()
    serialized_masks = []
    for mask in validation_masks:
        if mask.ndim == 1:
            serialized_masks.append(mask.tolist())
        else:
            serialized_masks.append(
                [torch.where(mask[:, rx])[0].tolist() for rx in range(mask.shape[1])]
            )
    (output_dir / "validation_masks.json").write_text(
        json.dumps(
            {"masking_mode": args.masking_mode, "masks": serialized_masks},
            indent=2,
        )
        + "\n"
    )
    if fixed_training_mask is not None:
        (output_dir / "training_masked_tx.txt").write_text(
            ",".join(map(str, fixed_training_mask.tolist())) + "\n"
        )
        assert disjoint_mask is not None
        (output_dir / "disjoint_masked_tx.txt").write_text(
            ",".join(map(str, disjoint_mask.tolist())) + "\n"
        )
    (output_dir / "train_global_indices.txt").write_text(
        "\n".join(map(str, train_indices)) + "\n"
    )

    model = ComplexUNet2d(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        levels=args.levels,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        if initial["model_config"] != {
            "in_channels": 2,
            "out_channels": 1,
            "base_channels": args.base_channels,
            "levels": args.levels,
        }:
            raise ValueError("initial checkpoint architecture does not match")
        model.load_state_dict(initial["model"])
        print(f"Initialized model from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        f"Model parameters: {model.num_parameters:,} | "
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
            config={**vars(args), "train_indices": train_indices, "parameters": model.num_parameters},
        )

    frame_order: list[int] = []
    generator = torch.Generator().manual_seed(args.seed + 17)
    running_loss = 0.0
    best_validation = float("inf")
    last_metrics: dict[str, float] = {}
    train_start = time.time()
    if args.validate_at_start:
        val_metrics = evaluate(
            model,
            validation_frames,
            validation_masks,
            validation_depths,
            device=device,
            batch_size=args.slices_per_step,
        )
        last_metrics = val_metrics
        best_validation = val_metrics["mse"]
        print(
            f"validation step=0: mse={val_metrics['mse']:.4f} "
            f"corr={val_metrics['correlation']:.3f} "
            f"energy={val_metrics['energy']:.3f} "
            f"recip_corr={val_metrics['reciprocal_correlation']:.3f}",
            flush=True,
        )
        if args.checkpoint_policy in ("both", "best"):
            torch.save(
                checkpoint_payload(model, args, 0, val_metrics),
                output_dir / "best.pt",
            )
        if wandb_run is not None:
            wandb_run.log(
                {f"val/{key}": value for key, value in val_metrics.items()},
                step=0,
            )
    for step in range(args.steps):
        if not frame_order:
            frame_order = torch.randperm(
                len(train_indices), generator=generator
            ).tolist()
        frame_position = frame_order.pop()
        if train_frames is None and args.stream_depth_windows:
            target_cpu = load_depth_window(
                array,
                train_indices[frame_position],
                args.slices_per_step,
                generator,
            )
            target = target_cpu.to(device, non_blocking=True)
        elif train_frames is None:
            frame, _ = load_frame(array, train_indices[frame_position])
            depth_indices = torch.randperm(frame.shape[0], generator=generator)[
                : args.slices_per_step
            ]
            target = frame[depth_indices].to(device, non_blocking=True)
        else:
            frame = train_frames[frame_position]
            depth_indices = torch.randperm(frame.shape[0], generator=generator)[
                : args.slices_per_step
            ]
            target = frame[depth_indices].to(device, non_blocking=True)
        if fixed_training_mask is None:
            if args.masking_mode == "scanline":
                masked = torch.randperm(180, generator=generator)[
                    : args.mask_count
                ].sort().values.to(device)
            else:
                noise = torch.rand(180, 180, generator=generator)
                masked_tx = noise.topk(
                    args.mask_count, dim=0, largest=False
                ).indices
                masked = torch.zeros(180, 180, dtype=torch.bool)
                masked.scatter_(0, masked_tx, True)
                masked = masked.to(device)
        else:
            masked = fixed_training_mask.to(device)
        input_real, input_imag, _ = model_inputs(target, masked)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_real, pred_imag = model(input_real, input_imag)
        spatial_mask = as_spatial_mask(masked, 180)
        loss_indices = (
            torch.ones(180, 180, dtype=torch.bool, device=device)
            if args.loss_region == "all"
            else spatial_mask
        )
        metrics = masked_metrics(pred_real, pred_imag, target, loss_indices)
        loss = (
            (pred_real[:, 0][:, loss_indices] - target[:, 0][:, loss_indices]).square()
            + (pred_imag[:, 0][:, loss_indices] - target[:, 1][:, loss_indices]).square()
        ).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        running_loss += float(loss.detach().cpu())

        log_data = {
            "train/masked_mse": metrics["mse"],
            "train/masked_energy": metrics["energy"],
            "train/masked_correlation": metrics["correlation"],
            "train/phase_error_rad": metrics["phase_error"],
            "train/grad_norm": float(grad_norm.detach().cpu()),
            "train/lr": lr,
            "train/step": step + 1,
            "train/mask_count": args.mask_count,
        }
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - train_start
            print(
                f"step {step + 1:4d}/{args.steps} "
                f"loss={running_loss / args.log_every:.4f} "
                f"corr={metrics['correlation']:.3f} energy={metrics['energy']:.3f} "
                f"{elapsed / (step + 1):.3f}s/step",
                flush=True,
            )
            running_loss = 0.0

        should_validate = (step + 1) % args.validate_every == 0 or step + 1 == args.steps
        if should_validate:
            val_metrics = evaluate(
                model,
                validation_frames,
                validation_masks,
                validation_depths,
                device=device,
                batch_size=args.slices_per_step,
            )
            last_metrics = val_metrics
            log_data.update({f"val/{key}": value for key, value in val_metrics.items()})
            print(
                f"validation step={step + 1}: mse={val_metrics['mse']:.4f} "
                f"corr={val_metrics['correlation']:.3f} energy={val_metrics['energy']:.3f} "
                f"recip_corr={val_metrics['reciprocal_correlation']:.3f}",
                flush=True,
            )
            if val_metrics["mse"] < best_validation:
                best_validation = val_metrics["mse"]
                if args.checkpoint_policy in ("both", "best"):
                    torch.save(
                        checkpoint_payload(model, args, step + 1, val_metrics),
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
    summary = {
        "run_name": args.run_name,
        "steps": args.steps,
        "parameters": model.num_parameters,
        "elapsed_seconds": time.time() - train_start,
        "best_validation_mse": best_validation,
        **{f"validation/{key}": value for key, value in last_metrics.items()},
    }
    if fixed_training_mask is not None:
        assert disjoint_mask is not None
        seen_indices = train_indices[: min(2, len(train_indices))]
        if train_frames is None:
            seen_pairs = {index: load_frame(array, index) for index in seen_indices}
            seen_frames = {index: pair[0] for index, pair in seen_pairs.items()}
            seen_rms = {index: pair[1] for index, pair in seen_pairs.items()}
        else:
            assert train_rms is not None
            seen_frames = {
                index: train_frames[train_indices.index(index)]
                for index in seen_indices
            }
            seen_rms = {
                index: train_rms[train_indices.index(index)]
                for index in seen_indices
            }
        full_depths = torch.arange(array.shape[-1], dtype=torch.long)
        grid_cases = {
            "seen_frame/training_mask": (
                seen_frames,
                seen_rms,
                [fixed_training_mask],
            ),
            "seen_frame/disjoint_mask": (
                seen_frames,
                seen_rms,
                [disjoint_mask],
            ),
            "unseen_frame/training_mask": (
                validation_frames,
                validation_rms,
                [fixed_training_mask],
            ),
            "unseen_frame/disjoint_mask": (
                validation_frames,
                validation_rms,
                [disjoint_mask],
            ),
        }
        for case_name, (case_frames, case_rms, case_masks) in grid_cases.items():
            case_metrics = evaluate_physical(
                model,
                case_frames,
                case_rms,
                case_masks,
                full_depths,
                device=device,
                batch_size=args.slices_per_step,
            )
            summary.update(
                {f"{case_name}/{key}": value for key, value in case_metrics.items()}
            )
            print(f"{case_name}: {case_metrics}", flush=True)
    if args.beamform_params:
        first_index = args.validation_frames[0]
        beam_metrics, image_path = beamform_summary(
            model,
            validation_frames[first_index],
            validation_rms[first_index],
            beamforming_mask,
            args,
            output_dir,
            device,
        )
        summary.update(beam_metrics)
        summary["beamforming_mask"] = beamforming_mask_label
        if wandb_run is not None:
            import wandb

            wandb_run.log(
                {
                    **beam_metrics,
                    "beamforming/comparison": wandb.Image(str(image_path)),
                },
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
