"""Evaluate complex U-Net completion across frames, masks, and FSA regions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import zarr

from scripts.train_nv_complex_unet import (
    as_spatial_mask,
    build_disjoint_mask,
    build_disjoint_spatial_mask,
    load_frame,
    make_masks,
    make_receive_tube_masks,
    model_inputs,
    read_indices,
)
from ssl_ultrasound_representation.mae.model.complex_unet import ComplexUNet2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-zarr", required=True)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--training-mask-file")
    mask_group.add_argument(
        "--reference-mask-file",
        help="Reference geometry for a model trained with dynamic masks.",
    )
    parser.add_argument(
        "--masking-mode",
        choices=("scanline", "receive_tube"),
        default="scanline",
    )
    parser.add_argument("--mask-count", type=int)
    parser.add_argument("--mask-seed", type=int, default=20260824)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-frames", type=int, nargs="+", required=True)
    parser.add_argument("--random-mask-count", type=int, default=4)
    parser.add_argument("--random-mask-seed", type=int, default=20260824)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-step", type=int, default=802)
    return parser.parse_args()


class Accumulator:
    def __init__(self) -> None:
        self.error = 0.0
        self.target = 0.0
        self.pred = 0.0
        self.cross_real = 0.0
        self.cross_imag = 0.0
        self.count = 0

    def update(
        self,
        pred_real: torch.Tensor,
        pred_imag: torch.Tensor,
        target_real: torch.Tensor,
        target_imag: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        pred_real = pred_real.float() * scale
        pred_imag = pred_imag.float() * scale
        target_real = target_real.float() * scale
        target_imag = target_imag.float() * scale
        self.error += float(
            ((pred_real - target_real).square() + (pred_imag - target_imag).square())
            .sum()
            .cpu()
        )
        self.target += float((target_real.square() + target_imag.square()).sum().cpu())
        self.pred += float((pred_real.square() + pred_imag.square()).sum().cpu())
        self.cross_real += float(
            (pred_real * target_real + pred_imag * target_imag).sum().cpu()
        )
        self.cross_imag += float(
            (pred_imag * target_real - pred_real * target_imag).sum().cpu()
        )
        self.count += target_real.numel()

    def metrics(self) -> dict[str, float]:
        target = max(self.target, 1e-12)
        pred = max(self.pred, 1e-12)
        return {
            "raw_rel_mse": self.error / target,
            "raw_energy": self.pred / target,
            "raw_correlation": math.hypot(self.cross_real, self.cross_imag)
            / math.sqrt(pred * target),
            "raw_phase_error": abs(math.atan2(self.cross_imag, self.cross_real)),
            "raw_complex_mse": self.error / max(self.count, 1),
        }


def select_region(
    value: torch.Tensor, masked: torch.Tensor, region: str
) -> torch.Tensor:
    spatial_mask = as_spatial_mask(masked, value.shape[-1])
    if region == "full":
        region_mask = spatial_mask
    if region == "hidden_kxk":
        region_mask = spatial_mask & spatial_mask.transpose(0, 1)
    if region == "reciprocal_observable":
        region_mask = spatial_mask & ~spatial_mask.transpose(0, 1)
    if region not in {"full", "hidden_kxk", "reciprocal_observable"}:
        raise ValueError(f"unknown region: {region}")
    return value[:, region_mask]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ComplexUNet2d(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    fixed_mask_path = args.training_mask_file or args.reference_mask_file
    fixed_label = "training" if args.training_mask_file else "reference"
    if args.masking_mode == "scanline":
        if fixed_mask_path is None:
            raise ValueError("scanline evaluation requires a mask file")
        fixed_mask = torch.tensor(
            sorted(read_indices(fixed_mask_path)), dtype=torch.long
        )
        mask_count = len(fixed_mask)
        disjoint_mask = (
            build_disjoint_mask(fixed_mask, 180) if 2 * mask_count <= 180 else None
        )
        random_masks = make_masks(
            args.random_mask_count,
            mask_count,
            seed=args.random_mask_seed,
            width=180,
        )
    else:
        if fixed_mask_path is not None:
            raise ValueError("receive_tube evaluation generates spatial masks by seed")
        if args.mask_count is None:
            raise ValueError("receive_tube evaluation requires --mask-count")
        mask_count = args.mask_count
        fixed_mask = make_receive_tube_masks(
            1, mask_count, seed=args.mask_seed, width=180
        )[0]
        disjoint_mask = (
            build_disjoint_spatial_mask(fixed_mask)
            if 2 * mask_count <= 180
            else None
        )
        random_masks = make_receive_tube_masks(
            args.random_mask_count,
            mask_count,
            seed=args.random_mask_seed,
            width=180,
        )
    masks = {fixed_label: fixed_mask}
    if disjoint_mask is not None:
        masks["disjoint"] = disjoint_mask
    for index, mask in enumerate(random_masks):
        masks[f"random_{index:02d}"] = mask

    regions = ("full", "reciprocal_observable", "hidden_kxk")
    totals = {
        mask_name: {region: Accumulator() for region in regions}
        for mask_name in masks
    }
    array = zarr.open(args.data_zarr, mode="r", path="rf_iq")
    for frame_index in args.global_frames:
        frame, rms = load_frame(array, frame_index)
        for mask_name, masked_cpu in masks.items():
            masked = masked_cpu.to(device)
            for depths in torch.arange(frame.shape[0]).split(args.batch_size):
                target = frame[depths].to(device, non_blocking=True)
                input_real, input_imag, _ = model_inputs(target, masked)
                pred_real, pred_imag = model(input_real, input_imag)
                pred_real = pred_real[:, 0]
                pred_imag = pred_imag[:, 0]
                scale = rms[depths].to(device)[:, None]
                for region in regions:
                    totals[mask_name][region].update(
                        select_region(pred_real, masked, region),
                        select_region(pred_imag, masked, region),
                        select_region(target[:, 0], masked, region),
                        select_region(target[:, 1], masked, region),
                        scale,
                    )
        print(f"evaluated frame {frame_index}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "global_frames": args.global_frames,
        "mask_count": mask_count,
        "masking_mode": args.masking_mode,
        "masks": {
            name: (
                mask.tolist()
                if mask.ndim == 1
                else [
                    torch.where(mask[:, rx])[0].tolist()
                    for rx in range(mask.shape[1])
                ]
            )
            for name, mask in masks.items()
        },
        "metrics": {
            mask_name: {
                region: accumulator.metrics()
                for region, accumulator in region_totals.items()
            }
            for mask_name, region_totals in totals.items()
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generalization.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)

    if args.wandb_project and args.wandb_run_id:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must",
            dir=str(output_dir / "wandb"),
        )
        flattened = {}
        for mask_name, region_metrics in summary["metrics"].items():
            for region, metrics in region_metrics.items():
                for metric, value in metrics.items():
                    key = f"broad/{mask_name}/{region}/{metric}"
                    flattened[key] = value
                    run.summary[key] = value
        run.log(flattened, step=args.wandb_step)
        run.summary["broad/frames"] = args.global_frames
        run.finish()


if __name__ == "__main__":
    main()
