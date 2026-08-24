"""Run full-aperture beamforming for one trained complex U-Net checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import zarr

from scripts.train_nv_complex_unet import (
    beamform_summary,
    load_frame,
    make_receive_tube_masks,
    read_indices,
)
from ssl_ultrasound_representation.mae.model.complex_unet import ComplexUNet2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-zarr", required=True)
    parser.add_argument("--beamform-params", required=True)
    parser.add_argument("--mask-file")
    parser.add_argument(
        "--masking-mode",
        choices=("scanline", "receive_tube"),
        default="scanline",
    )
    parser.add_argument("--mask-count", type=int)
    parser.add_argument("--mask-seed", type=int, default=20260824)
    parser.add_argument("--mask-label", default="disjoint")
    parser.add_argument("--global-frame", type=int, default=830)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slices-per-step", type=int, default=8)
    parser.add_argument("--beamform-tx-step", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ComplexUNet2d(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    array = zarr.open(args.data_zarr, mode="r", path="rf_iq")
    frame, rms = load_frame(array, args.global_frame)
    if args.masking_mode == "scanline":
        if not args.mask_file:
            raise ValueError("scanline evaluation requires --mask-file")
        masked = torch.tensor(sorted(read_indices(args.mask_file)), dtype=torch.long)
    else:
        if args.mask_count is None:
            raise ValueError("receive_tube evaluation requires --mask-count")
        masked = make_receive_tube_masks(
            1, args.mask_count, seed=args.mask_seed, width=180
        )[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_args = SimpleNamespace(
        slices_per_step=args.slices_per_step,
        beamform_params=args.beamform_params,
        beamform_tx_step=args.beamform_tx_step,
        validation_frames=[args.global_frame],
    )
    metrics, image_path = beamform_summary(
        model,
        frame,
        rms,
        masked,
        eval_args,
        output_dir,
        device,
    )
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(metrics)
    summary["beamforming_mask"] = args.mask_label
    summary["beamforming_tx_step"] = args.beamform_tx_step
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)

    if args.wandb_project and args.wandb_run_id:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="must",
            dir=str(output_dir / "wandb"),
        )
        run.log(
            {
                **metrics,
                "beamforming/comparison_full_aperture": wandb.Image(str(image_path)),
            },
            step=int(summary["steps"]) + 1,
        )
        for key, value in metrics.items():
            run.summary[key] = value
        run.summary["beamforming_tx_step"] = args.beamform_tx_step
        run.finish()


if __name__ == "__main__":
    main()
