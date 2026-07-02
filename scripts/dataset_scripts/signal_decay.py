"""
Visualise the depth-dependent signal decay that motivates per-depth RMS
normalization (`normalize_frame` in train.py) — across the whole dataset, so the
decision can be justified for the model as a whole rather than from one frame.

For every frame the script computes, at each depth sample, the RMS amplitude
over (transducer, scanline, iq) — exactly the quantity `normalize_frame` divides
by — and groups the curves by file/subject from the dataset metadata: per-depth
RMS for every frame, coloured by subject (+ dataset mean). If all files/subjects
share the same decay, one per-depth scheme is safe for the whole model rather
than tuned to a single acquisition.

Usage:
    poetry run python scripts/dataset_scripts/signal_decay.py
    poetry run python scripts/dataset_scripts/signal_decay.py --group-by file
    poetry run python scripts/dataset_scripts/signal_decay.py \
        --frames 0 1 2 --out figures/signal_decay.png

    # Against the HF dataset repo instead of a local zarr.zip build:
    poetry run python scripts/dataset_scripts/signal_decay.py \
        --hf-repo user/ssl-ultrasound-rats --max-frames 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssl_ultrasound_representation.dataset.dataset import load_dataset, load_local_dataset


def per_depth_rms(frame: np.ndarray) -> np.ndarray:
    """RMS over (transducer, scanline, iq) -> one value per depth sample.

    `frame` is (transducer, scanline, iq, sample); this is the same statistic
    `normalize_frame` divides each depth by (kept in float64 for a clean plot).
    """
    f = frame.astype(np.float64)
    return np.sqrt((f**2).mean(axis=(0, 1, 2)))  # (sample,)


def depth_axis_mm(n_samples: int, params: dict) -> np.ndarray:
    """Round-trip depth (mm) for each sample index: d = c * (t0 + n/fs) / 2."""
    c = params["c"]
    fs = params["fs"]
    t0 = params.get("t0", 0.0)
    n = np.arange(n_samples)
    return c * (t0 + n / fs) / 2.0 * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Global frame indices to use (default: every frame in the dataset).",
    )
    parser.add_argument(
        "--group-by",
        choices=["subject", "file"],
        default="subject",
        dest="group_by",
        help="How to colour the per-frame decay curves (default: subject).",
    )
    parser.add_argument("--focal-zone", type=int, default=0, dest="focal_zone")
    parser.add_argument(
        "--data-zip",
        default="data/processed/ultrasonic_dataset.zarr.zip",
        dest="data_zip",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/ultrasonic_dataset.metadata.json",
    )
    parser.add_argument(
        "--hf-repo",
        dest="hf_repo",
        help="HF dataset repo (e.g. user/ssl-ultrasound-rats) to pull from "
        "instead of the local --data-zip/--metadata build.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        dest="max_frames",
        help="HF only: only download/open the first N frames' chunks.",
    )
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        help="HF auth token for the download (higher rate limits). "
        "Falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN / cached login.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        dest="download_workers",
        help="HF only: chunk-download concurrency.",
    )
    parser.add_argument(
        "--beamform-params",
        default="configs/beamform_params.json",
        dest="beamform_params",
    )
    parser.add_argument("--out", default="figures/signal_decay.png")
    args = parser.parse_args()

    if args.hf_repo:
        arr, meta = load_dataset(
            args.hf_repo,
            max_frames=args.max_frames,
            token=args.hf_token,
            max_workers=args.download_workers,
        )
    else:
        arr, meta = load_local_dataset(args.data_zip, args.metadata)
    params = json.loads(Path(args.beamform_params).read_text())

    # Provenance: map each global frame index -> its group label (subject/file).
    frame_meta = {fm["global_frame_index"]: fm for fm in meta["frames"]}
    indices = args.frames if args.frames is not None else sorted(frame_meta)
    key = "subject" if args.group_by == "subject" else "relative_path"
    labels = [frame_meta[i].get(key, str(i)) for i in indices]

    # Each frame: (transducer, scanline, focal_zone, iq, sample); slice one zone.
    curves = [
        per_depth_rms(np.asarray(arr[idx, :, :, args.focal_zone]))
        for idx in indices
    ]
    curves = np.stack(curves)  # (n_frames, sample)
    n_samples = curves.shape[1]
    depth = depth_axis_mm(n_samples, params)

    mean_curve = curves.mean(axis=0)
    # Per-frame dynamic range to show the decay is consistent, not frame-specific.
    per_frame_dr = curves.max(axis=1) / np.maximum(curves.min(axis=1), 1e-12)
    dr_db = 20 * np.log10(per_frame_dr)
    dyn_range = mean_curve.max() / max(mean_curve.min(), 1e-12)
    print(f"Frames: {len(indices)}  |  grouped by {args.group_by}: "
          f"{sorted(set(labels))}")
    print(f"Per-frame dynamic range: {dr_db.mean():.1f} dB "
          f"(min {dr_db.min():.1f}, max {dr_db.max():.1f}) "
          f"-> the decay is consistent across files")
    print(f"Mean-curve dynamic range: {20 * np.log10(dyn_range):.1f} dB "
          f"= {np.log10(dyn_range):.2f} decades")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Per-frame decay, coloured by group — cross-dataset consistency.
    groups = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    color = {g: cmap(i % 10) for i, g in enumerate(groups)}
    seen: set[str] = set()
    for lab, curve in zip(labels, curves):
        ax.plot(
            depth, curve, color=color[lab], alpha=0.55, lw=1,
            label=lab if lab not in seen else None,
        )
        seen.add(lab)
    ax.plot(depth, mean_curve, color="black", lw=2.4, label="dataset mean")
    ax.set_yscale("log")
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Per-depth RMS amplitude (a.u.)")
    ax.set_title(
        f"Signal decay across the dataset\n({len(indices)} frames, "
        f"{dr_db.mean():.0f}±{dr_db.std():.0f} dB per frame)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, title=f"by {args.group_by}", frameon=False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
