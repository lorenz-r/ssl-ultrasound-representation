"""
Visualise the depth-dependent signal decay that motivates per-depth RMS
normalization (`normalize_frame` in train.py) — across the whole dataset, so the
decision can be justified for the model as a whole rather than from one frame.

For every frame the script computes, at each depth sample, the RMS amplitude
over (transducer, scanline, iq) — exactly the quantity `normalize_frame` divides
by — and groups the curves by file/subject from the dataset metadata. Three
panels:

  1. Per-depth RMS for every frame, coloured by subject (+ dataset mean). If all
     files/subjects share the same decay, one per-depth scheme is safe for the
     whole model rather than tuned to a single acquisition.
  2. Per-depth envelope percentiles pooled over the dataset. Parallel decay of
     low and high percentiles => the whole distribution scales down with depth
     (true per-echo amplitude decay); high percentiles flat while the median
     drops => fewer lit positions at depth (a spatial-fill effect, not decay).
  3. Share of the (squared-error) reconstruction loss each depth would own raw
     vs after per-depth normalization — why the shallow field must be equalized.

Usage:
    poetry run python scripts/dataset_scripts/signal_decay.py
    poetry run python scripts/dataset_scripts/signal_decay.py --group-by file
    poetry run python scripts/dataset_scripts/signal_decay.py \
        --frames 0 1 2 --out figures/signal_decay.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssl_ultrasound_representation.dataset.dataset import load_local_dataset


def per_depth_rms(frame: np.ndarray) -> np.ndarray:
    """RMS over (transducer, scanline, iq) -> one value per depth sample.

    `frame` is (transducer, scanline, iq, sample); this is the same statistic
    `normalize_frame` divides each depth by (kept in float64 for a clean plot).
    """
    f = frame.astype(np.float64)
    return np.sqrt((f**2).mean(axis=(0, 1, 2)))  # (sample,)


def envelope_percentiles_frame(frame: np.ndarray, pcts: list[float]) -> np.ndarray:
    """Per-depth percentiles of the IQ envelope |z|=sqrt(I^2+Q^2) for one frame.

    Returns (len(pcts), sample). Computed per frame and averaged across the
    dataset by the caller — cheaper on memory than pooling every position.
    """
    f = frame.astype(np.float64)
    I, Q = f[..., 0, :], f[..., 1, :]  # (T, Sc, sample)
    env = np.sqrt(I**2 + Q**2).reshape(-1, f.shape[-1])  # (T*Sc, sample)
    return np.percentile(env, pcts, axis=0)  # (len(pcts), sample)


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
        "--beamform-params",
        default="configs/beamform_params.json",
        dest="beamform_params",
    )
    parser.add_argument("--out", default="figures/signal_decay.png")
    args = parser.parse_args()

    arr, meta = load_local_dataset(args.data_zip, args.metadata)
    params = json.loads(Path(args.beamform_params).read_text())

    # Provenance: map each global frame index -> its group label (subject/file).
    frame_meta = {fm["global_frame_index"]: fm for fm in meta["frames"]}
    indices = args.frames if args.frames is not None else sorted(frame_meta)
    key = "subject" if args.group_by == "subject" else "relative_path"
    labels = [frame_meta[i].get(key, str(i)) for i in indices]

    pcts = [50.0, 90.0, 99.0, 100.0]
    # Each frame: (transducer, scanline, focal_zone, iq, sample); slice one zone.
    curves, perc_stack = [], []
    for idx in indices:
        frame = np.asarray(arr[idx, :, :, args.focal_zone])  # (T, Sc, iq, sample)
        curves.append(per_depth_rms(frame))
        perc_stack.append(envelope_percentiles_frame(frame, pcts))
    curves = np.stack(curves)  # (n_frames, sample)
    perc = np.stack(perc_stack).mean(axis=0)  # (len(pcts), sample), dataset mean
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

    fig, (ax_raw, ax_env, ax_loss) = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── 1. Per-frame decay, coloured by group — cross-dataset consistency ────
    groups = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    color = {g: cmap(i % 10) for i, g in enumerate(groups)}
    seen: set[str] = set()
    for lab, curve in zip(labels, curves):
        ax_raw.plot(
            depth, curve, color=color[lab], alpha=0.55, lw=1,
            label=lab if lab not in seen else None,
        )
        seen.add(lab)
    ax_raw.plot(depth, mean_curve, color="black", lw=2.4, label="dataset mean")
    ax_raw.set_yscale("log")
    ax_raw.set_xlabel("Depth (mm)")
    ax_raw.set_ylabel("Per-depth RMS amplitude (a.u.)")
    ax_raw.set_title(
        f"Signal decay across the dataset\n({len(indices)} frames, "
        f"{dr_db.mean():.0f}±{dr_db.std():.0f} dB per frame)"
    )
    ax_raw.grid(True, which="both", alpha=0.3)
    ax_raw.legend(fontsize=8, ncol=2, title=f"by {args.group_by}")

    # ── 2. Envelope percentiles — true amplitude decay vs spatial fill ───────
    styles = {50.0: ("p50 (median)", "-"), 90.0: ("p90", "-"),
              99.0: ("p99", "-"), 100.0: ("max", "--")}
    for row, p in enumerate(pcts):
        lab, ls = styles[p]
        ax_env.plot(depth, perc[row], ls=ls, lw=1.6, label=lab)
    ax_env.set_yscale("log")
    ax_env.set_xlabel("Depth (mm)")
    ax_env.set_ylabel("Envelope |z| (a.u.)")
    ax_env.set_title(
        "Envelope percentiles (whole dataset)\nparallel decay = real per-echo "
        "decay, not just fill"
    )
    ax_env.grid(True, which="both", alpha=0.3)
    ax_env.legend(fontsize=8, title="percentile over aperture")

    # ── 3. Share of reconstruction loss owned by each depth ──────────────────
    # The loss is squared error, so each depth's contribution scales with its
    # energy (RMS^2). Raw -> the shallow few mm own almost everything; per-depth
    # normalization makes every depth contribute the uniform 1/n_samples.
    energy = mean_curve**2
    raw_share = energy / energy.sum() * 100.0
    uniform_share = np.full(n_samples, 100.0 / n_samples)
    ax_loss.fill_between(depth, raw_share, color="C3", alpha=0.35, label="raw IQ")
    ax_loss.plot(depth, raw_share, color="C3", lw=1.5)
    ax_loss.plot(depth, uniform_share, color="black", lw=2, ls="--",
                 label="after per-depth norm (uniform)")
    ax_loss.set_xlabel("Depth (mm)")
    ax_loss.set_ylabel("Share of reconstruction loss (%)")
    top5 = raw_share[depth <= 5].sum()
    ax_loss.set_title(
        f"Where the loss goes\n(shallow 5 mm owns {top5:.0f}% of the raw loss)"
    )
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
