"""
Focal-zone ablation for a single multi-focal acquisition.

Pick one MultiFocal DATA_*.mat file (by index) and beamform every focal zone
once, then render the requested combinations of zones side by side. Every panel
shares one 0 dB reference (the full multi-focal image), so brightness is
directly comparable: a zone is "crucial" if removing it darkens or blurs a depth
band, and "redundant" if its combinations look the same with or without it.

Modes:
    subsets       every non-empty combination of zones (the full power set)
    isolated      each zone alone, plus the full image
    leave-one-out the full image, plus the full image with one zone removed

Usage:
    poetry run python scripts/dataset_scripts/focal_zone_comparison.py --index 0
    poetry run python scripts/dataset_scripts/focal_zone_comparison.py \
        --index 2 --mode leave-one-out --out figures/fz_leave_one_out.png
"""

from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pymust
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_beamformed_pymust import (
    beamform_per_zone,
    load_metadata,
    load_rf_frame,
    stitch_zones,
    to_db,
)


def list_multifocal_files(data_dir: Path) -> list[Path]:
    """Sorted list of MultiFocal acquisitions under data_dir (index -> file)."""
    all_files = sorted(data_dir.rglob("DATA_*.mat"))
    return [f for f in all_files if "multifocal" in f.name.lower()]


def zone_subsets(focal_depths: list[float], mode: str) -> list[tuple[int, ...]]:
    """
    Build the list of zone-index combinations to render for a given mode.

    Zone indices are positions into the depth-sorted `focal_depths`.
    """
    n = len(focal_depths)
    idx = list(range(n))
    full = tuple(idx)

    if mode == "subsets":
        # every non-empty subset, ordered by size then shallowest-first
        subs: list[tuple[int, ...]] = []
        for r in range(1, n + 1):
            subs.extend(combinations(idx, r))
        return subs
    if mode == "isolated":
        return [(i,) for i in idx] + [full]
    if mode == "leave-one-out":
        if n <= 1:
            return [full]
        return [full] + [tuple(k for k in idx if k != i) for i in idx]
    raise ValueError(f"unknown mode: {mode}")


def _subset_title(subset: tuple[int, ...], focal_depths: list[float], full_n: int) -> str:
    """Human label, e.g. 'Z1+Z3  (17.7+41.4 mm)' or '… (full)'."""
    zones = "+".join(f"Z{i + 1}" for i in subset)
    depths = "+".join(f"{focal_depths[i] * 1e3:.1f}" for i in subset)
    tag = "  (full)" if len(subset) == full_n else ""
    return f"{zones}{tag}\n{depths} mm"


def compare(
    data_dir: Path,
    index: int,
    out: Path,
    mode: str = "subsets",
    nx: int = 256,
    nz: int = 384,
    frame: int = 0,
    scanline_step: int = 1,
    max_depth_m: float | None = 45e-3,
) -> None:
    files = list_multifocal_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No MultiFocal DATA_*.mat files under {data_dir}")
    if not 0 <= index < len(files):
        listing = "\n".join(f"  [{i}] {f}" for i, f in enumerate(files))
        raise IndexError(
            f"--index {index} out of range (0..{len(files) - 1}). Available:\n{listing}"
        )

    fpath = files[index]
    print(f"[{index}] {fpath}")

    meta = load_metadata(fpath)
    rf_raw = load_rf_frame(fpath, frame=frame)

    print("  beamforming each focal zone (once) …", flush=True)
    focal_depths, zone_imgs, x_grid, z_grid = beamform_per_zone(
        meta,
        rf_raw,
        nx=nx,
        nz=nz,
        scanline_step=scanline_step,
        max_depth_m=max_depth_m,
    )
    n = len(focal_depths)
    print(f"  {n} focal zones at {[round(d * 1e3, 1) for d in focal_depths]} mm")

    # shared 0 dB reference = peak of the full multi-focal composite, so every
    # panel is brightness-comparable and dropped zones visibly lose signal.
    full_final = stitch_zones(zone_imgs, focal_depths, z_grid)
    env_ref = np.abs(full_final)
    env_ref, _ = pymust.tgc(env_ref)
    ref_max = float(np.maximum(env_ref, 1e-9).max())

    subsets = zone_subsets(focal_depths, mode)

    panels = []
    for subset in subsets:
        sub_depths = [focal_depths[i] for i in subset]
        final = stitch_zones(zone_imgs, sub_depths, z_grid)
        panels.append((subset, to_db(final, ref_max=ref_max)))

    # ── layout ──
    n_panels = len(panels)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(-1)

    extent = [
        x_grid[0] * 1e3,
        x_grid[-1] * 1e3,
        z_grid[-1] * 1e3,
        z_grid[0] * 1e3,
    ]
    for ax, (subset, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=-50, vmax=0, extent=extent, aspect="equal")
        ax.set_title(_subset_title(subset, focal_depths, n), fontsize=8)
        ax.set_xlabel("Lateral (mm)", fontsize=7)
        ax.set_ylabel("Depth (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        # mark the focal depths of the zones included in this panel
        for i in subset:
            ax.axhline(focal_depths[i] * 1e3, color="tab:cyan", lw=0.6, alpha=0.5)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Focal-zone comparison — {fpath.parent.name}/{fpath.stem}  "
        f"(mode: {mode}, shared 0 dB reference)",
        fontsize=11,
        y=1.0,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n→ {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rats/"))
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which MultiFocal file to use (sorted order; pass an out-of-range "
        "index to print the available list).",
    )
    parser.add_argument(
        "--mode",
        choices=["subsets", "isolated", "leave-one-out"],
        default="subsets",
        help="Which zone combinations to render (default: subsets = full power set).",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nz", type=int, default=384)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--scanline-step",
        type=int,
        default=1,
        help="Use every Nth scan line (>1 speeds up at the cost of lateral sampling).",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=45.0,
        help="Reconstruction depth cap in mm. Defaults to 45 mm so all focal "
        "zones are visible. Pass 0 to use the full configured depth.",
    )
    args = parser.parse_args()
    out = args.out or Path(f"figures/focal_zone_comparison_{args.index}_{args.mode}.png")
    max_depth_m = None if args.max_depth_mm == 0 else args.max_depth_mm * 1e-3
    compare(
        args.data_dir,
        args.index,
        out,
        mode=args.mode,
        nx=args.nx,
        nz=args.nz,
        frame=args.frame,
        scanline_step=args.scanline_step,
        max_depth_m=max_depth_m,
    )
