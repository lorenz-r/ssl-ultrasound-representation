"""
Export the acquisition geometry needed to DAS-beamform the processed zarr IQ
frames (215 decimated samples) without access to the raw .mat files.

The zarr build (`build_mae_dataset.py`) discards the Verasonics metadata; this
script reads it from one reference .mat file (geometry is identical across the
dataset — the build script enforces a common grid) and writes a small JSON
consumed by `ssl_ultrasound_representation.utils.beamform`.

Usage:
    poetry run python scripts/dataset_scripts/export_beamform_params.py \\
        --mat-file data/rats/rat1/DATA_MultiFocal_20190318_100333.mat \\
        --out data/processed/beamform_params.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_beamformed_pymust import load_metadata  # noqa: E402
from build_mae_dataset import (  # noqa: E402
    DEPTH_CUTOFF,
    DOWNSAMPLE_FACTOR,
    N_SAMPLES_OUT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/beamform_params.json")
    )
    args = parser.parse_args()

    meta = load_metadata(args.mat_file)

    origins = sorted({round(g["origin_x"], 6) for g in meta["groups"]})
    focal_depths = sorted({round(g["focus_m"], 6) for g in meta["groups"]})
    origin_index = {o: i for i, o in enumerate(origins)}
    focal_index = {fz: i for i, fz in enumerate(focal_depths)}

    n_elem = meta["n_elem"]
    # delays_s[focal_zone][scanline][element]; None = element inactive in TX
    delays: list[list[list[float | None]]] = [
        [[None] * n_elem for _ in origins] for _ in focal_depths
    ]
    for g in meta["groups"]:
        sl = origin_index[round(g["origin_x"], 6)]
        fz = focal_index[round(g["focus_m"], 6)]
        full = np.full(n_elem, np.nan)
        apod = g["apod"]
        active = np.where(apod > 0)[0]
        phys = (g["tx_aperture"] - 1) + active
        valid = phys < n_elem
        full[phys[valid]] = g["delay_s"][active[valid]]
        delays[fz][sl] = [None if np.isnan(d) else float(d) for d in full]

    # Deepest sample actually present in the zarr (truncated at DEPTH_CUTOFF)
    data_depth_limit_m = meta["c"] * (meta["t0"] + DEPTH_CUTOFF / meta["fs"]) / 2.0

    payload = {
        "c": meta["c"],
        "fc": meta["fc"],
        "fs": meta["fs"] / DOWNSAMPLE_FACTOR,  # decimated rate of the zarr samples
        "pitch": meta["pitch"],
        "n_elements": n_elem,
        "t0": meta["t0"],
        "n_samples": N_SAMPLES_OUT,
        "start_depth_m": meta["start_depth_m"],
        "data_depth_limit_m": data_depth_limit_m,
        "scanline_origins_m": [float(o) for o in origins],
        "focal_depths_m": [float(f) for f in focal_depths],
        "tx_delays_s": delays,
        "source_mat_file": args.mat_file.name,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.0f} kB)")
    print(
        f"  fc={payload['fc']/1e6:.2f} MHz, fs(decimated)={payload['fs']/1e6:.2f} MHz, "
        f"pitch={payload['pitch']*1e3:.3f} mm, t0={payload['t0']*1e6:.2f} us"
    )
    print(
        f"  depth: {payload['start_depth_m']*1e3:.1f} - "
        f"{payload['data_depth_limit_m']*1e3:.1f} mm, "
        f"{len(origins)} scanlines, {len(focal_depths)} focal zones"
    )


if __name__ == "__main__":
    main()
