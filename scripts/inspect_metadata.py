"""
Print metadata from all MultiFocal .mat files under data/rats.

Usage:
    poetry run python scripts/inspect_metadata.py
    poetry run python scripts/inspect_metadata.py --data-dir data/rats --verbose
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# reuse helpers from the beamformer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from visualize_beamformed_pymust import load_metadata, _scalar


# ── HDF5 structure walker ────────────────────────────────────────────────────


def _walk_hdf5(node: Any, prefix: str = "", depth: int = 0, max_depth: int = 3) -> list[str]:
    lines = []
    if depth > max_depth:
        return lines
    items = list(node.items()) if hasattr(node, "items") else []
    for i, (name, obj) in enumerate(items):
        connector = "└─" if i == len(items) - 1 else "├─"
        path = f"{prefix}{connector} {name}"
        if isinstance(obj, h5py.Dataset):
            lines.append(f"{path}  [{obj.dtype}  {obj.shape}]")
        elif isinstance(obj, h5py.Group):
            lines.append(f"{path}/")
            ext = "   " if i == len(items) - 1 else "│  "
            lines.extend(_walk_hdf5(obj, prefix + ext, depth + 1, max_depth))
    return lines


def print_hdf5_structure(filepath: Path, max_depth: int = 3) -> None:
    print(f"\n{'─'*60}")
    print(f"HDF5 structure  ({filepath.name})")
    print(f"{'─'*60}")
    with h5py.File(filepath, "r") as f:
        top_keys = [k for k in f.keys() if not k.startswith("#")]
        for line in _walk_hdf5(f, max_depth=max_depth):
            print(line)
        print(f"\nTop-level keys: {top_keys}")


# ── pretty metadata printer ─────────────────────────────────────────────────


def print_metadata(filepath: Path) -> None:
    print(f"\n{'═'*60}")
    print(f"  {filepath.parent.name} / {filepath.name}")
    print(f"{'═'*60}")

    try:
        meta = load_metadata(filepath)
    except Exception as exc:
        print(f"  ERROR loading metadata: {exc}")
        return

    groups = meta["groups"]
    focal_depths_mm = sorted({round(g["focus_m"] * 1e3, 2) for g in groups})
    scan_lines = len({round(g["origin_x"] * 1e6) for g in groups})
    n_events = sum(len(g["rx_events"]) for g in groups)

    print(f"  Speed of sound      : {meta['c']:.0f} m/s")
    print(f"  Centre frequency    : {meta['fc']/1e6:.2f} MHz")
    print(f"  Wavelength          : {meta['lam']*1e3:.4f} mm")
    print(f"  Element pitch       : {meta['pitch']*1e3:.4f} mm")
    print(f"  # elements          : {meta['n_elem']}")
    print(f"  Element x span      : {meta['elem_x'][0]*1e3:.2f} … {meta['elem_x'][-1]*1e3:.2f} mm")
    print(f"  Sampling frequency  : {meta['fs']/1e6:.2f} MHz")
    print(f"  Samples / event     : {meta['n_samp']}")
    print(f"  Start depth         : {meta['start_depth_m']*1e3:.2f} mm")
    print(f"  End depth           : {meta['end_depth_m']*1e3:.2f} mm")
    print(f"  t0 (round-trip)     : {meta['t0']*1e6:.2f} µs")
    print()
    print(f"  Focal zones (mm)    : {focal_depths_mm}")
    print(f"  Scan lines          : {scan_lines}")
    print(f"  TX-RX groups        : {len(groups)}")
    print(f"  Total RX events     : {n_events}")
    print(f"  RX events / group   : {n_events / max(len(groups), 1):.1f}")

    # quick sanity on group structure
    rx_counts = [len(g["rx_events"]) for g in groups]
    unique_counts = sorted(set(rx_counts))
    print(f"  RX halves per group : {unique_counts}")

    # sample group info
    g0 = groups[0]
    print(f"\n  [Sample group 0]")
    print(f"    origin_x  = {g0['origin_x']*1e3:.3f} mm")
    print(f"    focus_m   = {g0['focus_m']*1e3:.3f} mm")
    print(f"    tx_aperture (start element) = {g0['tx_aperture']}")
    print(f"    apod nonzero elements       = {int(np.sum(g0['apod'] > 0))}")
    delay_s = g0["delay_s"]
    active_d = delay_s[delay_s > 0]
    if active_d.size:
        print(f"    TX delays (active)          = {active_d.min()*1e6:.3f} … {active_d.max()*1e6:.3f} µs")


# ── RcvData shape ────────────────────────────────────────────────────────────


def print_rcvdata_info(filepath: Path) -> None:
    with h5py.File(filepath, "r") as _f:
        f: Any = _f
        try:
            rcv_ref = f["RcvData"][()].flat[0]
            rcv = f[rcv_ref]
            print(f"  RcvData shape       : {rcv.shape}  (frames × channels × samples)")
            print(f"  RcvData dtype       : {rcv.dtype}")
        except Exception as exc:
            print(f"  RcvData             : could not inspect ({exc})")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Print metadata from MultiFocal .mat files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/rats"))
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also print HDF5 tree structure for each file")
    parser.add_argument("--hdf5-depth", type=int, default=2,
                        help="Max depth for HDF5 tree (--verbose only)")
    args = parser.parse_args()

    all_files = sorted(args.data_dir.rglob("DATA_*.mat"))
    mat_files = [f for f in all_files if "multifocal" in f.name.lower()]

    if not mat_files:
        print(f"No MultiFocal DATA_*.mat files found under {args.data_dir}")
        return

    print(f"Found {len(mat_files)} MultiFocal file(s) under {args.data_dir}\n")

    for fpath in mat_files:
        print_metadata(fpath)
        print_rcvdata_info(fpath)
        if args.verbose:
            print_hdf5_structure(fpath, max_depth=args.hdf5_depth)

    print(f"\n{'─'*60}")
    print("Done.")


if __name__ == "__main__":
    main()
