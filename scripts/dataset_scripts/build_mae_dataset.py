"""
Aggregate Stanford RF MultiFocal ultrasound .mat files into a single 6D Zarr
array optimised for streaming during self-supervised MAE training.

Output shape (axes):
    (Frames, Transducers, Scanlines, FocalZones, IQ=2, Samples=215)

Sample axis is the original fast-time axis truncated at 860 samples and then
decimated by a factor of 4. IQ index 0 = In-phase, 1 = Quadrature.

Usage:
    poetry run python scripts/dataset_scripts/build_mae_dataset.py \\
        --input_dir data/rats \\
        --output_file data/processed/ultrasonic_dataset.zarr.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pymust
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# Reuse the *exact* metadata parser from the reference beamforming script. The
# RF read + sub-aperture combination are reimplemented below in a streaming
# variant that never materialises the multi-GB per-frame RF buffer (the
# reference `load_rf_frame` allocates ~2.4 GB float32 per call — fine for a
# single visualisation, fatal on a memory-capped batch job).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_beamformed_pymust import load_metadata  # noqa: E402

# ── Fixed processing constants ──────────────────────────────────────────────────
DEPTH_CUTOFF = 860           # fast-time samples kept (everything beyond is noise)
DOWNSAMPLE_FACTOR = 4
N_SAMPLES_OUT = DEPTH_CUTOFF // DOWNSAMPLE_FACTOR  # 215
CHUNKS = (1, 32, 32, 1, 2, N_SAMPLES_OUT)
ARRAY_NAME = "rf_iq"


# ── File discovery / metadata helpers ───────────────────────────────────────────
def discover_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*.mat") if "multifocal" in p.name.lower()
    )


def relative_parts(filepath: Path, input_dir: Path) -> tuple[str, str | None]:
    """Subject = first folder under input_dir; subfolder = intermediate path."""
    rel = filepath.relative_to(input_dir)
    parts = rel.parts
    subject = parts[0] if len(parts) > 1 else ""
    subfolder = "/".join(parts[1:-1]) if len(parts) > 2 else None
    return subject, subfolder


def file_n_frames(filepath: Path) -> int:
    with h5py.File(filepath, "r") as _f:
        f: Any = _f
        rcv = f[f["RcvData"][()].flat[0]]
        return int(rcv.shape[0])


def file_grid(meta: dict) -> tuple[list[float], list[float]]:
    """Unique scanline origins and focal depths defined by this file's groups."""
    origins = sorted({round(g["origin_x"], 6) for g in meta["groups"]})
    focal_depths = sorted({round(g["focus_m"], 6) for g in meta["groups"]})
    return origins, focal_depths


# ── Per-frame processing ────────────────────────────────────────────────────────
def _combine_rx_halves_streaming(
    group: dict, rcv_ds: Any, frame_idx: int, meta: dict
) -> np.ndarray:
    """
    Memory-frugal twin of `visualize_beamformed_pymust._combine_rx_halves`.

    Reads each receive-event slice (`n_ch × n_samp`, a few MB) directly from an
    open h5py `RcvData` dataset instead of from a pre-materialised float32 copy
    of the entire `(n_rx, total_samples)` per-frame buffer. The numerics, group
    semantics, and `pymust.rf2iq` call are identical to the reference helper.
    """
    n_samp, n_elem, fs, fc = meta["n_samp"], meta["n_elem"], meta["fs"], meta["fc"]
    iq192 = np.zeros((n_samp, n_elem), dtype=np.complex64)
    count = np.zeros(n_elem, dtype=np.float32)

    for ev_i, rx_ap in group["rx_events"]:
        s = ev_i * n_samp
        rf = rcv_ds[frame_idx, :, s : s + n_samp]  # h5py read; int16, (n_ch, n_samp)
        if rf.shape[1] != n_samp:
            continue
        iq = pymust.rf2iq(rf.T.astype(np.float64), fs, fc)  # (n_samp, n_ch)
        n_ch = iq.shape[1]
        # HVMux receive aperture maps hardware channels onto physical elements
        # *cyclically*, not linearly: element (ap0+i) is read by channel
        # (ap0+i) mod n_ch (cf. the [aperture:end, 1:aperture-1] reorder in
        # ProcessDataMultiFocal_L12_3v.m). For rx_ap=65 that means channels
        # 65-128 -> elements 65-128 and channels 1-64 -> elements 129-192.
        phys = (rx_ap - 1) + np.arange(n_ch)          # destination physical element
        src = ((rx_ap - 1) + np.arange(n_ch)) % n_ch  # source hardware channel
        valid = phys < n_elem
        iq192[:, phys[valid]] += iq[:, src[valid]]
        count[phys[valid]] += 1.0

    np.divide(iq192, count[np.newaxis, :], out=iq192, where=count[np.newaxis, :] > 0)
    return iq192


def process_frame(
    filepath: Path,
    frame_idx: int,
    meta: dict,
    origin_index: dict[float, int],
    focal_index: dict[float, int],
    n_elem: int,
    n_sl: int,
    n_fz: int,
) -> np.ndarray:
    """
    Return one frame shaped (n_elem, n_sl, n_fz, 2, 215) float32.

    Opens the .mat file via h5py and reads only the per-event RF slices each
    group actually needs (a few MB total per group, ~2.4 GB avoided per frame).
    Per (scanline, focal-zone) group:
        1. stream + combine the two RX sub-apertures (IQ demodulated by pymust.rf2iq)
        2. truncate fast-time axis at DEPTH_CUTOFF samples
        3. decimate by DOWNSAMPLE_FACTOR
        4. split complex into (I, Q) and place into the output grid
    """
    out = np.zeros((n_elem, n_sl, n_fz, 2, N_SAMPLES_OUT), dtype=np.float32)
    with h5py.File(filepath, "r") as _f:
        f: Any = _f
        rcv = f[f["RcvData"][()].flat[0]]
        for group in meta["groups"]:
            sl_i = origin_index[round(group["origin_x"], 6)]
            fz_i = focal_index[round(group["focus_m"], 6)]

            iq = _combine_rx_halves_streaming(group, rcv, frame_idx, meta)

            if iq.shape[0] < DEPTH_CUTOFF:
                pad = np.zeros(
                    (DEPTH_CUTOFF - iq.shape[0], iq.shape[1]), dtype=iq.dtype
                )
                iq = np.vstack([iq, pad])
            iq = iq[:DEPTH_CUTOFF, :][::DOWNSAMPLE_FACTOR, :]  # (215, n_elem)

            iq_t = iq.T  # (n_elem, 215)
            out[:, sl_i, fz_i, 0, :] = iq_t.real
            out[:, sl_i, fz_i, 1, :] = iq_t.imag
    return out


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Root folder to recursively scan for MultiFocal .mat files.")
    parser.add_argument("--output_file", type=Path, required=True,
                        help="Destination .zarr.zip path (e.g. ultrasonic_dataset.zarr.zip).")
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_file: Path = args.output_file.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input_dir is not a directory: {input_dir}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mat_files = discover_files(input_dir)
    if not mat_files:
        raise SystemExit(f"No MultiFocal .mat files found under {input_dir}")
    print(f"Discovered {len(mat_files)} MultiFocal .mat files.", flush=True)

    # ── Pass 1: index files, validate per-frame shape, count total frames ──
    records: list[dict] = []
    n_elem_ref = n_sl_ref = n_fz_ref = None
    origins_ref: list[float] | None = None
    focal_ref: list[float] | None = None
    total_frames = 0

    for fp in tqdm(mat_files, desc="Indexing"):
        try:
            meta = load_metadata(fp)
            origins, focal_depths = file_grid(meta)
            nf = file_n_frames(fp)
        except Exception as exc:
            print(f"SKIP (unreadable): {fp} — {exc}", flush=True)
            traceback.print_exc()
            continue

        if n_elem_ref is None:
            n_elem_ref, n_sl_ref, n_fz_ref = (
                meta["n_elem"], len(origins), len(focal_depths)
            )
            origins_ref, focal_ref = origins, focal_depths
        elif (meta["n_elem"], len(origins), len(focal_depths)) != (
            n_elem_ref, n_sl_ref, n_fz_ref
        ):
            print(
                f"SKIP (shape mismatch): {fp} — "
                f"got (n_elem={meta['n_elem']}, n_sl={len(origins)}, "
                f"n_fz={len(focal_depths)}); expected "
                f"({n_elem_ref}, {n_sl_ref}, {n_fz_ref})",
                flush=True,
            )
            continue

        records.append({"path": fp, "meta": meta, "n_frames": nf})
        total_frames += nf

    if total_frames == 0:
        raise SystemExit("No usable frames found.")

    print(
        f"Allocating Zarr: ({total_frames}, {n_elem_ref}, {n_sl_ref}, "
        f"{n_fz_ref}, 2, {N_SAMPLES_OUT}) float32, chunks={CHUNKS}",
        flush=True,
    )

    assert origins_ref is not None and focal_ref is not None
    origin_index = {o: i for i, o in enumerate(origins_ref)}
    focal_index = {fz: i for i, fz in enumerate(focal_ref)}

    # ── Pass 2: process frames, write into Zarr ZipStore ──
    store = zarr.ZipStore(str(output_file), mode="w")
    try:
        root = zarr.group(store=store)
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
        arr = root.create_dataset(
            ARRAY_NAME,
            shape=(total_frames, n_elem_ref, n_sl_ref, n_fz_ref, 2, N_SAMPLES_OUT),
            chunks=CHUNKS,
            dtype="float32",
            compressor=compressor,
        )
        arr.attrs.put({
            "axes": ["frame", "transducer", "scanline", "focal_zone", "iq", "sample"],
            "iq_index": {"in_phase": 0, "quadrature": 1},
            "depth_cutoff_samples": DEPTH_CUTOFF,
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "scanline_origins_m": list(origins_ref),
            "focal_depths_m": list(focal_ref),
        })

        frame_metadata: list[dict] = []
        global_idx = 0
        written_files = 0

        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
            for rec in records:
                fp: Path = rec["path"]
                meta = rec["meta"]
                nf = rec["n_frames"]
                subject, subfolder = relative_parts(fp, input_dir)
                rel_path = str(fp.relative_to(input_dir))

                file_ok = True
                frames_done_in_file = 0
                for local_i in range(nf):
                    try:
                        frame_arr = process_frame(
                            fp, local_i, meta, origin_index, focal_index,
                            n_elem_ref, n_sl_ref, n_fz_ref,
                        )
                        arr[global_idx] = frame_arr
                        frame_metadata.append({
                            "global_frame_index": global_idx,
                            "filename": fp.name,
                            "relative_path": rel_path,
                            "subject": subject,
                            "subfolder": subfolder,
                            "file_local_frame_index": local_i,
                        })
                        global_idx += 1
                        frames_done_in_file += 1
                        pbar.update(1)
                    except Exception as exc:
                        print(f"\nSKIP frame {local_i} of {fp}: {exc}", flush=True)
                        traceback.print_exc()
                        file_ok = False
                        pbar.update(nf - local_i)
                        break
                if file_ok:
                    written_files += 1
                # Release this file's meta (groups[] holds numpy arrays); over
                # 80+ files the cached metas would otherwise tie up ~100s of MB.
                rec.pop("meta", None)
                del meta
    finally:
        store.close()

    # ── Companion JSON metadata ──
    metadata_path = output_file.parent / (
        output_file.name.replace(".zarr.zip", "") + ".metadata.json"
    )
    payload = {
        "array_name": ARRAY_NAME,
        "shape": [global_idx, n_elem_ref, n_sl_ref, n_fz_ref, 2, N_SAMPLES_OUT],
        "axes": ["frame", "transducer", "scanline", "focal_zone", "iq", "sample"],
        "chunks": list(CHUNKS),
        "dtype": "float32",
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": Blosc.SHUFFLE},
        "depth_cutoff_samples": DEPTH_CUTOFF,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "n_samples_out": N_SAMPLES_OUT,
        "iq_index": {"in_phase": 0, "quadrature": 1},
        "scanline_origins_m": list(origins_ref),
        "focal_depths_m": list(focal_ref),
        "input_dir": str(input_dir),
        "n_source_files_indexed": len(records),
        "n_source_files_written": written_files,
        "frames": frame_metadata,
    }
    metadata_path.write_text(json.dumps(payload, indent=2))

    print(f"\nWrote {global_idx} frames → {output_file}", flush=True)
    print(f"Metadata → {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
