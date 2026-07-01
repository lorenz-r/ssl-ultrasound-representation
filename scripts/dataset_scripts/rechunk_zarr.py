"""
Rechunk an existing MultiFocal ultrasound `.zarr.zip` to **one chunk per frame**
without re-running the beamforming build.

The original builds chunked each frame into 108 spatial files
((1, 32, 32, 1, 2, 215)); with >~92 frames that exceeds HuggingFace's limit of
10,000 files per directory. This rewrites the array with full-axis spatial chunks
((1, n_elem, n_sl, n_fz, 2, n_samp)) → exactly one file per frame, and updates the
companion metadata.json `chunks` field to match. Frame values are copied verbatim.

Usage:
    poetry run python scripts/dataset_scripts/rechunk_zarr.py \\
        --input  data/processed/us_mf_dataset.zarr.zip \\
        --output data/processed/us_mf_dataset_rechunked.zarr.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import zarr
from numcodecs import Blosc
from tqdm import tqdm

ARRAY_NAME = "rf_iq"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Source .zarr.zip (108-files-per-frame layout).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .zarr.zip (one file per frame).")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Companion metadata.json to update in place with the new "
                             "chunks (default: <input stem>.metadata.json next to input).")
    args = parser.parse_args()

    in_path: Path = args.input.resolve()
    out_path: Path = args.output.resolve()
    if not in_path.is_file():
        raise SystemExit(f"input not found: {in_path}")
    if out_path.exists():
        raise SystemExit(f"output already exists (refusing to overwrite): {out_path}")

    src_store = zarr.ZipStore(str(in_path), mode="r")
    dst_store = zarr.ZipStore(str(out_path), mode="w")
    try:
        src = zarr.open_group(store=src_store, mode="r")[ARRAY_NAME]
        n_frames, n_elem, n_sl, n_fz, n_iq, n_samp = src.shape
        new_chunks = (1, n_elem, n_sl, n_fz, n_iq, n_samp)
        print(f"shape={src.shape}  old_chunks={src.chunks}  new_chunks={new_chunks}",
              flush=True)

        root = zarr.group(store=dst_store)
        dst = root.create_dataset(
            ARRAY_NAME,
            shape=src.shape,
            chunks=new_chunks,
            dtype=src.dtype,
            compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
        )
        dst.attrs.put(dict(src.attrs))  # preserve axes / iq_index / origins / …

        for f in tqdm(range(n_frames), desc="Rechunking", unit="frame"):
            dst[f] = src[f]  # read one frame (108 chunks), write it as a single chunk
    finally:
        src_store.close()
        dst_store.close()

    # Keep metadata.json's advertised chunk shape honest.
    meta_path = args.metadata or in_path.with_name(
        in_path.name.replace(".zarr.zip", "") + ".metadata.json"
    )
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta["chunks"] = list(new_chunks)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Updated chunks in {meta_path}", flush=True)
    else:
        print(f"WARNING: metadata not found at {meta_path}; its 'chunks' field is now "
              f"stale — regenerate or edit it to {list(new_chunks)}.", flush=True)

    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
