"""Sequentially convert local NV-Raw2Insights Parquet to MAE-ready Zarr.

The converter is designed for a Slurm node's local SSD: Parquet is scanned
exactly once in shard order, then training performs random access only against
the local Zarr directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc, blosc
from tqdm import tqdm

from ssl_ultrasound_representation.dataset.nv_raw2insights import (
    DEFAULT_NV_ROOT,
    NvidiaRaw2InsightsDataset,
    iter_nv_raw2insights_row_group,
    nv_iq_to_mae_frame,
    nv_sample_to_fsa_params,
)


ARRAY_NAME = "rf_iq"
GEOMETRY_COLUMNS = (
    "iq_real",
    "iq_imag",
    "bmode_extent",
    "fs",
    "fc",
    "fd",
    "t0",
    "c0",
    "elpos",
)


def _convert_row_group(
    task: dict,
    output_store: str,
    sample_stride: int,
) -> tuple[int, dict | None]:
    """Decode one Parquet row group and write its disjoint Zarr frame chunks."""
    blosc.set_nthreads(1)
    array = zarr.open(output_store, mode="r+", path=ARRAY_NAME)
    params = None
    written = 0
    for sample in iter_nv_raw2insights_row_group(
        task["path"],
        row_group=task["row_group"],
        split=task["split"],
        split_index=task["split_index"],
        columns=GEOMETRY_COLUMNS,
    ):
        if written >= task["n_rows"]:
            break
        array[task["global_index"] + written] = nv_iq_to_mae_frame(
            sample["iq_real"],
            sample["iq_imag"],
            sample_stride=sample_stride,
            keep_focal_zone=True,
        )
        if task["global_index"] == 0 and written == 0:
            params = nv_sample_to_fsa_params(
                sample,
                sample_stride=sample_stride,
            )
        written += 1
    if written != task["n_rows"]:
        raise RuntimeError(
            f"wrote {written} rows from {task['path']} row group "
            f"{task['row_group']}; expected {task['n_rows']}"
        )
    return written, params


def _build_tasks(
    input_root: Path,
    split_sizes: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []
    frame_metadata: list[dict] = []
    global_index = 0
    for split in ("train", "validation"):
        split_index = 0
        remaining = split_sizes[split]
        for path in sorted((input_root / "data").glob(f"{split}-*.parquet")):
            metadata = pq.read_metadata(path)
            for row_group in range(metadata.num_row_groups):
                if remaining == 0:
                    break
                n_rows = min(metadata.row_group(row_group).num_rows, remaining)
                tasks.append(
                    {
                        "path": str(path),
                        "row_group": row_group,
                        "split": split,
                        "split_index": split_index,
                        "global_index": global_index,
                        "n_rows": n_rows,
                    }
                )
                for row_offset in range(n_rows):
                    frame_metadata.append(
                        {
                            "global_frame_index": global_index + row_offset,
                            "split": split,
                            "subject": split,
                            "relative_path": (
                                f"{split}/{path.name}:{split_index + row_offset}"
                            ),
                            "filename": path.name,
                            "file_local_frame_index": split_index + row_offset,
                        }
                    )
                split_index += n_rows
                global_index += n_rows
                remaining -= n_rows
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError(f"missing {remaining} expected {split} rows")
    return tasks, frame_metadata


def output_is_complete(store_path: Path, metadata_path: Path, expected: int) -> bool:
    if not store_path.is_dir() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        array = zarr.open(str(store_path), mode="r", path=ARRAY_NAME)
    except Exception:
        return False
    return metadata.get("n_frames_written") == expected and array.shape[0] == expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_NV_ROOT)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--beamform-params", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Parquet row groups to decode (4 is suitable for 24 GB RAM).",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Smoke-test cap; omit for the full 830/93 conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1:
        raise SystemExit("--sample-stride must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.output_store.suffix != ".zarr":
        raise SystemExit("--output-store must end in .zarr")
    if args.max_samples_per_split is not None and args.max_samples_per_split < 1:
        raise SystemExit("--max-samples-per-split must be >= 1")

    split_sizes = {
        split: min(
            len(
                NvidiaRaw2InsightsDataset(
                    args.input_root,
                    split=split,
                    mode="sample",
                    columns=("fs",),
                )
            ),
            args.max_samples_per_split or 10**9,
        )
        for split in ("train", "validation")
    }
    n_frames = sum(split_sizes.values())
    if output_is_complete(args.output_store, args.metadata, n_frames):
        print(f"Reusing complete local NV Zarr: {args.output_store}", flush=True)
        return

    args.output_store.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.beamform_params.parent.mkdir(parents=True, exist_ok=True)
    partial_store = args.output_store.with_name(
        f"{args.output_store.name}.partial.{os.getpid()}"
    )
    if partial_store.exists():
        shutil.rmtree(partial_store)
    if args.output_store.exists():
        shutil.rmtree(args.output_store)

    n_samples = len(range(0, 1024, args.sample_stride))
    shape = (n_frames, 180, 180, 1, 2, n_samples)
    chunks = (1, 180, 180, 1, 2, min(64, n_samples))
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    root = zarr.open_group(str(partial_store), mode="w")
    array = root.create_dataset(
        ARRAY_NAME,
        shape=shape,
        chunks=chunks,
        dtype="float32",
        compressor=compressor,
    )
    array.attrs.put(
        {
            "acquisition_type": "fsa",
            "axes": [
                "frame",
                "transducer",
                "scanline",
                "focal_zone",
                "iq",
                "sample",
            ],
            "scanline_axis_semantics": "fsa_transmit_index",
            "iq_index": {"in_phase": 0, "quadrature": 1},
            "source": "NV-Raw2Insights-US",
            "sample_stride": args.sample_stride,
            "source_shape": [180, 180, 1024],
        }
    )

    tasks, frame_metadata = _build_tasks(args.input_root, split_sizes)
    params: dict | None = None
    converted = 0
    with tqdm(total=n_frames, desc="Converting NV Parquet", unit="sample") as progress:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _convert_row_group,
                    task,
                    str(partial_store),
                    args.sample_stride,
                )
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                n_written, task_params = future.result()
                converted += n_written
                params = task_params or params
                progress.update(n_written)

    if converted != n_frames or params is None:
        raise RuntimeError(f"converted {converted} samples, expected {n_frames}")

    os.replace(partial_store, args.output_store)
    metadata = {
        "array_name": ARRAY_NAME,
        "acquisition_type": "fsa",
        "source_dataset": str(args.input_root),
        "shape": list(shape),
        "axes": [
            "frame",
            "transducer",
            "scanline",
            "focal_zone",
            "iq",
            "sample",
        ],
        "scanline_axis_semantics": "fsa_transmit_index",
        "chunks": list(chunks),
        "dtype": "float32",
        "sample_stride": args.sample_stride,
        "publisher_split_sizes": split_sizes,
        "n_frames_written": converted,
        "frames": frame_metadata,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2))
    args.beamform_params.write_text(json.dumps(params, indent=2))
    print(f"Wrote {args.output_store}", flush=True)
    print(f"Wrote {args.metadata}", flush=True)
    print(f"Wrote {args.beamform_params}", flush=True)


if __name__ == "__main__":
    main()
