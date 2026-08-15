"""Fetch the per-pixel label maps for all 923 frames — MEMORY.md §8 item 2.

WI-13's crop-level anatomy targets (`contains_cyst`, crop `cyst_fraction`, crop
`sos_mean`/`sos_std`) are blocked on the cluster because only the K-sector store
survived: `frame_metadata.json` carries frame-level *scalars*, while the labels
those targets need are the per-pixel `segmentation_map` and `sound_speed_map`
that live in the source NVIDIA rows (see `eval/store_features.py`'s docstring).

Downloading the corpus again is not required. The payload is `iq_real`/`iq_imag`
at 265.4 MB/frame; **everything else is 2.1 MB/frame** (PLAN.md §2.2), and
parquet is columnar, so a column-pruned read over HTTP range requests fetches
~1.9 GB instead of 271 GB. This is the same technique every number in PLAN.md §2
was measured with, made durable.

What it writes (`.npy` per column, memory-mapped, so nothing is ever fully
resident and a partial run resumes at shard granularity):

    <out>/seg.npy            (923, 507, 456) uint8   — 0 tissue, 1 cyst
    <out>/sos.npy            (923,  32,  32) float32 — m/s, on its own extent
    <out>/bmode.npy          (923, 507, 456) float32 — DAS envelope, not log-compressed
    <out>/bmode_focused.npy  (923, 507, 456) float32 — focused-transmit envelope
    <out>/scalars.json       phase_error, t0, frame_idx, extents, frame_ids
    <out>/manifest.json      per-shard completion + the verification result

Frame ids are `"{split}/{index}"` — the same convention `scripts/wi8_build.py`
assigns, so row *i* here is the frame at store index *i*. That is asserted, not
assumed: `--verify` recomputes `cyst_fraction`, `sos_mean`, `sos_std`,
`phase_error` and `t0` from the fetched maps and compares them against the
store's `frame_metadata.json`. A misalignment would silently attach the wrong
anatomy to every crop, so the check runs by default and is fatal.

Usage (a `data` instance — this is pure I/O, no GPU):

    source /opt/venvs/.venv/bin/activate
    HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
      python scripts/wi13_fetch_label_maps.py --out /mnt/home/benbarkow/stores/wi8_label_maps

`--drop bmode_focused` (repeatable) skips a column if disk is tight; the
anatomy targets need only `seg` and `sos`, which are 217 MB of the 1.9 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, list_repo_files

HF_REPO = "nvidia/NV-Raw2Insights-US"
SPLITS = ("train", "validation")
STORE_METADATA = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr/frame_metadata.json"

# name -> (parquet column, per-frame shape, dtype). Shapes are PLAN.md §2.2's,
# verified against the Arrow schema; a row that disagrees raises rather than
# reshaping into nonsense.
MAP_COLUMNS: dict[str, tuple[str, tuple[int, int], type]] = {
    "seg": ("segmentation_map", (507, 456), np.uint8),
    "sos": ("sound_speed_map", (32, 32), np.float32),
    "bmode": ("bmode", (507, 456), np.float32),
    "bmode_focused": ("bmode_focused", (507, 456), np.float32),
}
# Scalars and the two extents. Cheap, and `phase_error`/`t0` are what the
# alignment check keys on.
SCALAR_COLUMNS = ("phase_error", "t0", "frame_idx")
EXTENT_COLUMNS = ("bmode_extent", "sound_speed_extent")


def _as_array(col) -> pa.Array:
    """A ChunkedArray with one row group behind it, as a plain Array."""
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks() if col.num_chunks == 1 else pa.concat_arrays(col.chunks)
    return col


def nested_to_ndarray(col, shape: tuple[int, int], dtype) -> np.ndarray:
    """Decode a `list<list<T>>` column to `(n, *shape)` without a Python loop.

    `as_py()` on a 507x456 cell builds 231k Python objects; over 923 frames and
    three float columns that dominates the runtime by orders of magnitude.
    `flatten()` walks the offsets in C and hands back one buffer.
    """
    arr = _as_array(col)
    n = len(arr)
    flat = arr.flatten().flatten()
    expected = n * shape[0] * shape[1]
    if len(flat) != expected:
        raise ValueError(f"expected {expected} values for {n} frames of {shape}, got {len(flat)}")
    return flat.to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(n, *shape)


def shard_list(repo: str, splits=SPLITS) -> list[tuple[str, str]]:
    """`(split, path)` for every parquet shard, in the order WI-8 indexed them."""
    files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    out: list[tuple[str, str]] = []
    for split in splits:
        out.extend(
            (split, f) for f in sorted(f for f in files if f.split("/")[-1].startswith(f"{split}-"))
        )
    return out


def split_sizes(store_metadata: dict, splits=SPLITS) -> dict[str, int]:
    """Frame count per split, taken from the store's own metadata.

    Deliberately not from `load_dataset_builder`: the arrays written here are
    indexed *against the store*, so the store is the authority on how many rows
    each split has. If the two ever disagreed, the verification pass is where
    that should surface.
    """
    sizes = dict.fromkeys(splits, 0)
    for fid in store_metadata:
        split = fid.split("/")[0]
        if split in sizes:
            sizes[split] += 1
    return sizes


class MapWriter:
    """Memory-mapped `(N, ...)` arrays, one per column, written shard by shard."""

    def __init__(self, out: Path, n_frames: int, columns: list[str]):
        self.out = out
        self.n_frames = n_frames
        self.columns = columns
        self.arrays: dict[str, np.memmap] = {}
        for name in columns:
            _, shape, dtype = MAP_COLUMNS[name]
            path = out / f"{name}.npy"
            mode = "r+" if path.exists() else "w+"
            self.arrays[name] = np.lib.format.open_memmap(
                path, mode=mode, dtype=dtype, shape=(n_frames, *shape)
            )

    def write(self, name: str, start: int, block: np.ndarray) -> None:
        self.arrays[name][start : start + len(block)] = block

    def flush(self) -> None:
        for a in self.arrays.values():
            a.flush()


def fetch(
    out: Path,
    repo: str = HF_REPO,
    columns: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Column-pruned fetch of every shard into `out`. Resumable at shard level."""
    columns = columns or list(MAP_COLUMNS)
    out.mkdir(parents=True, exist_ok=True)
    store_metadata = json.loads(Path(STORE_METADATA).read_text())
    sizes = split_sizes(store_metadata)
    n_total = sum(sizes.values())
    print(f"corpus: {sizes} = {n_total} frames | columns: {', '.join(columns)}")

    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if resume and manifest_path.exists()
        else {"repo": repo, "columns": columns, "shards": {}, "sizes": sizes}
    )
    if sorted(manifest.get("columns", [])) != sorted(columns):
        # A resumed run with a different column set would leave the new column's
        # array zero-filled for every shard already marked done.
        raise SystemExit(
            f"manifest was built for columns {manifest['columns']}, "
            f"asked for {columns} — use --no-resume or a different --out"
        )
    done: dict[str, dict] = manifest["shards"]

    scalars: dict[str, dict[str, float]] = {}
    scalars_path = out / "scalars.json"
    if resume and scalars_path.exists():
        scalars = json.loads(scalars_path.read_text())["per_frame"]

    fs = HfFileSystem()
    read_columns = [MAP_COLUMNS[c][0] for c in columns]
    read_columns += list(SCALAR_COLUMNS) + list(EXTENT_COLUMNS)
    writer = MapWriter(out, n_total, columns)

    split_offset = {"train": 0, "validation": sizes["train"]}
    index_in_split = dict.fromkeys(SPLITS, 0)
    extents: dict[str, list[float]] = {}
    t_start, n_seen, bytes_est = time.time(), 0, 0

    for split, shard in shard_list(repo, tuple(sizes)):
        if limit is not None and n_seen >= limit:
            break
        record = done.get(shard)
        if record is not None:
            index_in_split[split] += record["n_rows"]
            n_seen += record["n_rows"]
            continue

        t0 = time.time()
        with fs.open(f"datasets/{repo}/{shard}", "rb") as fh:
            table = pq.ParquetFile(fh).read(columns=read_columns)
        n_rows = table.num_rows
        base = split_offset[split] + index_in_split[split]

        for name in columns:
            src, shape, dtype = MAP_COLUMNS[name]
            block = nested_to_ndarray(table[src], shape, dtype)
            writer.write(name, base, block)
            bytes_est += block.nbytes
        for i in range(n_rows):
            fid = f"{split}/{index_in_split[split] + i}"
            scalars[fid] = {
                "phase_error": float(table["phase_error"][i].as_py()),
                "t0": float(table["t0"][i].as_py()),
                "frame_idx": int(table["frame_idx"][i].as_py()),
            }
        for name in EXTENT_COLUMNS:
            extents.setdefault(name, [float(v) for v in table[name][0].as_py()])

        writer.flush()
        index_in_split[split] += n_rows
        n_seen += n_rows
        done[shard] = {"split": split, "n_rows": n_rows, "start": base}
        manifest["shards"] = done
        manifest_path.write_text(json.dumps(manifest, indent=1))
        elapsed = time.time() - t_start
        print(
            f"[{n_seen:4d}/{n_total}] {shard.split('/')[-1]:34s} "
            f"{n_rows:2d} rows in {time.time() - t0:5.1f}s | "
            f"{bytes_est / 1e6 / max(elapsed, 1e-9):5.1f} MB/s decoded | "
            f"{elapsed / 60:5.1f} min elapsed",
            flush=True,
        )

    frame_ids = [f"train/{i}" for i in range(sizes["train"])] + [
        f"validation/{i}" for i in range(sizes["validation"])
    ]
    if extents:
        manifest["extents"] = extents
    scalars_path.write_text(
        json.dumps({"frame_ids": frame_ids, "extents": extents, "per_frame": scalars}, indent=1)
    )
    manifest["n_frames"] = n_total
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} — {sum(a.nbytes for a in writer.arrays.values()) / 1e9:.2f} GB on disk")
    return manifest


def verify(out: Path, tol: float = 1e-4) -> dict:
    """Re-derive the store's frame-level scalars from the fetched maps.

    The store's `frame_metadata.json` was written by an entirely separate pass
    (WI-8, over the streaming reader). Reproducing its `cyst_fraction`,
    `sos_mean`, `sos_std`, `phase_error` and `t0` from these arrays therefore
    checks *frame alignment*, which is the only thing that can silently
    invalidate every crop-level label built on top.
    """
    meta = json.loads(Path(STORE_METADATA).read_text())
    scalars = json.loads((out / "scalars.json").read_text())
    frame_ids = scalars["frame_ids"]
    seg = np.load(out / "seg.npy", mmap_mode="r")
    sos = np.load(out / "sos.npy", mmap_mode="r")

    worst = {"cyst_fraction": 0.0, "sos_mean": 0.0, "sos_std": 0.0, "phase_error": 0.0, "t0": 0.0}
    n_empty = 0
    for i, fid in enumerate(frame_ids):
        want = meta[fid]
        got_seg = np.asarray(seg[i], dtype=np.float64)
        got_sos = np.asarray(sos[i], dtype=np.float64)
        n_empty += int(got_seg.max() == 0)
        checks = {
            "cyst_fraction": (float(got_seg.mean()), want["cyst_fraction"]),
            "sos_mean": (float(got_sos.mean()), want["sos_mean"]),
            "sos_std": (float(got_sos.std()), want["sos_std"]),
            "phase_error": (scalars["per_frame"][fid]["phase_error"], want["phase_error"]),
            "t0": (scalars["per_frame"][fid]["t0"], want["t0"]),
        }
        for key, (got, exp) in checks.items():
            rel = abs(got - exp) / max(abs(exp), 1e-12)
            worst[key] = max(worst[key], rel)
            if rel > tol:
                raise SystemExit(
                    f"MISALIGNED at {fid} ({key}): fetched {got!r} vs store {exp!r} "
                    f"(rel {rel:.2e}). Frame ordering does not match the store — "
                    f"do not build crop labels on this."
                )
        if want["segmentation_empty"] != (got_seg.max() == 0):
            raise SystemExit(f"MISALIGNED at {fid}: segmentation_empty disagrees with the map")

    result = {
        "n_frames": len(frame_ids),
        "worst_relative_error": worst,
        "empty_segmentation_frames": n_empty,
        "tolerance": tol,
    }
    print(json.dumps(result, indent=1))
    print(f"✔ all {len(frame_ids)} frames agree with the store's own metadata")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/home/benbarkow/stores/wi8_label_maps")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument(
        "--drop", action="append", default=[], choices=sorted(MAP_COLUMNS), help="skip a column"
    )
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames (smoke run)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    columns = [c for c in MAP_COLUMNS if c not in args.drop]
    if not args.verify_only:
        manifest = fetch(out, args.repo, columns, args.limit, resume=not args.no_resume)
        if args.limit is not None:
            print("partial fetch (--limit) — skipping verification")
            return
    if not args.no_verify:
        result = verify(out)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["verification"] = result
        manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
