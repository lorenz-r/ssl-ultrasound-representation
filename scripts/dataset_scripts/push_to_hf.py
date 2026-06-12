"""
Push the aggregated MultiFocal ultrasound dataset to HuggingFace.

The local build artefact is a single `ultrasonic_dataset.zarr.zip`, but uploading
it as one opaque blob forces every consumer to download the *whole* array. Instead
this script unpacks the zip and uploads the zarr as an **unzipped directory tree**
(`ultrasonic_dataset.zarr/`, one file per chunk) so consumers can fetch only the
chunks they need — e.g. just the first N frames (see `max_frames` in
`src/dataset.py`).

Uploads to a HF dataset repo:
  * ultrasonic_dataset.zarr/          — 6D float32 array, exploded into per-chunk files
  * ultrasonic_dataset.metadata.json  — per-frame provenance
  * README.md                          — auto-generated dataset card (unless --no-readme)

The repo is created (or reused if it exists) as public by default. Authentication
falls through `--token` -> $HF_TOKEN -> $HUGGING_FACE_HUB_TOKEN -> cached
`huggingface-cli login`.

Usage:
    poetry run python scripts/dataset_scripts/push_to_hf.py \\
        --repo_id user/ssl-ultrasound-rats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import zarr
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.dataset import (  # noqa: E402
    DEFAULT_ARRAY_NAME,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_ZARR_DIRNAME,
    DEFAULT_ZARR_FILENAME,
)

DEFAULT_ZIP_PATH = REPO_ROOT / "data" / "processed" / DEFAULT_ZARR_FILENAME
DEFAULT_META_PATH = REPO_ROOT / "data" / "processed" / DEFAULT_METADATA_FILENAME


# ── validation ────────────────────────────────────────────────────────────────
def validate_inputs(zip_path: Path, meta_path: Path) -> tuple[dict, tuple[int, ...]]:
    """Open both artefacts, check they agree, return (metadata, array_shape)."""
    if not zip_path.is_file():
        raise SystemExit(f"zarr.zip not found: {zip_path}")
    if not meta_path.is_file():
        raise SystemExit(f"metadata.json not found: {meta_path}")

    metadata = json.loads(meta_path.read_text())

    store = zarr.ZipStore(str(zip_path), mode="r")
    try:
        group = zarr.open_group(store=store, mode="r")
        if DEFAULT_ARRAY_NAME not in group:
            raise SystemExit(
                f"zarr group has no '{DEFAULT_ARRAY_NAME}' array; "
                f"present arrays: {list(group.array_keys())}"
            )
        arr = group[DEFAULT_ARRAY_NAME]
        arr_shape = tuple(arr.shape)
        if len(arr_shape) != 6:
            raise SystemExit(f"expected 6D array, got shape {arr_shape}")
    finally:
        store.close()

    meta_shape = tuple(metadata.get("shape", ()))
    if meta_shape != arr_shape:
        raise SystemExit(
            f"shape mismatch: metadata says {meta_shape}, zarr says {arr_shape}"
        )
    n_frames_listed = len(metadata.get("frames", []))
    if n_frames_listed != arr_shape[0]:
        raise SystemExit(
            f"metadata['frames'] has {n_frames_listed} entries but zarr has "
            f"{arr_shape[0]} frames"
        )
    return metadata, arr_shape


# ── zip → directory tree ────────────────────────────────────────────────────────
def extract_zarr_zip(zip_path: Path, dest_dir: Path) -> int:
    """
    Explode a zarr `.zip` into a plain directory tree under `dest_dir`.

    A `zarr.ZipStore` stores each key (`.zgroup`, `rf_iq/.zarray`,
    `rf_iq/0.0.0.0.0.0`, …) as a flat zip entry whose name *is* the store key, so
    a straight extract reproduces an equivalent `DirectoryStore` layout — exactly
    what `upload_folder` needs to expose one HF file per chunk. Returns the number
    of chunk/metadata files written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        zf.extractall(dest_dir)
    return len(members)


# ── README generation ─────────────────────────────────────────────────────────
def generate_readme(repo_id: str, metadata: dict, zip_size_bytes: int) -> str:
    frames = metadata["frames"]
    shape = metadata["shape"]
    axes = metadata.get("axes", [])
    chunks = metadata.get("chunks", [])

    subject_counts = Counter(f["subject"] for f in frames)
    pair_counts = Counter((f["subject"], f.get("subfolder") or "") for f in frames)

    def fmt_bytes(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} GB"

    subject_rows = "\n".join(
        f"| `{s}` | {n} |" for s, n in sorted(subject_counts.items())
    )
    subfolder_rows = "\n".join(
        f"| `{s}` | `{sf or '—'}` | {n} |"
        for (s, sf), n in sorted(pair_counts.items())
    )

    return f"""---
pretty_name: SSL Ultrasound Representation (Stanford RF MultiFocal)
license: other
task_categories:
- feature-extraction
tags:
- ultrasound
- self-supervised
- rf-data
- mae
- jepa
size_categories:
- n<1K
---

# SSL Ultrasound Representation Dataset

IQ-demodulated multi-focal ultrasound channel data from the Stanford Ultrasound RF
Channel dataset, aggregated for self-supervised pre-training (MAE / JEPA /
contrastive). The signals are **not** beamformed — each frame retains its
per-transducer channel data after sub-aperture combination and IQ demodulation.

## Files

| File | Description |
|------|-------------|
| `{DEFAULT_ZARR_DIRNAME}/` | 6D float32 zarr array, Blosc-LZ4 compressed, stored as an unzipped directory tree (one file per chunk, ~{fmt_bytes(zip_size_bytes)} total). One chunk per frame on the leading axis → consumers can fetch just the first N frames. |
| `{DEFAULT_METADATA_FILENAME}` | Per-frame provenance (subject, file, subfolder, indices). |
| `README.md` | This card (auto-generated by `push_to_hf.py`). |

## Array layout

- **Name:** `{DEFAULT_ARRAY_NAME}`
- **Shape:** `{tuple(shape)}`
- **Axes:** `{axes}`
- **Chunks:** `{tuple(chunks)}` (one frame per chunk on the leading axis — aligned with DataLoader workers).
- **Dtype:** `float32`
- **IQ index:** `0` = in-phase, `1` = quadrature.

## Processing pipeline

1. Source: Verasonics Vantage `.mat` files with `MultiFocal` in the filename
   (each file holds a 2-frame cine via `Resource.RcvBuffer.numFrames = 2`).
2. Sub-aperture combination + IQ demodulation via `pymust.rf2iq`.
3. Fast-time axis truncated at **{metadata.get("depth_cutoff_samples")} samples**
   (everything beyond is noise/zeros), then decimated by
   **{metadata.get("downsample_factor")}×** → **{metadata.get("n_samples_out")}** output samples per signal.
4. Stacked into a single 6D array with chunks optimised for 3D MAE patching.

See `scripts/dataset_scripts/build_mae_dataset.py` in the source repo for the
exact pipeline; `scripts/dataset_scripts/visualize_beamformed_pymust.py` beamforms
the same channel data for *visualisation only* (the stored array stays unbeamformed).

## Frame counts

By subject:

| Subject | Frames |
|---------|--------|
{subject_rows}

By subject × subfolder:

| Subject | Subfolder | Frames |
|---------|-----------|--------|
{subfolder_rows}

## Loading

```python
from src.dataset import build_split_datasets

splits, meta = build_split_datasets(
    "{repo_id}",
    split_strategy="file",  # or "subject"
    cache_dir="/tmp/hf_cache",
    max_frames=16,          # optional: fetch only the first 16 frames' chunks
)
train_ds = splits["train"]
```

`max_frames` exploits the per-frame chunking: only the chunks for the first N
frames are downloaded, so you can iterate on a small box without pulling the whole
array. Omit it to fetch everything.

The split helpers (`split_by_subject`, `split_by_file`) avoid cross-split leakage:
both frames of the same 2-frame cine always end up in the same split, and (for
`split_by_subject`) no subject appears in two splits.
"""


# ── upload ────────────────────────────────────────────────────────────────────
def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    return None  # HfApi falls back to cached `huggingface-cli login`


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True,
                        help="Target HF dataset repo, e.g. user/ssl-ultrasound-rats")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH,
                        help=f"Path to zarr.zip (default: {DEFAULT_ZIP_PATH})")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META_PATH,
                        help=f"Path to metadata.json (default: {DEFAULT_META_PATH})")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private (default: public).")
    parser.add_argument("--no-readme", action="store_true",
                        help="Skip auto-generating and uploading README.md.")
    parser.add_argument("--token", default=None,
                        help="HF token. Falls through $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN "
                             "/ cached huggingface-cli login if omitted.")
    parser.add_argument("--commit-message", default=None,
                        help="Override the per-file commit message.")
    parser.add_argument("--revision", default=None,
                        help="Target branch on the dataset repo (default: main).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and preview the README; no network writes.")
    args = parser.parse_args()

    zip_path: Path = args.zip.resolve()
    meta_path: Path = args.metadata.resolve()

    print(f"Validating inputs…", flush=True)
    metadata, arr_shape = validate_inputs(zip_path, meta_path)
    n_frames = arr_shape[0]
    n_subjects = len({f["subject"] for f in metadata["frames"]})
    zip_size = zip_path.stat().st_size
    print(f"  zarr.zip:     {zip_path}  ({zip_size / 1e9:.2f} GB, shape {arr_shape})")
    print(f"  metadata:     {meta_path}  ({n_frames} frames, {n_subjects} subjects)")

    readme_text = None
    if not args.no_readme:
        readme_text = generate_readme(args.repo_id, metadata, zip_size)

    commit_message = args.commit_message or (
        f"Upload ultrasonic dataset ({n_frames} frames, {n_subjects} subjects)"
    )

    if args.dry_run:
        print("\n[dry-run] would create repo:")
        print(f"  repo_id  = {args.repo_id}")
        print(f"  private  = {args.private}")
        print(f"  type     = dataset")
        print("[dry-run] would upload:")
        print(f"  - {DEFAULT_ZARR_DIRNAME}/  (unzipped chunk tree from "
              f"{zip_path.name}, {zip_size / 1e9:.2f} GB)")
        print(f"  - {meta_path.name}")
        if readme_text is not None:
            print(f"  - README.md  ({len(readme_text)} chars)")
            print("\n[dry-run] README preview (first 40 lines):")
            print("\n".join(readme_text.splitlines()[:40]))
            print("…")
        print(f"\n[dry-run] commit message: {commit_message!r}")
        return

    token = resolve_token(args.token)
    api = HfApi(token=token)

    print(f"\nEnsuring repo exists: {args.repo_id} (private={args.private})…", flush=True)
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(
            f"Failed to create/access repo {args.repo_id!r}: {exc}\n"
            f"Hint: pass --token, set $HF_TOKEN, or run `huggingface-cli login`."
        )

    def _upload(local: Path, name: str, content: bytes | None = None) -> None:
        print(f"Uploading {name}…", flush=True)
        kwargs = dict(
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            revision=args.revision,
        )
        if content is not None:
            api.upload_file(path_or_fileobj=content, **kwargs)
        else:
            api.upload_file(path_or_fileobj=str(local), **kwargs)

    if readme_text is not None:
        _upload(Path("README.md"), "README.md", content=readme_text.encode("utf-8"))
    _upload(meta_path, DEFAULT_METADATA_FILENAME)

    # Explode the zip into a per-chunk directory tree and upload it as a folder so
    # each chunk becomes its own HF file (enables partial / first-N-frame fetches).
    with tempfile.TemporaryDirectory(prefix="zarr_upload_") as tmp:
        zarr_dir = Path(tmp) / DEFAULT_ZARR_DIRNAME
        print(f"Unpacking {zip_path.name} → {DEFAULT_ZARR_DIRNAME}/ …", flush=True)
        n_files = extract_zarr_zip(zip_path, zarr_dir)
        print(f"Uploading {DEFAULT_ZARR_DIRNAME}/ ({n_files} chunk files)…", flush=True)
        api.upload_folder(
            folder_path=str(zarr_dir),
            path_in_repo=DEFAULT_ZARR_DIRNAME,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            revision=args.revision,
        )

    print(f"\nDone. Repo: https://huggingface.co/datasets/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
