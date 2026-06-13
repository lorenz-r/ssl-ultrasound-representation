"""
HuggingFace-backed dataset loader for the aggregated MultiFocal ultrasound zarr.

The companion build script (`scripts/dataset_scripts/build_mae_dataset.py`)
produces a local `ultrasonic_dataset.zarr.zip` + `ultrasonic_dataset.metadata.json`;
the push script (`scripts/dataset_scripts/push_to_hf.py`) unpacks the zip and
uploads the zarr as an *unzipped directory tree* (one file per chunk) so that
clients can fetch only the chunks they actually need:
  * `ultrasonic_dataset.zarr/` — 6D float32 array exploded into per-chunk files
    `(frames, transducers, scanlines, focal_zones, 2, samples)`
  * `ultrasonic_dataset.metadata.json` — per-frame provenance

Because the leading (frame) axis is chunked one-frame-per-chunk, this module can
download just the chunks for the first `max_frames` frames instead of the whole
array — handy for quick experiments on a memory/disk-capped box. The files are
cached on local SSD (not `$HOME`); the module exposes a raw `(zarr.Array,
metadata)` handle for notebooks / EDA, and provides leakage-safe splits + a thin
`torch.utils.data.Dataset` wrapper for SSL training.

Typical usage:
    from ssl_ultrasound_representation.dataset import build_split_datasets
    splits, meta = build_split_datasets(
        "user/repo", split_strategy="subject", max_frames=16
    )
    train_ds = splits["train"]
"""

from __future__ import annotations

import json
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

import zarr
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_ZARR_FILENAME = "ultrasonic_dataset.zarr.zip"  # local build artefact (zipped)
DEFAULT_ZARR_DIRNAME = "ultrasonic_dataset.zarr"       # path_in_repo on HF (unzipped tree)
DEFAULT_METADATA_FILENAME = "ultrasonic_dataset.metadata.json"
DEFAULT_ARRAY_NAME = "rf_iq"
DEFAULT_CACHE_DIR = "/tmp/hf_cache"
CACHE_ENV_VAR = "HF_CACHE_DIR"
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# The zarr is exploded into thousands of small per-chunk files, so download
# speed is dominated by request concurrency rather than per-file bandwidth.
DEFAULT_DOWNLOAD_WORKERS = 16


# ── cache resolution ──────────────────────────────────────────────────────────
def resolve_cache_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Pick the HF cache root. Priority: explicit arg > $HF_CACHE_DIR > /tmp/hf_cache."""
    if explicit is not None:
        path = Path(explicit)
    else:
        env_val = os.environ.get(CACHE_ENV_VAR)
        path = Path(env_val) if env_val else Path(DEFAULT_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── auth resolution ───────────────────────────────────────────────────────────
def resolve_hf_token(explicit: str | None = None) -> str | None:
    """
    Pick the HF auth token. Priority: explicit arg > $HF_TOKEN >
    $HUGGING_FACE_HUB_TOKEN > None (huggingface_hub then falls back to a cached
    `huggingface-cli login`). An authenticated download raises rate limits and
    avoids anonymous throttling.
    """
    if explicit:
        return explicit
    for var in TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


# ── low-level accessors ───────────────────────────────────────────────────────
def _open_zarr_array(
    zip_path: str | os.PathLike, array_name: str = DEFAULT_ARRAY_NAME
) -> zarr.Array:
    store = zarr.ZipStore(str(zip_path), mode="r")
    group = zarr.open_group(store=store, mode="r")
    return group[array_name]


def _open_zarr_dir_array(
    zarr_dir: str | os.PathLike, array_name: str = DEFAULT_ARRAY_NAME
) -> zarr.Array:
    store = zarr.DirectoryStore(str(zarr_dir))
    group = zarr.open_group(store=store, mode="r")
    return group[array_name]


def load_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    zarr_dirname: str = DEFAULT_ZARR_DIRNAME,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    array_name: str = DEFAULT_ARRAY_NAME,
    repo_type: str = "dataset",
    max_frames: int | None = None,
    token: str | None = None,
    max_workers: int = DEFAULT_DOWNLOAD_WORKERS,
) -> tuple[zarr.Array, dict]:
    """
    Download the zarr (as a per-chunk directory tree) + metadata from a HF repo
    and open the array.

    Authentication: `token` (or $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN / cached
    `huggingface-cli login`) is used for every request — authenticated downloads
    get higher rate limits and avoid anonymous throttling. `max_workers` sets the
    chunk-download concurrency (the array is thousands of small files).

    When `max_frames` is set, only the chunks belonging to the first `max_frames`
    frames are fetched (plus the structural `.zarray`/`.zgroup` files), so the
    whole dataset is never materialised on disk. The returned `metadata` is
    trimmed to the loaded frames (its `frames` list and a new `n_frames_loaded`
    key reflect the cap), keeping the split helpers leakage-safe and bounded to
    what is actually available locally. Frames beyond the cap remain part of the
    array's declared shape but read back as the zarr fill value (0.0).
    """
    cache_root = resolve_cache_dir(cache_dir)
    token = resolve_hf_token(token)

    # Metadata first: it tells us the total frame count and lets us build the
    # per-frame chunk globs before pulling any (potentially large) chunk files.
    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename=metadata_filename,
        revision=revision,
        cache_dir=str(cache_root),
        repo_type=repo_type,
        token=token,
    )
    metadata = json.loads(Path(meta_path).read_text())
    n_total = int(metadata["shape"][0])

    array_prefix = f"{zarr_dirname}/{array_name}"
    if max_frames is None:
        n_load = n_total
        allow_patterns = [f"{zarr_dirname}/*"]
    else:
        n_load = max(0, min(int(max_frames), n_total))
        # The frame axis is chunked one-frame-per-chunk, so the leading chunk
        # index of every chunk for frame `f` is exactly `f` (e.g. `f.0.0.0.0.0`,
        # `f.1.2.0.0.0`, …). `{f}.*` selects all of that frame's spatial chunks.
        allow_patterns = [
            f"{zarr_dirname}/.zgroup",
            f"{zarr_dirname}/.zattrs",
            f"{array_prefix}/.zarray",
            f"{array_prefix}/.zattrs",
            *(f"{array_prefix}/{f}.*" for f in range(n_load)),
        ]

    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=str(cache_root),
        allow_patterns=allow_patterns,
        token=token,
        max_workers=max_workers,
    )
    arr = _open_zarr_dir_array(Path(snapshot_dir) / zarr_dirname, array_name=array_name)

    if max_frames is not None and n_load < n_total:
        metadata = dict(metadata)
        metadata["frames"] = [
            f for f in metadata.get("frames", [])
            if f["global_frame_index"] < n_load
        ]
        metadata["n_frames_loaded"] = n_load
    return arr, metadata


def load_local_dataset(
    zip_path: str | os.PathLike,
    metadata_path: str | os.PathLike,
    *,
    array_name: str = DEFAULT_ARRAY_NAME,
) -> tuple[zarr.Array, dict]:
    """Open a zarr.zip + metadata.json already on disk (no HF round-trip)."""
    arr = _open_zarr_array(zip_path, array_name=array_name)
    metadata = json.loads(Path(metadata_path).read_text())
    return arr, metadata


# ── split helpers ─────────────────────────────────────────────────────────────
def _allocate_ratios(
    items: list, ratios: tuple[float, float, float]
) -> tuple[list, list, list]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    return (
        items[:n_train],
        items[n_train : n_train + n_val],
        items[n_train + n_val :],
    )


def _assert_disjoint(groups: dict[str, list], label: str) -> None:
    seen: dict[Any, str] = {}
    for name, items in groups.items():
        for x in items:
            if x in seen:
                raise ValueError(
                    f"{label} {x!r} listed in both {seen[x]!r} and {name!r}"
                )
            seen[x] = name


def split_by_subject(
    metadata: dict,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    train_subjects: list[str] | None = None,
    val_subjects: list[str] | None = None,
    test_subjects: list[str] | None = None,
) -> dict[str, list[int]]:
    """
    Split global frame indices grouped by subject. No subject appears in two splits.

    Pass any of `train_subjects` / `val_subjects` / `test_subjects` to override the
    random allocation explicitly (useful when subject count is small).
    """
    frames = metadata["frames"]
    by_subject: dict[str, list[int]] = defaultdict(list)
    for fm in frames:
        by_subject[fm["subject"]].append(fm["global_frame_index"])
    subjects = sorted(by_subject.keys())

    explicit = any(
        s is not None for s in (train_subjects, val_subjects, test_subjects)
    )
    if explicit:
        train_s = list(train_subjects or [])
        val_s = list(val_subjects or [])
        test_s = list(test_subjects or [])
        for grp in (train_s, val_s, test_s):
            for s in grp:
                if s not in by_subject:
                    raise ValueError(
                        f"Subject {s!r} not present in metadata; have {subjects}"
                    )
        _assert_disjoint(
            {"train": train_s, "val": val_s, "test": test_s}, "subject"
        )
    else:
        if len(subjects) < 3:
            warnings.warn(
                f"split_by_subject called with only {len(subjects)} subjects "
                f"({subjects}); random allocation will be degenerate. Pass "
                f"explicit train/val/test_subjects for control.",
                stacklevel=2,
            )
        rng = random.Random(seed)
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        train_s, val_s, test_s = _allocate_ratios(shuffled, ratios)

    return {
        "train": sorted(i for s in train_s for i in by_subject[s]),
        "val": sorted(i for s in val_s for i in by_subject[s]),
        "test": sorted(i for s in test_s for i in by_subject[s]),
    }


def split_by_file(
    metadata: dict,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    group_by_subject: bool = True,
) -> dict[str, list[int]]:
    """
    Split at unique-file granularity. Both frames of the same `.mat` cine end up
    in the same split (each Verasonics file holds `numFrames=2` temporal frames).

    When `group_by_subject=True` (default), files are stratified within each
    subject so every subject is represented in every split.
    """
    frames = metadata["frames"]
    by_file: dict[str, list[int]] = defaultdict(list)
    file_subject: dict[str, str] = {}
    for fm in frames:
        rp = fm["relative_path"]
        by_file[rp].append(fm["global_frame_index"])
        file_subject[rp] = fm["subject"]

    rng = random.Random(seed)
    if group_by_subject:
        subject_files: dict[str, list[str]] = defaultdict(list)
        for rp, s in file_subject.items():
            subject_files[s].append(rp)
        train_files: list[str] = []
        val_files: list[str] = []
        test_files: list[str] = []
        for s in sorted(subject_files):
            files = sorted(subject_files[s])
            rng.shuffle(files)
            tr, vl, te = _allocate_ratios(files, ratios)
            train_files += tr
            val_files += vl
            test_files += te
    else:
        files = sorted(by_file)
        rng.shuffle(files)
        train_files, val_files, test_files = _allocate_ratios(files, ratios)

    return {
        "train": sorted(i for f in train_files for i in by_file[f]),
        "val": sorted(i for f in val_files for i in by_file[f]),
        "test": sorted(i for f in test_files for i in by_file[f]),
    }


# ── PyTorch wrapper (lazy torch import) ───────────────────────────────────────
class UltrasoundMAEDataset:
    """
    PyTorch Dataset returning one IQ-demodulated (unbeamformed) frame per item.

    Item shape: `(transducers, scanlines, focal_zones, 2, samples)`, or with
    `focal_zone` set, `(transducers, scanlines, 2, samples)` — the zone slice
    is then taken *at the zarr level*, which reads only that zone's chunks
    (the focal axis is chunked with size 1): 3x less I/O and decompression
    than slicing after a full-frame read.

    `preload=True` materialises all frames into one shared-memory tensor at
    construction (the transform still runs per access). At ~63 MB/frame for a
    single focal zone the whole dataset fits in CPU RAM, removing every
    per-epoch disk read and Blosc decompression — without it the GPU idles on
    chunk decompression, and each DataLoader worker respawn at an epoch
    boundary pays the first-frame latency again.

    Pass `indices` from a split helper to materialise a sub-dataset without
    copying the zarr array.
    """

    def __init__(
        self,
        arr: zarr.Array,
        indices: list[int] | None = None,
        dtype: Any = None,  # torch.float32 by default
        transform: Callable | None = None,
        focal_zone: int | None = None,
        preload: bool = False,
    ):
        import torch  # noqa: F401 — required, deferred to keep the module torch-free

        self._torch = torch
        self.arr = arr
        self.indices = (
            list(range(arr.shape[0])) if indices is None else list(indices)
        )
        self.dtype = dtype if dtype is not None else torch.float32
        self.transform = transform
        self.focal_zone = focal_zone

        self._cache = None
        if preload and self.indices:
            from tqdm import tqdm

            frames = [
                self._read_frame(g)
                for g in tqdm(self.indices, desc="Preloading frames", unit="frame")
            ]
            # One copy in shared memory, visible to all DataLoader workers.
            self._cache = torch.stack(frames).share_memory_()

    def _read_frame(self, global_idx: int):
        if self.focal_zone is None:
            frame = self.arr[global_idx]
        else:
            frame = self.arr[global_idx, :, :, self.focal_zone]
        return self._torch.from_numpy(frame).to(self.dtype)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._cache is not None:
            tensor = self._cache[i]
        else:
            tensor = self._read_frame(self.indices[i])
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


# ── convenience constructor ───────────────────────────────────────────────────
def build_split_datasets(
    repo_id: str | None = None,
    *,
    local_zip: str | os.PathLike | None = None,
    local_metadata: str | os.PathLike | None = None,
    split_strategy: Literal["subject", "file"] = "subject",
    cache_dir: str | os.PathLike | None = None,
    revision: str | None = None,
    transform: Callable | None = None,
    max_frames: int | None = None,
    focal_zone: int | None = None,
    preload: bool = False,
    token: str | None = None,
    **split_kwargs,
) -> tuple[dict[str, UltrasoundMAEDataset], dict]:
    """
    One-call entry point for training scripts.

    Either pass `repo_id` (HF dataset repo) OR both `local_zip` and `local_metadata`
    (for testing against a local build before any upload).

    `max_frames` (HF path only) caps how many frames are downloaded/opened — only
    those frames' chunks are fetched, and the splits are restricted to them.

    `token` (HF path only) authenticates the download for higher rate limits;
    falls back to $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN / cached login when None.

    `focal_zone` / `preload` are forwarded to `UltrasoundMAEDataset` (zarr-level
    zone slicing and one-off in-RAM materialisation respectively).

    Extra kwargs are forwarded to the chosen split function.
    """
    if repo_id is not None:
        arr, meta = load_dataset(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            max_frames=max_frames,
            token=token,
        )
    elif local_zip is not None and local_metadata is not None:
        if max_frames is not None:
            warnings.warn(
                "max_frames is ignored when loading from a local zip "
                "(the whole array is already on disk).",
                stacklevel=2,
            )
        arr, meta = load_local_dataset(local_zip, local_metadata)
    else:
        raise ValueError(
            "Pass `repo_id`, or both `local_zip` and `local_metadata`."
        )

    if split_strategy == "subject":
        splits = split_by_subject(meta, **split_kwargs)
    elif split_strategy == "file":
        splits = split_by_file(meta, **split_kwargs)
    else:
        raise ValueError(f"unknown split_strategy: {split_strategy!r}")

    datasets = {
        name: UltrasoundMAEDataset(
            arr,
            indices=ix,
            transform=transform,
            focal_zone=focal_zone,
            preload=preload,
        )
        for name, ix in splits.items()
    }
    return datasets, meta
