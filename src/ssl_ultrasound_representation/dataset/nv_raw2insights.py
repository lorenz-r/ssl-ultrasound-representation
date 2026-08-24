"""Local reader for NVIDIA NV-Raw2Insights-US Parquet shards.

The published IQ axes are ``(tx, rx, sample)``.  MAE mode returns the layout
used by this project's FSA pipeline: ``(rx, tx, iq, sample)`` with no singleton
focal-zone axis, matching ``UltrasoundMAEDataset(..., focal_zone=0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_NV_ROOT = Path(
    "/mnt/projects/drl4cvb/us_ssl/nv-us/NV-Raw2Insights-US"
)

ARRAY_SHAPES: dict[str, tuple[int, ...]] = {
    "iq_real": (180, 180, 1024),
    "iq_imag": (180, 180, 1024),
    "bmode": (507, 456),
    "bmode_focused": (507, 456),
    "bmode_extent": (4,),
    "sound_speed_map": (32, 32),
    "sound_speed_extent": (4,),
    "segmentation_map": (507, 456),
    "elpos": (3, 180),
}

SCALAR_COLUMNS = (
    "phase_error",
    "fs",
    "fc",
    "fd",
    "t0",
    "c0",
    "frame_idx",
)
ALL_COLUMNS = tuple(ARRAY_SHAPES) + SCALAR_COLUMNS
IQ_COLUMNS = ("iq_real", "iq_imag")


@dataclass(frozen=True)
class ParquetRow:
    path: Path
    row_group: int
    row_in_group: int
    split_index: int


def _nested_scalar_to_numpy(
    scalar: pa.Scalar,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Flatten Arrow list storage without constructing nested Python lists."""
    values = scalar.values
    while (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    ):
        values = values.values
    return values.to_numpy(zero_copy_only=False).reshape(shape)


def _decode_batch_row(
    batch: pa.RecordBatch,
    columns: Sequence[str],
    *,
    split: str,
    split_index: int,
    parquet_path: Path,
) -> dict:
    sample: dict = {
        "split": split,
        "split_index": split_index,
        "parquet_file": str(parquet_path),
    }
    for name in columns:
        scalar = batch.column(batch.schema.get_field_index(name))[0]
        if name in ARRAY_SHAPES:
            sample[name] = _nested_scalar_to_numpy(scalar, ARRAY_SHAPES[name])
        else:
            sample[name] = scalar.as_py()
    return sample


def iter_nv_raw2insights_samples(
    root: str | Path = DEFAULT_NV_ROOT,
    *,
    split: Literal["train", "validation"] = "train",
    columns: Sequence[str] = ALL_COLUMNS,
):
    """Sequentially stream a split, decoding every Parquet row exactly once."""
    root = Path(root)
    paths = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {split!r} shards under {root / 'data'}")

    split_index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            for sample in iter_nv_raw2insights_row_group(
                path,
                row_group=row_group,
                split=split,
                split_index=split_index,
                columns=columns,
            ):
                yield sample
                split_index += 1


def iter_nv_raw2insights_row_group(
    path: str | Path,
    *,
    row_group: int,
    split: str,
    split_index: int,
    columns: Sequence[str] = ALL_COLUMNS,
):
    """Decode one row group, suitable for shard-level parallel conversion."""
    path = Path(path)
    parquet = pq.ParquetFile(path)
    for row_offset, batch in enumerate(
        parquet.iter_batches(
            batch_size=1,
            row_groups=[row_group],
            columns=list(columns),
            use_threads=False,
        )
    ):
        yield _decode_batch_row(
            batch,
            columns,
            split=split,
            split_index=split_index + row_offset,
            parquet_path=path,
        )


def nv_iq_to_mae_frame(
    iq_real: np.ndarray,
    iq_imag: np.ndarray,
    *,
    sample_stride: int = 4,
    keep_focal_zone: bool = False,
) -> np.ndarray:
    """Convert NV ``(tx, rx, sample)`` IQ to the project's FSA MAE layout."""
    if iq_real.shape != iq_imag.shape:
        raise ValueError(
            f"real/imag shape mismatch: {iq_real.shape} vs {iq_imag.shape}"
        )
    if iq_real.ndim != 3:
        raise ValueError(f"expected (tx, rx, sample), got {iq_real.shape}")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    real = iq_real[:, :, ::sample_stride].transpose(1, 0, 2)
    imag = iq_imag[:, :, ::sample_stride].transpose(1, 0, 2)
    frame = np.stack((real, imag), axis=2).astype(np.float32, copy=False)
    if keep_focal_zone:
        frame = frame[:, :, np.newaxis, :, :]
    return np.ascontiguousarray(frame)


def nv_sample_to_fsa_params(sample: dict, *, sample_stride: int = 1) -> dict:
    """Map NV geometry to the JSON contract consumed by ``ZoneBeamformer``."""
    n_tx, n_rx, n_samples_raw = sample["iq_real"].shape
    if n_tx != n_rx:
        raise ValueError(f"expected square FSA TX/RX grid, got {(n_tx, n_rx)}")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    element_x = np.asarray(sample["elpos"], dtype=np.float64)[0]
    pitch = float(np.median(np.diff(element_x)))
    extent_mm = np.asarray(sample["bmode_extent"], dtype=np.float64)
    x_min_m, x_max_m, z_max_m, z_min_m = extent_mm * 1e-3

    tx_delays: list[list[float | None]] = []
    for tx_index in range(n_tx):
        delays: list[float | None] = [None] * n_rx
        delays[tx_index] = 0.0
        tx_delays.append(delays)

    return {
        "acquisition_type": "fsa",
        "beamforming_mode": "synthetic_aperture",
        "c": float(sample["c0"]),
        "fc": float(sample["fd"]),
        "fs": float(sample["fs"]) / sample_stride,
        "pitch": pitch,
        "n_elements": n_rx,
        "t0": float(sample["t0"]),
        "n_samples": len(range(0, n_samples_raw, sample_stride)),
        "start_depth_m": float(z_min_m),
        "data_depth_limit_m": float(z_max_m),
        "scanline_origins_m": element_x[:n_tx].tolist(),
        "focal_depths_m": [0.0],
        "tx_delays_s": [tx_delays],
        "source": "NV-Raw2Insights-US",
        "source_center_frequency_hz": float(sample["fc"]),
        "bmode_lateral_extent_m": [float(x_min_m), float(x_max_m)],
        "sample_stride": sample_stride,
    }


class NvidiaRaw2InsightsDataset:
    """Index local NV-Raw2Insights Parquet shards as a PyTorch-style dataset.

    ``mode='sample'`` returns a dictionary of requested raw fields.
    ``mode='mae'`` returns a float32 tensor in ``(rx, tx, iq, sample)`` layout.

    Parquet row groups contain roughly ten very large examples.  Reads are
    streamed in one-row Arrow batches to bound Python-side memory, but random
    access still has substantial I/O cost.  For training, use a sequential or
    shard-aware sampler rather than fully random per-sample access.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_NV_ROOT,
        *,
        split: Literal["train", "validation"] = "train",
        mode: Literal["sample", "mae"] = "sample",
        columns: Sequence[str] | None = None,
        indices: Iterable[int] | None = None,
        sample_stride: int = 4,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.sample_stride = sample_stride
        self.transform = transform

        if not self.root.is_dir():
            raise FileNotFoundError(f"NV dataset root does not exist: {self.root}")
        if mode not in {"sample", "mae"}:
            raise ValueError(f"unknown mode: {mode!r}")
        if sample_stride < 1:
            raise ValueError("sample_stride must be >= 1")

        requested = tuple(columns or (IQ_COLUMNS if mode == "mae" else ALL_COLUMNS))
        unknown = sorted(set(requested) - set(ALL_COLUMNS))
        if unknown:
            raise ValueError(f"unknown NV columns: {unknown}")
        if mode == "mae" and not set(IQ_COLUMNS).issubset(requested):
            raise ValueError("mode='mae' requires iq_real and iq_imag")
        self.columns = requested

        paths = sorted((self.root / "data").glob(f"{split}-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no {split!r} Parquet shards under {self.root / 'data'}"
            )

        rows: list[ParquetRow] = []
        split_index = 0
        for path in paths:
            metadata = pq.read_metadata(path)
            for row_group in range(metadata.num_row_groups):
                n_rows = metadata.row_group(row_group).num_rows
                rows.extend(
                    ParquetRow(path, row_group, row_in_group, split_index + row_in_group)
                    for row_in_group in range(n_rows)
                )
                split_index += n_rows

        if indices is None:
            self.rows = rows
        else:
            selected = list(indices)
            for index in selected:
                if index < 0 or index >= len(rows):
                    raise IndexError(
                        f"{split} index {index} outside [0, {len(rows)})"
                    )
            self.rows = [rows[index] for index in selected]

    def __len__(self) -> int:
        return len(self.rows)

    def _read_row(self, location: ParquetRow) -> dict:
        parquet = pq.ParquetFile(location.path)
        batches = parquet.iter_batches(
            batch_size=1,
            row_groups=[location.row_group],
            columns=list(self.columns),
            use_threads=True,
        )
        try:
            batch = next(islice(batches, location.row_in_group, None))
        except StopIteration as exc:
            raise RuntimeError(f"could not read indexed row {location}") from exc

        return _decode_batch_row(
            batch,
            self.columns,
            split=self.split,
            split_index=location.split_index,
            parquet_path=location.path,
        )

    def __getitem__(self, index: int):
        sample = self._read_row(self.rows[index])
        if self.mode == "sample":
            return sample

        frame = nv_iq_to_mae_frame(
            sample["iq_real"],
            sample["iq_imag"],
            sample_stride=self.sample_stride,
        )
        try:
            import torch
        except ImportError as exc:
            raise ImportError("mode='mae' requires PyTorch") from exc
        tensor = torch.from_numpy(frame)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor


def build_nv_raw2insights_datasets(
    root: str | Path = DEFAULT_NV_ROOT,
    *,
    mode: Literal["sample", "mae"] = "mae",
    sample_stride: int = 4,
    transform: Callable | None = None,
    train_indices: Iterable[int] | None = None,
    validation_indices: Iterable[int] | None = None,
) -> dict[str, NvidiaRaw2InsightsDataset]:
    """Build datasets using NVIDIA's published 830/93 train/validation split."""
    return {
        "train": NvidiaRaw2InsightsDataset(
            root,
            split="train",
            mode=mode,
            indices=train_indices,
            sample_stride=sample_stride,
            transform=transform,
        ),
        "validation": NvidiaRaw2InsightsDataset(
            root,
            split="validation",
            mode=mode,
            indices=validation_indices,
            sample_stride=sample_stride,
            transform=transform,
        ),
    }
