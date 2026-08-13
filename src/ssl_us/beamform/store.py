"""Zarr-backed K-sector store: the WI-8 build's output and WI-13's input.

PLAN.md §3.7 specifies the store this module implements: zarr **v2** (matching
the repo's `zarr<3` pin), one array `fsa` of shape ``(N, 2, Z, X, K)`` float32
chunked **one frame per chunk**, plus the frame-invariant geometry — the
``sector_id`` map and the ``occupancy`` counts — written **once** rather than
per frame. At the dataset's native (507, 456) grid with K=16 that is
29.59 MB/frame, ~27.3 GB for all 923 frames.

Two things this module deliberately does *not* do:

- **It never normalizes.** The store holds **raw** sector sums, because
  normalizing at build time breaks the K=16→8→4→2→1 ladder's exact pairwise-sum
  identity (PLAN.md §3.2 / WI-5, and see :mod:`ssl_us.data.transforms`). The
  occupancy counts needed to normalize at read time are stored alongside, so
  :func:`ssl_us.data.transforms.reduce_and_normalize` is the caller's one
  correct path.
- **It stores one chunk per frame**, not per crop. `build_mae_dataset.py` in the
  old repo learned the chunk-count blowup the hard way (PLAN.md §3.7); 923
  chunks stays far under HF's 10k-files-per-directory limit.

:func:`open_store` is the reader WI-13 needs (`eval/matrix.py`'s
``load_bootstrap`` is the shape to match); :class:`SectorStore` is used by both
sides so the layout is defined exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr

from ..data import geometry as geo
from .sectors import SectorGeometry

# Blosc keeps a global internal thread pool. The WI-8 build writes frames from
# the main thread while HuggingFace's streaming reader has its own fsspec /
# aiohttp threads live, and in that combination a write can block forever in
# uninterruptible I/O — observed during the first smoke build: all three frames
# beamformed, but only two reached the manifest, the third wedged inside the
# chunk write. zarr's own docs recommend disabling blosc's threading when the
# caller is already multithreaded. Compression is nowhere near the bottleneck
# here (~3 s of GPU and minutes of download per frame), so this costs nothing.
numcodecs.blosc.use_threads = False

FSA_ARRAY = "fsa"
OCCUPANCY_ARRAY = "occupancy"
SECTOR_ID_ARRAY = "sector_id"
METADATA_FILE = "frame_metadata.json"


@dataclass
class StoreSpec:
    """Everything needed to create a store, and everything a reader must know."""

    n_frames: int
    nz: int = geo.BMODE_NZ
    nx: int = geo.BMODE_NX
    k: int = 16
    n_channels: int = 2  # (I, Q)

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return (self.n_channels, self.nz, self.nx, self.k)

    @property
    def frame_nbytes(self) -> int:
        return int(np.prod(self.frame_shape)) * 4  # float32


class SectorStore:
    """Reader/writer for the zarr K-sector store.

    Use :meth:`create` to make a new one and :func:`open_store` to read an
    existing one. Frames are addressed by integer index; the index -> frame_id
    mapping lives in the group attrs, so a reader never has to guess the order
    the build happened to run in.
    """

    def __init__(self, root: zarr.Group, path: Path):
        self.root = root
        self.path = Path(path)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        path: str | Path,
        spec: StoreSpec,
        frame_ids: list[str],
        beamform_params: dict,
        geometry: SectorGeometry | None = None,
        overwrite: bool = False,
    ) -> SectorStore:
        """Create the store's arrays and write the frame-invariant geometry.

        Args:
            path: destination directory.
            spec: shapes/dtypes, see :class:`StoreSpec`.
            frame_ids: index -> frame_id, length ``spec.n_frames``. Recorded in
                attrs so readers can map either way without re-deriving it.
            beamform_params: recorded verbatim into attrs — the beamformer
                settings that produced the data (WI-6's requirement, and what
                WI-9's dataset card must quote).
            geometry: the sector/occupancy maps to store once. Optional only so
                tests can skip building the full 41.6 MB map; the real build
                always passes it.
            overwrite: replace an existing store at ``path``.
        """
        if len(frame_ids) != spec.n_frames:
            raise ValueError(
                f"frame_ids has {len(frame_ids)} entries but spec.n_frames is {spec.n_frames}"
            )
        path = Path(path)
        root = zarr.open_group(str(path), mode="w" if overwrite else "w-")

        root.create_dataset(
            FSA_ARRAY,
            shape=(spec.n_frames, *spec.frame_shape),
            # One frame per chunk: the access pattern is whole-frame, and it is
            # what keeps the file count at N rather than N x crops.
            chunks=(1, *spec.frame_shape),
            dtype="float32",
        )
        root.attrs["spec"] = {
            "n_frames": spec.n_frames,
            "nz": spec.nz,
            "nx": spec.nx,
            "k": spec.k,
            "n_channels": spec.n_channels,
        }
        root.attrs["frame_ids"] = list(frame_ids)
        root.attrs["beamform_params"] = beamform_params
        root.attrs["z_mm"] = geo.bmode_z_axis_mm().tolist()
        root.attrs["x_mm"] = geo.bmode_x_axis_mm().tolist()
        # Raw sums, not normalized — stated in the store itself so a consumer
        # that never reads PLAN.md still cannot miss it.
        root.attrs["normalized"] = False
        root.attrs["normalization_note"] = (
            "Raw sector sums. Divide by `occupancy` only AFTER reducing along "
            "the K axis to your target K, via ssl_us.data.transforms."
            "reduce_and_normalize — normalizing before reducing breaks the "
            "K-ladder identity (PLAN.md 3.2)."
        )

        if geometry is not None:
            cls._write_geometry(root, geometry)
        return cls(root, path)

    @staticmethod
    def _write_geometry(root: zarr.Group, geometry: SectorGeometry) -> None:
        root.create_dataset(
            OCCUPANCY_ARRAY, data=geometry.occupancy, chunks=geometry.occupancy.shape
        )
        root.create_dataset(
            SECTOR_ID_ARRAY, data=geometry.sector_id, chunks=geometry.sector_id.shape
        )
        root.attrs["sector_geometry"] = {
            "k": geometry.k,
            "f_number": geometry.f_number,
            "theta_max_rad": float(geometry.theta_max_rad),
        }

    # ── frame I/O ───────────────────────────────────────────────────────────

    def write_frame(self, index: int, array: np.ndarray) -> None:
        """Write one frame's ``(2, Z, X, K)`` raw sector sums at ``index``."""
        expected = tuple(self.root[FSA_ARRAY].shape[1:])
        if array.shape != expected:
            raise ValueError(f"frame shape {array.shape} != store's {expected}")
        self.root[FSA_ARRAY][index] = array.astype(np.float32, copy=False)

    def read_frame(self, index: int) -> np.ndarray:
        """Read one frame's raw ``(2, Z, X, K)`` sector sums."""
        return np.asarray(self.root[FSA_ARRAY][index])

    # ── accessors ───────────────────────────────────────────────────────────

    @property
    def spec(self) -> StoreSpec:
        d = dict(self.root.attrs["spec"])
        return StoreSpec(**d)

    @property
    def frame_ids(self) -> list[str]:
        return list(self.root.attrs["frame_ids"])

    @property
    def beamform_params(self) -> dict:
        return dict(self.root.attrs["beamform_params"])

    @property
    def occupancy(self) -> np.ndarray:
        """``(Z, X, K)`` uint16 element counts — the divisor for normalization."""
        return np.asarray(self.root[OCCUPANCY_ARRAY])

    @property
    def sector_id(self) -> np.ndarray:
        """``(Z, X, n_el)`` int8 sector membership, -1 outside the aperture."""
        return np.asarray(self.root[SECTOR_ID_ARRAY])

    def index_of(self, frame_id: str) -> int:
        return self.frame_ids.index(frame_id)

    def __len__(self) -> int:
        return int(self.root[FSA_ARRAY].shape[0])

    # ── sidecar metadata ────────────────────────────────────────────────────

    def write_frame_metadata(self, metadata: dict[str, dict]) -> None:
        """Per-frame labels/metadata sidecar (PLAN.md §3.7's `manifest.json`)."""
        (self.path / METADATA_FILE).write_text(json.dumps(metadata, indent=1))

    def read_frame_metadata(self) -> dict[str, dict]:
        path = self.path / METADATA_FILE
        return json.loads(path.read_text()) if path.exists() else {}


def open_store(path: str | Path, mode: str = "r") -> SectorStore:
    """Open an existing store. ``mode='r'`` (default) is read-only."""
    path = Path(path)
    return SectorStore(zarr.open_group(str(path), mode=mode), path)
