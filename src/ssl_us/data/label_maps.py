"""Per-pixel label maps for the corpus, fetched separately from the K-sector store.

`frame_metadata.json` beside the store carries frame-level *scalars* only, which
is why `eval/store_features.py` reports WI-13's crop-level anatomy targets as
blocked. The maps those targets need — `segmentation_map` (507, 456) and
`sound_speed_map` (32, 32) — are a 2.1 MB/frame slice of the source rows next to
a 265 MB/frame IQ payload, so `scripts/data/fetch_nvidia_label_maps.py` pulls them by
column and writes them here as `.npy`. This module is the read side.

Everything is memory-mapped and indexed by the store's own `"{split}/{index}"`
frame ids, so `seg[maps.index_of(["train/7"])]` is the mask of the frame the
store holds at that id — an equality the fetch script asserts against the
store's independently-written metadata before this module will open the
directory.

Resampling the sound-speed map is the one non-trivial read: it is on a
*different* extent from the B-mode grid (PLAN.md §2.2 — and in metres, not
millimetres), so `geometry.resample_sos_to_bmode_grid` does a real coordinate
transform. It runs once per frame and is cached, because crop labels hit the
same frame `crops_per_frame` times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as geo

DEFAULT_ROOT = Path("/mnt/home/benbarkow/stores/wi8_label_maps")

# Crop-level anatomy labels, named exactly as `crops.py` names them so a probe
# cannot silently target a differently-defined version of the same quantity.
CONTAINS_CYST_THRESHOLD = 0.01


@dataclass
class LabelMaps:
    """Memory-mapped per-pixel maps, aligned to store frame ids."""

    root: Path
    frame_ids: list[str]
    arrays: dict[str, np.memmap]
    scalars: dict[str, dict[str, float]]
    extents: dict[str, list[float]] = field(default_factory=dict)
    _index: dict[str, int] = field(default_factory=dict, repr=False)
    _sos_cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._index:
            self._index = {fid: i for i, fid in enumerate(self.frame_ids)}

    def __len__(self) -> int:
        return len(self.frame_ids)

    @property
    def seg(self) -> np.memmap:
        return self.arrays["seg"]

    @property
    def sos(self) -> np.memmap:
        return self.arrays["sos"]

    def index_of(self, frame_ids) -> np.ndarray:
        try:
            return np.array([self._index[f] for f in frame_ids], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"frame id {exc.args[0]!r} is not in {self.root}") from exc

    def sos_on_grid(self, frame_id: str) -> np.ndarray:
        """`sound_speed_map` for one frame, resampled onto the (507, 456) grid."""
        i = self._index[frame_id]
        cached = self._sos_cache.get(i)
        if cached is None:
            cached = geo.resample_sos_to_bmode_grid(np.asarray(self.sos[i], dtype=np.float32))
            self._sos_cache[i] = cached
        return cached

    def crop_labels(
        self,
        frame_ids,
        origins: np.ndarray,
        size: int,
    ) -> dict[str, np.ndarray]:
        """Anatomy labels for crops given as `(row, col)` top-left origins.

        `origins` is what `StoreCropBatch` records, which is why it is the
        argument rather than the mm centres: recovering the origin from
        `z_centre`/`x_centre` means inverting the grid mapping approximately,
        and a half-pixel slip there shifts every mask.
        """
        origins = np.asarray(origins, dtype=np.int64)
        frame_ids = list(frame_ids)
        if len(origins) != len(frame_ids):
            raise ValueError(f"{len(origins)} origins for {len(frame_ids)} frame ids")
        n = len(origins)
        cyst = np.empty(n, dtype=np.float32)
        sos_mean = np.empty(n, dtype=np.float32)
        sos_std = np.empty(n, dtype=np.float32)
        for i, (fid, (top, left)) in enumerate(zip(frame_ids, origins, strict=True)):
            j = self._index[fid]
            seg_c = self.seg[j, top : top + size, left : left + size]
            sos_c = self.sos_on_grid(fid)[top : top + size, left : left + size]
            if seg_c.shape != (size, size):
                raise ValueError(f"crop at {(int(top), int(left))} runs off frame {fid}")
            cyst[i] = float(seg_c.mean())
            sos_mean[i] = float(sos_c.mean())
            sos_std[i] = float(sos_c.std())
        return {
            "cyst_fraction": cyst,
            "contains_cyst": (cyst > CONTAINS_CYST_THRESHOLD).astype(np.int64),
            "sos_mean": sos_mean,
            "sos_std": sos_std,
        }

    def crop_masks(self, frame_ids, origins: np.ndarray, size: int) -> np.ndarray:
        """The per-pixel masks `crop_labels` reduces to scalars — `(n, size, size)` uint8.

        Same slicing, same `origins` contract, same bounds check; the only
        difference is that nothing is averaged away. `crop_labels` answers
        "how much cyst is in this crop", which is what the `contains_cyst` and
        `cyst_fraction_crop` matrix cells probe; this answers *where*, which is
        what WI-19's segmentation probe needs as its target.
        """
        origins = np.asarray(origins, dtype=np.int64)
        frame_ids = list(frame_ids)
        if len(origins) != len(frame_ids):
            raise ValueError(f"{len(origins)} origins for {len(frame_ids)} frame ids")
        out = np.empty((len(origins), size, size), dtype=np.uint8)
        for i, (fid, (top, left)) in enumerate(zip(frame_ids, origins, strict=True)):
            j = self._index[fid]
            seg_c = self.seg[j, top : top + size, left : left + size]
            if seg_c.shape != (size, size):
                raise ValueError(f"crop at {(int(top), int(left))} runs off frame {fid}")
            out[i] = seg_c
        return out

    def to_bootstrap(self, frame_ids=None) -> dict:
        """The dict shape `eval/matrix.load_bootstrap` produces, over real frames.

        Lets the WI-2 harness — which was written against a `bmode` subset —
        run over the whole corpus without an npz repack. `bmode` is loaded
        eagerly for the requested frames because the crop sampler indexes it
        positionally; keep the frame list to a subset if memory is tight
        (923 frames of `bmode` is 854 MB).
        """
        ids = list(frame_ids) if frame_ids is not None else list(self.frame_ids)
        idx = self.index_of(ids)
        if "bmode" not in self.arrays:
            raise KeyError(f"{self.root} was fetched without the `bmode` column")
        sos = np.asarray(self.sos[idx], dtype=np.float32)
        flat = sos.reshape(len(idx), -1)
        frame_labels = {
            "phase_error": np.array(
                [self.scalars[f]["phase_error"] for f in ids], dtype=np.float32
            ),
            "t0_group": np.array(
                [geo.t0_group(self.scalars[f]["t0"]) for f in ids], dtype=np.int64
            ),
            "sos_mean_frame": flat.mean(1).astype(np.float32),
            "sos_std_frame": flat.std(1).astype(np.float32),
        }
        return {
            "bmode": np.asarray(self.arrays["bmode"][idx], dtype=np.float32),
            "seg": np.asarray(self.seg[idx]),
            "sos": sos,
            "frame_id": np.array(ids, dtype=object).astype(str),
            "frame_labels": frame_labels,
            "frame_labels_by_id": {
                fid: {k: float(v[i]) for k, v in frame_labels.items()} for i, fid in enumerate(ids)
            },
        }


def open_label_maps(root: str | Path = DEFAULT_ROOT, columns=None) -> LabelMaps:
    """Open a fetched label-map directory, refusing an incomplete one.

    A partially-fetched directory is dangerous rather than merely useless: the
    `.npy` files are allocated at full size up front, so the frames not yet
    written read back as zeros — a plausible-looking empty mask and a 0 m/s
    sound speed. The manifest's shard record is the authority on how many rows
    are real, and anything short of the full corpus raises here.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path} — run scripts/data/fetch_nvidia_label_maps.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    scalars_blob = json.loads((root / "scalars.json").read_text())
    frame_ids = list(scalars_blob["frame_ids"])

    written = sum(s["n_rows"] for s in manifest["shards"].values())
    if written != len(frame_ids):
        raise ValueError(
            f"{root} holds {written} of {len(frame_ids)} frames — the rest read back as "
            f"zeros, which is indistinguishable from an empty mask. Finish the fetch."
        )
    if "verification" not in manifest:
        raise ValueError(
            f"{root} has not been verified against the store's frame_metadata.json — "
            f"re-run scripts/data/fetch_nvidia_label_maps.py --verify-only"
        )

    wanted = list(columns) if columns is not None else list(manifest["columns"])
    missing = [c for c in wanted if not (root / f"{c}.npy").exists()]
    if missing:
        raise FileNotFoundError(f"{root} is missing columns {missing}")
    arrays = {c: np.load(root / f"{c}.npy", mmap_mode="r") for c in wanted}
    return LabelMaps(
        root=root,
        frame_ids=frame_ids,
        arrays=arrays,
        scalars=scalars_blob["per_frame"],
        extents=scalars_blob.get("extents", {}),
    )
