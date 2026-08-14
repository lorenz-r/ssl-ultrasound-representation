"""Store-backed inputs for the WI-13 results matrix.

`eval/matrix.py` (WI-2) bootstrapped the whole protocol against `bmode`: single-
channel (507, 456) frames with per-pixel `seg` and `sos` maps alongside. WI-13
runs the same protocol against the real K-sector store and a *trained* encoder,
which differs in three ways that this module exists to absorb:

1. **Crops are multi-channel.** A K-sector crop is ``(2K, size, size)`` — I- and
   Q-blocks over K receive sectors — not ``(size, size)``. `data/crops.py`'s
   `sample_corpus` is 2-D by construction, so sampling happens here instead.
2. **Pre-processing must match training exactly**, or the frozen features are
   measuring the mismatch rather than the representation. The one correct path
   is `reduce_and_normalize` (reduce the K-ladder, *then* normalize by
   occupancy — never the reverse, PLAN.md §3.2), then `(…, K)` moved to the
   channel axis as I-block-then-Q-block. That mirrors `GpuAugmentor.__call__`
   with the augmentation removed.
3. **Only frame-level labels survive.** `frame_metadata.json` beside the store
   carries scalars per frame (`t0`, `phase_error`, `sos_mean`, `sos_std`,
   `cyst_fraction`); the per-pixel `seg` and `sos` maps live in the source
   NVIDIA frames, which are not on the cluster. Crop-level anatomy targets are
   therefore unavailable from the store alone — see `scripts/wi13_matrix.py`,
   which reports them as explicitly-blocked cells rather than dropping them.

Both arms are encoded through the identical un-augmented path, so the Policy A
vs Policy B contrast is a difference in learned weights and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..beamform.store import open_store
from ..data import geometry as geo
from ..data import transforms as tf
from ..models.resnet import BACKBONE_DIM, ResNetBackbone
from ..models.vicreg import VICReg


@dataclass
class StoreCropSpec:
    """Deterministic crop sampling over store frames.

    Mirrors `LoaderConfig`'s geometry (PLAN.md §3.2c/d) so eval crops are drawn
    from the same legal region training drew from: `K <= 4` gets 128 px at full
    depth, `K >= 8` gets 64 px below `z_min_mm`.
    """

    k: int = 4
    crop_size: int = 128
    z_min_mm: float = 0.0
    crops_per_frame: int = 16
    seed: int = 0

    @property
    def in_channels(self) -> int:
        return 2 * self.k


@dataclass
class StoreCropBatch:
    """Crops plus the labels the store can actually support."""

    images: np.ndarray  # (N, 2K, size, size) float32
    frame_id: np.ndarray  # (N,) str
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    # (N, 2) int — the (row, col) each crop was cut from. Kept because the mm
    # centres in `labels` are a lossy view of it: recovering the origin from
    # them means inverting the grid mapping, which tests and any re-extraction
    # would otherwise have to redo approximately.
    origins: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.images)


# Frame-level targets recoverable from `frame_metadata.json`. Crop-level
# anatomy targets are deliberately absent — see the module docstring.
FRAME_LABEL_KEYS = ("phase_error", "sos_mean", "sos_std", "cyst_fraction")


def frame_labels_from_metadata(metadata: dict, frame_ids: list[str]) -> dict[str, np.ndarray]:
    """Frame-level label vectors aligned to `frame_ids`.

    `t0` becomes `t0_group` — the discrete acquisition-group id — because the
    raw offset is a continuous nuisance parameter and §4.1 probes the group.
    """
    out: dict[str, np.ndarray] = {}
    for key in FRAME_LABEL_KEYS:
        out[key] = np.array([metadata[f][key] for f in frame_ids], dtype=np.float32)
    out["t0_group"] = np.array(
        [geo.t0_group(metadata[f]["t0"]) for f in frame_ids], dtype=np.int64
    )
    # Named `*_frame` to match §4.1's naming, where the unsuffixed form is the
    # crop-level target of the same quantity.
    out["sos_mean_frame"] = out.pop("sos_mean")
    out["sos_std_frame"] = out.pop("sos_std")
    return out


def _origin_bounds(spec: StoreCropSpec, nz: int, nx: int) -> tuple[int, int, int, int]:
    z_lo = geo.z_mm_to_row(spec.z_min_mm) if spec.z_min_mm > 0 else 0
    z_hi = nz - spec.crop_size
    x_hi = nx - spec.crop_size
    if z_hi < z_lo or x_hi < 0:
        raise ValueError(
            f"crop {spec.crop_size} does not fit in ({nz}, {nx}) above z={spec.z_min_mm} mm"
        )
    return z_lo, z_hi, 0, x_hi


def sample_store_crops(
    store_path: str | Path,
    frame_ids: list[str],
    spec: StoreCropSpec,
) -> StoreCropBatch:
    """Draw `crops_per_frame` un-augmented crops from each frame in `frame_ids`.

    Returns crops already in the encoder's layout, ``(N, 2K, size, size)``, with
    geometric labels (`z_centre`, `x_centre`, in mm) attached per crop and
    frame-level labels broadcast across their frame's crops.
    """
    store = open_store(store_path)
    metadata = store.read_frame_metadata()
    occupancy = np.asarray(store.occupancy)  # (Z, X, K_store)
    nz, nx = store.spec.nz, store.spec.nx
    z_lo, z_hi, x_lo, x_hi = _origin_bounds(spec, nz, nx)

    rng = np.random.default_rng(spec.seed)
    crops, ids, z_centres, x_centres, origins = [], [], [], [], []

    for fid in frame_ids:
        frame = store.read_frame(store.index_of(fid))  # (2, Z, X, K_store)
        z0s = rng.integers(z_lo, z_hi + 1, size=spec.crops_per_frame)
        x0s = rng.integers(x_lo, x_hi + 1, size=spec.crops_per_frame)
        for z0, x0 in zip(z0s, x0s, strict=True):
            zs, xs = slice(z0, z0 + spec.crop_size), slice(x0, x0 + spec.crop_size)
            # Reduce the K-ladder and normalize by occupancy in one step, in
            # that order. The reverse — normalizing at K=16 then reducing —
            # is the documented wrong path (PLAN.md §3.2).
            v = tf.reduce_and_normalize(
                frame[:, zs, xs, :], occupancy[zs, xs, :], target_k=spec.k
            )  # (2, size, size, k)
            # (2, size, size, k) -> (2k, size, size), I-block then Q-block,
            # matching GpuAugmentor's `movedim(-1, 2).flatten(1, 2)`.
            crops.append(np.moveaxis(v, -1, 1).reshape(spec.in_channels, *v.shape[1:3]))
            ids.append(fid)
            z_centres.append(geo.row_to_z_mm(z0 + spec.crop_size / 2))
            x_centres.append(geo.col_to_x_mm(x0 + spec.crop_size / 2))
            origins.append((z0, x0))

    labels: dict[str, np.ndarray] = {
        "z_centre": np.asarray(z_centres, dtype=np.float32),
        "x_centre": np.asarray(x_centres, dtype=np.float32),
    }
    per_frame = frame_labels_from_metadata(metadata, frame_ids)
    frame_index = {fid: i for i, fid in enumerate(frame_ids)}
    row = np.array([frame_index[f] for f in ids])
    for key, values in per_frame.items():
        labels[key] = values[row]

    return StoreCropBatch(
        images=np.stack(crops).astype(np.float32),
        frame_id=np.asarray(ids, dtype=object),
        labels=labels,
        origins=np.asarray(origins, dtype=int),
    )


class VICRegCheckpointEncoder:
    """Frozen backbone features from a WI-12 checkpoint.

    Reads the 512-d *backbone* output, not the projector: PLAN.md §4.7 probes
    the backbone, and the projector is discarded after pre-training.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        in_channels: int,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = VICReg(
            ResNetBackbone(in_channels), backbone_dim=BACKBONE_DIM, projector_dim=2048
        )
        model.load_state_dict(state["model"])
        self.epoch = int(state.get("epoch", -1))
        self.model = model.backbone.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(crops[i : i + self.batch_size]).to(self.device)
            out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out)
