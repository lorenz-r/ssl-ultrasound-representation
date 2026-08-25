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
3. **Only frame-level labels survive the store.** `frame_metadata.json` beside
   it carries scalars per frame (`t0`, `phase_error`, `sos_mean`, `sos_std`,
   `cyst_fraction`); the per-pixel `seg` and `sos` maps live in the source
   NVIDIA rows. Those are now fetched separately — `data/label_maps.py`, built
   by `scripts/data/fetch_nvidia_label_maps.py` — and `LabelMaps.crop_labels` turns
   `StoreCropBatch.origins` into the crop-level anatomy targets. Until that is
   wired into `scripts/eval/nvidia_matrix.py` those cells are still reported as
   explicitly-blocked rather than dropped.

Both arms are encoded through the identical un-augmented path, so the Policy A
vs Policy B contrast is a difference in learned weights and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..beamform.store import open_store
from ..data import geometry as geo
from ..data import transforms as tf
from ..models.resnet import ResNetBackbone
from ..models.vicreg import VICReg, build_backbone


@dataclass(frozen=True)
class StoreGrid:
    """The z/x pixel axes a store was actually beamformed on.

    Crop centres are reported in mm, so something has to map ``row -> z_mm``.
    WI-13 used `data/geometry`'s bmode helpers directly, which is right for the
    NVIDIA store and *silently wrong* for any other grid: WI-16's Stanford
    variants are 253x484 @ 79.05 um and 304x582 @ 65.7 um over 0-19.9 mm, and
    read through the bmode axis every crop centre comes out on a 0-40 mm scale
    that no pixel in the store ever occupied. `SectorStore` records its own
    axes in attrs, so :meth:`from_store` is the non-guessing path.
    """

    z_mm: np.ndarray
    x_mm: np.ndarray

    @classmethod
    def from_store(cls, store) -> StoreGrid:
        return cls(z_mm=np.asarray(store.z_mm), x_mm=np.asarray(store.x_mm))

    @classmethod
    def bmode(cls) -> StoreGrid:
        """NVIDIA's (507, 456) bmode grid — the default for callers with no store."""
        return cls(z_mm=geo.bmode_z_axis_mm(), x_mm=geo.bmode_x_axis_mm())

    def row_to_z_mm(self, row: np.ndarray | float) -> np.ndarray | float:
        return np.interp(row, np.arange(len(self.z_mm)), self.z_mm)

    def col_to_x_mm(self, col: np.ndarray | float) -> np.ndarray | float:
        return np.interp(col, np.arange(len(self.x_mm)), self.x_mm)

    def z_mm_to_row(self, z_mm: float) -> int:
        return int(round(float(np.interp(z_mm, self.z_mm, np.arange(len(self.z_mm))))))


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
    # Off by default: it adds a label key, and `cluster_report` scores against
    # whatever keys it is handed, so switching it on unconditionally would
    # change what every already-published WI-13 column reports. WI-16 turns it
    # on because echogenicity is one of the four probe axes Stanford *has*
    # (PLAN.md WI-16) — it is unlabelled tissue, so brightness is ground truth
    # derived from the image rather than from a simulator.
    derive_echogenicity: bool = False
    # Multiplies every crop after reduce+normalize. 1.0 (no-op) within a single
    # corpus, where every frame already shares one amplitude scale; it exists
    # for *cross-corpus* evaluation — see `corpus_intensity_scale`.
    intensity_scale: float = 1.0

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
        # Counted from `frame_id`, not `images`: once features are extracted
        # the caller may drop the images (they are 3.5 GB at corpus scale) and
        # keep the batch for its labels, and a length of 0 there would silently
        # empty every per-target mask built from it.
        return len(self.frame_id)


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


def _origin_bounds(
    spec: StoreCropSpec, nz: int, nx: int, grid: StoreGrid | None = None
) -> tuple[int, int, int, int]:
    grid = grid or StoreGrid.bmode()
    z_lo = grid.z_mm_to_row(spec.z_min_mm) if spec.z_min_mm > 0 else 0
    z_hi = nz - spec.crop_size
    x_hi = nx - spec.crop_size
    if z_hi < z_lo or x_hi < 0:
        raise ValueError(
            f"crop {spec.crop_size} does not fit in ({nz}, {nx}) above z={spec.z_min_mm} mm"
        )
    return z_lo, z_hi, 0, x_hi


def sample_store_origins(
    frame_ids: list[str],
    spec: StoreCropSpec,
    nz: int,
    nx: int,
    grid: StoreGrid | None = None,
) -> tuple[list[str], np.ndarray]:
    """The `(frame_id, (row, col))` of every crop, before any pixel is read.

    Split out from the materialization so a corpus can be processed in chunks
    without changing which crops get drawn: the generator is consumed here, in
    one pass over the full frame list, so chunk boundaries cannot shift the
    stream. Sampling per chunk instead would re-seed for every chunk and give a
    different — and silently different — set of crops.
    """
    z_lo, z_hi, x_lo, x_hi = _origin_bounds(spec, nz, nx, grid)
    rng = np.random.default_rng(spec.seed)
    ids: list[str] = []
    origins: list[tuple[int, int]] = []
    for fid in frame_ids:
        z0s = rng.integers(z_lo, z_hi + 1, size=spec.crops_per_frame)
        x0s = rng.integers(x_lo, x_hi + 1, size=spec.crops_per_frame)
        for z0, x0 in zip(z0s, x0s, strict=True):
            ids.append(fid)
            origins.append((int(z0), int(x0)))
    return ids, np.asarray(origins, dtype=int).reshape(-1, 2)


def crop_echogenicity_db(crop: np.ndarray, eps: float = 1e-12) -> float:
    """Mean log-envelope brightness of one ``(2, size, size, k)`` reduced crop.

    The envelope is taken **per sector and then averaged over K** (incoherent),
    not from a coherent sum across sectors: the sectors are receive
    sub-apertures of the same scatterers, so a coherent re-sum would measure
    inter-sector phase agreement — an aperture-coherence quantity that varies
    with aberration — on top of the brightness this axis is supposed to be.
    Returned in dB, on the store's own raw-sum scale, so it is comparable
    across crops of one corpus but is not an absolute physical quantity.

    The mean is taken in the **linear** domain and converted to dB once, rather
    than averaging per-pixel dB. Pixels whose sectors are all empty are exactly
    0 (nothing was summed into them — see `transforms.normalize`), and in the
    per-pixel-dB form those pixels pin to the `eps` floor, which is a constant
    independent of amplitude. A crop containing them then fails to shift by a
    full 20*log10(s) when the corpus scale changes: measured on a 4x4 test crop
    with a quarter of its pixels empty, the shift came out at exactly 3/4 of
    the correct value. That would make `echogenicity` partly a measure of how
    much dead aperture a crop contains, and would make it respond to
    `intensity_scale` non-uniformly across crops.
    """
    env = np.sqrt(crop[0].astype(np.float64) ** 2 + crop[1].astype(np.float64) ** 2)
    return float(20.0 * np.log10(env.mean() + eps))


def sample_store_crops(
    store_path: str | Path,
    frame_ids: list[str],
    spec: StoreCropSpec,
    label_maps=None,
    only_frames: list[str] | None = None,
    frame_label_fn=frame_labels_from_metadata,
) -> StoreCropBatch:
    """Draw `crops_per_frame` un-augmented crops from each frame in `frame_ids`.

    Returns crops already in the encoder's layout, ``(N, 2K, size, size)``, with
    geometric labels (`z_centre`, `x_centre`, in mm) attached per crop and
    frame-level labels broadcast across their frame's crops.

    Pass `label_maps` (a `data.label_maps.LabelMaps`) to also attach the
    crop-level anatomy targets. They are suffixed `_crop` rather than given
    §4.1's bare names because `cyst_fraction` is already taken here by the
    *frame-level* quantity from `frame_metadata.json`, and every K column
    measured so far reports it under that name — silently changing what the
    name means would make old and new columns look comparable when they are
    not.

    `only_frames` materializes just those frames while still drawing origins
    for all of `frame_ids`, so a caller short of memory can walk the corpus in
    chunks and get exactly the crops the whole-corpus call would have given.
    The full set is 3.5 GB at 830 frames x 8 crops, against a 4 GB cgroup on
    the `data` instance — which is how this argument came to exist.

    `frame_label_fn` turns the metadata sidecar into frame-level label vectors.
    It defaults to :func:`frame_labels_from_metadata`, which reads NVIDIA's
    simulator ground truth; WI-16's Stanford store has none of those keys and
    passes :func:`ssl_us.eval.stanford_targets.stanford_frame_labels` instead.
    """
    store = open_store(store_path)
    metadata = store.read_frame_metadata()
    occupancy = np.asarray(store.occupancy)  # (Z, X, K_store)
    nz, nx = store.spec.nz, store.spec.nx

    grid = StoreGrid.from_store(store)

    all_ids, all_origins = sample_store_origins(frame_ids, spec, nz, nx, grid)
    wanted = set(frame_ids if only_frames is None else only_frames)
    keep = np.array([fid in wanted for fid in all_ids], dtype=bool)
    by_frame: dict[str, list[tuple[int, int]]] = {}
    for fid, (z0, x0) in zip(all_ids, all_origins, strict=True):
        if fid in wanted:
            by_frame.setdefault(fid, []).append((int(z0), int(x0)))

    ids = [fid for fid, k in zip(all_ids, keep, strict=True) if k]
    origins_arr = all_origins[keep]
    n = len(ids)
    crops = np.empty((n, spec.in_channels, spec.crop_size, spec.crop_size), dtype=np.float32)
    z_centres = np.empty(n, dtype=np.float32)
    x_centres = np.empty(n, dtype=np.float32)
    echo_db = np.empty(n, dtype=np.float32) if spec.derive_echogenicity else None

    # Frame-major so each frame is decompressed once, but written back into
    # the crop-major order the origins were drawn in.
    positions: dict[str, list[int]] = {}
    for i, fid in enumerate(ids):
        positions.setdefault(fid, []).append(i)

    for fid, frame_origins in by_frame.items():
        frame = store.read_frame(store.index_of(fid))  # (2, Z, X, K_store)
        for (z0, x0), out_i in zip(frame_origins, positions[fid], strict=True):
            zs, xs = slice(z0, z0 + spec.crop_size), slice(x0, x0 + spec.crop_size)
            # Reduce the K-ladder and normalize by occupancy in one step, in
            # that order. The reverse — normalizing at K=16 then reducing —
            # is the documented wrong path (PLAN.md §3.2).
            v = tf.reduce_and_normalize(
                frame[:, zs, xs, :], occupancy[zs, xs, :], target_k=spec.k
            )  # (2, size, size, k)
            # (2, size, size, k) -> (2k, size, size), I-block then Q-block,
            # matching GpuAugmentor's `movedim(-1, 2).flatten(1, 2)`.
            if spec.intensity_scale != 1.0:
                v = v * spec.intensity_scale
            crops[out_i] = np.moveaxis(v, -1, 1).reshape(spec.in_channels, *v.shape[1:3])
            z_centres[out_i] = grid.row_to_z_mm(z0 + spec.crop_size / 2)
            x_centres[out_i] = grid.col_to_x_mm(x0 + spec.crop_size / 2)
            if echo_db is not None:
                # From `v`, the reduced+normalized crop, i.e. the same pixels
                # the encoder is handed — not the raw store sums.
                echo_db[out_i] = crop_echogenicity_db(v)

    labels: dict[str, np.ndarray] = {"z_centre": z_centres, "x_centre": x_centres}
    if echo_db is not None:
        labels["echogenicity"] = echo_db
    per_frame = frame_label_fn(metadata, frame_ids)
    frame_index = {fid: i for i, fid in enumerate(frame_ids)}
    row = np.array([frame_index[f] for f in ids])
    for key, values in per_frame.items():
        labels[key] = values[row]

    if label_maps is not None:
        anatomy = label_maps.crop_labels(ids, origins_arr, spec.crop_size)
        labels["cyst_fraction_crop"] = anatomy["cyst_fraction"]
        labels["contains_cyst"] = anatomy["contains_cyst"]
        labels["sos_mean_crop"] = anatomy["sos_mean"]
        labels["sos_std_crop"] = anatomy["sos_std"]
        # Carried so the caller can honour PLAN.md §2.6: the 75 frames whose
        # `segmentation_map` is entirely empty are a mix of true negatives and
        # unlabelled edge-touching cysts, and belong out of the cyst probes
        # rather than in them as zeros.
        labels["segmentation_empty"] = np.array(
            [bool(metadata[f]["segmentation_empty"]) for f in ids], dtype=bool
        )

    return StoreCropBatch(
        images=crops,
        frame_id=np.asarray(ids, dtype=object),
        labels=labels,
        origins=origins_arr,
    )


def corpus_median_abs(
    store_path: str | Path,
    spec: StoreCropSpec,
    n_frames: int = 8,
    frame_label_fn=frame_labels_from_metadata,
) -> float:
    """Median |crop value| over a fixed sample of a store's frames.

    A robust one-number summary of a corpus's amplitude scale, at the same K
    and crop size the run will use (both change the magnitude: `reduce_ladder`
    sums before `normalize` divides).
    """
    store = open_store(store_path)
    ids = list(store.frame_ids)[:n_frames]
    probe_spec = replace(spec, intensity_scale=1.0)
    batch = sample_store_crops(store_path, ids, probe_spec, frame_label_fn=frame_label_fn)
    return float(np.median(np.abs(batch.images)))


def corpus_intensity_scale(
    store_path: str | Path,
    reference_store_path: str | Path,
    spec: StoreCropSpec,
    n_frames: int = 8,
    frame_label_fn=frame_labels_from_metadata,
    reference_frame_label_fn=frame_labels_from_metadata,
) -> float:
    """Scalar mapping `store_path`'s amplitude scale onto the reference's.

    **Why a frozen encoder needs this to cross corpora at all.** Nothing in the
    pipeline standardizes intensity: `reduce_and_normalize` divides by
    occupancy, which makes sector *means*, but leaves the corpus's own
    amplitude units intact. Within one corpus that is correct and invisible —
    every frame shares the scale. Across corpora it is neither: measured at
    K=4/128 px, NVIDIA crops have |x| mean 0.52 and Stanford's 321, a factor of
    ~600. And the mismatch is *not* absorbed downstream: the backbone runs in
    `.eval()`, so its BatchNorm layers divide by the running variance recorded
    during NVIDIA training, not by the batch's own. A 600x input drives every
    post-BN activation far outside the range the weights were fitted on, and
    the resulting transfer score measures that mismatch rather than the
    representation.

    **One scalar for the whole corpus, never per crop.** Per-crop
    standardization would divide out exactly the quantity `echogenicity`
    probes, turning a real axis into a constant and its R^2 into noise about
    zero. A single corpus-level factor is a change of units; it moves every
    crop together and leaves all *relative* structure — the depth trend, the
    brightness spread, the session differences — untouched.
    """
    ref = corpus_median_abs(
        reference_store_path, spec, n_frames, frame_label_fn=reference_frame_label_fn
    )
    own = corpus_median_abs(store_path, spec, n_frames, frame_label_fn=frame_label_fn)
    if not np.isfinite(own) or own == 0.0:
        raise ValueError(f"corpus {store_path} has median |crop| = {own!r}; cannot scale")
    return ref / own


def extract_features(
    store_path: str | Path,
    frame_ids: list[str],
    spec: StoreCropSpec,
    encoders: dict,
    *,
    label_maps=None,
    frame_chunk: int = 0,
    tag: str = "wi13",
) -> tuple[StoreCropBatch, dict[str, np.ndarray]]:
    """Features for every encoder, materializing crops a chunk of frames at a time.

    Returns `(batch, {encoder: features})` where `batch` carries the labels,
    frame ids and origins for the whole corpus but **no images** — those are
    the 3.5 GB, and nothing downstream of feature extraction needs them.

    Chunking exists for the `data` instance's 4 GB cgroup; ``frame_chunk=0``
    materializes everything at once. ⚠ Encoders that are *fitted* rather than
    applied (`pca_raw`) fit one basis per chunk, so their scores are not
    comparable across chunk sizes — `scripts/eval/nvidia_matrix.py` says so in its
    output and records it in `results.json`.

    Lives here rather than in `scripts/eval/nvidia_matrix.py` for the reason
    `build_encoders` does: scripts cannot import one another, and the second
    caller (`scripts/eval/geometry_control.py`) must extract features by the
    identical path or its control is measuring the extraction rather than the
    geometry.
    """
    chunk = frame_chunk or len(frame_ids)
    parts, feats = [], {name: [] for name in encoders}
    for start in range(0, len(frame_ids), chunk):
        sub = frame_ids[start : start + chunk]
        b = sample_store_crops(
            store_path, frame_ids, spec, label_maps=label_maps, only_frames=list(sub)
        )
        for name, fn in encoders.items():
            feats[name].append(fn(b.images))
        b.images = np.empty((0,), dtype=np.float32)  # freed before the next chunk
        parts.append(b)
        if chunk < len(frame_ids):
            print(
                f"[{tag}] features {min(start + chunk, len(frame_ids))}/{len(frame_ids)} frames",
                flush=True,
            )

    batch = StoreCropBatch(
        images=np.empty((0,), dtype=np.float32),
        frame_id=np.concatenate([p.frame_id for p in parts]),
        labels={key: np.concatenate([p.labels[key] for p in parts]) for key in parts[0].labels},
        origins=np.concatenate([p.origins for p in parts]),
    )
    return batch, {name: np.concatenate(v) for name, v in feats.items()}


class VICRegCheckpointEncoder:
    """Frozen backbone features from a WI-12 checkpoint.

    Reads the *backbone* output, not the projector: PLAN.md §4.7 probes the
    backbone, and the projector is discarded after pre-training. That is 512-d
    for ResNet-18 and 384-d for the WI-18 ViT-S arm.

    The architecture is read from `run_config.json` beside the checkpoint,
    following `MAECheckpointEncoder`: once an arm can vary its backbone, an
    assumed one fails as a shape error at best and loads a silently different
    model at worst. **Pre-WI-18 checkpoints have no `backbone` key and default
    to `resnet18`**, so every existing column is unchanged.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        in_channels: int,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        checkpoint = Path(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg_path = checkpoint.parent / "run_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        name = cfg.get("backbone", "resnet18")
        if name == "resnet18":
            backbone = ResNetBackbone(in_channels)
        else:
            backbone = build_backbone(
                name, in_channels, int(cfg["crop_size"]), int(cfg.get("patch_size", 0))
            )
        # Read too, not assumed: the projector is discarded for probing, but it
        # is part of the state dict, so a mismatch is a load error rather than a
        # wrong number. 2048 is every run's value to date.
        model = VICReg(
            backbone, backbone_dim=backbone.dim, projector_dim=int(cfg.get("projector_dim", 2048))
        )
        model.load_state_dict(state["model"])
        self.backbone_name = name
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


class MAECheckpointEncoder:
    """Frozen encoder features from a WI-14 MAE checkpoint.

    Calls `MAE2D.encode` — the **unmasked** path. `forward` masks 75 % of the
    tokens because that is the pretext task, and probing it would score the
    reconstruction problem rather than the representation, which is not what
    WI-14's comparison against VICReg is asking.

    The architecture is read from `run_config.json` beside the checkpoint
    rather than assumed, because patch size, mask ratio and crop size are all
    run-time choices and a mismatch would fail as a shape error at best and a
    silently different model at worst.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        in_channels: int,
        device: str = "cuda",
        batch_size: int = 256,
    ):
        from ..models.mae2d import MAE2D

        checkpoint = Path(checkpoint)
        config_path = checkpoint.parent / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} is required to rebuild the MAE architecture for {checkpoint}"
            )
        cfg = json.loads(config_path.read_text())
        if cfg["in_channels"] != in_channels:
            raise ValueError(
                f"checkpoint was trained with in_channels={cfg['in_channels']}, "
                f"but the crops supply {in_channels} — wrong K for this checkpoint"
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = MAE2D(
            in_channels=in_channels,
            img_size=cfg["crop_size"],
            patch_size=cfg["patch_size"],
            mask_ratio=cfg["mask_ratio"],
            # Does not affect `encode` (it only transforms the loss target),
            # but carried through so the rebuilt model matches the run record.
            norm_pix_loss=cfg.get("norm_pix_loss", True),
        )
        model.load_state_dict(state["model"])
        self.epoch = int(state.get("epoch", -1))
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(crops[i : i + self.batch_size]).to(self.device)
            out.append(self.model.encode(x).float().cpu().numpy())
        return np.concatenate(out)


class IJEPACheckpointEncoder:
    """Frozen encoder features from a WI-17 I-JEPA checkpoint.

    Calls `IJEPA.encode`, which is the **target (EMA) encoder** on the whole
    unmasked crop, mean-pooled — the network I-JEPA evaluates and publishes,
    and the counterpart of reading VICReg's backbone rather than its projector.
    `forward` runs the masked pretext task and probing it would score the
    prediction problem instead of the representation.

    Like `MAECheckpointEncoder`, the architecture is read from
    `run_config.json` beside the checkpoint rather than assumed: patch size,
    embed dim and predictor shape are all run-time choices, and WI-17 derives
    the patch size from the crop, so a hardcoded default would silently rebuild
    a different model for the K >= 8 arm.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        in_channels: int,
        device: str = "cuda",
        batch_size: int = 256,
        encoder: str = "target",
        random_init: bool = False,
        seed: int = 0,
    ):
        from ..models.ijepa import IJEPA

        checkpoint = Path(checkpoint)
        config_path = checkpoint.parent / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} is required to rebuild the I-JEPA architecture for {checkpoint}"
            )
        cfg = json.loads(config_path.read_text())
        if cfg["in_channels"] != in_channels:
            raise ValueError(
                f"checkpoint was trained with in_channels={cfg['in_channels']}, "
                f"but the crops supply {in_channels} — wrong K for this checkpoint"
            )
        if random_init:
            torch.manual_seed(seed)
        model = IJEPA(
            in_channels=in_channels,
            img_size=cfg["crop_size"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg.get("embed_dim", 384),
            depth=cfg.get("depth", 12),
            heads=cfg.get("heads", 6),
            predictor_dim=cfg.get("predictor_dim", 192),
            predictor_depth=cfg.get("predictor_depth", 6),
            predictor_heads=cfg.get("predictor_heads", 6),
        )
        if random_init:
            # The architecture floor for the I-JEPA column: same ViT, same
            # crops, never trained. `random_resnet18` cannot play that role for
            # a ViT arm — a random CNN and a random transformer do not encode
            # crop position equally well, so scoring a ViT against a CNN floor
            # would credit (or blame) the architecture for the objective. The
            # shape is read from the trained run's own `run_config.json`, so
            # the floor is the *same* model with the training removed.
            self.epoch = 0
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            self.epoch = int(state.get("epoch", -1))
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.encoder = encoder

    @torch.no_grad()
    def __call__(self, crops: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(crops), self.batch_size):
            x = torch.from_numpy(crops[i : i + self.batch_size]).to(self.device)
            out.append(self.model.encode(x, encoder=self.encoder).float().cpu().numpy())
        return np.concatenate(out)
