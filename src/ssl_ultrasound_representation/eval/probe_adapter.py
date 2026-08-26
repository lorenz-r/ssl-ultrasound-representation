"""Expose `RFConvMAE` to the frozen-feature probe protocol in `ssl_us.eval.probes`.

The probe protocol (PCA capacity control -> GroupKFold by frame -> nested CV for
the penalty -> bootstrap CI) is reused verbatim by importing it. What cannot be
reused is `ssl_us.eval.store_features`: that path crops a **beamformed K-sector**
store and hands encoders `(2K, crop, crop)` image tensors, whereas `RFConvMAE`
consumes **pre-beamform channel data** `(2, n_fast, n_lat)`. Feeding one to the
other would not be a harder evaluation, it would be a different input space. So
this module supplies the features and the labels, and delegates every scoring
decision to `probes.probe`.

Three things this gets right that a naive adapter would not:

1. **The mask channel is zeroed.** `RFConvMAE` takes `(I, Q, mask_flag)`. Probing
   with a live mask would score the reconstruction pretext instead of the
   representation — the same reason `MAECheckpointEncoder` calls `MAE2D.encode`
   rather than `forward`.
2. **Normalization matches training.** `train_rf_mae` applies per-depth RMS
   (`train.normalize_frame`) before the model sees anything; features extracted
   from raw counts would be a different distribution than the encoder was fit on.
3. **The architecture is rebuilt from the checkpoint's own `model_config`**, never
   assumed — c0, depths, dilations and global_ctx are all run-time choices.

Stage tag: `probes.probe` demands one so `assert_no_label_leak` can run. This data
is not beamformed at all, so no beamformer consumed a label and `c0_const` (the
"leaks nothing" tag) is the honest value. `gt_sos` would be a lie here — there is
no delay-and-sum step to consume the sound-speed map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ssl_ultrasound_representation.mae.model.rf_conv_mae import (
    RFConvMAE,
    RFMaeConfig,
    fold_frame,
)
from ssl_ultrasound_representation.train import normalize_frame

# Frame-level axes this corpus can support. The crop-level geometry targets
# (`z_centre`, `x_centre`) from the NVIDIA matrix have no analogue here: a
# feature vector covers a whole frame, so there is no crop centre to regress.
NV_FRAME_TARGETS: list[tuple[str, str, str]] = [
    ("phase_error", "frame", "ridge"),
    ("t0_group", "frame", "logistic"),
    ("sos_mean_frame", "frame", "ridge"),
    ("sos_std_frame", "frame", "ridge"),
    ("cyst_fraction", "frame", "ridge"),
]

LABEL_COLUMNS = ("phase_error", "t0", "sound_speed_map", "segmentation_map")


class RFConvMAEEncoder:
    """Frozen `RFConvMAE` features: one vector per frame.

    Mirrors `ssl_us.eval.store_features.MAECheckpointEncoder` in contract — build
    from a checkpoint, then call on data and get `(N, D)` back — but consumes
    frames in this project's layout, `(transducer, scanline, 2, sample)`.

    A frame folds into `n_lat` independent rows; each row is encoded to the
    bottleneck and mean-pooled, then the rows are averaged into the frame vector.
    That is the same reduction `MAE2D.encode` performs over tokens, one level up.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        lateral: str | None = None,
        row_batch: int = 16,
        random_init: bool = False,
        seed: int = 0,
    ):
        """`random_init=True` rebuilds the same architecture with untrained weights.

        This is the floor, and it is not decoration: an anisotropic dilated conv
        stack imposes real structure on RF data before any training, so a probe
        score is only evidence of *learning* if it clears the identical untrained
        network on the identical frames. Same role as `random_resnet18` in
        `ssl_us.eval.encoders`.
        """
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if "model_config" not in ckpt:
            raise KeyError(
                f"{checkpoint} has no 'model_config' — cannot rebuild the architecture. "
                "Checkpoints written by train_rf_mae always carry one."
            )
        cfg = RFMaeConfig(**ckpt["model_config"])
        if random_init:
            torch.manual_seed(seed)
        model = RFConvMAE(cfg)
        if not random_init:
            model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.row_batch = row_batch
        self.epoch = -1 if random_init else int(ckpt.get("epoch", -1))
        self.random_init = random_init
        # The fold used at training time decides which axis became n_lat; probing
        # on the other one would encode a transposed field of view.
        self.lateral = lateral or (ckpt.get("config") or {}).get("lateral", "transducer")

    @property
    def dim(self) -> int:
        return self.cfg.widths[-1]

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> np.ndarray:
        """`(transducer, scanline, 2, sample)` -> `(dim,)` float32."""
        rows = fold_frame(frame.unsqueeze(0), self.lateral)  # (n_rows, 2, H, W)
        chunks = []
        for i in range(0, rows.shape[0], self.row_batch):
            x = rows[i : i + self.row_batch].to(self.device)
            # Zero mask plane: nothing is hidden, so this is the unmasked path.
            x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
            chunks.append(self.model.latent(x, pool="mean").float().cpu())
        return torch.cat(chunks).mean(dim=0).numpy()


def extract_frame_features(
    dataset,
    encoder: RFConvMAEEncoder,
    frame_ids: Sequence[int],
    *,
    normalize: bool = True,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each frame once. Returns `(features (N, D), frame_ids (N,))`.

    `dataset` is indexable and yields `(transducer, scanline, 2, sample)` tensors
    — an `UltrasoundMAEDataset`, or anything with the same contract.
    """
    feats = []
    it = range(len(frame_ids))
    if progress:
        from tqdm import tqdm

        it = tqdm(it, desc="encoding frames", unit="frame")
    for i in it:
        frame = dataset[i]
        if normalize:
            frame = normalize_frame(frame)
        feats.append(encoder(frame))
    return np.stack(feats), np.asarray(frame_ids, dtype=str)


def build_frame_labels(
    nv_root: str | Path,
    splits: Iterable[str] = ("train", "validation"),
    *,
    progress: bool = True,
) -> dict:
    """Frame-level probe labels streamed from the NV Parquet shards.

    Only the label columns are requested, so the 180x180x1024 IQ payload is never
    decoded — this is minutes, not hours. The MAE store built by
    `build_nv_raw2insights_mae_dataset.py` deliberately carries IQ only
    (`mode="mae"`), which is why the labels have to come from here.

    Returns `{split: {"frame_index": [...], "<target>": [...]}}`, ordered by the
    split index the converter assigned, so position i lines up with frame i of
    that split in the zarr.
    """
    from ssl_ultrasound_representation.dataset.nv_raw2insights import (
        iter_nv_raw2insights_samples,
    )

    out: dict[str, dict[str, list]] = {}
    for split in splits:
        rows: dict[str, list] = {
            "frame_index": [], "phase_error": [], "t0": [],
            "sos_mean_frame": [], "sos_std_frame": [],
            "cyst_fraction": [], "segmentation_empty": [],
        }
        it = iter_nv_raw2insights_samples(nv_root, split=split, columns=LABEL_COLUMNS)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, desc=f"labels[{split}]", unit="frame")
        for i, s in enumerate(it):
            sos = np.asarray(s["sound_speed_map"], dtype=np.float64)
            seg = np.asarray(s["segmentation_map"])
            rows["frame_index"].append(i)
            rows["phase_error"].append(float(s["phase_error"]))
            rows["t0"].append(float(s["t0"]))
            rows["sos_mean_frame"].append(float(sos.mean()))
            rows["sos_std_frame"].append(float(sos.std()))
            rows["cyst_fraction"].append(float((seg > 0).mean()))
            rows["segmentation_empty"].append(bool(seg.max() == 0))
        out[split] = rows
    return out


def derive_t0_group(t0: np.ndarray) -> np.ndarray:
    """`t0` is a small set of discrete acquisition settings — index them.

    Kept explicit rather than binned: if the corpus really has continuous t0 this
    produces one class per frame, which `probe` then rejects for having too few
    members per group. That is the correct loud failure.
    """
    uniq = np.unique(np.round(np.asarray(t0, dtype=float), 12))
    return np.searchsorted(uniq, np.round(np.asarray(t0, dtype=float), 12))


def labels_to_arrays(split_labels: dict) -> dict[str, np.ndarray]:
    """Label dict from :func:`build_frame_labels` -> probe-ready arrays."""
    out = {k: np.asarray(v) for k, v in split_labels.items()}
    out["t0_group"] = derive_t0_group(out["t0"])
    return out


@dataclass
class ProbeRow:
    encoder: str
    payload: dict


def run_probe_matrix(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    frame_ids: np.ndarray,
    *,
    encoder_name: str = "rf_conv_mae",
    targets: Sequence[tuple[str, str, str]] = tuple(NV_FRAME_TARGETS),
    stage: str = "c0_const",
    seed: int = 0,
) -> tuple[list[dict], dict[str, str]]:
    """Run every legal target through `ssl_us.eval.probes.probe`.

    Returns `(rows, not_run)`. Nothing is dropped silently: a target that cannot
    run lands in `not_run` with the reason, matching `nvidia_matrix.py`'s
    "no silent truncation" rule.
    """
    from ssl_us.eval.guards import LabelLeakageError, filter_targets
    from ssl_us.eval.probes import probe

    from ssl_us.eval.probes import CAPACITY_DIMS

    legal = set(filter_targets(stage, [t for t, _, _ in targets]))
    rows: list[dict] = []
    not_run: dict[str, str] = {
        t: f"blocked by label-leakage guard on stage {stage!r}"
        for t, _, _ in targets
        if t not in legal
    }

    # The protocol's capacity control is a PCA to 64 dims, and PCA needs at least
    # that many samples. Crop-level probing has frames x crops_per_frame samples so
    # it never hits this; a frame-level probe has one sample per frame, so a split
    # with <= 64 frames cannot run the protocol as specified. Say that plainly here
    # rather than letting sklearn report it as an svd_solver argument error.
    n_samples = len(features)
    if n_samples <= CAPACITY_DIMS and features.shape[1] > CAPACITY_DIMS:
        reason = (
            f"frame-level probe has {n_samples} samples but the protocol's capacity "
            f"control is PCA to {CAPACITY_DIMS} dims, which needs > {CAPACITY_DIMS}. "
            f"Use a split with more frames (NV: train=830, validation=93), or accept "
            f"a different protocol than nvidia_matrix.py's and say so explicitly."
        )
        for t, _, _ in targets:
            not_run.setdefault(t, reason)
        return rows, not_run

    for target, level, kind in targets:
        if target not in legal:
            continue
        if target not in labels:
            not_run[target] = "label not present in the labels file"
            continue
        y = np.asarray(labels[target])
        mask = np.ones(len(y), dtype=bool)
        # Same exclusion as the NVIDIA matrix: an empty segmentation map is an
        # unknown mix of true negatives and unlabelled edge-touching cysts.
        if target == "cyst_fraction" and "segmentation_empty" in labels:
            mask = ~np.asarray(labels["segmentation_empty"], dtype=bool)
        try:
            r = probe(
                features[mask], y[mask], frame_ids[mask], target,
                stage=stage, level=level, kind=kind, seed=seed,
            )
        except (ValueError, LabelLeakageError) as exc:
            not_run[target] = f"{type(exc).__name__}: {exc}"
            continue
        rows.append({
            "encoder": encoder_name, **r.row(),
            "score_lo": r.score_lo, "score_hi": r.score_hi,
            "n_frames": int(mask.sum()),
        })
    return rows, not_run


def write_report(rows: list[dict], not_run: dict, payload_extra: dict, out_dir: Path) -> None:
    """`results.json` + `results.md`, shaped like `nvidia_matrix.py`'s output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(
        {"results": rows, "not_run": not_run, **payload_extra}, indent=1))

    cols = ["encoder", "target", "level", "metric", "score", "ci95", "n_eff", "n_feat"]
    lines = ["# RFConvMAE frozen-feature probes", "", "## Cells", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    lines += ["", "## Cells not run", "", "| cell | reason |", "|---|---|"]
    lines += [f"| {c} | {why} |" for c, why in not_run.items()]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
