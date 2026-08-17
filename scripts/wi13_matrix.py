"""WI-13 — results matrix over the real K-sector store and the WI-12 encoders.

PLAN.md §5.2 WI-13 specifies
``(K ∈ {16,8,4,1}) × (policy A, B) × (§4.1 axes) × (probe, cluster)``. This
script runs **one K column per invocation** — both arms plus the baselines —
and reports the Policy A vs Policy B contrast as the deliverable WI-13's
done-condition names.

    python scripts/wi13_matrix.py --k 4  --out figures/wi13_k4
    python scripts/wi13_matrix.py --k 16 --out figures/wi13_k16

Each K needs its own trained pair, because K sets the stem's input channel
count (`Conv2d(2K, 64, …)`) — a K = 16 cell is a different model, not a
different evaluation of the K = 4 one. `scripts/wi13_kablation.sh` trains them.
Crop geometry is **derived from K** via `LoaderConfig.for_arm`, the same call
the training loader makes, so the encoder is never evaluated on crops it was
not trained on. Cells that were not run are listed in `results.json` under
`not_run` with a reason, per the done-condition's "no silent truncation".

**The crop-level anatomy axes need the per-pixel maps**, which are not in the
store — `scripts/wi13_fetch_label_maps.py` fetches them and `--label-maps`
points here. With them, `contains_cyst`, `cyst_fraction_crop` and
`sos_mean_crop` run; without them they stay in `not_run` as before. The cyst
probes exclude crops from the 75 empty-`segmentation_map` frames (PLAN.md
§2.6), and say so in the output rather than silently shrinking.

Crops are materialized `--frame-chunk` frames at a time. The full set is 3.5 GB
at 830 frames x 8 crops, against a **4 GB cgroup** on the `data` instance —
note `free` reports the host's 31 GB, not the limit that actually applies.
Origins are drawn for the whole corpus before any pixel is read, so chunking
cannot change which crops are used.

**Probing runs on the `validation` split by default** — the 93 frames the SSL
encoders never saw. VICReg pre-training is label-free, so probing the train
frames would not leak *labels*, and the usual SSL convention would allow it;
but those images did shape the weights, so held-out frames are the honest
comparison. `--split train` runs the in-sample version for contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.data.label_maps import DEFAULT_ROOT as LABEL_MAPS_DEFAULT_ROOT
from ssl_us.data.label_maps import open_label_maps
from ssl_us.data.splits import outer_split
from ssl_us.eval import features as feat
from ssl_us.eval import viz
from ssl_us.eval.cluster import cluster_report
from ssl_us.eval.guards import LabelLeakageError, filter_targets
from ssl_us.eval.probes import probe
from ssl_us.eval.store_features import (
    MAECheckpointEncoder,
    StoreCropBatch,
    StoreCropSpec,
    VICRegCheckpointEncoder,
    sample_store_crops,
)

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
CKPT_ROOT = "/mnt/home/benbarkow/checkpoints"
DEFAULT_LABEL_MAPS = LABEL_MAPS_DEFAULT_ROOT

# (target, level, kind) — the §4.1 axes the store can support on its own.
TARGETS = [
    ("phase_error", "frame", "ridge"),
    ("t0_group", "frame", "logistic"),
    ("sos_mean_frame", "frame", "ridge"),
    ("sos_std_frame", "frame", "ridge"),
    ("cyst_fraction", "frame", "ridge"),
    ("z_centre", "crop", "ridge"),
    ("x_centre", "crop", "ridge"),
]

# §4.1's crop-level anatomy axes. These need the per-pixel `seg` / `sos` maps,
# which are not in the store — they are fetched separately by
# `scripts/wi13_fetch_label_maps.py` and read through `data/label_maps.py`.
# Available only when that directory is present; otherwise reported in
# `not_run`, never silently dropped.
CROP_ANATOMY_TARGETS = [
    ("contains_cyst", "crop", "logistic"),
    ("cyst_fraction_crop", "crop", "ridge"),
    ("sos_mean_crop", "crop", "ridge"),
]

UNAVAILABLE_REASON = {
    "contains_cyst": "needs per-pixel segmentation map — run scripts/wi13_fetch_label_maps.py",
    "cyst_fraction_crop": "crop-level needs the segmentation map; frame-level cyst_fraction is run",
    "sos_mean_crop": "crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run",
}

# PLAN.md §2.6: 75 of 923 frames ship an entirely empty `segmentation_map`, and
# NVIDIA's own issue list says cysts touching the frame edge are not labelled.
# So an empty mask is an unknown mix of true negatives and unlabelled
# positives; including those crops would inject that mix straight into the
# primary tissue axis as zeros. Excluded from the cyst probes, and the
# exclusion is reported rather than assumed.
CYST_TARGETS = frozenset({"contains_cyst", "cyst_fraction_crop"})

PHASE_ERROR_ORACLE_R2 = 0.585  # measured: R^2(phase_error ~ sos_mean + sos_std)


def _extract_features(args, frame_ids, spec, label_maps, encoders):
    """Features for every encoder, materializing crops a chunk of frames at a time.

    Returns `(batch, {encoder: features})` where `batch` carries the labels,
    frame ids and origins for the whole corpus but **no images** — those are
    the 3.5 GB, and nothing downstream of feature extraction needs them.
    """
    chunk = args.frame_chunk or len(frame_ids)
    parts, feats = [], {name: [] for name in encoders}
    for start in range(0, len(frame_ids), chunk):
        sub = frame_ids[start : start + chunk]
        b = sample_store_crops(
            args.store, frame_ids, spec, label_maps=label_maps, only_frames=list(sub)
        )
        for name, fn in encoders.items():
            feats[name].append(fn(b.images))
        b.images = np.empty((0,), dtype=np.float32)  # freed before the next chunk
        parts.append(b)
        if chunk < len(frame_ids):
            print(
                f"[wi13] features {min(start + chunk, len(frame_ids))}/{len(frame_ids)} frames",
                flush=True,
            )

    batch = StoreCropBatch(
        images=np.empty((0,), dtype=np.float32),
        frame_id=np.concatenate([p.frame_id for p in parts]),
        labels={
            key: np.concatenate([p.labels[key] for p in parts]) for key in parts[0].labels
        },
        origins=np.concatenate([p.origins for p in parts]),
    )
    return batch, {name: np.concatenate(v) for name, v in feats.items()}


def target_subsets(targets, labels: dict, n: int) -> dict:
    """Per-target crop mask. Everything is in, except the cyst probes.

    Applied at the probe call site rather than inside `probe`, so the protocol
    is untouched and the exclusion stays visible next to the reason for it.
    """
    subsets = {t: np.ones(n, dtype=bool) for t, _, _ in targets}
    empty = labels.get("segmentation_empty")
    if empty is not None:
        keep = ~np.asarray(empty, dtype=bool)
        for t in subsets:
            if t in CYST_TARGETS:
                subsets[t] = keep
    return subsets


# Checkpoint directory layouts, tried in order. The K-ablation cells
# `wi13_kablation.sh` trains carry K in their directory name; K=4 was trained
# earlier by WI-12 and lives under the wi12_* name, which predates the
# convention. Ordered so an explicit per-K directory always wins.
CKPT_TEMPLATES = (
    "wi13_vicreg_k{k}_policy{policy}",
    "wi12_vicreg_policy{policy}",
)

# ⚠ The wi12_* fallback carries no `{k}`, so it matches *every* K, and WI-12
# only ever trained K=4. Left ungated it reports a trained pair for K=8 — which
# does not exist — and the availability probe below then drops "K=8 column"
# from `not_run`, i.e. exactly the silent truncation WI-13's done-condition
# forbids. It is a wrong *answer*, not a crash: loading it for a real `--k 8`
# run would fail loudly on the stem's channel count in `load_state_dict`, so
# the only thing it ever corrupted is the not-run bookkeeping.
LEGACY_TEMPLATE_K = 4


def find_checkpoint(root: str | Path, k: int, policy: str) -> Path | None:
    for template in CKPT_TEMPLATES:
        if "{k}" not in template and k != LEGACY_TEMPLATE_K:
            continue
        candidate = Path(root) / template.format(k=k, policy=policy) / "last.pt"
        if candidate.exists():
            return candidate
    return None


def build_encoders(args, in_channels: int) -> dict:
    """Encoder name -> callable. Both arms go through the identical path."""
    encoders = {}
    for policy in ("A", "B"):
        ckpt = find_checkpoint(args.checkpoint_root, args.k, policy)
        if ckpt is None:
            print(f"[wi13] policy {policy}: no K={args.k} checkpoint found — skipping")
            continue
        print(f"[wi13] policy {policy}: {ckpt}")
        encoders[f"vicreg_{policy}"] = VICRegCheckpointEncoder(
            ckpt, in_channels=in_channels, device=args.device
        )
    # WI-14's MAE baselines, when trained for this K. Both target variants are
    # probed because they are not interchangeable on speckle: `norm_pix_loss`
    # divides each patch by its own std, and the per-patch scale is the *only*
    # predictable component here (neighbour correlation 0.87 lateral / 0.64
    # axial, against 0.000 for the normalised within-patch pattern). The
    # normalised arm therefore trains on pure noise, and the raw arm is the
    # fair MAE baseline for this data even though it is not the paper default.
    for suffix, name in ((f"wi14_mae_k{args.k}", "mae_norm"),
                         (f"wi14_mae_k{args.k}_rawtarget", "mae_raw")):
        mae_ckpt = Path(args.checkpoint_root) / suffix / "last.pt"
        if mae_ckpt.exists():
            print(f"[wi13] {name}: {mae_ckpt}")
            encoders[name] = MAECheckpointEncoder(
                mae_ckpt, in_channels=in_channels, device=args.device
            )
        else:
            print(f"[wi13] no {name} checkpoint for K={args.k} — skipping")

    # The floor: same architecture, same input, never trained. If a trained arm
    # cannot beat this, it learned nothing that the architecture did not supply.
    encoders["random_resnet18"] = feat.RandomResNet18(
        in_channels=in_channels, seed=args.seed, device=args.device
    )
    encoders["pca_raw"] = lambda c: feat.pca_features(c, seed=args.seed)
    encoders["trivial"] = feat.trivial_features
    return encoders


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--checkpoint-root", default=CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default="figures/wi13_k4")
    p.add_argument("--k", type=int, default=4)
    # Default None, then derived from K below — hardcoding 128/0.0 would have
    # silently evaluated the K=16 arm on 128 px full-depth crops it was never
    # trained on (it trains at 64 px below z=8 mm).
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=None, dest="z_min_mm")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument(
        "--frame-chunk",
        type=int,
        default=100,
        dest="frame_chunk",
        help="materialize crops this many frames at a time (0 = all at once)",
    )
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--reference-encoder", default="vicreg_A", dest="reference_encoder")
    p.add_argument("--projections", nargs="+", default=["pca"])
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--label-maps",
        default=str(DEFAULT_LABEL_MAPS),
        dest="label_maps",
        help="per-pixel seg/sos maps; '' disables the crop-level anatomy targets",
    )
    args = p.parse_args()

    # Geometry comes from LoaderConfig.for_arm — the same call the *training*
    # loader uses — rather than a copy of §3.2c/d's rule here. Two copies would
    # drift, and the failure is silent: the encoder would simply be evaluated on
    # crops it never saw. Explicit flags still win, for ablations that
    # deliberately vary one of them.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    if args.z_min_mm is None:
        args.z_min_mm = arm.z_min_mm

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(args.store)
    splits = outer_split(np.array(store.frame_ids))
    frame_ids = splits[args.split]
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]

    spec = StoreCropSpec(
        k=args.k,
        crop_size=args.crop_size,
        z_min_mm=args.z_min_mm,
        crops_per_frame=args.crops_per_frame,
        seed=args.seed,
    )
    print(
        f"[wi13] K={spec.k} crop={spec.crop_size} in_ch={spec.in_channels} "
        f"split={args.split} frames={len(frame_ids)} "
        f"crops/frame={spec.crops_per_frame}",
        flush=True,
    )

    # The per-pixel maps are optional: without them the crop-level anatomy
    # targets stay in `not_run` with their reason, exactly as before.
    label_maps, maps_error = None, None
    if args.label_maps:
        try:
            label_maps = open_label_maps(args.label_maps)
            print(f"[wi13] label maps: {label_maps.root} ({len(label_maps)} frames)")
        except (FileNotFoundError, ValueError) as exc:
            maps_error = str(exc)
            print(f"[wi13] no usable label maps ({exc}) — crop anatomy targets stay blocked")

    # Sampled once and shared by every encoder, so the contrast is a difference
    # in weights rather than in which crops each encoder happened to see.
    #
    # Materialized in frame chunks, because the whole set is 3.5 GB at 830
    # frames x 8 crops against a 4 GB cgroup on the `data` instance — the first
    # attempt was OOM-killed. Origins are drawn for the full corpus up front
    # (`sample_store_origins`), so the crops are identical to what a
    # single-shot call would give; only the peak memory differs.
    encoders = build_encoders(args, spec.in_channels)
    # `pca_raw` is the one encoder that is *fitted* rather than applied: it runs
    # a fresh PCA on whatever array it is handed, so under chunking it fits one
    # basis per chunk instead of one over the corpus, and its scores shift by up
    # to 0.026 (measured, K=4 train). Every other encoder is a per-crop forward
    # pass and is chunk-independent. Validation is 93 frames, i.e. one chunk, so
    # this only ever bites the 830-frame train split — which is exactly how it
    # went unnoticed. Chunking exists for the `data` instance's 4 GB cgroup;
    # anywhere with room for 3.5 GB should pass `--frame-chunk 0`.
    chunked = bool(args.frame_chunk) and args.frame_chunk < len(frame_ids)
    if chunked and "pca_raw" in encoders:
        print(
            f"[wi13] ⚠ pca_raw is fitted per chunk ({args.frame_chunk} frames of "
            f"{len(frame_ids)}), not over the corpus — its scores are not "
            f"comparable to a --frame-chunk 0 column. Recorded in results.json.",
            flush=True,
        )
    batch, feats_cache = _extract_features(
        args, frame_ids, spec, label_maps, encoders
    )
    zc = batch.labels["z_centre"]
    print(
        f"[wi13] crops {len(batch)} | depth {zc.min():.1f}-{zc.max():.1f} mm",
        flush=True,
    )

    targets = list(TARGETS)
    if label_maps is not None:
        targets += CROP_ANATOMY_TARGETS
        pos = batch.labels["contains_cyst"]
        keep = ~batch.labels["segmentation_empty"]
        print(
            f"[wi13] crop anatomy: cyst-positive {pos.mean() * 100:.1f}% of crops | "
            f"cyst probes drop {int((~keep).sum())} crops from "
            f"{len({f for f, e in zip(batch.frame_id, ~keep, strict=True) if e})} "
            f"empty-segmentation frames (PLAN.md §2.6)",
            flush=True,
        )

    legal = filter_targets(args.stage, [t for t, _, _ in targets])
    blocked_by_stage = [t for t, _, _ in targets if t not in legal]
    if blocked_by_stage:
        print(f"[wi13] stage {args.stage!r} BLOCKS (label leak): {blocked_by_stage}")

    results = []
    subsets = target_subsets(targets, batch.labels, len(batch))

    for name, f in feats_cache.items():
        print(f"\n[{name}] features {f.shape}", flush=True)
        for target, level, kind in targets:
            if target not in legal:
                continue
            m = subsets[target]
            try:
                r = probe(
                    f[m], batch.labels[target][m], batch.frame_id[m], target,
                    stage=args.stage, level=level, kind=kind, seed=args.seed,
                )
            except (ValueError, LabelLeakageError) as exc:
                print(f"  {target}: SKIPPED ({exc})")
                continue
            # `row()`'s ci95 is a *formatted string* for the markdown table, so
            # the bounds are carried separately — comparing the string would
            # compare characters and silently produce a nonsense verdict.
            row = {"encoder": name, **r.row(),
                   "score_lo": r.score_lo, "score_hi": r.score_hi,
                   "n_crops": int(m.sum())}
            if target == "phase_error":
                row["oracle_r2"] = PHASE_ERROR_ORACLE_R2
                row["beats_oracle"] = bool(r.score > PHASE_ERROR_ORACLE_R2)
            results.append(row)
            print(
                f"  {target:<16} {r.metric:<18} {r.score:+.3f}  "
                f"ci95 [{r.score_lo:+.3f},{r.score_hi:+.3f}]  n_eff={r.n_eff}"
            )

    # ── Policy A vs Policy B, the WI-13 deliverable ──────────────────────────
    contrast = []
    by_key = {(r["encoder"], r["target"]): r for r in results}
    for target, level, _kind in targets:
        a, b = by_key.get(("vicreg_A", target)), by_key.get(("vicreg_B", target))
        if not (a and b):
            continue
        rand = by_key.get(("random_resnet18", target))
        mae = by_key.get(("mae_raw", target)) or by_key.get(("mae_norm", target))
        contrast.append({
            "target": target,
            "level": level,
            "metric": a["metric"],
            "policy_a": a["score"],
            "policy_b": b["score"],
            "delta_b_minus_a": b["score"] - a["score"],
            # Overlapping CIs mean the arms are not separated by this cell,
            # which is a result and needs saying rather than ranking anyway.
            "separated": bool(
                a["score_hi"] < b["score_lo"] or b["score_hi"] < a["score_lo"]
            ),
            "random_baseline": rand["score"] if rand else None,
            # WI-14: does joint embedding actually beat masked autoencoding here?
            "mae": mae["score"] if mae else None,
            "best_vicreg_beats_mae": (
                None if mae is None else bool(max(a["score"], b["score"]) > mae["score"])
            ),
            # A trained arm that cannot clear an untrained one of the same
            # architecture has learned nothing for this target.
            "beats_random": (
                None if rand is None
                else bool(max(a["score"], b["score"]) > rand["score"])
            ),
        })

    ref = args.reference_encoder
    if ref not in feats_cache:
        ref = next(iter(feats_cache))
    clusters = cluster_report(feats_cache[ref], batch.labels)

    figs, geometry_r2 = [], None
    for method in args.projections:
        sub = np.arange(len(batch))
        if len(sub) > args.max_points:
            sub = np.random.default_rng(0).choice(len(batch), args.max_points, replace=False)
        labs = {k: v[sub] for k, v in batch.labels.items()}
        xy = viz.project(feats_cache[ref][sub], method)
        figs.append(viz.plot_embedding(
            xy, labs, f"{ref} — {method.upper()}",
            out_dir / f"embed_{ref}_{method}.png"))
        resid, geometry_r2 = viz.regress_out(
            feats_cache[ref], batch.labels, ["z_centre", "x_centre"])
        print(f"\n[wi13] geometry (z,x) explains {geometry_r2 * 100:.1f}% of {ref} variance")
        figs.append(viz.plot_embedding(
            viz.project(resid[sub], method), labs,
            f"{ref} — {method.upper()} (geometry regressed out)",
            out_dir / f"embed_{ref}_{method}_residual.png"))

    not_run = {}
    if label_maps is None:
        not_run.update(UNAVAILABLE_REASON)
        if maps_error:
            not_run["label_maps"] = maps_error
    # Only report a K column as missing if it really has no trained pair — the
    # ablation has since trained K=16 and K=1, and a hardcoded list would keep
    # claiming otherwise in every artifact this writes.
    for k in (16, 8, 4, 1):
        if k == args.k:
            continue
        if all(find_checkpoint(args.checkpoint_root, k, p) for p in ("A", "B")):
            continue
        not_run[f"K={k} column"] = (
            "K sets the stem's input channel count, so this cell needs its own "
            "VICReg run (~1.5 h/arm); no trained pair found for this K"
        )
    for t in blocked_by_stage:
        not_run[t] = f"blocked by label-leakage guard on stage {args.stage!r}"

    payload = {
        "wi": 13, "k": spec.k, "crop_size": spec.crop_size, "stage": args.stage,
        "split": args.split, "n_frames": len(frame_ids), "n_crops": len(batch),
        "crops_per_frame": spec.crops_per_frame,
        "frame_chunk": args.frame_chunk,
        "pca_raw_fitted_per_chunk": bool(chunked and "pca_raw" in encoders),
        "label_maps": str(label_maps.root) if label_maps is not None else None,
        "cyst_probe_exclusion": (
            None if label_maps is None
            else {
                "reason": "PLAN.md §2.6 — empty segmentation_map is an unknown mix "
                          "of true negatives and unlabelled edge-touching cysts",
                "targets": sorted(CYST_TARGETS),
                "crops_dropped": int((~subsets["contains_cyst"]).sum()),
                "crops_kept": int(subsets["contains_cyst"].sum()),
            }
        ),
        "encoders": list(encoders), "reference_encoder": ref,
        "geometry_variance_explained": geometry_r2,
        "clusters": clusters, "results": results,
        "policy_contrast": contrast, "not_run": not_run,
        "figures": [str(f) for f in figs],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_tables(results, contrast, not_run, out_dir / "results.md")
    print(f"\n[wi13] wrote {out_dir/'results.json'} and {out_dir/'results.md'}")
    return 0


def _write_tables(results, contrast, not_run, path: Path) -> None:
    lines = ["# WI-13 results matrix (K=4)", "", "## Policy A vs Policy B", ""]
    cols = ["target", "level", "metric", "policy_a", "policy_b",
            "delta_b_minus_a", "separated", "random_baseline", "beats_random",
            "mae", "best_vicreg_beats_mae"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in contrast:
        lines.append("| " + " | ".join(
            f"{c[k]:+.3f}" if isinstance(c.get(k), float) else str(c.get(k, ""))
            for k in cols) + " |")

    lines += ["", "## All cells", ""]
    cols2 = ["encoder", "target", "level", "metric", "score", "ci95", "n_eff", "n_feat"]
    lines += ["| " + " | ".join(cols2) + " |", "|" + "---|" * len(cols2)]
    for r in results:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols2) + " |")

    lines += ["", "## Cells not run", ""]
    lines += ["| cell | reason |", "|---|---|"]
    for cell, reason in not_run.items():
        lines.append(f"| {cell} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
