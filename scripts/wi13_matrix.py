"""WI-13 — results matrix over the real K-sector store and the WI-12 encoders.

PLAN.md §5.2 WI-13 specifies
``(K ∈ {16,8,4,1}) × (policy A, B) × (§4.1 axes) × (probe, cluster)``. This
script runs the **K = 4 column** — the two arms WI-12 actually trained — with
the baselines, and reports the Policy A vs Policy B contrast as the deliverable
WI-13's done-condition names.

    python scripts/wi13_matrix.py --out figures/wi13_k4

Other K columns need their own VICReg runs: K sets the stem's input channel
count (`Conv2d(2K, 64, …)`), so a K = 16 cell is a different model, not a
different evaluation of this one. Cells that were not run are listed in
`results.json` under `not_run` with a reason, per the done-condition's "no
silent truncation".

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
from ssl_us.data.splits import outer_split
from ssl_us.eval import features as feat
from ssl_us.eval import viz
from ssl_us.eval.cluster import cluster_report
from ssl_us.eval.guards import LabelLeakageError, filter_targets
from ssl_us.eval.probes import probe
from ssl_us.eval.store_features import (
    StoreCropSpec,
    VICRegCheckpointEncoder,
    sample_store_crops,
)

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
CKPT_ROOT = "/mnt/home/benbarkow/checkpoints"

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

# §4.1 axes that need the per-pixel `seg` / `sos` maps. Those live in the source
# NVIDIA frames, which are not on the cluster — `data/` is empty and only the
# beamformed store survived. Reported, never silently dropped.
UNAVAILABLE_TARGETS = {
    "contains_cyst": "needs per-pixel segmentation map (source NVIDIA frames absent)",
    "cyst_fraction_crop": "crop-level needs the segmentation map; frame-level cyst_fraction is run",
    "sos_mean_crop": "crop-level needs the per-pixel SoS map; frame-level sos_mean_frame is run",
}

PHASE_ERROR_ORACLE_R2 = 0.585  # measured: R^2(phase_error ~ sos_mean + sos_std)


def build_encoders(args, in_channels: int) -> dict:
    """Encoder name -> callable. Both arms go through the identical path."""
    encoders = {}
    for policy in ("A", "B"):
        ckpt = Path(args.checkpoint_root) / f"wi12_vicreg_policy{policy}" / "last.pt"
        if not ckpt.exists():
            print(f"[wi13] policy {policy}: no checkpoint at {ckpt} — skipping")
            continue
        encoders[f"vicreg_{policy}"] = VICRegCheckpointEncoder(
            ckpt, in_channels=in_channels, device=args.device
        )
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
    p.add_argument("--crop-size", type=int, default=128, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=0.0, dest="z_min_mm")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--reference-encoder", default="vicreg_A", dest="reference_encoder")
    p.add_argument("--projections", nargs="+", default=["pca"])
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

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

    # Sampled once and shared by every encoder, so the contrast is a difference
    # in weights rather than in which crops each encoder happened to see.
    batch = sample_store_crops(args.store, frame_ids, spec)
    zc = batch.labels["z_centre"]
    print(
        f"[wi13] crops {batch.images.shape} | depth {zc.min():.1f}-{zc.max():.1f} mm",
        flush=True,
    )

    legal = filter_targets(args.stage, [t for t, _, _ in TARGETS])
    blocked_by_stage = [t for t, _, _ in TARGETS if t not in legal]
    if blocked_by_stage:
        print(f"[wi13] stage {args.stage!r} BLOCKS (label leak): {blocked_by_stage}")

    encoders = build_encoders(args, spec.in_channels)
    results, feats_cache = [], {}

    for name, fn in encoders.items():
        f = fn(batch.images)
        feats_cache[name] = f
        print(f"\n[{name}] features {f.shape}", flush=True)
        for target, level, kind in TARGETS:
            if target not in legal:
                continue
            try:
                r = probe(
                    f, batch.labels[target], batch.frame_id, target,
                    stage=args.stage, level=level, kind=kind, seed=args.seed,
                )
            except (ValueError, LabelLeakageError) as exc:
                print(f"  {target}: SKIPPED ({exc})")
                continue
            # `row()`'s ci95 is a *formatted string* for the markdown table, so
            # the bounds are carried separately — comparing the string would
            # compare characters and silently produce a nonsense verdict.
            row = {"encoder": name, **r.row(),
                   "score_lo": r.score_lo, "score_hi": r.score_hi}
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
    for target, level, _kind in TARGETS:
        a, b = by_key.get(("vicreg_A", target)), by_key.get(("vicreg_B", target))
        if not (a and b):
            continue
        rand = by_key.get(("random_resnet18", target))
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

    not_run = dict(UNAVAILABLE_TARGETS)
    for k in (16, 8, 1):
        not_run[f"K={k} column"] = (
            "K sets the stem's input channel count, so this cell needs its own "
            "VICReg run (~1.5 h/arm); only K=4 was trained in WI-12"
        )
    for t in blocked_by_stage:
        not_run[t] = f"blocked by label-leakage guard on stage {args.stage!r}"

    payload = {
        "wi": 13, "k": spec.k, "crop_size": spec.crop_size, "stage": args.stage,
        "split": args.split, "n_frames": len(frame_ids), "n_crops": len(batch),
        "crops_per_frame": spec.crops_per_frame,
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
            "delta_b_minus_a", "separated", "random_baseline", "beats_random"]
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
