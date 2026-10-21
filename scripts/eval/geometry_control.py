"""Geometry control — are the crop-level anatomy scores just *where the crop came from*?

§1.5 is the project's strongest evidence for its actual thesis: Policy A wins
all three crop-level anatomy targets at every K and on both splits, with CI
separation in 17 of 18 cells. It is also **uncontrolled for crop position**.
Cysts have a spatial distribution, `z_centre` alone probes at +0.699, and the
open question is how much of `contains_cyst` is anatomy and how much is
geometry. docs/MEMORY.md §8 item 3.

This script answers it by running the same probes twice on the same features:

1. **plain** — the §1.5 protocol, reproduced here so the comparison is not
   across two runs;
2. **residualised** — features with `z_centre` and `x_centre` linearly
   regressed out first (`eval/viz.regress_out`, binned one-hot per axis, the
   same transform the WI-13 figures use).

and by adding one encoder that makes the question concrete: **`geometry_only`**,
whose entire feature vector is the crop's `(z_centre, x_centre)`. If that
two-number encoder scores near vicreg_A on an anatomy target, the target is
positional and no amount of representation quality is being measured.

    python scripts/eval/geometry_control.py --k 4 --split train \
        --out outputs/nvidia/geometry_k4_train

⚠ **Two limits of the control, stated rather than assumed.** The residualiser
is fit on all crops, including the ones in each CV test fold — it uses no
label, so it cannot leak the target, but it is not a within-fold transform
either. And it removes *additive per-axis* dependence: a genuine `z x x`
interaction survives residualisation and would still be counted as anatomy.
Both are recorded in `results.json`.

`z_centre` and `x_centre` are probed alongside the anatomy targets as the
control's own sanity check: after residualisation they must collapse toward
zero, and if they do not, the transform is not doing what this script claims.
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
from ssl_us.eval import viz
from ssl_us.eval.encoders import build_encoders
from ssl_us.eval.guards import LabelLeakageError, filter_targets
from ssl_us.eval.probes import probe
from ssl_us.eval.store_features import StoreCropSpec, extract_features

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
CKPT_ROOT = "/mnt/home/benbarkow/checkpoints"

# The three §1.5 targets, plus the two geometry axes as the transform's own check.
ANATOMY_TARGETS = [
    ("contains_cyst", "crop", "logistic"),
    ("cyst_fraction_crop", "crop", "ridge"),
    ("sos_mean_crop", "crop", "ridge"),
]
GEOMETRY_TARGETS = [
    ("z_centre", "crop", "ridge"),
    ("x_centre", "crop", "ridge"),
]
GEOMETRY_FACTORS = ["z_centre", "x_centre"]

# PLAN.md §2.6, same exclusion as `nvidia_matrix.py`: an empty segmentation map is
# an unknown mix of true negatives and unlabelled edge-touching cysts.
CYST_TARGETS = frozenset({"contains_cyst", "cyst_fraction_crop"})


def _probe_all(feats, batch, targets, subsets, legal, args, variant):
    rows = []
    for name, f in feats.items():
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
                print(f"  [{variant}] {name}/{target}: SKIPPED ({exc})")
                continue
            rows.append({
                "encoder": name, "variant": variant, **r.row(),
                "score_lo": r.score_lo, "score_hi": r.score_hi,
                "n_crops": int(m.sum()),
            })
            print(
                f"  [{variant}] {name:<17} {target:<19} {r.score:+.3f} "
                f"ci95 [{r.score_lo:+.3f},{r.score_hi:+.3f}] n_eff={r.n_eff}",
                flush=True,
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--checkpoint-root", default=CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default="outputs/nvidia/geometry_k4")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=None, dest="z_min_mm")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--split", default="train", choices=["validation", "train"])
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument("--frame-chunk", type=int, default=100, dest="frame_chunk")
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--n-bins", type=int, default=10, dest="n_bins")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-maps", default=str(LABEL_MAPS_DEFAULT_ROOT), dest="label_maps")
    args = p.parse_args()

    # Geometry from for_arm, exactly as nvidia_matrix.py does, so this control is
    # run on the crops the §1.5 numbers were measured on.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    if args.z_min_mm is None:
        args.z_min_mm = arm.z_min_mm

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(args.store)
    frame_ids = outer_split(np.array(store.frame_ids))[args.split]
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]

    # The per-pixel maps are not optional here: without them there are no
    # crop-level anatomy targets and nothing to control.
    label_maps = open_label_maps(args.label_maps)

    spec = StoreCropSpec(
        k=args.k, crop_size=args.crop_size, z_min_mm=args.z_min_mm,
        crops_per_frame=args.crops_per_frame, seed=args.seed,
    )
    print(
        f"[geom] K={spec.k} crop={spec.crop_size} in_ch={spec.in_channels} "
        f"split={args.split} frames={len(frame_ids)} crops/frame={spec.crops_per_frame}\n"
        f"[geom] label maps: {label_maps.root} ({len(label_maps)} frames)",
        flush=True,
    )

    encoders = build_encoders(args, spec.in_channels, tag="geom")
    batch, feats = extract_features(
        args.store, frame_ids, spec, encoders,
        label_maps=label_maps, frame_chunk=args.frame_chunk, tag="geom",
    )

    # The control that needs no encoder at all: two numbers per crop. Added
    # after extraction because its "features" are the labels themselves.
    feats["geometry_only"] = np.stack(
        [batch.labels[f].astype(np.float32) for f in GEOMETRY_FACTORS], axis=1
    )

    targets = ANATOMY_TARGETS + GEOMETRY_TARGETS
    legal = filter_targets(args.stage, [t for t, _, _ in targets])
    subsets = {t: np.ones(len(batch), dtype=bool) for t, _, _ in targets}
    empty = np.asarray(batch.labels["segmentation_empty"], dtype=bool)
    for t in CYST_TARGETS:
        subsets[t] = ~empty
    print(
        f"[geom] crops {len(batch)} | cyst-positive "
        f"{batch.labels['contains_cyst'].mean() * 100:.1f}% | cyst probes drop "
        f"{int(empty.sum())} crops from empty-segmentation frames (PLAN.md §2.6)",
        flush=True,
    )

    # Residualise every encoder's features against the same two factors. The
    # R^2 reported per encoder is how much of that latent was pure geometry.
    residual_feats, geometry_r2 = {}, {}
    for name, f in feats.items():
        resid, r2 = viz.regress_out(f, batch.labels, GEOMETRY_FACTORS, n_bins=args.n_bins)
        residual_feats[name], geometry_r2[name] = resid, float(r2)
    print("\n[geom] variance explained by (z,x) per encoder:")
    for name, r2 in sorted(geometry_r2.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<17} {r2 * 100:5.1f}%")

    print("\n[geom] probing", flush=True)
    results = _probe_all(feats, batch, targets, subsets, legal, args, "plain")
    results += _probe_all(residual_feats, batch, targets, subsets, legal, args, "residual")

    # ── the deliverable: does §1.5's claim survive the control? ──────────────
    by_key = {(r["encoder"], r["target"], r["variant"]): r for r in results}
    verdict = []
    for target, _level, _kind in ANATOMY_TARGETS:
        for variant in ("plain", "residual"):
            a = by_key.get(("vicreg_A", target, variant))
            b = by_key.get(("vicreg_B", target, variant))
            rand = by_key.get(("random_resnet18", target, variant))
            triv = by_key.get(("trivial", target, variant))
            geo = by_key.get(("geometry_only", target, "plain"))
            if a is None:
                continue
            row = {
                "target": target, "variant": variant, "metric": a["metric"],
                "vicreg_A": a["score"], "vicreg_B": b["score"] if b else None,
                "random_resnet18": rand["score"] if rand else None,
                "trivial": triv["score"] if triv else None,
                # Position alone, always from the plain run — residualising the
                # geometry features against geometry would leave nothing.
                "geometry_only": geo["score"] if geo else None,
            }
            if b is not None:
                row["a_beats_b"] = bool(a["score"] > b["score"])
                row["a_b_separated"] = bool(
                    a["score_hi"] < b["score_lo"] or b["score_hi"] < a["score_lo"]
                )
            if rand is not None:
                row["a_beats_random"] = bool(a["score"] > rand["score"])
                row["a_random_separated"] = bool(rand["score_hi"] < a["score_lo"])
            if geo is not None:
                row["a_beats_geometry_only"] = bool(a["score"] > geo["score"])
                row["a_geometry_separated"] = bool(geo["score_hi"] < a["score_lo"])
            verdict.append(row)

    payload = {
        "wi": 13, "control": "geometry", "k": spec.k, "crop_size": spec.crop_size,
        "split": args.split, "stage": args.stage, "n_frames": len(frame_ids),
        "n_crops": len(batch), "crops_per_frame": spec.crops_per_frame,
        "frame_chunk": args.frame_chunk, "n_bins": args.n_bins,
        "factors": GEOMETRY_FACTORS,
        "encoders": list(feats),
        "geometry_variance_explained": geometry_r2,
        "results": results,
        "verdict": verdict,
        "caveats": {
            "residualiser_fit_on_all_crops": (
                "regress_out is fit on every crop, including each CV test fold. It uses "
                "no label, so it cannot leak the target — it can only remove signal — "
                "but it is not a within-fold transform."
            ),
            "additive_only": (
                "binned one-hot per axis removes additive per-axis dependence; a genuine "
                "z x x interaction survives and would still be counted as anatomy."
            ),
            "pca_raw_fitted_per_chunk": bool(
                args.frame_chunk and args.frame_chunk < len(frame_ids)
            ),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_table(verdict, geometry_r2, out_dir / "results.md", spec.k, args.split)
    print(f"\n[geom] wrote {out_dir/'results.json'} and {out_dir/'results.md'}")
    return 0


def _write_table(verdict, geometry_r2, path: Path, k: int, split: str) -> None:
    lines = [
        f"# WI-13 geometry control (K={k}, {split} split)", "",
        "Anatomy probes with and without `(z_centre, x_centre)` regressed out of "
        "the features, plus `geometry_only` — an encoder whose entire feature "
        "vector is the crop's position.", "",
    ]
    cols = ["target", "variant", "metric", "vicreg_A", "vicreg_B", "random_resnet18",
            "trivial", "geometry_only", "a_beats_b", "a_b_separated",
            "a_beats_geometry_only", "a_geometry_separated"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in verdict:
        lines.append("| " + " | ".join(
            f"{row[c]:+.3f}" if isinstance(row.get(c), float) else str(row.get(c, ""))
            for c in cols) + " |")
    lines += ["", "## Variance explained by (z, x) per encoder", "",
              "| encoder | geometry R² |", "|---|---|"]
    for name, r2 in sorted(geometry_r2.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {r2 * 100:.1f}% |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
