"""Results-matrix driver — runnable on `bmode` with a random encoder, before any
beamformer or trained model exists (PLAN.md 4.6).

    python -m ssl_us.eval.matrix --data bmode_subset.npz --out outputs/diagnostics/eval_boot

This is the harness bootstrap: `bmode` is already a (507, 456) image on the same
grid as `segmentation_map`, so crop sampling, label attachment, both guards, the
probe protocol, clustering, embeddings and every baseline can be exercised end to
end with no beamforming at all. When the K=16 store lands, only the loader changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data import geometry as geo
from ..data.crops import CropSpec, sample_corpus
from ..data.splits import inner_probe_split, outer_split, split_audit
from . import features as feat
from . import viz
from .cluster import cluster_report
from .guards import LabelLeakageError, filter_targets
from .probes import probe

# (target, level, kind) — regression-first per PLAN.md 4.1.
TARGETS = [
    ("contains_cyst", "crop", "logistic"),
    ("cyst_fraction", "crop", "ridge"),
    ("sos_mean_frame", "frame", "ridge"),
    ("sos_std_frame", "frame", "ridge"),
    ("sos_mean", "crop", "ridge"),
    ("phase_error", "frame", "ridge"),
    ("z_centre", "crop", "ridge"),
    ("x_centre", "crop", "ridge"),
    ("t0_group", "frame", "logistic"),
]

PHASE_ERROR_ORACLE_R2 = 0.585  # measured: R^2(phase_error ~ sos_mean + sos_std)


def load_bootstrap(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=True)
    sos = d["sos"]
    flat = sos.reshape(len(sos), -1)
    frame_id = d["frame_id"].astype(str)
    frame_labels = {
        "phase_error": d["phase_error"].astype(np.float32),
        "t0_group": np.array([geo.t0_group(t) for t in d["t0"]], dtype=np.int64),
        "sos_mean_frame": flat.mean(1).astype(np.float32),
        "sos_std_frame": flat.std(1).astype(np.float32),
    }
    return {
        "bmode": d["bmode"],
        "seg": d["seg"],
        "sos": sos,
        "frame_id": frame_id,
        "frame_labels": frame_labels,
        # Same labels keyed by frame id, for the WI-10 split-audit figure —
        # it needs to look a label up per split member, not per row.
        "frame_labels_by_id": {
            fid: {k: float(v[i]) for k, v in frame_labels.items()}
            for i, fid in enumerate(frame_id)
        },
    }


def log_compress(bmode: np.ndarray, dr_db: float = 60.0) -> np.ndarray:
    """Per-frame log compression to [-dr, 0] dB. The signal stage the baselines see."""
    out = np.empty_like(bmode, dtype=np.float32)
    for i, f in enumerate(bmode):
        db = 20 * np.log10(np.maximum(f, 1e-9))
        out[i] = np.clip(db - db.max(), -dr_db, 0.0)
    return out


def run(config: dict) -> dict:
    out_dir = Path(config["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = config["stage"]

    data = load_bootstrap(config["data"])
    images = log_compress(data["bmode"]) if config["log_compress"] else data["bmode"]

    # ── splits, guarded ──────────────────────────────────────────────────────
    outer = outer_split(data["frame_id"])
    inner = inner_probe_split(outer["train"], seed=config["seed"])
    audit = split_audit(data["frame_id"], outer, inner)
    print(f"split audit: {json.dumps(audit)}")
    # Emitted unconditionally, per PLAN.md WI-10's done-condition: the audit is
    # only useful if it is produced on every run rather than on request.
    audit_fig = viz.plot_split_audit(
        outer,
        out_dir / "split_audit.png",
        inner=inner,
        frame_labels=data.get("frame_labels_by_id"),
        label_keys=("sos_mean_frame", "phase_error"),
    )

    # ── crops ────────────────────────────────────────────────────────────────
    spec = CropSpec(
        size=config["crop_size"],
        z_min_mm=config["z_min_mm"],
        n_per_frame=config["crops_per_frame"],
        seed=config["seed"],
    )
    batch = sample_corpus(
        images, data["seg"], data["sos"], data["frame_id"], data["frame_labels"], spec
    )
    zc = batch.labels["z_centre"]
    print(
        f"crops: {len(batch)} from {len(images)} frames | size {spec.size} | "
        f"depth {zc.min():.1f}-{zc.max():.1f} mm | "
        f"cyst-positive {batch.labels['contains_cyst'].mean() * 100:.1f}%"
    )

    legal = filter_targets(stage, [t for t, _, _ in TARGETS])
    if len(legal) < len(TARGETS):
        blocked = [t for t, _, _ in TARGETS if t not in legal]
        print(f"stage {stage!r} BLOCKS targets (label leak): {blocked}")

    # ── features ─────────────────────────────────────────────────────────────
    encoders = config["encoders"]
    results, feats_cache = [], {}
    for name in encoders:
        fn = feat.EXTRACTORS[name]
        f = fn(batch.images)
        feats_cache[name] = f
        print(f"[{name}] features {f.shape}")
        for target, level, kind in TARGETS:
            if target not in legal:
                continue
            try:
                r = probe(
                    f, batch.labels[target], batch.frame_id, target,
                    stage=stage, level=level, kind=kind, seed=config["seed"],
                )
            except (ValueError, LabelLeakageError) as exc:
                print(f"  {target}: SKIPPED ({exc})")
                continue
            row = {"encoder": name, **r.row()}
            if target == "phase_error":
                row["oracle_r2"] = PHASE_ERROR_ORACLE_R2
                row["beats_oracle"] = bool(r.score > PHASE_ERROR_ORACLE_R2)
            results.append(row)
            print(
                f"  {target:<16} {r.metric:<18} {r.score:+.3f}  "
                f"ci95 [{r.score_lo:+.3f},{r.score_hi:+.3f}]  n_eff={r.n_eff}"
            )

    # ── clustering + embeddings on the reference encoder ─────────────────────
    ref = config["reference_encoder"]
    clusters = cluster_report(feats_cache[ref], batch.labels)
    print(f"clustering ({ref}): {json.dumps(clusters, indent=1)}")

    figs = [audit_fig]
    for method in config["projections"]:
        sub = np.arange(len(batch))
        if len(sub) > config["max_points"]:
            sub = np.random.default_rng(0).choice(len(batch), config["max_points"], replace=False)
        xy = viz.project(feats_cache[ref][sub], method)
        labs = {k: v[sub] for k, v in batch.labels.items()}
        figs.append(viz.plot_embedding(xy, labs, f"{ref} — {method.upper()}",
                                       out_dir / f"embed_{ref}_{method}.png"))
        resid, r2 = viz.regress_out(feats_cache[ref], batch.labels, ["z_centre", "x_centre"])
        print(f"geometry (z,x) explains {r2 * 100:.1f}% of {ref} feature variance")
        xy_r = viz.project(resid[sub], method)
        figs.append(
            viz.plot_embedding(
                xy_r, labs, f"{ref} — {method.upper()} (geometry regressed out)",
                out_dir / f"embed_{ref}_{method}_residual.png",
            )
        )

    payload = {
        "stage": stage, "crop_size": spec.size, "n_crops": len(batch),
        "n_frames": len(images), "split_audit": audit,
        "geometry_variance_explained": r2, "clusters": clusters,
        "results": results, "figures": [str(f) for f in figs],
        "blocked_targets": [t for t, _, _ in TARGETS if t not in legal],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_table(results, out_dir / "results.md")
    print(f"\nwrote {out_dir/'results.json'} and {out_dir/'results.md'}")
    return payload


def _write_table(results: list[dict], path: Path) -> None:
    if not results:
        path.write_text("_no results_\n")
        return
    cols = ["encoder", "target", "level", "metric", "score", "ci95", "n_eff", "n_feat"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def _cli() -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="npz from the bootstrap fetch")
    p.add_argument("--out", default="outputs/diagnostics/eval_boot")
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--crop-size", type=int, default=128, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=0.0, dest="z_min_mm")
    p.add_argument("--crops-per-frame", type=int, default=8, dest="crops_per_frame")
    p.add_argument("--encoders", nargs="+",
                   default=["random_resnet18", "pca_raw", "speckle_stats", "trivial"])
    p.add_argument("--reference-encoder", default="random_resnet18", dest="reference_encoder")
    p.add_argument("--projections", nargs="+", default=["pca"])
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--no-log-compress", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = vars(p.parse_args())
    a["log_compress"] = not a.pop("no_log_compress")
    return a


if __name__ == "__main__":
    run(_cli())
