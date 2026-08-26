"""NVIDIA qualitative half: encoder-vs-encoder embeddings and clustering.

`nvidia_matrix.py` answers "how much of label *y* is linearly decodable from
encoder *e*" for every (e, y) pair, but it draws its pictures for **one**
encoder (`--reference-encoder`, i.e. `vicreg_A`) and with **one** projection
(PCA). So the project's most reproducible finding — Policy A carries lateral
position at R² +0.52 while Policy B carries exactly none, −0.001 — exists only
as two numbers in a table. This script produces the figure.

    python scripts/eval/nvidia_qualitative.py --k 4 --split validation \
        --out outputs/nvidia/k4_validation_qualitative

Three things, all off the same cached features:

1. **Encoder grid** (`viz.plot_encoder_grid`) — colour key down, encoder
   across, so the comparison is read along a row. Every encoder sees the
   identical crops: origins are sampled once for the whole corpus from the same
   seed as the probe column, so a difference between two panels is a difference
   in weights and nothing else.
2. **`cluster_report` for every encoder**, not just the reference — the
   quantitative counterpart to the grid, and the only unsupervised number in the
   project that currently exists for one encoder alone.
3. **PCA, t-SNE and UMAP**, each in a raw and a geometry-regressed variant,
   because depth dominates the latent and a projection of the raw features is
   substantially a picture of the crop's z coordinate.

⚠ **These figures illustrate the probe table; they cannot overturn it.** t-SNE
and UMAP neighbourhoods are not metric and their axes carry no units, so a blob
that looks unstructured is not evidence that a target is undecodable — PCA is
linear and the probes are ridge, and both already disagree with the eye on
`cyst_fraction`. The probe R² is therefore printed on each panel, from the
matching `nvidia_matrix.py` column, so the picture is always read next to the
number it illustrates.

Feature extraction is the expensive step and is identical to the probe column's,
so `--feature-cache` stores it and later figure iterations are free.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.data.label_maps import open_label_maps
from ssl_us.data.splits import outer_split
from ssl_us.eval import viz
from ssl_us.eval.cluster import cluster_report
from ssl_us.eval.store_features import StoreCropSpec, extract_features

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_matrix():
    """Import `scripts/eval/nvidia_matrix.py` by path — `scripts` is not a package.

    Reused rather than reimplemented on purpose: the target lists and the
    crop-subset rules are what make these features the *same* features the
    probe column used. (`build_encoders` and `extract_features` themselves have
    since moved into `ssl_us.eval` for the same reason, and are imported
    directly.)
    """
    spec = importlib.util.spec_from_file_location(
        "nvidia_matrix", SCRIPT_DIR / "nvidia_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Crops per frame is a property of the split, not a knob: these are the values
# every committed column used, and the features only match the probe table if
# they match here too.
CROPS_PER_FRAME = {"validation": 32, "train": 8}

# Confound first, same rule as `viz.DEFAULT_COLOUR_ORDER` — the depth panel is
# what the tissue panels have to be read against.
DEFAULT_COLOUR_KEYS = ("z_centre", "x_centre", "sos_mean_frame", "cyst_fraction")

# The four that make the argument: the two policies, the fair MAE baseline, and
# the untrained floor. `pca_raw`/`trivial` are in the cluster table but not the
# grid — seven columns of scatter is unreadable and they are not the contrast.
DEFAULT_GRID_ENCODERS = ("vicreg_A", "vicreg_B", "mae_raw", "random_resnet18")


def probe_notes(path: Path, encoders, colour_keys) -> dict:
    """`{encoder: {target: 'R²=+0.52'}}` from a `nvidia_matrix.py` results.json."""
    if not path.exists():
        print(f"[qual] no probe results at {path} — panels will carry no R²")
        return {}
    rows = json.loads(path.read_text())["results"]
    notes: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["encoder"] in encoders and r["target"] in colour_keys:
            metric = "R²" if r["metric"] == "r2" else r["metric"]
            notes.setdefault(r["encoder"], {})[r["target"]] = f"{metric}={r['score']:+.3f}"
    return notes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    matrix = _load_matrix()
    p.add_argument("--store", default=matrix.DEFAULT_STORE)
    p.add_argument("--checkpoint-root", default=matrix.CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--label-maps", default=str(matrix.DEFAULT_LABEL_MAPS), dest="label_maps")
    p.add_argument("--out", default="outputs/nvidia/k4_validation_qualitative")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--crops-per-frame", type=int, default=None, dest="crops_per_frame")
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=None, dest="z_min_mm")
    p.add_argument("--frame-chunk", type=int, default=100, dest="frame_chunk")
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument("--projections", nargs="+", default=["pca", "tsne", "umap"])
    p.add_argument("--colour-by", nargs="+", default=list(DEFAULT_COLOUR_KEYS), dest="colour_by")
    p.add_argument("--encoders", nargs="+", default=list(DEFAULT_GRID_ENCODERS))
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--no-residual", action="store_true", dest="no_residual")
    p.add_argument("--probe-results", default=None, dest="probe_results")
    p.add_argument("--feature-cache", default=None, dest="feature_cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Geometry from K, exactly as the training loader and the probe column do.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    if args.z_min_mm is None:
        args.z_min_mm = arm.z_min_mm
    if args.crops_per_frame is None:
        args.crops_per_frame = CROPS_PER_FRAME[args.split]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(args.store)
    frame_ids = outer_split(np.array(store.frame_ids))[args.split]
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]

    spec = StoreCropSpec(
        k=args.k, crop_size=args.crop_size, z_min_mm=args.z_min_mm,
        crops_per_frame=args.crops_per_frame, seed=args.seed,
    )
    print(
        f"[qual] K={spec.k} crop={spec.crop_size} split={args.split} "
        f"frames={len(frame_ids)} crops/frame={spec.crops_per_frame}",
        flush=True,
    )

    label_maps = None
    if args.label_maps:
        try:
            label_maps = open_label_maps(args.label_maps)
            print(f"[qual] label maps: {label_maps.root} ({len(label_maps)} frames)")
        except (FileNotFoundError, ValueError) as exc:
            print(f"[qual] no usable label maps ({exc})")

    encoders = matrix.build_encoders(args, spec.in_channels)
    cache = Path(args.feature_cache) if args.feature_cache else None
    if cache is not None and cache.exists():
        print(f"[qual] features from cache {cache}")
        blob = np.load(cache, allow_pickle=True)
        feats = {k[2:]: blob[k] for k in blob.files if k.startswith("f_")}
        labels = {k[2:]: blob[k] for k in blob.files if k.startswith("l_")}
        n = len(next(iter(labels.values())))
    else:
        t0 = time.time()
        batch, feats = extract_features(
            args.store, frame_ids, spec, encoders,
            label_maps=label_maps, frame_chunk=args.frame_chunk, tag="qual",
        )
        labels, n = batch.labels, len(batch)
        print(f"[qual] features for {len(feats)} encoders, {n} crops, {time.time() - t0:.0f}s")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                **{f"f_{k}": v for k, v in feats.items()},
                **{f"l_{k}": np.asarray(v) for k, v in labels.items()},
            )
            print(f"[qual] cached features -> {cache}")

    # ── clustering, for every encoder ────────────────────────────────────────
    clusters = {}
    for name, f in feats.items():
        t0 = time.time()
        clusters[name] = cluster_report(f, labels, max_points=args.max_points, seed=args.seed)
        km = clusters[name]["kmeans"]
        best = max(km, key=lambda k: km[k].get("ari_depth_quartile", 0.0))
        print(
            f"[qual] cluster {name:<16} best-ARI(depth) {best} "
            f"{km[best].get('ari_depth_quartile', float('nan')):+.3f} "
            f"sil {km[best]['silhouette']:+.3f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── the grid figures ─────────────────────────────────────────────────────
    grid = [e for e in args.encoders if e in feats]
    missing = [e for e in args.encoders if e not in feats]
    if missing:
        print(f"[qual] not available for K={args.k}, dropped from the grid: {missing}")
    keys = [k for k in args.colour_by if k in labels]

    # One subsample, shared by every encoder and every method, for the same
    # reason the crops are shared: so the only thing that varies is the weights.
    rng = np.random.default_rng(args.seed)
    sub = np.arange(n) if n <= args.max_points else rng.choice(n, args.max_points, replace=False)
    sub_labels = {k: np.asarray(labels[k])[sub] for k in keys}

    notes_path = Path(
        args.probe_results
        or f"outputs/nvidia/k{args.k}_{args.split}_anatomy/results.json"
    )
    notes = probe_notes(notes_path, grid, keys)

    variants = [("", {name: feats[name] for name in grid})]
    if not args.no_residual:
        resid, r2 = {}, {}
        for name in grid:
            r, v = viz.regress_out(feats[name], labels, ["z_centre", "x_centre"])
            resid[name], r2[name] = r, v
        print("[qual] geometry (z,x) explains " + ", ".join(
            f"{k} {v * 100:.1f}%" for k, v in r2.items()))
        variants.append(("_residual", resid))

    figures = []
    for suffix, source in variants:
        for method in args.projections:
            t0 = time.time()
            xys = {name: viz.project(source[name][sub], method, seed=args.seed)
                   for name in grid}
            title = (
                f"K={args.k} {args.split} — {method.upper()}"
                + (" — geometry (z,x) regressed out" if suffix else "")
            )
            path = viz.plot_encoder_grid(
                xys, sub_labels, keys, title,
                out_dir / f"grid_{method}{suffix}.png",
                cell_notes=notes,
            )
            figures.append(str(path))
            print(f"[qual] {path}  ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "wi": 13, "kind": "qualitative", "k": args.k, "split": args.split,
        "crop_size": spec.crop_size, "crops_per_frame": spec.crops_per_frame,
        "n_crops": int(n), "n_plotted": int(len(sub)),
        "encoders": list(feats), "grid_encoders": grid, "colour_keys": keys,
        "projections": args.projections, "probe_results": str(notes_path),
        "clusters": clusters, "figures": figures,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_cluster_table(clusters, out_dir / "clusters.md")
    print(f"\n[qual] wrote {out_dir/'results.json'}, {out_dir/'clusters.md'} "
          f"and {len(figures)} figures")
    return 0


def _write_cluster_table(clusters: dict, path: Path) -> None:
    """k-means silhouette and label agreement, every encoder in one table."""
    refs = sorted({
        key[4:] for c in clusters.values() for e in c["kmeans"].values()
        for key in e if key.startswith("ari_")
    })
    lines = [
        "# WI-13 — unsupervised clustering, per encoder",
        "",
        "k-means over standardised features. `silhouette` is cluster separation;",
        "`ari_*` is agreement with a label the clustering never saw. Depth quartile",
        "is the confound — a high ARI there means the encoder clusters by *where*",
        "the crop came from, which is not a representation of tissue.",
        "",
    ]
    for k in ("k=2", "k=4", "k=8", "k=16"):
        if not any(k in c["kmeans"] for c in clusters.values()):
            continue
        lines += [f"## {k}", "",
                  "| encoder | silhouette | " + " | ".join(f"ari_{r}" for r in refs) + " |",
                  "|---|---|" + "---|" * len(refs)]
        for name, c in clusters.items():
            e = c["kmeans"].get(k)
            if e is None:
                continue
            cells = " | ".join(f"{e.get(f'ari_{r}', float('nan')):+.3f}" for r in refs)
            lines.append(f"| {name} | {e['silhouette']:+.3f} | {cells} |")
        lines.append("")
    lines += ["## DBSCAN", "",
              "| encoder | eps | clusters | noise fraction |", "|---|---|---|---|"]
    for name, c in clusters.items():
        d = c["dbscan"]
        lines.append(
            f"| {name} | {d['eps']:.3f} | {d['n_clusters']} | {d['noise_fraction']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
