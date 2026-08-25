"""Stanford qualitative half: encoder-vs-encoder embeddings on real tissue.

The Stanford counterpart of `scripts/eval/nvidia_qualitative.py`, and it exists for
the same reason that one does. `stanford_matrix.py` produces the transfer table but
draws its pictures for **one** encoder (`vicreg_A`) with **one** projection
(PCA). So the project's most reproducible finding — Policy A carries lateral
position while Policy B carries none — currently exists on real tissue only as
numbers: Policy B's `x_centre` R² stays under 0.026 across all ten matched
cells while Policy A reaches +0.664 at K=16 (docs/MEMORY.md §14.4). This script is
the figure.

    python scripts/eval/stanford_qualitative.py --k 4 --variant mm_matched
    python scripts/eval/stanford_qualitative.py --k 16 --variant mm_matched

Three things, all off one feature extraction:

1. **Encoder grid** (`viz.plot_encoder_grid`) — colour key down, encoder
   across, so the comparison is read along a row. Every encoder sees the
   identical crops, because the spec comes from `stanford_matrix.band_spec` and the
   origins are drawn once for the whole corpus.
2. **`cluster_report` for every encoder**, not just the reference.
3. **PCA, t-SNE and UMAP**, each raw and geometry-regressed, because depth
   dominates the latent and a raw projection is substantially a picture of the
   crop's z coordinate — on Stanford that is 12.7 % of Policy A's variance and
   **35.1 %** of Policy B's, against 35.9 % for an untrained ResNet (§14.7).

**The amplitude match applies here too, and is not optional.** These are the
same frozen NVIDIA-pretrained encoders reading a corpus whose crops are ~600x
brighter, so without `corpus_intensity_scale` the pictures show a units
mismatch rather than a representation (§14.3). It is resolved through
`stanford_matrix.resolve_intensity_scale`, so figure and table always agree.

⚠ **These figures illustrate the probe table; they cannot overturn it.** t-SNE
and UMAP neighbourhoods are not metric and their axes carry no units, so a blob
that looks unstructured is not evidence a target is undecodable. The probe R²
from the matching `stanford_matrix.py` column is printed on each panel so the
picture is always read next to the number it illustrates.
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
from ssl_us.eval import viz
from ssl_us.eval.cluster import cluster_report
from ssl_us.eval.stanford_targets import stanford_frame_labels
from ssl_us.eval.store_features import sample_store_crops

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_matrix():
    """Import `scripts/eval/stanford_matrix.py` by path — `scripts` is not a package.

    Reused rather than reimplemented: `band_spec`, `resolve_intensity_scale`
    and `depth_bands` are what make these the *same* crops the probe column
    used, and a second copy would drift from it silently.
    """
    spec = importlib.util.spec_from_file_location(
        "stanford_matrix", SCRIPT_DIR / "stanford_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Confound first, same rule as WI-13's: the depth panel is what the tissue
# panels have to be read against. `date_id` rather than `session_id` for the
# nuisance row — 10 acquisition days plot legibly where 30 sessions do not, and
# `session_id` is still scored in the cluster table.
DEFAULT_COLOUR_KEYS = ("z_centre", "x_centre", "echogenicity", "date_id")

# The four that make the argument: the two policies, the fair MAE baseline, and
# the untrained floor. `pca_raw`/`trivial` are in the cluster table but not the
# grid — seven columns of scatter is unreadable and they are not the contrast.
DEFAULT_GRID_ENCODERS = ("vicreg_A", "vicreg_B", "mae_raw", "random_resnet18")


def probe_notes(path: Path, band: str, encoders, colour_keys) -> dict:
    """`{encoder: {target: 'R²=+0.52'}}` from a `stanford_matrix.py` results.json.

    Filtered to one `band`, unlike WI-13's: a Stanford column holds both the
    full and the restricted depth band, and printing the wrong band's R² on a
    panel would caption the figure with a different experiment's number.
    """
    if not path.exists():
        print(f"[qual] no probe results at {path} — panels will carry no R²")
        return {}
    notes: dict[str, dict[str, str]] = {}
    for r in json.loads(path.read_text())["results"]:
        if r["band"] == band and r["encoder"] in encoders and r["target"] in colour_keys:
            metric = "R²" if r["metric"] == "r2" else r["metric"]
            notes.setdefault(r["encoder"], {})[r["target"]] = f"{metric}={r['score']:+.3f}"
    return notes


def main() -> int:
    matrix = _load_matrix()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=matrix.VARIANTS, default="mm_matched")
    p.add_argument("--store", default=None)
    p.add_argument("--checkpoint-root", default=matrix.CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--band",
        default=None,
        help="which depth band to draw; defaults to the headline band for this K "
        "(full at K<=4, restricted at K>=8)",
    )
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument("--projections", nargs="+", default=["pca", "tsne", "umap"])
    p.add_argument("--colour-by", nargs="+", default=list(DEFAULT_COLOUR_KEYS),
                   dest="colour_by")
    p.add_argument("--encoders", nargs="+", default=list(DEFAULT_GRID_ENCODERS))
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--no-residual", action="store_true", dest="no_residual")
    p.add_argument("--probe-results", default=None, dest="probe_results")
    p.add_argument("--feature-cache", default=None, dest="feature_cache")
    p.add_argument("--intensity-scale", choices=["matched", "raw"], default="matched",
                   dest="intensity_scale")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.store is None:
        args.store = matrix.STORE_TEMPLATE.format(variant=args.variant)
    # Geometry from K, exactly as the training loader and the probe column do.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    bands = matrix.depth_bands(args.k, arm.z_min_mm)
    band = args.band or next(iter(bands))
    if band not in bands:
        raise SystemExit(f"band {band!r} not available at K={args.k}; have {list(bands)}")
    out_dir = Path(args.out or f"outputs/stanford/{args.variant}_k{args.k}_qual")
    out_dir.mkdir(parents=True, exist_ok=True)

    store = open_store(args.store)
    frame_ids = list(store.frame_ids)
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]

    scale = matrix.resolve_intensity_scale(args)
    spec = matrix.band_spec(args, bands[band], scale)
    print(
        f"[qual] {args.variant} K={spec.k} crop={spec.crop_size} band={band} "
        f"frames={len(frame_ids)} crops/frame={spec.crops_per_frame}",
        flush=True,
    )

    encoders = matrix.build_encoders(args, spec.in_channels, tag="stanford-qual")
    cache = Path(args.feature_cache) if args.feature_cache else None
    if cache is not None and cache.exists():
        print(f"[qual] features from cache {cache}")
        blob = np.load(cache, allow_pickle=True)
        feats = {k[2:]: blob[k] for k in blob.files if k.startswith("f_")}
        labels = {k[2:]: blob[k] for k in blob.files if k.startswith("l_")}
        n = len(next(iter(labels.values())))
    else:
        t0 = time.time()
        batch = sample_store_crops(
            args.store, frame_ids, spec, frame_label_fn=stanford_frame_labels
        )
        feats = {name: fn(batch.images) for name, fn in encoders.items()}
        batch.images = np.empty((0,), dtype=np.float32)
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

    suffix = "" if args.intensity_scale == "matched" else "_rawscale"
    notes_path = Path(
        args.probe_results
        or f"outputs/stanford/{args.variant}_k{args.k}{suffix}/results.json"
    )
    notes = probe_notes(notes_path, band, grid, keys)

    variants = [("", {name: feats[name] for name in grid})]
    geometry_r2 = {}
    if not args.no_residual:
        resid = {}
        for name in grid:
            resid[name], geometry_r2[name] = viz.regress_out(
                feats[name], labels, ["z_centre", "x_centre"]
            )
        print("[qual] geometry (z,x) explains " + ", ".join(
            f"{k} {v * 100:.1f}%" for k, v in geometry_r2.items()))
        variants.append(("_residual", resid))

    figures = []
    for fig_suffix, source in variants:
        for method in args.projections:
            t0 = time.time()
            xys = {name: viz.project(source[name][sub], method, seed=args.seed)
                   for name in grid}
            title = (
                f"Stanford {args.variant} — K={args.k} {band} band — {method.upper()}"
                + (" — geometry (z,x) regressed out" if fig_suffix else "")
            )
            path = viz.plot_encoder_grid(
                xys, sub_labels, keys, title,
                out_dir / f"grid_{method}{fig_suffix}.png",
                cell_notes=notes,
            )
            figures.append(str(path))
            print(f"[qual] {path}  ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "wi": 16, "kind": "qualitative", "corpus": "stanford_fsa",
        "variant": args.variant, "k": args.k, "band": band,
        "crop_size": spec.crop_size, "crops_per_frame": spec.crops_per_frame,
        "intensity_scale_mode": args.intensity_scale, "intensity_scale": scale,
        "n_frames": len(frame_ids), "n_crops": int(n), "n_plotted": int(len(sub)),
        "encoders": list(feats), "grid_encoders": grid, "colour_keys": keys,
        "projections": args.projections, "probe_results": str(notes_path),
        "geometry_variance_explained": geometry_r2,
        "clusters": clusters, "figures": figures,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_cluster_table(clusters, args, band, out_dir / "clusters.md")
    print(f"\n[qual] wrote {out_dir / 'results.json'}, {out_dir / 'clusters.md'} "
          f"and {len(figures)} figures")
    return 0


def _write_cluster_table(clusters: dict, args, band: str, path: Path) -> None:
    """k-means silhouette and label agreement, every encoder in one table."""
    refs = sorted({
        key[4:] for c in clusters.values() for e in c["kmeans"].values()
        for key in e if key.startswith("ari_")
    })
    lines = [
        f"# WI-16 — unsupervised clustering on Stanford ({args.variant}, K={args.k}, "
        f"{band} band)",
        "",
        "k-means over standardised features of **real in-vivo rat tissue**, using",
        "encoders pre-trained only on NVIDIA simulation. `silhouette` is cluster",
        "separation; `ari_*` is agreement with a label the clustering never saw.",
        "",
        "`ari_depth_quartile` is the confound — a high value means the encoder",
        "clusters by *where* the crop came from, not by tissue. `ari_session` is the",
        "transfer-specific one: clustering that tracks acquisition session is reading",
        "the machine and the animal's session, not anatomy.",
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
