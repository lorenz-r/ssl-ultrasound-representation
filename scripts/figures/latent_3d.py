"""3-D latent-space figures for the Stanford transfer corpus.

One point per **frame** (117 in-vivo rat acquisitions), projected to three
dimensions by PCA and by t-SNE, coloured by a tissue-composition axis. This is
the "what does the latent space look like" slide: the probe tables say how much
of an axis is decodable, and this says whether it is laid out in a way a reader
can see.

    python scripts/figures/latent_3d.py                       # default arm, both themes
    python scripts/figures/latent_3d.py --encoder ijepa
    python scripts/figures/latent_3d.py --variant lambda_matched --k 16

Needs the beamformed Stanford store and the trained checkpoints, so it runs on
the cluster (`.venv`, a GPU is used if present). Features are cached to
`--cache` so re-styling the figure does not re-extract them.

⚠ **On "body fat".** The obvious colour axis for this corpus does not exist in
it. Stanford's FSA acquisitions come from the Telichko et al. 2022 hepatic-
steatosis study, and that study *does* publish per-animal liver fat percentage,
ex vivo speed of sound and steatosis grade for 20 rats in `SoSDataAll.xlsx`.
**None of it can be attached to a frame.** figshare flattened the per-rat
directory tree that the dataset's own `mainLoader.m` walks, so every published
filename is `DATA_FSA_<date>_<time>.mat` — a timestamp and nothing else — and
the corpus sidecar can only record an *inferred* `session` (acquisitions
clustered by timestamp: 30 over 117 files). Verified directly against the
store's `frame_metadata.json`, whose keys are `session`, `source_file`, `kind`,
`date`, `t0_s`. Colouring these points by fat percentage would mean inventing
the acquisition→animal join, so this figure does not do it. See
docs/MEMORY.md §10.3 and `ssl_us.eval.stanford_targets`.

What it colours by instead, and why that is the honest substitute:

* **echogenicity** — mean B-mode brightness of the frame's crops. On real
  ultrasound this is the *imaging biomarker steatosis is read from*: fat
  infiltration raises hepatic echogenicity, which is exactly why a
  fat-percentage study images these animals in the first place. It is derived
  from the image rather than from a label file, so it is available for every
  frame. It is a **proxy for tissue composition, not a fat measurement**, and
  the figure says so on its face.
* **acquisition day** — the nuisance axis. If the latent organises by *when*
  the scan happened rather than by what is in it, the tissue reading is
  confounded, and that is worth seeing beside the first panel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.eval import slides
from ssl_us.eval.stanford_targets import stanford_frame_labels
from ssl_us.eval.store_features import sample_store_crops

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
OUT = REPO / "outputs" / "presentation"

NAMES = {
    "vicreg_A": "VICReg A (ResNet-18)",
    "vicreg_B": "VICReg B (ResNet-18)",
    "vicreg_vit_A": "VICReg A (ViT-S)",
    "ijepa": "I-JEPA (ViT-S)",
    "mae_norm": "MAE (norm-pixel)",
    "mae_raw": "MAE (raw-pixel)",
    "random_resnet18": "random ResNet-18",
    "random_vit": "random ViT-S",
}


def _load_matrix():
    spec = importlib.util.spec_from_file_location(
        "stanford_matrix", REPO / "scripts" / "eval" / "stanford_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── features ────────────────────────────────────────────────────────────────

def frame_latents(args) -> dict:
    """Per-frame latents for every encoder, plus the frame-level colour axes.

    Crops are pooled to frames by the mean. The alternative — one point per
    crop — makes a denser and prettier cloud, but the question asked here is
    about *frames*, and a per-crop plot would show mostly the within-frame
    depth gradient, which is the strongest axis in this latent space and would
    swamp the between-frame differences the figure is about.
    """
    matrix = _load_matrix()
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    crop_size = args.crop_size or arm.crop_size
    bands = matrix.depth_bands(args.k, arm.z_min_mm)
    z_min = bands[next(iter(bands))]

    ns = argparse.Namespace(**vars(args))
    ns.crop_size = crop_size
    scale = matrix.resolve_intensity_scale(ns)
    spec = matrix.band_spec(ns, z_min, scale)

    ids = list(open_store(args.store).frame_ids)
    batch = sample_store_crops(args.store, ids, spec, frame_label_fn=stanford_frame_labels)
    print(f"[latent] {len(batch)} crops from {len(ids)} frames", flush=True)

    encoders = matrix.build_encoders(ns, spec.in_channels, tag="latent-3d")
    order = [f for f in batch.frame_id]
    uniq = list(dict.fromkeys(order))
    index = {f: i for i, f in enumerate(uniq)}
    rows = np.array([index[f] for f in order])

    def pool(v: np.ndarray) -> np.ndarray:
        out = np.zeros((len(uniq),) + v.shape[1:], dtype=np.float64)
        cnt = np.zeros(len(uniq))
        for i, r in enumerate(rows):
            out[r] += v[i]
            cnt[r] += 1
        return out / cnt.reshape((-1,) + (1,) * (v.ndim - 1))

    feats = {}
    for name, fn in encoders.items():
        f = np.asarray(fn(batch.images), dtype=np.float64)
        feats[name] = pool(f)
        print(f"[latent]   {name}: {feats[name].shape}", flush=True)

    echo = pool(np.asarray(batch.labels["echogenicity"], dtype=np.float64))
    meta = json.loads((Path(args.store) / "frame_metadata.json").read_text())
    days = sorted({meta[f]["date"] for f in uniq})
    day_idx = np.array([days.index(meta[f]["date"]) for f in uniq], dtype=float)

    return {
        "frames": uniq,
        "features": feats,
        "echogenicity": echo,
        "day_index": day_idx,
        "days": days,
        "k": args.k,
        "variant": args.variant,
        "crop_size": crop_size,
        "crops_per_frame": spec.crops_per_frame,
        "intensity_scale_mode": args.intensity_scale,
    }


def project(feats: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    """Feature matrix -> (N, 3)."""
    x = feats - feats.mean(0, keepdims=True)
    if method == "pca":
        return PCA(n_components=3, random_state=seed).fit_transform(x)
    # PCA-reduce before t-SNE: at 512-d with 117 points the distances t-SNE
    # consumes are dominated by noise directions, and this is the standard
    # preconditioning step for exactly that reason.
    pre = PCA(n_components=min(30, x.shape[0] - 1, x.shape[1]),
              random_state=seed).fit_transform(x)
    return TSNE(n_components=3, perplexity=15, init="pca", random_state=seed,
                max_iter=1500).fit_transform(pre)


# ── figures ─────────────────────────────────────────────────────────────────

def _scatter3d(ax, xyz, values, theme, *, cmap, size=58):
    t = slides.tokens(theme)
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap=cmap,
                    s=size, depthshade=False, edgecolors=t["surface"], linewidths=1.2)
    ax.set_facecolor(t["surface"])
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.set_pane_color((0, 0, 0, 0))
        pane_axis.line.set_color(t["axis"])
        pane_axis._axinfo["grid"]["color"] = t["grid"]
        pane_axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(colors=t["muted"], labelsize=8)
    for setter in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
        setter([])
    ax.view_init(elev=20, azim=-58)
    # 3-D axes leave a wide empty margin by default; zoom fills the cell.
    ax.set_box_aspect((1, 1, 0.92), zoom=1.18)
    return sc


def fig_latent_3d(data: dict, theme: str, encoder: str) -> Path:
    """PCA and t-SNE, side by side, coloured by the tissue-composition axis."""
    t = slides.use(theme)
    cmap = slides.sequential_cmap(theme, for_markers=True)
    feats = data["features"][encoder]
    echo = data["echogenicity"]

    fig = plt.figure(figsize=(15.0, 7.0))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    for ax, method, label in zip(axes, ("pca", "tsne"), ("PCA", "t-SNE"), strict=True):
        xyz = project(feats, method)
        sc = _scatter3d(ax, xyz, echo, theme, cmap=cmap)
        # No axis labels: the axes carry no scale a reader can use, and on a
        # 3-D panel the z-label lands on top of the neighbouring colourbar.
        ax.set_title(f"{label} — 3 components", fontsize=12.5,
                     color=t["secondary"], pad=-2)

    cb = fig.colorbar(sc, ax=axes, fraction=0.020, pad=0.02, shrink=0.62)
    cb.set_label("frame echogenicity  (mean B-mode brightness, dB)",
                 fontsize=10.5, color=t["secondary"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=t["muted"], labelsize=9)

    slides.tight(fig, top=0.94, bottom=0.02, left=0.01, right=0.88, wspace=0.02)
    return slides.save(fig, OUT, f"01_latent_3d_{encoder}", theme)


def _probe_scores(data: dict, encoder: str) -> dict:
    """The published probe scores for this encoder, so the figure's caption and
    the results table cannot drift apart. Empty if the column is not committed."""
    path = REPO / "results" / "stanford" / f"{data['variant']}_k{data['k']}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    out = {}
    for row in doc["results"]:
        if row["encoder"] == encoder and row.get("band", "full") == "full":
            out[row["target"]] = row["score"]
    return out


def fig_latent_3d_nuisance(data: dict, theme: str, encoder: str) -> Path:
    """The same projection, coloured by tissue vs by acquisition day.

    The numbers are printed on the figure on purpose. Acquisition day is *not*
    visually obvious in this embedding, but it is decodable well above chance —
    so a reader invited to judge by eye alone would conclude the opposite of
    what the probe says. t-SNE is not a measurement; the caption carries the
    measurement.
    """
    t = slides.use(theme)
    cmap = slides.sequential_cmap(theme, for_markers=True)
    xyz = project(data["features"][encoder], "tsne")
    scores = _probe_scores(data, encoder)

    fig = plt.figure(figsize=(15.0, 7.2))
    # The probe score goes in each panel's own title rather than in one shared
    # line: acquisition day is NOT visually obvious in this embedding but is
    # decodable well above chance, so a reader judging by eye alone would
    # conclude the opposite of what the probe says. The number belongs beside
    # the picture it qualifies.
    echo_note = (f"  —  probe R² {scores['echogenicity']:+.2f}"
                 if "echogenicity" in scores else "")
    day_note = (f"  —  {scores['date_id']:.2f} balanced accuracy vs "
                f"{1 / len(data['days']):.2f} chance" if "date_id" in scores else "")
    panels = [(f"frame echogenicity, tissue{echo_note}", data["echogenicity"]),
              (f"acquisition day 1-{len(data['days'])}, nuisance{day_note}",
               data["day_index"] + 1)]
    for i, (label, values) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = _scatter3d(ax, xyz, values, theme, cmap=cmap)
        ax.set_title(label, fontsize=12, color=t["primary"], pad=-14)
        cb = fig.colorbar(sc, ax=ax, fraction=0.026, pad=0.02, shrink=0.58)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, colors=t["muted"], labelsize=9)

    slides.tight(fig, top=0.96, bottom=0.02, left=0.01, right=0.97, wspace=0.10)
    return slides.save(fig, OUT, f"02_latent_3d_nuisance_{encoder}", theme)


def fig_latent_3d_encoders(data: dict, theme: str, encoders: list[str]) -> Path:
    """Small multiples: the same frames, the same colouring, four latents.

    **PCA, deliberately, not t-SNE.** This panel exists to show collapse, and
    t-SNE cannot show it: it normalises the global variance structure away and
    will spread a near-1-D representation into a comfortable-looking blob. PCA
    preserves that structure, so a latent holding most of its variance in one
    direction is drawn as the line it actually is. Each panel is annotated with
    the share of variance its first component holds, computed from the very
    features plotted, so the picture and the number cannot disagree.
    """
    t = slides.use(theme)
    cmap = slides.sequential_cmap(theme, for_markers=True)
    echo = data["echogenicity"]
    present = [e for e in encoders if e in data["features"]]

    fig = plt.figure(figsize=(16.0, 6.0))
    for i, name in enumerate(present):
        ax = fig.add_subplot(1, len(present), i + 1, projection="3d")
        feats = data["features"][name]
        x = feats - feats.mean(0, keepdims=True)
        pca = PCA(n_components=3, random_state=0).fit(x)
        sc = _scatter3d(ax, pca.transform(x), echo, theme, cmap=cmap, size=34)
        share = pca.explained_variance_ratio_[0]
        ax.set_title(NAMES.get(name, name), fontsize=12, color=t["primary"], pad=-4)
        ax.text2D(0.5, -0.04, f"PC1 holds {share * 100:.0f} % of the variance",
                  transform=ax.transAxes, ha="center", fontsize=10.5,
                  color=t["primary"] if share > 0.6 else t["muted"])

    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.013, pad=0.015, shrink=0.55)
    cb.set_label("echogenicity", fontsize=10, color=t["secondary"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=t["muted"], labelsize=9)

    slides.tight(fig, top=0.93, bottom=0.12, left=0.01, right=0.92, wspace=0.03)
    return slides.save(fig, OUT, "02_latent_3d_encoders", theme)


def table_latent_variance(data: dict, encoders: list[str]) -> Path:
    """Variance share per component, for the encoders drawn as small multiples.

    Computed from the same frame-pooled features the plot uses, so the table
    and the picture cannot disagree. These are **frame-pooled** shares and are
    deliberately not the crop-level numbers in the RankMe column: pooling
    removes the within-frame depth gradient, which is the largest single axis
    in this latent space, so the two are different measurements of different
    things and quoting one for the other would overstate the collapse.
    """
    import csv
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    head = ["encoder", "feature dim", "PC1 var %", "PC1-3 var %", "components for 90% var"]
    rows = []
    for name in [e for e in encoders if e in data["features"]]:
        feats = np.asarray(data["features"][name])
        x = feats - feats.mean(0, keepdims=True)
        ratio = PCA(n_components=min(x.shape) - 1, random_state=0).fit(x).explained_variance_ratio_
        rows.append([NAMES.get(name, name), feats.shape[1],
                     f"{ratio[0] * 100:.1f}", f"{ratio[:3].sum() * 100:.1f}",
                     int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)])
    note = (f"frame-pooled Stanford latents · store wi16_{data['variant']}_k16.zarr · "
            f"K={data['k']} · {len(data['frames'])} frames · "
            "pooled over crops, so NOT comparable to the crop-level RankMe column")
    md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md += ["", f"_{note}_", ""]
    (tables / "latent_variance.md").write_text("\n".join(md))
    with (tables / "latent_variance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)
    return tables / "latent_variance.md"


def main() -> int:
    matrix = _load_matrix()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=matrix.VARIANTS, default="mm_matched")
    p.add_argument("--store", default=None)
    p.add_argument("--nvidia-store", default=matrix.NVIDIA_STORE, dest="nvidia_store")
    p.add_argument("--checkpoint-root", default=matrix.CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--intensity-scale", choices=["matched", "raw"], default="matched",
                   dest="intensity_scale")
    p.add_argument("--encoder", default="vicreg_A")
    p.add_argument("--panel-encoders", nargs="+", dest="panel_encoders",
                   default=["vicreg_A", "vicreg_vit_A", "ijepa", "vicreg_B"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external", action="store_true")
    p.add_argument("--external-model", default=None, dest="external_model")
    p.add_argument("--theme", choices=slides.THEMES, default="light")
    p.add_argument("--cache", default=None,
                   help="npz of pooled frame latents; written if absent, reused if present")
    args = p.parse_args()
    if args.store is None:
        args.store = matrix.STORE_TEMPLATE.format(variant=args.variant)

    cache = Path(args.cache) if args.cache else (
        REPO / "outputs" / "presentation" / f".latents_{args.variant}_k{args.k}.npz")
    if cache.exists():
        raw = np.load(cache, allow_pickle=True)
        data = {k: raw[k].item() if raw[k].ndim == 0 else raw[k] for k in raw.files}
        data["features"] = dict(data["features"])
        print(f"[latent] reusing {cache}")
    else:
        data = frame_latents(args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{k: np.array(v, dtype=object) if isinstance(v, (dict, list)) else v
                           for k, v in data.items()})
        print(f"[latent] cached {cache}")

    for theme in [args.theme]:
        print(f"  {fig_latent_3d(data, theme, args.encoder).relative_to(REPO)}")
        print(f"  {fig_latent_3d_nuisance(data, theme, args.encoder).relative_to(REPO)}")
        print(f"  {fig_latent_3d_encoders(data, theme, args.panel_encoders).relative_to(REPO)}")
    print(f"  {table_latent_variance(data, args.panel_encoders).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
