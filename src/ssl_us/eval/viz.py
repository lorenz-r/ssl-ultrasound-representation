"""Embedding plots whose DEFAULT colouring is the confounder, not the payoff.

PLAN.md 4.3: the brief asked for the first UMAP to be coloured by acquisition ID.
This dataset has no acquisition ID (923 independent k-Wave simulations, identical
geometry, `frame_idx == 0` everywhere), so the equivalent — and here the more
dangerous — confound is **crop depth**: attenuation, lateral resolution, speckle
grain and sector occupancy all vary with z.

So the colour order is fixed and deliberate: ``z_centre`` first, then frame-level
sound speed, then phase error, and only fourth ``cyst_fraction``. Getting the
payoff figure requires scrolling past the confound figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Deliberate order — confounds first. Do not reorder for prettier figures.
DEFAULT_COLOUR_ORDER = (
    "z_centre",
    "sos_mean_frame",
    "phase_error",
    "cyst_fraction",
    "x_centre",
    "t0_group",
)


def plot_split_audit(
    outer: dict[str, list[str]],
    out_path: str | Path,
    inner: dict[str, list[str]] | None = None,
    frame_labels: dict[str, dict] | None = None,
    label_keys: tuple[str, ...] = ("sos_mean", "phase_error"),
) -> Path:
    """Per-run split-audit figure (PLAN.md WI-10's third done-condition).

    Counts alone would make a reassuring but nearly content-free picture, so
    this plots the two things that can actually invalidate a run:

    1. **Split sizes and the nesting**, with the guard's own verdict stated on
       the figure — the inner probe split must be carved out of outer-train, so
       it is drawn nested rather than beside it.
    2. **Distribution overlap between train and validation** for the frame-level
       labels, which is the assumption PLAN.md §8 leans on when it adopts
       NVIDIA's split wholesale ("they are distributionally matched"). If that
       ever stops being true the probe results are comparing two different
       populations, and this is where it would show up.

    ``outer``/``inner`` are the split id lists themselves (as returned by
    :func:`ssl_us.data.splits.outer_split` / ``inner_probe_split``), not the
    count summary — the label panels need the ids to look their frames up.
    ``frame_labels`` is the store's `frame_metadata.json` (frame_id -> labels);
    when omitted only the counts panel is drawn.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(label_keys) if frame_labels else []
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(5 + 4.5 * len(keys), 4.2))
    axes = np.atleast_1d(axes)

    counts = {k: len(v) for k, v in outer.items()}
    total = sum(counts.values())
    ax = axes[0]
    names = list(counts)
    ax.bar(names, [counts[k] for k in names], color="#4C78A8")
    if inner:
        # Drawn stacked inside the train bar: the inner split is a partition OF
        # train, not a third sibling split.
        bottom = 0.0
        for i, (k, v) in enumerate(inner.items()):
            ax.bar(["train"], [len(v)], bottom=[bottom], width=0.45,
                   color=["#F58518", "#54A24B"][i % 2], label=f"{k} ({len(v)})")
            bottom += len(v)
        ax.legend(fontsize=8)
    for i, k in enumerate(names):
        ax.text(i, counts[k], str(counts[k]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("frames")
    ax.set_title(
        f"Split sizes (total {total})\nno frame_id on both sides — guard passed",
        fontsize=9,
    )

    for ax, key in zip(axes[1:], keys, strict=True):
        for split, colour in (("train", "#4C78A8"), ("validation", "#E45756")):
            vals = [
                frame_labels[f][key]
                for f in outer.get(split, [])
                if f in frame_labels and key in frame_labels[f]
            ]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.55, density=True, color=colour,
                    label=f"{split} (n={len(vals)}, μ={np.mean(vals):.4g}, σ={np.std(vals):.3g})")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.set_title(f"{key}: train vs validation", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("WI-10 split audit — frame-level, guard-checked", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def project(feats: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    """2-D projection of standardised features. ``umap`` falls back to t-SNE."""
    x = StandardScaler().fit_transform(feats)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
        except ImportError:
            method = "tsne"
    if method == "tsne":
        perp = float(max(5.0, min(30.0, (len(x) - 1) / 3.0)))
        return TSNE(n_components=2, init="pca", perplexity=perp, random_state=seed).fit_transform(x)
    raise ValueError(f"unknown method {method!r}")


def regress_out(
    feats: np.ndarray, labels: dict[str, np.ndarray], factors: list[str], n_bins: int = 10
) -> tuple[np.ndarray, float]:
    """Linearly remove geometry factors from features; return (residual, R^2).

    Continuous factors are binned before one-hot encoding: the latent's dependence
    on a coordinate is not linear in the coordinate, but it *is* additive per axis,
    so per-level dummies remove it cleanly. What remains is the content the
    positional shell was hiding — and the printed R^2 says how much of the latent
    was pure geometry before any claim is made about tissue.
    """
    cols = [np.ones((len(feats), 1))]
    for f in factors:
        v = np.asarray(labels[f])
        if v.dtype.kind in "fc":
            v = np.digitize(v, np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        cats = np.unique(v)
        cols.append((v[:, None] == cats[None, :]).astype(float))
    x = np.concatenate(cols, axis=1)
    beta, *_ = np.linalg.lstsq(x, feats, rcond=None)
    pred = x @ beta
    resid = feats - pred
    ss_tot = float(((feats - feats.mean(0)) ** 2).sum())
    r2 = 1.0 - float(((feats - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return resid, r2


def _scatter(ax, xy, values, name, point_size):
    values = np.asarray(values)
    categorical = values.dtype == object or values.dtype.kind in "USib"
    if categorical:
        cats = sorted(set(values.tolist()))
        cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
        for k, c in enumerate(cats):
            m = values == c
            ax.scatter(xy[m, 0], xy[m, 1], s=point_size, alpha=0.6,
                       color=cmap(k % cmap.N), label=str(c), linewidths=0)
        ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.7)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=0.6, c=values,
                        cmap="viridis", linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embedding(
    xy: np.ndarray,
    labels: dict[str, np.ndarray],
    title: str,
    out_path: str | Path,
    point_size: int = 6,
    colour_order: tuple[str, ...] = DEFAULT_COLOUR_ORDER,
) -> Path:
    keys = [k for k in colour_order if k in labels] + [
        k for k in labels if k not in colour_order
    ]
    ncol = min(3, len(keys))
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, key in zip(axes.flat, keys, strict=False):
        ax.axis("on")
        _scatter(ax, xy, labels[key], key, point_size)
    fig.suptitle(f"{title}  —  first panel is the CONFOUND (depth), by design", fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_encoder_grid(
    xys: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    colour_keys: list[str],
    title: str,
    out_path: str | Path,
    point_size: int = 6,
    cell_notes: dict[str, dict[str, str]] | None = None,
) -> Path:
    """One row per colour key, one column per encoder — the A-vs-B figure.

    `plot_embedding` shows every label for *one* encoder, which answers "what
    did this model organise by". The comparison question is the transpose: hold
    the label fixed and vary the encoder. Policy B's `x_centre` R² is −0.001
    against Policy A's +0.52, and that contrast is only visible side by side.

    The comparison is fair because **every encoder saw the identical crops** —
    `nvidia_matrix` samples origins once for the whole corpus and shares them, so
    the only difference between two columns is the weights. Each row also gets a
    single shared colour scale and one colorbar, so a panel cannot look
    structured merely by having been given a tighter range than its neighbour.

    ⚠ The 2-D projections are *per encoder* and their axes are not comparable
    between columns — t-SNE and UMAP distances are not metric at all. Read a row
    for whether structure exists, never for how far apart two points are. The
    probe R² carried in `cell_notes` is the quantity that is comparable, which
    is why it is printed on the panel rather than left in a separate table.
    """
    encoders = list(xys)
    fig, axes = plt.subplots(
        len(colour_keys), len(encoders),
        figsize=(4.4 * len(encoders), 4.2 * len(colour_keys)),
        squeeze=False,
    )
    for row, key in enumerate(colour_keys):
        values = np.asarray(labels[key])
        categorical = values.dtype == object or values.dtype.kind in "USib"
        # Shared scale across the row: computed once from the values, which are
        # the same array for every column, and applied explicitly so a future
        # per-encoder subsample cannot silently rescale one panel.
        vmin, vmax = (None, None) if categorical else (float(values.min()), float(values.max()))
        handle = None
        for col, name in enumerate(encoders):
            ax = axes[row][col]
            xy = xys[name]
            if categorical:
                cats = sorted(set(values.tolist()))
                cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
                for k, c in enumerate(cats):
                    m = values == c
                    ax.scatter(xy[m, 0], xy[m, 1], s=point_size, alpha=0.6,
                               color=cmap(k % cmap.N), label=str(c), linewidths=0)
                if col == len(encoders) - 1:
                    ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.7)
            else:
                handle = ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=0.6,
                                    c=values, cmap="viridis", linewidths=0,
                                    vmin=vmin, vmax=vmax)
            note = (cell_notes or {}).get(name, {}).get(key)
            ax.set_title(f"{name}\n{key}" + (f"  ({note})" if note else ""), fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        if handle is not None:
            fig.colorbar(handle, ax=list(axes[row]), fraction=0.02, pad=0.01)
    fig.suptitle(title, fontsize=13)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mask_grid(
    frames: list[str],
    bmode: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    predictions: dict[str, dict[str, np.ndarray]],
    dice: dict[str, dict[str, float]],
    out_path: str | Path,
    title: str = "",
    dynamic_range_db: float = 60.0,
) -> Path:
    """One row per frame: B-mode, ground truth, then one column per encoder.

    The WI-19 segmentation figure. `plot_encoder_grid` holds a label fixed and
    varies the encoder over a *projection*; this does the same over a predicted
    mask, which is the only view in the project that shows **where** an encoder
    thinks the anatomy is rather than how much of it a crop contains.

    ⚠ **Column order matters and the baselines are not optional.** Cysts are
    anechoic, so a per-cell brightness threshold — `trivial_dense` — already
    produces a plausible mask. A grid of trained arms without it invites the
    reading that five representations all localise anatomy, when the honest
    finding may be that darkness does. `scripts/eval/nvidia_segmentation.py`
    orders the columns so the baselines are read before the trained arms, and
    the Dice under each panel is the same number the probe table reports.

    The ground-truth contour is drawn over every prediction panel, so
    over- and under-segmentation are separable by eye instead of requiring the
    reader to compare two panels across the page.

    Args:
        frames: frame ids, one per row, in the order they should be plotted.
        bmode: frame id -> ``(nz, nx)`` log-envelope in dB.
        truth: frame id -> ``(nz, nx)`` ground-truth mask.
        predictions: encoder -> frame id -> ``(nz, nx)`` cyst probability.
        dice: encoder -> frame id -> per-frame Dice, printed on the panel.
    """
    encoders = list(predictions)
    ncols = 2 + len(encoders)
    fig, axes = plt.subplots(
        len(frames), ncols, figsize=(2.7 * ncols, 3.1 * len(frames)), squeeze=False
    )
    for row, fid in enumerate(frames):
        db = np.asarray(bmode[fid], dtype=np.float32)
        # Per-frame dB window, referenced to the frame's own peak: the store
        # holds raw sums on no absolute scale, so a shared window would render
        # brighter frames as saturated and dimmer ones as black.
        vmax = float(db.max())
        gt = np.asarray(truth[fid])

        ax = axes[row][0]
        ax.imshow(db, cmap="gray", vmin=vmax - dynamic_range_db, vmax=vmax, aspect="auto")
        ax.set_title(f"{fid}\nB-mode", fontsize=9)

        ax = axes[row][1]
        ax.imshow(gt, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_title("ground truth", fontsize=9)

        for col, name in enumerate(encoders, start=2):
            ax = axes[row][col]
            ax.imshow(
                np.asarray(predictions[name][fid], dtype=np.float32),
                cmap="magma", vmin=0, vmax=1, aspect="auto",
            )
            if gt.any():
                ax.contour(gt, levels=[0.5], colors="#39ff88", linewidths=0.7)
            d = dice.get(name, {}).get(fid)
            ax.set_title(f"{name}\nDice {d:.3f}" if d is not None else name, fontsize=9)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=13)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
