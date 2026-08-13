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
