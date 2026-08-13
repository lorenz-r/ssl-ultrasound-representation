"""Unsupervised clustering against the label axes (PLAN.md 4.4).

k-means over a sweep of k, DBSCAN with eps taken from the k-NN distance elbow.
Scored with ARI/NMI against `contains_cyst` and `t0_group`, plus silhouette.

The depth axis is included on purpose: if clusters align with `z_centre` far better
than with any tissue label, the latent has organised by geometry and the clustering
result is about depth, not tissue.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _knn_elbow_eps(x: np.ndarray, k: int = 8) -> float:
    """DBSCAN eps from the knee of the sorted k-NN distance curve."""
    nn = NearestNeighbors(n_neighbors=min(k, len(x) - 1)).fit(x)
    d = np.sort(nn.kneighbors(x)[0][:, -1])
    # knee = point of maximum distance from the chord joining the curve's endpoints
    i = np.arange(len(d))
    chord = d[0] + (d[-1] - d[0]) * i / max(len(d) - 1, 1)
    return float(d[int(np.argmax(d - chord))])


def cluster_report(
    feats: np.ndarray,
    labels: dict[str, np.ndarray],
    ks: tuple[int, ...] = (2, 4, 8, 16),
    max_points: int = 4000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(feats), max_points, replace=False)
        if len(feats) > max_points
        else np.arange(len(feats))
    )
    x = StandardScaler().fit_transform(feats[idx])

    ref = {}
    if "contains_cyst" in labels:
        ref["cyst"] = labels["contains_cyst"][idx].astype(int)
    if "t0_group" in labels:
        ref["t0_group"] = labels["t0_group"][idx].astype(int)
    if "z_centre" in labels:  # the confound, quantised — included deliberately
        z = labels["z_centre"][idx]
        ref["depth_quartile"] = np.digitize(z, np.quantile(z, [0.25, 0.5, 0.75]))

    out: dict = {"n_points": int(len(idx)), "kmeans": {}, "dbscan": {}}
    for k in ks:
        if k >= len(idx):
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        sil = float(silhouette_score(x, lab)) if len(set(lab)) > 1 else float("nan")
        entry = {"silhouette": sil}
        for name, y in ref.items():
            entry[f"ari_{name}"] = float(adjusted_rand_score(y, lab))
            entry[f"nmi_{name}"] = float(normalized_mutual_info_score(y, lab))
        out["kmeans"][f"k={k}"] = entry

    eps = _knn_elbow_eps(x)
    lab = DBSCAN(eps=eps, min_samples=8).fit_predict(x)
    n_clusters = int(len(set(lab)) - (1 if -1 in lab else 0))
    entry = {
        "eps": round(eps, 4),
        "n_clusters": n_clusters,
        "noise_fraction": float((lab == -1).mean()),
    }
    for name, y in ref.items():
        entry[f"ari_{name}"] = float(adjusted_rand_score(y, lab))
        entry[f"nmi_{name}"] = float(normalized_mutual_info_score(y, lab))
    out["dbscan"] = entry
    return out
