"""Frozen-feature probes with the PLAN.md 4.4 protocol enforced.

Every frame-level cell in the results matrix uses, without exception:

* ridge/logistic penalty selected by **inner** cross-validation, never a fixed value;
* **k-fold CV grouped by frame_id** over the train frames — not a single fit scored
  on the 93-frame validation split, whose R^2 standard error is wide enough to
  swallow most effects of interest;
* **bootstrap confidence intervals** on every cell (n_eff = frames, not crops);
* **PCA to 64 dimensions** as a fixed capacity control, so a 2048-d feature space
  and a 512-d one are compared on equal terms rather than on width.

Regression-first: ridge with R^2/MAE for continuous axes, logistic only for
``contains_cyst`` and ``t0_group``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .guards import assert_no_label_leak

RIDGE_ALPHAS = np.logspace(-3, 5, 17)
LOGIT_CS = np.logspace(-4, 3, 15)
CAPACITY_DIMS = 64


@dataclass
class ProbeResult:
    target: str
    level: str  # "crop" | "frame"
    kind: str  # "ridge" | "logistic"
    metric: str  # "r2" | "balanced_accuracy"
    score: float
    score_lo: float
    score_hi: float
    secondary: dict[str, float]
    n_eff: int
    n_features: int
    stage: str

    def row(self) -> dict:
        return {
            "target": self.target,
            "level": self.level,
            "metric": self.metric,
            "score": round(self.score, 4),
            "ci95": f"[{self.score_lo:.3f}, {self.score_hi:.3f}]",
            **{k: round(v, 4) for k, v in self.secondary.items()},
            "n_eff": self.n_eff,
            "n_feat": self.n_features,
            "stage": self.stage,
        }


def capacity_control(x: np.ndarray, dims: int = CAPACITY_DIMS, seed: int = 0) -> np.ndarray:
    """Project features to a fixed dimensionality so widths are comparable.

    Without this, ridge rewards the wider representation on capacity alone and the
    results matrix silently ranks feature spaces by their dimension.
    """
    if x.shape[1] <= dims:
        return x
    return PCA(n_components=dims, random_state=seed).fit_transform(x)


def pool_to_frames(
    x: np.ndarray, frame_id: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool crop features to one vector per frame, with the frame's label."""
    order = np.argsort(frame_id.astype(str), kind="stable")
    xs, fs, ys = x[order], frame_id.astype(str)[order], y[order]
    uniq, starts = np.unique(fs, return_index=True)
    xp = np.stack([xs[a:b].mean(0) for a, b in zip(starts, [*starts[1:], len(fs)], strict=True)])
    yp = np.array([ys[a] for a in starts])
    return xp, yp, uniq


def _bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, scorer, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(scorer(y[idx], pred[idx]))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _cv_predict(x, y, groups, kind, seed):
    """Out-of-fold predictions with an inner-CV-selected penalty per outer fold."""
    n_groups = len(np.unique(groups))
    n_splits = int(min(5, n_groups))
    if n_splits < 2:
        raise ValueError(f"need >=2 groups for grouped CV, got {n_groups}")
    outer = GroupKFold(n_splits=n_splits)
    pred = np.empty(len(y), dtype=float if kind == "ridge" else int)
    prob = np.full(len(y), np.nan)

    for tr, te in outer.split(x, y, groups):
        inner_groups = groups[tr]
        n_inner = int(min(4, len(np.unique(inner_groups))))
        best, best_score = None, -np.inf
        grid = RIDGE_ALPHAS if kind == "ridge" else LOGIT_CS
        for p in grid:
            if n_inner < 2:
                best = p
                break
            scores = []
            for itr, ite in GroupKFold(n_splits=n_inner).split(x[tr], y[tr], inner_groups):
                m = _make(kind, p, seed)
                if kind == "logistic" and len(np.unique(y[tr][itr])) < 2:
                    continue
                m.fit(x[tr][itr], y[tr][itr])
                yh = m.predict(x[tr][ite])
                scores.append(
                    r2_score(y[tr][ite], yh)
                    if kind == "ridge"
                    else balanced_accuracy_score(y[tr][ite], yh)
                )
            if scores and np.mean(scores) > best_score:
                best_score, best = float(np.mean(scores)), p
        m = _make(kind, best if best is not None else grid[len(grid) // 2], seed)
        m.fit(x[tr], y[tr])
        pred[te] = m.predict(x[te])
        if kind == "logistic" and len(np.unique(y[tr])) == 2:
            prob[te] = m.predict_proba(x[te])[:, 1]
    return pred, prob


def _make(kind: str, param: float, seed: int):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=param, random_state=seed))
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=param, max_iter=2000, random_state=seed),
    )


def probe(
    x: np.ndarray,
    y: np.ndarray,
    frame_id: np.ndarray,
    target: str,
    *,
    stage: str,
    level: str = "crop",
    kind: str = "ridge",
    apply_capacity_control: bool = True,
    seed: int = 0,
) -> ProbeResult:
    """Run one probe cell. Raises before fitting if the stage leaks the target."""
    assert_no_label_leak(stage, target)

    if apply_capacity_control:
        x = capacity_control(x, seed=seed)

    if level == "frame":
        x, y, groups = pool_to_frames(x, frame_id, y)
    else:
        groups = frame_id.astype(str)

    if kind == "logistic":
        y = y.astype(int)

    pred, prob = _cv_predict(x, y, groups, kind, seed)

    if kind == "ridge":
        score = float(r2_score(y, pred))
        lo, hi = _bootstrap_ci(y, pred, r2_score, seed=seed)
        secondary = {"mae": float(mean_absolute_error(y, pred))}
    else:
        score = float(balanced_accuracy_score(y, pred))
        lo, hi = _bootstrap_ci(y, pred, balanced_accuracy_score, seed=seed)
        secondary = {}
        if len(np.unique(y)) == 2 and not np.isnan(prob).all():
            secondary["auc"] = float(roc_auc_score(y, prob))

    return ProbeResult(
        target=target,
        level=level,
        kind=kind,
        metric="r2" if kind == "ridge" else "balanced_accuracy",
        score=score,
        score_lo=lo,
        score_hi=hi,
        secondary=secondary,
        n_eff=len(y),
        n_features=x.shape[1],
        stage=stage,
    )
