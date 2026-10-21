"""Per-pixel segmentation probe on frozen features (WI-19).

`probes.probe` scores one number per crop. This scores one number per *cell* of
the dense grid and reports Dice at full pixel resolution, which is the question
`contains_cyst` cannot answer: not whether a crop contains a cyst but where the
cyst is. The protocol follows PLAN.md 4.4 wherever the shapes allow and deviates
in three places that are named here rather than left to be discovered:

1. **Grouped CV is unchanged.** `GroupKFold` over `frame_id`, with the group
   vector repeated once per cell. Cells within a crop are massively correlated
   and crops within a frame more so; the effective sample size is frames.

2. **The bootstrap resamples frames, not rows.** `probes._bootstrap_ci` draws
   rows, which is right when a row is a crop and catastrophic when a row is a
   pixel: over the ~380 k cells of a validation column it would report a CI
   roughly two orders of magnitude too tight. Dice is aggregated from
   per-frame (intersection, |pred|, |true|) triples, so resampling frames is
   both exact and cheap.

3. **The penalty grid is coarser and the fit is subsampled.** Nested CV over
   `probes.LOGIT_CS`' 15 values on millions of rows does not finish; this uses
   5 values and at most `max_fit_cells` rows per fit. Scoring still uses every
   held-out cell. A dense cell is therefore *not* interchangeable with a pooled
   one and the two are reported in separate tables.

`class_weight="balanced"` is on. Cyst pixels are a small minority, so an
unweighted logistic head at a 0.5 threshold collapses to all-background for
every arm — Dice 0 everywhere, which measures the base rate rather than the
representation. The weighting is applied identically to every encoder including
the baselines, so it cannot favour one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .guards import assert_no_label_leak
from .probes import CAPACITY_DIMS

# Five values, not `probes.LOGIT_CS`' fifteen. See the module docstring: the
# nested grid is run per outer fold over hundreds of thousands of rows.
DENSE_LOGIT_CS = np.logspace(-3, 1, 5)

# The probe target's name, for the leakage guard and the results record. Not in
# `guards.LEAKED_BY_GT_SOS`: the segmentation map is not an input to either
# beamforming stage, so this passes on `gt_sos` as well as `c0_const`. It is
# routed through the guard anyway so that adding a stage which *does* consume
# it cannot silently produce a valid-looking number.
TARGET = "cyst_seg"


def dice_counts(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    """``(intersection, |pred|, |true|)`` for one boolean field."""
    return float(np.logical_and(pred, true).sum()), float(pred.sum()), float(true.sum())


def dice_from_counts(inter: float, psum: float, tsum: float) -> float:
    """Micro (aggregate) Dice. Empty prediction *and* empty truth scores 1.0."""
    denom = psum + tsum
    return 1.0 if denom == 0 else 2.0 * inter / denom


def iou_from_counts(inter: float, psum: float, tsum: float) -> float:
    union = psum + tsum - inter
    return 1.0 if union == 0 else inter / union


def upsample_maps(prob: np.ndarray, size: int) -> np.ndarray:
    """``(N, g, g) -> (N, size, size)`` bilinear, matching the sliding window's."""
    t = torch.from_numpy(np.ascontiguousarray(prob)).float()[:, None]
    return (
        F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
        .numpy()
        .astype(np.float32)
    )


def cell_targets(seg: np.ndarray, grid: int) -> np.ndarray:
    """``(N, size, size) uint8 -> (N, grid, grid) float`` cyst fraction per cell."""
    n, h, w = seg.shape
    ch, cw = h // grid, w // grid
    return seg.astype(np.float32).reshape(n, grid, ch, grid, cw).mean(axis=(2, 4))


@dataclass
class DenseHead:
    """A fitted channel-PCA plus linear head, applied to feature *maps*.

    Holds the PCA because the sliding window must reduce new windows with the
    basis the head was fitted against; refitting per frame would make the
    picture and the number two different models.
    """

    pca: PCA | None
    clf: object
    grid: int

    def _cells(self, maps: np.ndarray) -> np.ndarray:
        n, c, g, _ = maps.shape
        x = maps.transpose(0, 2, 3, 1).reshape(n * g * g, c).astype(np.float32)
        return self.pca.transform(x) if self.pca is not None else x

    def predict_proba_maps(self, maps: np.ndarray) -> np.ndarray:
        """``(N, C, g, g) -> (N, g, g)`` cyst probability."""
        n, _, g, _ = maps.shape
        return self.clf.predict_proba(self._cells(maps))[:, 1].reshape(n, g, g)


class _ConstantClf:
    """The degenerate head for a single-class training fold.

    A fold whose training cells are all background is rare on the real corpus
    but not impossible — cysts are sparse, `GroupKFold` splits by frame, and a
    fold can draw only frames whose sampled crops missed the lesion. sklearn
    raises there, which would abort a column mid-run and lose every encoder
    already probed. Predicting the one class it saw is the honest degenerate
    answer: it scores Dice 0 against any real mask, so the fold contributes a
    visible failure rather than a crash or a quietly dropped fold.
    """

    def __init__(self, label: int):
        self.label = int(label)

    def predict_proba(self, x):
        p = np.zeros((len(x), 2), dtype=float)
        p[:, self.label] = 1.0
        return p

    def predict(self, x):
        return np.full(len(x), self.label, dtype=int)


def _make_clf(c: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c, max_iter=1000, class_weight="balanced", random_state=seed
        ),
    )


def _fit_clf(x, y, c: float, seed: int):
    """`_make_clf` plus the single-class fallback."""
    classes = np.unique(y)
    if len(classes) < 2:
        return _ConstantClf(classes[0])
    m = _make_clf(c, seed)
    m.fit(x, y)
    return m


def _subsample(n: int, cap: int, rng) -> np.ndarray:
    return np.arange(n) if n <= cap else rng.choice(n, size=cap, replace=False)


def fit_dense_head(
    maps: np.ndarray,
    y_cell: np.ndarray,
    *,
    dims: int = CAPACITY_DIMS,
    c: float = 1.0,
    max_fit_cells: int = 200_000,
    seed: int = 0,
) -> DenseHead:
    """Fit channel-PCA + logistic head on ``(N, C, g, g)`` maps and ``(N, g, g)`` targets.

    `dims` is `probes.CAPACITY_DIMS`, applied to the *channel* axis for the same
    reason the pooled probe applies it to the feature axis: an 896-channel
    hypercolumn and a 384-channel token map must not be separated by width.
    """
    n, c_ch, g, _ = maps.shape
    rng = np.random.default_rng(seed)
    x = maps.transpose(0, 2, 3, 1).reshape(n * g * g, c_ch).astype(np.float32)
    y = (y_cell.reshape(-1) >= 0.5).astype(int)

    pca = None
    if c_ch > dims:
        idx = _subsample(len(x), max_fit_cells, rng)
        pca = PCA(n_components=dims, random_state=seed).fit(x[idx])
        x = pca.transform(x)

    idx = _subsample(len(x), max_fit_cells, rng)
    return DenseHead(pca=pca, clf=_fit_clf(x[idx], y[idx], c, seed), grid=g)


@dataclass
class DenseProbeResult:
    encoder: str
    metric: str  # "dice"
    score: float
    score_lo: float
    score_hi: float
    secondary: dict[str, float]
    n_frames: int
    n_crops: int
    n_cells: int
    n_features: int
    grid: int
    stage: str
    per_frame_dice: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "encoder": self.encoder,
            "metric": self.metric,
            "dice": round(self.score, 4),
            "ci95": f"[{self.score_lo:.3f}, {self.score_hi:.3f}]",
            **{k: round(v, 4) for k, v in self.secondary.items()},
            "n_frames": self.n_frames,
            "n_crops": self.n_crops,
            "n_feat": self.n_features,
            "grid": self.grid,
            "stage": self.stage,
        }


def _select_c(x, y, groups, seed) -> float:
    """Inner-CV penalty selection, scored by Dice on the held-in folds."""
    n_inner = int(min(3, len(np.unique(groups))))
    if n_inner < 2:
        return float(DENSE_LOGIT_CS[len(DENSE_LOGIT_CS) // 2])
    best, best_score = None, -np.inf
    for c in DENSE_LOGIT_CS:
        scores = []
        for itr, ite in GroupKFold(n_splits=n_inner).split(x, y, groups):
            if len(np.unique(y[itr])) < 2:
                continue
            m = _fit_clf(x[itr], y[itr], c, seed)
            scores.append(dice_from_counts(*dice_counts(m.predict(x[ite]) == 1, y[ite] == 1)))
        if scores and float(np.mean(scores)) > best_score:
            best_score, best = float(np.mean(scores)), float(c)
    return best if best is not None else float(DENSE_LOGIT_CS[len(DENSE_LOGIT_CS) // 2])


def dense_probe(
    maps: np.ndarray,
    seg: np.ndarray,
    frame_id: np.ndarray,
    *,
    encoder: str,
    stage: str,
    dims: int = CAPACITY_DIMS,
    max_fit_cells: int = 200_000,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[DenseProbeResult, np.ndarray]:
    """One dense column: out-of-fold masks plus Dice with a frame-level CI.

    Returns ``(result, oof_prob)`` where `oof_prob` is ``(N, g, g)`` — the
    out-of-fold cyst probability for every crop. The caller keeps it to pick
    figure frames by a stated rule rather than by eye.
    """
    assert_no_label_leak(stage, TARGET)

    n, c_ch, g, _ = maps.shape
    size = seg.shape[-1]
    groups_crop = np.asarray(frame_id).astype(str)
    y_cell = cell_targets(seg, g)
    y = (y_cell.reshape(-1) >= 0.5).astype(int)
    rng = np.random.default_rng(seed)

    x = maps.transpose(0, 2, 3, 1).reshape(n * g * g, c_ch).astype(np.float32)
    # Capacity control is fit once, on the whole column, rather than per fold.
    # It is unsupervised — it never sees `y` — so this leaks no label, and it
    # keeps the basis identical across folds, which is what lets a single
    # `DenseHead` from `fit_dense_head` be used for the whole-frame figure.
    if c_ch > dims:
        idx = _subsample(len(x), max_fit_cells, rng)
        pca = PCA(n_components=dims, random_state=seed).fit(x[idx])
        x = pca.transform(x).astype(np.float32)
    n_features = x.shape[1]

    groups_cell = np.repeat(groups_crop, g * g)
    uniq_frames = np.unique(groups_crop)
    n_splits = int(min(5, len(uniq_frames)))
    if n_splits < 2:
        raise ValueError(f"need >=2 frames for grouped CV, got {len(uniq_frames)}")

    oof = np.empty(len(x), dtype=np.float32)
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups_cell):
        fit_idx = tr[_subsample(len(tr), max_fit_cells, rng)]
        c = _select_c(x[fit_idx], y[fit_idx], groups_cell[fit_idx], seed)
        m = _fit_clf(x[fit_idx], y[fit_idx], c, seed)
        oof[te] = m.predict_proba(x[te])[:, 1]

    oof_prob = oof.reshape(n, g, g)

    # Scored at full pixel resolution, against the mask as fetched. A coarse
    # feature grid is penalised here for being coarse, which is the honest
    # reading: the figure shows the upsampled map, so the number must too.
    pred_px = upsample_maps(oof_prob, size) >= 0.5
    true_px = seg.astype(bool)

    per_frame: dict[str, tuple[float, float, float]] = {}
    for fid in uniq_frames:
        m = groups_crop == fid
        per_frame[fid] = dice_counts(pred_px[m], true_px[m])

    totals = np.array([per_frame[f] for f in uniq_frames], dtype=np.float64)
    score = dice_from_counts(*totals.sum(0))
    iou = iou_from_counts(*totals.sum(0))

    # Resample *frames*. Each frame's triple is a sufficient statistic for its
    # contribution to the aggregate, so this is exact, not an approximation.
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq_frames), size=len(uniq_frames))
        boot[b] = dice_from_counts(*totals[pick].sum(0))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    cell_pred = oof_prob.reshape(-1) >= 0.5
    secondary = {
        "iou": iou,
        # How much of the score is lost to the 16x16 readout rather than to the
        # representation: `dice` is scored after upsampling to `size`, this one
        # on the grid the head actually predicts on.
        "dice_cell": dice_from_counts(*dice_counts(cell_pred, y == 1)),
        "positive_rate": float(true_px.mean()),
    }

    result = DenseProbeResult(
        encoder=encoder,
        metric="dice",
        score=score,
        score_lo=lo,
        score_hi=hi,
        secondary=secondary,
        n_frames=len(uniq_frames),
        n_crops=n,
        n_cells=len(x),
        n_features=n_features,
        grid=g,
        stage=stage,
        per_frame_dice={
            f: dice_from_counts(*per_frame[f]) for f in uniq_frames
        },
    )
    return result, oof_prob
