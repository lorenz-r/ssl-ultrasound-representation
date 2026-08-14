"""Training-time collapse instrumentation, shipped *before* the first real
run (PLAN.md §4.7): singular-value spectrum + RankMe on the backbone,
per-dimension std on the projector, a gradient-isolated online probe, and
kNN evaluation. See `train/README.md` for the failure-signature table these
feed.

**Backbone vs projector, on purpose.** RankMe is computed on the 512-d
*backbone* feature (pre-projector); the per-dimension std histogram is
computed on the 2048-d *projector* output. Collapse appears in the backbone
spectrum well before the projector's variance hinge reacts — reporting
RankMe on the projector instead is the standard mistake this split exists to
avoid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def singular_value_spectrum(features: torch.Tensor | np.ndarray) -> np.ndarray:
    """Singular values of the (N, D) feature matrix (mean-centered), descending.

    One 512x512-scale SVD per epoch, on the backbone feature (PLAN.md §4.7
    Tier 1). Used directly by :func:`rankme`, and plottable on its own to see
    *where* along the spectrum a collapse is happening (early dims vs a long
    flat tail disappearing).
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    # Under AMP the backbone emits float16, which `np.linalg.svd` rejects
    # outright ("array type float16 is unsupported in linalg"). Promote here,
    # at the single point every caller funnels through, rather than asking
    # each training loop to remember to cast. float32 is enough for a
    # diagnostic spectrum and halves the copy against float64.
    features = np.asarray(features)
    if features.dtype != np.float64:
        features = features.astype(np.float32, copy=False)
    centered = features - features.mean(axis=0, keepdims=True)
    return np.linalg.svd(centered, compute_uv=False)


def rankme(singular_values: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe: ``exp(-sum(p_k log p_k))``, ``p_k = sigma_k / sum(sigma) + eps``.

    The primary collapse diagnostic (PLAN.md §4.7) — needs no labels. A
    healthy, full-rank backbone gives a value close to the feature
    dimensionality; a collapsed one (variance concentrated in a handful of
    directions) gives a value close to 1.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total + eps
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def per_dimension_std(z: torch.Tensor | np.ndarray) -> np.ndarray:
    """Per-dimension std of the projector output — the Tier-1 histogram."""
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    return z.std(axis=0)


# ── online probe, with the stop-gradient guarantee built in ────────────────


class OnlineProbe:
    """A linear head trained on **detached** backbone features, with its own
    optimizer, so it can be stepped every training iteration without ever
    contributing gradient to the encoder.

    PLAN.md §4.7: "Stop-gradient on the online probe is a correctness
    requirement, not a detail... without it the 'linear probe' silently
    becomes supervised fine-tuning and every frozen-feature number in the
    results matrix is invalid." The detach happens once, right here, so no
    call site can forget it.
    """

    def __init__(self, in_dim: int, n_classes: int, lr: float = 1e-2):
        self.head = nn.Linear(in_dim, n_classes)
        self.optimizer = torch.optim.SGD(self.head.parameters(), lr=lr)

    def step(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """One probe update from an already-computed encoder forward pass.

        `features` is detached *inside* this call — callers pass whatever
        the encoder just produced, tracked graph and all, and this function
        is where the stop-gradient boundary actually lives.
        """
        logits = self.head(features.detach())
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def balanced_accuracy(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        from sklearn.metrics import balanced_accuracy_score

        pred = self.head(features).argmax(dim=-1).cpu().numpy()
        return float(balanced_accuracy_score(labels.cpu().numpy(), pred))


def assert_no_grad_leak(
    encoder: nn.Module, probe: OnlineProbe, features: torch.Tensor, labels: torch.Tensor
) -> None:
    """Run one probe step and assert every encoder parameter's `.grad` is
    unaffected — the test PLAN.md §4.7 requires to exist, exposed as a
    reusable function so it can also run as a startup sanity check before a
    real training run, not only inside pytest.
    """
    grads_before = [p.grad.clone() if p.grad is not None else None for p in encoder.parameters()]
    probe.step(features, labels)
    for p, before in zip(encoder.parameters(), grads_before, strict=True):
        after = p.grad
        if before is None:
            assert after is None, "online probe backward leaked a gradient into a fresh param"
        else:
            assert torch.equal(before, after), "online probe backward changed an encoder gradient"


def knn_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Held-out kNN accuracy, no training required (PLAN.md §4.7 Tier 1)."""
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_features, train_labels)
    return float(clf.score(test_features, test_labels))
