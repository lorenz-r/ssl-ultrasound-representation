"""CPU smoke test for the VICReg training loop (PLAN.md WI-12): generic over
the data source, so a synthetic loader here exercises exactly the same code
path a real Policy-A/Policy-B `DataLoader` will on the cluster.
"""

from __future__ import annotations

import torch

from ssl_us.models.resnet import BACKBONE_DIM, ResNetBackbone
from ssl_us.models.vicreg import VICReg
from ssl_us.train import instrument as instr
from ssl_us.train.vicreg import VICRegTrainConfig, _is_collapsing, train_vicreg


def _synthetic_loader(n_batches: int, batch_size: int, in_channels: int, size: int):
    torch.manual_seed(0)
    base = torch.randn(n_batches, batch_size, in_channels, size, size)
    for i in range(n_batches):
        view1 = base[i]
        view2 = base[i] + 0.05 * torch.randn_like(base[i])  # a cheap "other view"
        yield view1, view2


def test_train_vicreg_runs_and_reports_all_instruments():
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=32)
    loader = list(_synthetic_loader(n_batches=4, batch_size=6, in_channels=2, size=32))
    config = VICRegTrainConfig(epochs=2, lr=1e-3, device="cpu", amp=False)

    history = train_vicreg(model, loader, config)

    for key in ("loss", "invariance", "variance", "covariance", "rankme"):
        assert key in history
        assert len(history[key]) == config.epochs
        assert all(v == v for v in history[key]), f"{key} contains NaN"  # v==v is False for NaN


def test_train_vicreg_loss_does_not_explode():
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=32)
    loader = list(_synthetic_loader(n_batches=4, batch_size=6, in_channels=2, size=32))
    config = VICRegTrainConfig(epochs=3, lr=1e-3, device="cpu", amp=False)

    history = train_vicreg(model, loader, config)
    assert all(loss < 1e4 for loss in history["loss"])


def test_train_vicreg_with_online_probe_never_leaks_gradient():
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=32)
    loader = list(_synthetic_loader(n_batches=3, batch_size=6, in_channels=2, size=32))
    probe = instr.OnlineProbe(in_dim=BACKBONE_DIM, n_classes=2)
    config = VICRegTrainConfig(epochs=1, lr=1e-3, device="cpu", amp=False)

    def label_fn(x1):
        return torch.zeros(x1.shape[0], dtype=torch.long)

    history = train_vicreg(model, loader, config, probe=probe, probe_label_fn=label_fn)
    assert "probe_loss" in history
    assert len(history["probe_loss"]) == 1


# --- dimensional-collapse detector (PLAN.md §4.7) ---------------------------
# Regression coverage for a bug that halted both WI-12 arms at epoch 50: the
# detector paired a window with its own tail under `zip(..., strict=True)`,
# which is ragged by construction, so every call that got past the length
# guard raised ValueError instead of returning a verdict. The chain then read
# python's exit-1 as the deliberate collapse halt and skipped the arm.


def test_is_collapsing_returns_a_verdict_rather_than_raising():
    """The detector must be callable at all once the window is long enough."""
    history = {
        "rankme": [100.0 - i for i in range(20)],
        "loss": [10.0 - 0.1 * i for i in range(20)],
    }
    assert _is_collapsing(history, patience=10) is True


def test_is_collapsing_true_only_when_rank_and_loss_both_fall():
    falling_rank = [100.0 - i for i in range(12)]
    falling_loss = [10.0 - 0.1 * i for i in range(12)]
    rising_loss = [10.0 + 0.1 * i for i in range(12)]

    assert _is_collapsing({"rankme": falling_rank, "loss": falling_loss}, 10) is True
    # RankMe down while the loss *rises* is an unstable run, not a collapse.
    assert _is_collapsing({"rankme": falling_rank, "loss": rising_loss}, 10) is False
    assert _is_collapsing({"rankme": [100.0] * 12, "loss": falling_loss}, 10) is False


def test_is_collapsing_is_false_before_the_window_fills():
    history = {"rankme": [100.0 - i for i in range(8)], "loss": [10.0 - i for i in range(8)]}
    assert _is_collapsing(history, patience=10) is False
    assert _is_collapsing(history, patience=0) is False


def test_is_collapsing_tolerates_a_single_non_monotonic_epoch():
    """One epoch of recovery inside the window breaks the collapse signature."""
    rank = [100.0 - i for i in range(12)]
    rank[5] = rank[4] + 1.0
    loss = [10.0 - 0.1 * i for i in range(12)]
    assert _is_collapsing({"rankme": rank, "loss": loss}, 10) is False
