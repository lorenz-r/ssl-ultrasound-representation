"""CPU smoke test for the VICReg training loop (PLAN.md WI-12): generic over
the data source, so a synthetic loader here exercises exactly the same code
path a real Policy-A/Policy-B `DataLoader` will on the cluster.
"""

from __future__ import annotations

import torch

from ssl_us.models.resnet import BACKBONE_DIM, ResNetBackbone
from ssl_us.models.vicreg import VICReg
from ssl_us.train import instrument as instr
from ssl_us.train.vicreg import VICRegTrainConfig, train_vicreg


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
