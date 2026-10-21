"""ResNet-18 stem adaptation + VICReg projector/loss (PLAN.md WI-12)."""

from __future__ import annotations

import pytest
import torch

from ssl_us.models.resnet import BACKBONE_DIM, ResNetBackbone
from ssl_us.models.vicreg import (
    VICReg,
    VICRegLossWeights,
    covariance_term,
    invariance_term,
    variance_term,
    vicreg_loss,
)


@pytest.mark.parametrize("in_channels", [1, 2, 32])
def test_resnet_backbone_accepts_arbitrary_in_channels(in_channels):
    net = ResNetBackbone(in_channels, seed=0)
    x = torch.randn(2, in_channels, 64, 64)
    out = net(x)
    assert out.shape == (2, BACKBONE_DIM)
    assert net.net.conv1.in_channels == in_channels


def test_resnet_backbone_never_loads_pretrained_weights():
    """Random-init only — pretrained RGB filters have no meaning on IQ channels."""
    a = ResNetBackbone(2, seed=0)
    b = ResNetBackbone(2, seed=0)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)  # same seed -> identical init
    c = ResNetBackbone(2, seed=1)
    diffs = [not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters(), strict=True)]
    assert any(diffs), "different seeds should give different random init"


def test_invariance_term_is_zero_for_identical_views():
    z = torch.randn(8, 16)
    assert invariance_term(z, z).item() == pytest.approx(0.0, abs=1e-6)


def test_invariance_term_is_positive_for_different_views():
    z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
    assert invariance_term(z1, z2).item() > 0


def test_variance_term_penalizes_constant_dimension():
    """A dimension with zero variance must incur (close to) the full hinge margin."""
    z = torch.zeros(32, 4)
    z[:, 0] = torch.randn(32) * 5  # only dim 0 has real variance
    loss = variance_term(z, margin=1.0)
    assert loss.item() > 0.5  # 3 of 4 dims sit near the max penalty


def test_variance_term_is_near_zero_when_std_exceeds_margin():
    z = torch.randn(256, 8) * 5.0  # std ~5, well above margin=1
    assert variance_term(z, margin=1.0).item() < 0.05


def test_covariance_term_penalizes_correlated_dimensions():
    base = torch.randn(256, 1)
    correlated = torch.cat([base, base + 0.01 * torch.randn(256, 1)], dim=1)
    independent = torch.randn(256, 2)
    assert covariance_term(correlated).item() > covariance_term(independent).item()


def test_vicreg_loss_returns_all_three_terms_separately():
    z1, z2 = torch.randn(16, 32), torch.randn(16, 32)
    total, terms = vicreg_loss(z1, z2)
    assert set(terms) == {"invariance", "variance", "covariance"}
    weights = VICRegLossWeights()
    expected = (
        weights.invariance * terms["invariance"]
        + weights.variance * terms["variance"]
        + weights.covariance * terms["covariance"]
    )
    torch.testing.assert_close(total, expected)


def test_vicreg_forward_returns_backbone_and_projector_outputs():
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=64)
    x = torch.randn(4, 2, 32, 32)
    h, z = model(x)
    assert h.shape == (4, BACKBONE_DIM)
    assert z.shape == (4, 64)


def test_vicreg_loss_is_differentiable_and_training_reduces_it():
    """A few gradient steps on random-but-fixed data must not increase the loss —
    a basic sanity check that the loss is correctly wired for backprop."""
    torch.manual_seed(0)
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=32)
    x1 = torch.randn(8, 2, 32, 32)
    x2 = x1 + 0.01 * torch.randn_like(x1)  # a near-duplicate "other view"
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        h1, z1 = model(x1)
        h2, z2 = model(x2)
        loss, _ = vicreg_loss(z1, z2)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] <= losses[0] + 1e-3
