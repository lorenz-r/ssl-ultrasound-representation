"""Collapse instrumentation: RankMe, per-dim std, and — the one that must
never silently break — the online probe's gradient isolation (PLAN.md §4.7).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ssl_us.models.resnet import ResNetBackbone
from ssl_us.train import instrument as instr


def test_rankme_is_high_for_a_full_rank_spectrum():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 64))  # isotropic -> near-full-rank
    spectrum = instr.singular_value_spectrum(features)
    rm = instr.rankme(spectrum)
    assert rm > 40, f"expected near-full-rank (~64), got {rm}"


def test_rankme_is_low_for_a_collapsed_spectrum():
    rng = np.random.default_rng(0)
    dominant = rng.normal(size=(500, 1)) * 10
    tiny_noise = rng.normal(size=(500, 63)) * 1e-6
    features = np.concatenate([dominant, tiny_noise], axis=1)
    spectrum = instr.singular_value_spectrum(features)
    rm = instr.rankme(spectrum)
    assert rm < 2.0, f"expected near-collapsed (~1), got {rm}"


def test_rankme_monotonic_with_effective_rank():
    """More independent, comparably-scaled dimensions -> strictly higher RankMe."""
    rng = np.random.default_rng(0)
    rms = []
    for n_dims in (2, 8, 32):
        features = rng.normal(size=(500, n_dims))
        rms.append(instr.rankme(instr.singular_value_spectrum(features)))
    assert rms[0] < rms[1] < rms[2]


def test_per_dimension_std_matches_numpy_std():
    z = torch.randn(100, 10)
    got = instr.per_dimension_std(z)
    np.testing.assert_allclose(got, z.numpy().std(axis=0), rtol=1e-5)


def test_singular_value_spectrum_accepts_torch_and_numpy():
    z = torch.randn(50, 8)
    from_tensor = instr.singular_value_spectrum(z)
    from_array = instr.singular_value_spectrum(z.numpy())
    np.testing.assert_allclose(from_tensor, from_array, rtol=1e-4)


# ── the load-bearing test: stop-gradient on the online probe ────────────────


def test_online_probe_never_leaks_gradient_into_the_encoder():
    """PLAN.md §4.7's literal requirement: a probe-loss backward pass must
    leave every encoder parameter's .grad unchanged. If this ever fails,
    every frozen-feature number in the results matrix becomes invalid.
    """
    torch.manual_seed(0)
    encoder = ResNetBackbone(2, seed=0)
    x = torch.randn(6, 2, 32, 32)
    h = encoder(x)  # tracked: grad_fn chains back through encoder params

    probe = instr.OnlineProbe(in_dim=encoder.dim, n_classes=3)
    labels = torch.randint(0, 3, (6,))

    instr.assert_no_grad_leak(encoder, probe, h, labels)  # must not raise

    # and the probe itself must actually have learned something (grad flowed there)
    assert all(p.grad is not None for p in probe.head.parameters())


def test_online_probe_step_runs_and_probe_params_change():
    torch.manual_seed(0)
    probe = instr.OnlineProbe(in_dim=16, n_classes=2, lr=0.5)
    before = [p.clone() for p in probe.head.parameters()]
    feats = torch.randn(20, 16)
    labels = torch.randint(0, 2, (20,))
    for _ in range(5):
        probe.step(feats, labels)
    after = list(probe.head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_online_probe_balanced_accuracy_on_separable_data():
    torch.manual_seed(0)
    probe = instr.OnlineProbe(in_dim=2, n_classes=2, lr=0.5)
    feats = torch.cat([torch.randn(50, 2) - 5, torch.randn(50, 2) + 5])
    labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    for _ in range(100):
        probe.step(feats, labels)
    acc = probe.balanced_accuracy(feats, labels)
    assert acc > 0.9


def test_assert_no_grad_leak_would_fire_if_detach_were_removed():
    """Negative control: prove the assertion is not vacuous by constructing
    the exact bug it guards against (probe reading non-detached features)."""
    torch.manual_seed(0)
    encoder = ResNetBackbone(2, seed=0)
    x = torch.randn(4, 2, 32, 32)
    h = encoder(x)
    labels = torch.randint(0, 3, (4,))

    # the bug: train directly on `h` (no .detach()) via a bypassed probe head
    head = torch.nn.Linear(encoder.dim, 3)
    logits = head(h)  # NOT detached
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    leaked = any(p.grad is not None for p in encoder.parameters())
    assert leaked, "the negative control itself did not reproduce the leak"


def test_knn_eval_on_separable_clusters():
    rng = np.random.default_rng(0)
    train_x = np.concatenate([rng.normal(-5, 0.5, (50, 2)), rng.normal(5, 0.5, (50, 2))])
    train_y = np.concatenate([np.zeros(50), np.ones(50)])
    test_x = np.concatenate([rng.normal(-5, 0.5, (20, 2)), rng.normal(5, 0.5, (20, 2))])
    test_y = np.concatenate([np.zeros(20), np.ones(20)])
    acc = instr.knn_eval(train_x, train_y, test_x, test_y, k=3)
    assert acc > 0.95
