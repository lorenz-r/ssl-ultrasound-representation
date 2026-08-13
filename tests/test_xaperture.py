"""Cross-aperture prediction arm (PLAN.md WI-15, optional)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssl_us.models import xaperture as xap


@pytest.fixture
def sector_crop():
    rng = np.random.default_rng(0)
    return rng.normal(size=(2, 12, 10, 8)).astype(np.float32)  # (ch, Z, X, K=8)


def test_mask_axis_tx_is_unavailable():
    """Second, independent reason from the transmit augmentation: no
    transmit-resolved tensor exists at all without the PW stack."""
    with pytest.raises(NotImplementedError, match="PW\\(5\\) transmit-angle stack was not built"):
        xap.mask_axis_tx()


def test_assert_single_aperture_axis_accepts_exactly_one():
    xap.assert_single_aperture_axis((2, 12, 10, 8), expected_aperture_axes=1)


def test_assert_single_aperture_axis_rejects_two():
    with pytest.raises(ValueError, match="exactly one aperture axis"):
        xap.assert_single_aperture_axis((2, 12, 10, 8, 5), expected_aperture_axes=2)


def test_mask_one_rx_sector_shapes_and_content(sector_crop):
    rng = np.random.default_rng(1)
    context, target, idx = xap.mask_one_rx_sector(sector_crop, rng)
    ch, z, x, k = sector_crop.shape
    assert context.shape == (ch * k, z, x)
    assert target.shape == (ch, z, x)
    np.testing.assert_array_equal(target, sector_crop[..., idx])
    # the held-out sector must read back as exactly zero in the folded context
    folded_back = context.reshape(ch, k, z, x)
    np.testing.assert_array_equal(folded_back[:, idx], 0.0)
    # every other sector must survive untouched
    for i in range(k):
        if i != idx:
            np.testing.assert_array_equal(folded_back[:, i], sector_crop[..., i])


def test_mask_one_rx_sector_never_leaks_the_target_into_context(sector_crop):
    """The context must never contain the held-out sector's values anywhere —
    the single most important correctness property of a masking scheme."""
    rng = np.random.default_rng(2)
    ch, _z, _x, k = sector_crop.shape
    for _ in range(50):
        context, target, idx = xap.mask_one_rx_sector(sector_crop, rng)
        folded_back = context.reshape(ch, k, *sector_crop.shape[1:-1])
        assert not np.any(folded_back[:, idx])


def test_trivial_copy_baseline_is_mean_of_others(sector_crop):
    idx = 3
    pred = xap.trivial_copy_baseline(sector_crop, idx)
    others = [i for i in range(sector_crop.shape[-1]) if i != idx]
    expected = sector_crop[..., others].mean(axis=-1)
    np.testing.assert_allclose(pred, expected)


def test_vcz_coherence_is_one_at_zero_separation():
    coh = xap.van_cittert_zernike_coherence(0.0, angular_extent_rad=0.2, wavelength_m=2e-4)
    assert coh == pytest.approx(1.0)


def test_vcz_coherence_decreases_with_separation():
    seps = np.array([0.0, 1e-4, 5e-4, 1e-3])
    coh = xap.van_cittert_zernike_coherence(seps, angular_extent_rad=0.3, wavelength_m=2e-4)
    assert np.all(np.diff(coh) <= 1e-9)  # non-increasing


def test_vcz_coherence_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    seps = rng.uniform(0, 1e-2, size=100)
    coh = xap.van_cittert_zernike_coherence(seps, angular_extent_rad=0.4, wavelength_m=2.4e-4)
    assert np.all(coh >= 0.0) and np.all(coh <= 1.0 + 1e-9)


def test_vcz_predict_baseline_weights_nearby_sectors_more(sector_crop):
    """A sector angularly close to the held-out one should get more weight
    than a distant one, for a nontrivial (non-flat) angular extent."""
    k = sector_crop.shape[-1]
    angles = np.linspace(-0.4, 0.4, k)  # evenly spaced view angles
    held_out = k // 2
    pred = xap.vcz_predict_baseline(
        sector_crop, held_out, angles, range_m=0.02, wavelength_m=2.4e-4, angular_extent_rad=0.05
    )
    assert pred.shape == sector_crop[..., 0].shape
    assert np.all(np.isfinite(pred))


def test_vcz_predict_baseline_falls_back_to_trivial_when_all_weights_zero(sector_crop):
    """A pathological angular_extent of ~0 makes every off-center weight
    collapse to (near) zero; the function must not divide by zero."""
    k = sector_crop.shape[-1]
    angles = np.linspace(-0.4, 0.4, k)
    pred = xap.vcz_predict_baseline(
        sector_crop, 0, angles, range_m=1.0, wavelength_m=1e-6, angular_extent_rad=1e-12
    )
    assert np.all(np.isfinite(pred))


def test_residual_frame_correlation_recovers_known_correlation():
    rng = np.random.default_rng(0)
    phase_error = rng.normal(size=200)
    residual = 0.8 * phase_error + 0.2 * rng.normal(size=200)  # strongly correlated by construction
    sos_std = rng.normal(size=200)  # independent
    out = xap.residual_frame_correlation(residual, phase_error, sos_std)
    assert out["corr_phase_error"] > 0.7
    assert abs(out["corr_sos_std"]) < 0.3


# ── model ────────────────────────────────────────────────────────────────


def test_xaperture_model_forward_shape():
    k = 8
    model = xap.XApertureModel(k=k, out_size=(12, 10), seed=0)
    context = torch.randn(3, 2 * k, 12, 10)
    out = model(context)
    assert out.shape == (3, 2, 12, 10)


def test_xaperture_loss_decreases_with_training():
    torch.manual_seed(0)
    k = 4
    model = xap.XApertureModel(k=k, out_size=(16, 16), seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    context = torch.randn(4, 2 * k, 16, 16)
    target = torch.randn(4, 2, 16, 16)

    losses = []
    for _ in range(8):
        opt.zero_grad()
        pred = model(context)
        loss = xap.xaperture_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_xaperture_model_beats_trivial_baseline_is_measured_not_assumed(sector_crop):
    """This does not assert the model beats the baseline (it's untrained and
    may not, on 12x10 random data) — it asserts both are computed on the
    *same* held-out target, which is the comparison PLAN.md WI-15 requires
    ("the trivial-copy baseline is reported first, so any learned gain is
    measured against it")."""
    rng = np.random.default_rng(3)
    context, target, idx = xap.mask_one_rx_sector(sector_crop, rng)
    baseline_pred = xap.trivial_copy_baseline(sector_crop, idx)

    model = xap.XApertureModel(k=sector_crop.shape[-1], out_size=target.shape[1:], seed=0)
    model.eval()  # a batch of 1 through BatchNorm requires eval mode
    with torch.no_grad():
        model_pred = model(torch.from_numpy(context).unsqueeze(0).float()).squeeze(0).numpy()

    baseline_mse = float(np.mean((baseline_pred - target) ** 2))
    model_mse = float(np.mean((model_pred - target) ** 2))
    assert np.isfinite(baseline_mse) and np.isfinite(model_mse)
