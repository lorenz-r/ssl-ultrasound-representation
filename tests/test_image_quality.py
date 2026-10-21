"""Speckle-invariant image-quality metrics (PLAN.md WI-4, revised gate).

All synthetic: the point of these metrics is that they do *not* depend on a
particular speckle realisation, so they can be pinned against distributions
with known closed-form answers rather than against stored images.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import image_quality as iq


def _rayleigh(rng, sigma, size):
    """Envelope of complex Gaussian noise == Rayleigh, i.e. real speckle."""
    re = rng.normal(0.0, sigma, size)
    im = rng.normal(0.0, sigma, size)
    return np.sqrt(re**2 + im**2)


# ── gCNR ─────────────────────────────────────────────────────────────────────


def test_gcnr_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    a = _rayleigh(rng, 1.0, 200_000)
    b = _rayleigh(rng, 1.0, 200_000)
    # Same distribution -> histograms overlap almost completely -> gCNR ~ 0.
    assert iq.gcnr(a, b) == pytest.approx(0.0, abs=0.02)


def test_gcnr_disjoint_distributions_is_one():
    a = np.linspace(0.0, 1.0, 10_000)
    b = np.linspace(5.0, 6.0, 10_000)
    assert iq.gcnr(a, b) == pytest.approx(1.0, abs=1e-6)


def test_gcnr_is_bounded_and_increases_with_separation():
    rng = np.random.default_rng(1)
    bg = _rayleigh(rng, 1.0, 100_000)
    scores = [iq.gcnr(_rayleigh(rng, s, 100_000), bg) for s in (1.0, 2.0, 4.0, 8.0)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores), f"gCNR should grow with contrast, got {scores}"


def test_gcnr_invariant_to_monotonic_transform():
    """The defining property: dB vs linear must not change the answer."""
    rng = np.random.default_rng(2)
    a = _rayleigh(rng, 0.5, 50_000)
    b = _rayleigh(rng, 2.0, 50_000)
    linear = iq.gcnr(a, b)
    db = iq.gcnr(20 * np.log10(a + 1e-12), 20 * np.log10(b + 1e-12))
    assert db == pytest.approx(linear, abs=0.02)


def test_gcnr_invariant_to_speckle_realisation():
    """Different speckle draws of the same physics must score the same."""
    scores = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        scores.append(iq.gcnr(_rayleigh(rng, 0.5, 80_000), _rayleigh(rng, 2.0, 80_000)))
    assert np.std(scores) < 0.02, f"gCNR varies across speckle draws: {scores}"


def test_gcnr_empty_region_is_nan():
    assert np.isnan(iq.gcnr(np.array([]), np.array([1.0, 2.0])))


# ── CNR ──────────────────────────────────────────────────────────────────────


def test_cnr_zero_contrast_is_negative_infinity_like():
    rng = np.random.default_rng(3)
    a = _rayleigh(rng, 1.0, 20_000)
    b = _rayleigh(rng, 1.0, 20_000)
    # Means nearly equal -> ratio near zero -> strongly negative dB.
    assert iq.cnr(a, b) < -20.0


def test_cnr_grows_with_separation():
    rng = np.random.default_rng(4)
    bg = _rayleigh(rng, 1.0, 20_000)
    assert iq.cnr(_rayleigh(rng, 4.0, 20_000), bg) > iq.cnr(_rayleigh(rng, 1.5, 20_000), bg)


# ── speckle SNR ──────────────────────────────────────────────────────────────


def test_speckle_snr_matches_rayleigh_constant():
    rng = np.random.default_rng(5)
    # Holds for any sigma — the ratio is scale-free.
    for sigma in (0.1, 1.0, 50.0):
        assert iq.speckle_snr(_rayleigh(rng, sigma, 400_000)) == pytest.approx(
            iq.RAYLEIGH_SNR, abs=0.02
        )


def test_rayleigh_constant_value():
    assert iq.RAYLEIGH_SNR == pytest.approx(1.9130, abs=1e-4)


def test_speckle_snr_detects_oversmoothing():
    """Smoothing raises SNR above the Rayleigh value — the failure it exists to catch."""
    rng = np.random.default_rng(6)
    img = _rayleigh(rng, 1.0, (400, 400))
    assert iq.speckle_snr(iq.boxcar_mean(img, 9)) > iq.speckle_snr(img)


def test_speckle_snr_degenerate_input_is_nan():
    assert np.isnan(iq.speckle_snr(np.array([1.0])))
    assert np.isnan(iq.speckle_snr(np.full(100, 3.0)))  # zero variance


def test_pooled_snr_is_biased_by_depth_gain_but_patched_is_not():
    """The reason speckle_snr_patched exists: a depth-varying mean (attenuation
    / TGC) depresses a single pooled mean/std well below the Rayleigh constant,
    while per-patch measurement is unaffected."""
    rng = np.random.default_rng(10)
    img = _rayleigh(rng, 1.0, (256, 256))
    # Apply a realistic depth gain ramp: 20 dB over the region.
    gain = np.linspace(1.0, 10.0, 256)[:, None]
    img = img * gain
    mask = np.ones_like(img, dtype=bool)

    pooled = iq.speckle_snr(img[mask])
    patched = iq.speckle_snr_patched(img, mask, patch=32)
    assert pooled < 1.6, f"expected pooled SNR to be dragged down, got {pooled}"
    assert patched == pytest.approx(iq.RAYLEIGH_SNR, abs=0.15), (
        f"per-patch SNR should recover the Rayleigh constant, got {patched}"
    )


def test_speckle_snr_patched_matches_rayleigh_on_flat_speckle():
    rng = np.random.default_rng(11)
    img = _rayleigh(rng, 2.0, (256, 256))
    snr = iq.speckle_snr_patched(img, np.ones_like(img, dtype=bool), patch=32)
    assert snr == pytest.approx(iq.RAYLEIGH_SNR, abs=0.1)


def test_speckle_snr_patched_skips_masked_out_patches():
    rng = np.random.default_rng(12)
    img = _rayleigh(rng, 1.0, (128, 128))
    mask = np.zeros_like(img, dtype=bool)
    mask[:32, :32] = True  # exactly one valid patch
    assert np.isfinite(iq.speckle_snr_patched(img, mask, patch=32))
    assert np.isnan(iq.speckle_snr_patched(img, np.zeros_like(mask), patch=32))


def test_speckle_snr_patched_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        iq.speckle_snr_patched(np.zeros((8, 8)), np.zeros((8, 9), dtype=bool))


# ── boxcar / structural NCC ──────────────────────────────────────────────────


def test_boxcar_mean_preserves_shape_and_constant():
    img = np.full((32, 48), 7.0)
    out = iq.boxcar_mean(img, 5)
    assert out.shape == img.shape
    assert out == pytest.approx(7.0)


def test_boxcar_k1_is_identity():
    rng = np.random.default_rng(7)
    img = rng.normal(size=(16, 16))
    assert iq.boxcar_mean(img, 1) == pytest.approx(img)


def test_structural_ncc_identical_images_is_one():
    rng = np.random.default_rng(8)
    img = 20 * np.log10(_rayleigh(rng, 1.0, (128, 128)) + 1e-9)
    img -= img.max()
    assert iq.structural_ncc(img, img) == pytest.approx(1.0, abs=1e-6)


def test_structural_ncc_sees_through_speckle_to_shared_structure():
    """Two independent speckle draws over the SAME structure: pixel-wise
    correlation is near zero, structural correlation is high. This is the
    whole reason the WI-4 gate moved off pixel-wise NCC."""
    rng = np.random.default_rng(9)
    zz, xx = np.mgrid[0:192, 0:192]
    # A mild -3 dB inclusion. Deliberately low-contrast: a strong lesion is so
    # dominant that even pixel-wise correlation finds it (measured: a -14 dB
    # disk scores 0.43 pixel-wise), which would not demonstrate anything.
    structure = np.where((zz - 96) ** 2 + (xx - 96) ** 2 < 40**2, 0.7, 1.0)

    def draw():
        img = structure * _rayleigh(rng, 1.0, structure.shape)
        db = 20 * np.log10(img + 1e-9)
        return db - db.max()

    a, b = draw(), draw()
    mask = (a > -40.0) | (b > -40.0)
    pixelwise = np.corrcoef(a[mask], b[mask])[0, 1]
    structural = iq.structural_ncc(a, b, window=15)
    assert pixelwise < 0.15, f"expected decorrelated speckle, got {pixelwise}"
    assert structural > 0.8, f"expected shared structure to survive, got {structural}"
    assert structural > 5 * pixelwise


def test_structural_ncc_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        iq.structural_ncc(np.zeros((8, 8)), np.zeros((8, 9)))
