"""Augmentation policies A and B (PLAN.md WI-11)."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.data import augment as aug

# ── Policy B: never axial flip ───────────────────────────────────────────────


@pytest.fixture
def z_marked_crop():
    """(2, Z=8, X=8, K=4) with a value that identifies which end is z=0."""
    crop = np.zeros((2, 8, 8, 4), dtype=np.float32)
    crop[:, 0, :, :] = 1.0  # shallow end marked
    crop[:, -1, :, :] = -1.0  # deep end marked
    return crop


def test_lateral_flip_never_touches_z_axis(z_marked_crop):
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        out = aug.lateral_flip(z_marked_crop, rng, x_axis=2, prob=0.5)
        assert out[0, 0, 0, 0] == 1.0
        assert out[0, -1, 0, 0] == -1.0


def test_policy_b_never_axial_flips_over_1e4_samples():
    """The literal WI-11 done-condition: no axial flip ever, over 10^4 draws.

    Uses a wide Z with quarter-sized shallow/deep marker bands and a max
    jitter well inside each band, so a +-4 px shift can never move deep
    content into the row being checked for shallow content (and vice versa)
    — isolating "was there an axial flip" from "jitter moved things a bit".
    """
    nz = 40
    crop = np.zeros((2, nz, 8, 4), dtype=np.float32)
    crop[:, : nz // 4, :, :] = 1.0  # shallow quarter
    crop[:, -nz // 4 :, :, :] = -1.0  # deep quarter
    rng = np.random.default_rng(1)
    config = aug.PolicyBConfig(jitter_px=4, flip_prob=0.5)
    for _ in range(10_000):
        out = aug.policy_b(crop, rng, z_axis=1, x_axis=2, config=config)
        assert out[0, 0, :, :].mean() > 0
        assert out[0, -1, :, :].mean() < 0


def test_lateral_flip_actually_flips_sometimes():
    crop = np.arange(8, dtype=np.float32).reshape(1, 1, 8, 1)  # (ch, z, x, k)
    rng = np.random.default_rng(0)
    flipped_seen, unflipped_seen = False, False
    for _ in range(200):
        out = aug.lateral_flip(crop, rng, x_axis=2, prob=0.5)
        if np.array_equal(out[0, 0, :, 0], crop[0, 0, ::-1, 0]):
            flipped_seen = True
        elif np.array_equal(out[0, 0, :, 0], crop[0, 0, :, 0]):
            unflipped_seen = True
    assert flipped_seen and unflipped_seen


def test_spatial_jitter_preserves_shape_and_is_bounded(z_marked_crop):
    rng = np.random.default_rng(2)
    for _ in range(50):
        out = aug.spatial_jitter(z_marked_crop, rng, z_axis=1, x_axis=2, max_px=4)
        assert out.shape == z_marked_crop.shape


def test_spatial_jitter_zero_max_px_is_identity(z_marked_crop):
    rng = np.random.default_rng(3)
    out = aug.spatial_jitter(z_marked_crop, rng, z_axis=1, x_axis=2, max_px=0)
    assert out is z_marked_crop


def test_spatial_jitter_shifts_a_ramp_by_a_bounded_amount():
    ramp = np.arange(20, dtype=np.float32).reshape(1, 20, 1, 1)  # (ch, z, x, k)
    rng = np.random.default_rng(4)
    for _ in range(50):
        out = aug.spatial_jitter(ramp, rng, z_axis=1, x_axis=2, max_px=3)
        # interior values (away from reflect-padded edges) must equal a
        # shifted copy of the ramp for some shift within [-3, 3].
        interior = out[0, 5:15, 0, 0]
        shifts = [np.arange(5 - d, 15 - d) for d in range(-3, 4)]
        assert any(np.allclose(interior, ramp[0, s, 0, 0]) for s in shifts)


# ── Policy A ──────────────────────────────────────────────────────────────


@pytest.fixture
def sector_scene():
    rng = np.random.default_rng(0)
    sums = rng.normal(size=(2, 6, 5, 16)).astype(np.float32)
    occupancy = rng.integers(1, 12, size=(6, 5, 16)).astype(np.uint16)
    return sums, occupancy


def test_random_sector_subset_keeps_a_contiguous_span(sector_scene):
    sums, occ = sector_scene
    rng = np.random.default_rng(0)
    for _ in range(200):
        s, o = aug.random_sector_subset(sums, occ, rng, min_span=8)
        keep_mask = ~np.all(s == 0, axis=tuple(range(s.ndim - 1)))
        n_kept = int(keep_mask.sum())
        assert 8 <= n_kept <= 16
        # circular contiguity: a single arc has exactly 0 (full circle) or 2
        # True/False transitions going around; a fragmented mask has >2.
        transitions = int(np.sum(keep_mask != np.roll(keep_mask, 1)))
        assert transitions in (0, 2)
        np.testing.assert_array_equal(o[..., ~keep_mask], 0)


def test_random_sector_subset_rejects_bad_min_span(sector_scene):
    sums, occ = sector_scene
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="min_span"):
        aug.random_sector_subset(sums, occ, rng, min_span=0)
    with pytest.raises(ValueError, match="min_span"):
        aug.random_sector_subset(sums, occ, rng, min_span=17)


def test_transmit_angle_pair_unavailable_per_plan_10_3b():
    """PLAN.md 10.3(b): the PW stack was not built, so this has no data
    source. It must fail loudly, not silently return a degenerate pair."""
    with pytest.raises(NotImplementedError, match="PW\\(5\\) stack was not built"):
        aug.transmit_angle_pair()


def test_sector_dropout_never_drops_every_sector(sector_scene):
    sums, _ = sector_scene
    rng = np.random.default_rng(0)
    for _ in range(500):
        out = aug.sector_dropout(sums, rng, p=0.99)
        kept = ~np.all(out == 0, axis=tuple(range(out.ndim - 1)))
        assert kept.sum() >= 1


def test_sector_dropout_p_zero_keeps_everything(sector_scene):
    sums, _ = sector_scene
    rng = np.random.default_rng(0)
    out = aug.sector_dropout(sums, rng, p=0.0)
    np.testing.assert_array_equal(out, sums)


def test_tgc_gain_jitter_is_depth_dependent_and_smooth():
    crop = np.ones((2, 40, 3, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    out = aug.tgc_gain_jitter(crop, rng, z_axis=1, max_db=6.0, n_control=4)
    gain_per_depth = out[0, :, 0, 0]
    assert gain_per_depth.std() > 0, "gain must vary with depth"
    assert gain_per_depth.min() > 0, "gain must stay positive (it's a multiplicative factor)"
    # smoothness: no wild sample-to-sample jump given only 4 control points over 40 depths
    assert np.abs(np.diff(gain_per_depth)).max() < 0.2


def test_tgc_gain_jitter_bounded_by_max_db():
    crop = np.ones((1, 50, 1, 1), dtype=np.float32)
    rng = np.random.default_rng(0)
    out = aug.tgc_gain_jitter(crop, rng, z_axis=1, max_db=3.0, n_control=6)
    db = 20 * np.log10(out[0, :, 0, 0])
    assert db.max() <= 3.0 + 1e-4
    assert db.min() >= -3.0 - 1e-4


def test_additive_complex_noise_matches_target_snr_on_average():
    rng = np.random.default_rng(0)
    crop = np.ones((2, 64, 64, 16), dtype=np.float32) * 5.0
    sig_rms = 5.0
    snrs = []
    for _ in range(30):
        out = aug.additive_complex_noise(crop, rng, snr_db_range=(20.0, 20.0))
        noise = out - crop
        noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
        snrs.append(20 * np.log10(sig_rms / noise_rms))
    assert np.mean(snrs) == pytest.approx(20.0, abs=1.0)


def test_random_k_reduction_returns_an_allowed_k(sector_scene):
    sums, occ = sector_scene
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        view, k = aug.random_k_reduction(sums, occ, rng, choices=(16, 8, 4, 2, 1))
        assert k in (16, 8, 4, 2, 1)
        assert view.shape[-1] == k
        seen.add(k)
    assert seen == {16, 8, 4, 2, 1}, "every configured K should appear over 200 draws"


def test_policy_a_output_shape_matches_returned_k(sector_scene):
    sums, occ = sector_scene
    rng = np.random.default_rng(0)
    for _ in range(20):
        view, k = aug.policy_a(sums, occ, rng, z_axis=1)
        assert view.shape == (*sums.shape[:-1], k)
        assert np.all(np.isfinite(view))


def test_policy_a_two_draws_differ():
    """Sanity: the augmentation is actually random, not a no-op."""
    rng = np.random.default_rng(0)
    sums = np.ones((2, 6, 5, 16), dtype=np.float32)
    occ = np.full((6, 5, 16), 4, dtype=np.uint16)
    view1, k1 = aug.policy_a(sums, occ, rng, z_axis=1)
    view2, k2 = aug.policy_a(sums, occ, rng, z_axis=1)
    assert k1 != k2 or not np.allclose(view1, view2)
