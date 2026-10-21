"""Augmentation policies A and B (PLAN.md WI-11), batched torch implementation.

Runs on CPU tensors — torch is a core dependency and these need no GPU, even
though training applies the very same functions on the GPU. That is the point
of having one implementation rather than a NumPy copy for tests: what is
verified here is what trains the model.

Layout throughout is the canonical ``(B, 2, Z, X, K)`` for sums and
``(B, Z, X, K)`` for occupancy.
"""

from __future__ import annotations

import pytest
import torch

from ssl_us.data import augment as aug


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Policy B: never axial flip ───────────────────────────────────────────────


@pytest.fixture
def z_marked_crop():
    """(B, 2, Z=8, X=8, K=4) with values identifying which end is z=0."""
    crop = torch.zeros((4, 2, 8, 8, 4))
    crop[:, :, 0, :, :] = 1.0  # shallow end marked
    crop[:, :, -1, :, :] = -1.0  # deep end marked
    return crop


def test_lateral_flip_never_touches_z_axis(z_marked_crop):
    g = gen(0)
    for _ in range(500):
        out = aug.lateral_flip(z_marked_crop, g, prob=0.5)
        assert torch.all(out[:, 0, 0, 0, 0] == 1.0)
        assert torch.all(out[:, 0, -1, 0, 0] == -1.0)


def test_policy_b_never_axial_flips_over_1e4_samples():
    """The literal WI-11 done-condition: no axial flip ever, over 10^4 draws.

    Wide Z with quarter-sized shallow/deep marker bands and a jitter well
    inside each band, so a ±4 px shift can never move deep content into the row
    checked for shallow content — isolating "was there an axial flip" from
    "jitter moved things a bit". Batched, so the 10^4 draws are one call.
    """
    nz, b = 40, 10_000
    crop = torch.zeros((b, 2, nz, 8, 4))
    crop[:, :, : nz // 4, :, :] = 1.0  # shallow quarter
    crop[:, :, -(nz // 4) :, :, :] = -1.0  # deep quarter
    out = aug.policy_b(crop, gen(1), config=aug.PolicyBConfig(jitter_px=4, flip_prob=0.5))
    assert torch.all(out[:, 0, 0, :, :].mean(dim=(1, 2)) > 0)
    assert torch.all(out[:, 0, -1, :, :].mean(dim=(1, 2)) < 0)


def test_lateral_flip_actually_flips_sometimes():
    crop = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8, 1).repeat(200, 1, 1, 1, 1)
    out = aug.lateral_flip(crop, gen(0), prob=0.5)
    flipped = (out[:, 0, 0, :, 0] == crop[0, 0, 0, :, 0].flip(0)).all(dim=1)
    unflipped = (out[:, 0, 0, :, 0] == crop[0, 0, 0, :, 0]).all(dim=1)
    assert bool(flipped.any()) and bool(unflipped.any())


def test_lateral_flip_is_decided_per_sample():
    """A per-batch decision would correlate the augmentation across a batch,
    and VICReg's variance/covariance terms are computed across the batch."""
    crop = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8, 1).repeat(64, 1, 1, 1, 1)
    out = aug.lateral_flip(crop, gen(0), prob=0.5)
    flipped = (out[:, 0, 0, :, 0] == crop[0, 0, 0, :, 0].flip(0)).all(dim=1)
    assert 0 < int(flipped.sum()) < 64


def test_spatial_jitter_preserves_shape(z_marked_crop):
    out = aug.spatial_jitter(z_marked_crop, gen(2), max_px=4)
    assert out.shape == z_marked_crop.shape


def test_spatial_jitter_zero_max_px_is_identity(z_marked_crop):
    assert aug.spatial_jitter(z_marked_crop, gen(3), max_px=0) is z_marked_crop


def test_spatial_jitter_shifts_a_ramp_by_a_bounded_amount():
    ramp = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20, 1, 1).repeat(50, 1, 1, 1, 1)
    out = aug.spatial_jitter(ramp, gen(4), max_px=3)
    # Interior values (away from the reflect-padded edges) must equal the ramp
    # shifted by some amount within [-3, 3].
    for i in range(out.shape[0]):
        interior = out[i, 0, 5:15, 0, 0]
        assert any(
            torch.allclose(interior, torch.arange(5 - d, 15 - d, dtype=torch.float32))
            for d in range(-3, 4)
        ), f"sample {i} is not a bounded shift of the ramp"


def test_spatial_jitter_shifts_differ_across_the_batch():
    ramp = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20, 1, 1).repeat(64, 1, 1, 1, 1)
    out = aug.spatial_jitter(ramp, gen(5), max_px=3)
    assert len({tuple(out[i, 0, :, 0, 0].tolist()) for i in range(64)}) > 1


# ── Policy A ──────────────────────────────────────────────────────────────


@pytest.fixture
def sector_scene():
    g = gen(0)
    sums = torch.randn((4, 2, 6, 5, 16), generator=g)
    occupancy = torch.randint(1, 12, (4, 6, 5, 16), generator=g).float()
    return sums, occupancy


def test_random_sector_subset_keeps_a_contiguous_span(sector_scene):
    sums, occ = sector_scene
    g = gen(0)
    for _ in range(100):
        s, o = aug.random_sector_subset(sums, occ, g, min_span=8)
        keep = ~(s == 0).all(dim=(1, 2, 3))  # (B, K)
        for b in range(keep.shape[0]):
            row = keep[b]
            assert 8 <= int(row.sum()) <= 16
            # Circular contiguity: one arc has exactly 0 or 2 transitions.
            transitions = int((row != row.roll(1)).sum())
            assert transitions in (0, 2)
            assert torch.all(o[b][..., ~row] == 0)


def test_random_sector_subset_spans_differ_across_the_batch(sector_scene):
    sums, occ = sector_scene
    s, _ = aug.random_sector_subset(sums, occ, gen(7), min_span=4)
    keep = ~(s == 0).all(dim=(1, 2, 3))
    assert len({tuple(keep[b].tolist()) for b in range(keep.shape[0])}) > 1


def test_random_sector_subset_rejects_bad_min_span(sector_scene):
    sums, occ = sector_scene
    with pytest.raises(ValueError, match="min_span"):
        aug.random_sector_subset(sums, occ, gen(0), min_span=0)
    with pytest.raises(ValueError, match="min_span"):
        aug.random_sector_subset(sums, occ, gen(0), min_span=17)


def test_transmit_angle_pair_unavailable_per_plan_10_3b():
    """PLAN.md 10.3(b): the PW stack was not built, so this has no data
    source. It must fail loudly, not silently return a degenerate pair."""
    with pytest.raises(NotImplementedError, match="PW\\(5\\) stack was not built"):
        aug.transmit_angle_pair()


def test_sector_dropout_never_drops_every_sector(sector_scene):
    sums, _ = sector_scene
    g = gen(0)
    for _ in range(200):
        out = aug.sector_dropout(sums, g, p=0.99)
        kept = ~(out == 0).all(dim=(1, 2, 3))
        assert torch.all(kept.sum(dim=1) >= 1)


def test_sector_dropout_p_zero_keeps_everything(sector_scene):
    sums, _ = sector_scene
    torch.testing.assert_close(aug.sector_dropout(sums, gen(0), p=0.0), sums)


def test_tgc_gain_jitter_is_depth_dependent_and_smooth():
    crop = torch.ones((2, 2, 40, 3, 4))
    out = aug.tgc_gain_jitter(crop, gen(0), max_db=6.0, n_control=4)
    gain = out[0, 0, :, 0, 0]
    assert gain.std() > 0, "gain must vary with depth"
    assert gain.min() > 0, "gain must stay positive (it's a multiplicative factor)"
    # Smoothness: no wild jump given only 4 control points over 40 depths.
    assert gain.diff().abs().max() < 0.2


def test_tgc_gain_jitter_bounded_by_max_db():
    crop = torch.ones((3, 1, 50, 1, 1))
    out = aug.tgc_gain_jitter(crop, gen(0), max_db=3.0, n_control=6)
    db = 20 * torch.log10(out[:, 0, :, 0, 0])
    assert db.max() <= 3.0 + 1e-4
    assert db.min() >= -3.0 - 1e-4


def test_tgc_curves_differ_across_the_batch():
    crop = torch.ones((32, 1, 40, 1, 1))
    out = aug.tgc_gain_jitter(crop, gen(0), max_db=6.0)
    curves = out[:, 0, :, 0, 0]
    assert not torch.allclose(curves[0], curves[1])


def test_additive_complex_noise_matches_target_snr_on_average():
    crop = torch.full((30, 2, 64, 64, 16), 5.0)
    out = aug.additive_complex_noise(crop, gen(0), snr_db_range=(20.0, 20.0))
    noise = (out - crop).double()
    noise_rms = noise.flatten(1).pow(2).mean(dim=1).sqrt()
    snr = 20 * torch.log10(5.0 / noise_rms)
    assert snr.mean().item() == pytest.approx(20.0, abs=1.0)


def test_additive_noise_snr_is_drawn_per_sample():
    crop = torch.full((64, 2, 16, 16, 4), 3.0)
    out = aug.additive_complex_noise(crop, gen(0), snr_db_range=(5.0, 30.0))
    rms = (out - crop).flatten(1).pow(2).mean(dim=1).sqrt()
    assert rms.std() > 0, "every sample got the same SNR"


def test_random_k_reduction_returns_an_allowed_k(sector_scene):
    sums, occ = sector_scene
    g = gen(0)
    seen = set()
    for _ in range(200):
        view, ks = aug.random_k_reduction(sums, occ, g, choices=(16, 8, 4, 2, 1), expand_to=16)
        assert view.shape[-1] == 16  # expanded to a fixed width
        seen.update(int(k) for k in ks)
    assert seen == {16, 8, 4, 2, 1}, "every configured K should appear over 200 draws"


def test_random_k_reduction_expands_by_repeating(sector_scene):
    """Zero-padding would let the model read k off the zero channels."""
    sums, occ = sector_scene
    view, ks = aug.random_k_reduction(sums, occ, gen(0), choices=(2,), expand_to=16)
    assert view.shape[-1] == 16
    assert torch.all(ks == 2)
    # Each of the 2 real sectors is repeated 8 times.
    torch.testing.assert_close(view[..., 0], view[..., 7])
    assert not torch.allclose(view[..., 0], view[..., 8])


def test_policy_a_output_has_the_arm_width(sector_scene):
    sums, occ = sector_scene
    g = gen(0)
    for _ in range(20):
        view, ks = aug.policy_a(sums, occ, g, expand_to=4)
        assert view.shape == (*sums.shape[:-1], 4)
        assert torch.isfinite(view).all()
        assert all(int(k) in (16, 8, 4, 2, 1) for k in ks)


def test_policy_a_two_draws_differ(sector_scene):
    """Sanity: the augmentation is actually random, not a no-op."""
    sums, occ = sector_scene
    g = gen(0)
    v1, _ = aug.policy_a(sums, occ, g, expand_to=4)
    v2, _ = aug.policy_a(sums, occ, g, expand_to=4)
    assert not torch.allclose(v1, v2)


def test_policy_a_is_reproducible_for_a_seed(sector_scene):
    sums, occ = sector_scene
    a, _ = aug.policy_a(sums, occ, gen(11), expand_to=4)
    b, _ = aug.policy_a(sums, occ, gen(11), expand_to=4)
    torch.testing.assert_close(a, b)
