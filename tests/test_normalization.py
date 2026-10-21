"""Dataset-level normalization (PLAN.md WI-7).

Both tests exist specifically to prevent regressing to the old repo's
`normalize_frame` (per-depth RMS division, PLAN.md 1.3), which is exactly
the operation the new pipeline forbids: it deletes the attenuation signal
that PLAN.md 4.5c is built to probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.data import normalization as norm


@pytest.fixture
def decaying_frames():
    """Synthetic (N, Z, X, K, 2) data with a known exponential depth decay
    and random phase — |I+jQ| is then exactly the decay curve, independent
    of the random phase, which is what makes this a clean test fixture."""
    rng = np.random.default_rng(0)
    n_frames, nz, nx, k = 20, 40, 8, 4
    z = np.arange(nz)
    amplitude = 10.0 * np.exp(-0.05 * z)  # decays ~2 orders of magnitude over nz
    phase = rng.uniform(0, 2 * np.pi, size=(n_frames, nz, nx, k))
    a = amplitude[None, :, None, None]
    iq = np.stack([a * np.cos(phase), a * np.sin(phase)], axis=-1).astype(np.float32)
    return iq, amplitude


def test_complex_magnitude_recovers_known_amplitude(decaying_frames):
    iq, amplitude = decaying_frames
    mag = norm.complex_magnitude(iq, channel_axis=-1)
    expected = np.broadcast_to(amplitude[None, :, None, None], mag.shape)
    np.testing.assert_allclose(mag, expected, rtol=1e-5)


def test_per_depth_rms_is_not_flat(decaying_frames):
    """The measured decay must survive — this is the axis PLAN.md 4.5c probes."""
    iq, amplitude = decaying_frames
    profile = norm.per_depth_rms(iq, depth_axis=1, channel_axis=-1)
    print(f"measured per-depth RMS: shallow={profile[0]:.3f} deep={profile[-1]:.3f}")
    assert profile.shape == amplitude.shape
    np.testing.assert_allclose(profile, amplitude, rtol=1e-5)
    # explicitly not flat: shallow must be far brighter than deep
    assert profile[0] / profile[-1] > 5.0


def test_apply_normalization_divides_by_exactly_one_scalar(decaying_frames):
    """The regression guard: normalizing must scale every depth by the *same*
    factor, not correct each depth toward a common level (old `normalize_frame`
    behaviour)."""
    iq, _ = decaying_frames
    scalar = norm.dataset_rms(iq, channel_axis=-1)
    normalized = norm.apply_normalization(iq, scalar)

    profile_before = norm.per_depth_rms(iq, depth_axis=1, channel_axis=-1)
    profile_after = norm.per_depth_rms(normalized, depth_axis=1, channel_axis=-1)

    # a single global divide: the ratio before/after must be the *same*
    # constant at every depth (namely `scalar`), not depth-dependent.
    ratios = profile_before / profile_after
    np.testing.assert_allclose(ratios, scalar, rtol=1e-5)
    assert np.ptp(ratios) < 1e-3 * scalar, "ratio varies across depth -> not a single scalar divide"

    # and the decay itself must still be there after normalizing
    assert profile_after[0] / profile_after[-1] > 5.0


def test_normalization_does_not_flatten_the_depth_profile(decaying_frames):
    """Named regression test for the exact old bug: dividing by a *per-depth*
    RMS (rather than one dataset-level scalar) would make this profile flat —
    assert that is NOT what happens.
    """
    iq, _ = decaying_frames
    scalar = norm.dataset_rms(iq, channel_axis=-1)
    normalized = norm.apply_normalization(iq, scalar)
    profile_after = norm.per_depth_rms(normalized, depth_axis=1, channel_axis=-1)

    # the old, forbidden operation: divide each depth by its own RMS
    profile_before = norm.per_depth_rms(iq, depth_axis=1, channel_axis=-1)
    old_bug_result = profile_before / profile_before  # == ones, i.e. perfectly flat

    assert not np.allclose(profile_after, profile_after[0]), "profile is flat: attenuation deleted"
    assert np.allclose(old_bug_result, 1.0)  # sanity: that really is what flat looks like


def test_dataset_rms_sampling_estimates_the_population_value(decaying_frames):
    iq, _ = decaying_frames
    full = norm.dataset_rms(iq, channel_axis=-1)
    sampled = norm.dataset_rms(iq, channel_axis=-1, sample_size=10, seed=0)
    assert sampled == pytest.approx(full, rel=0.3)


def test_compute_normalization_stats_bundles_both_artifacts(decaying_frames):
    iq, _ = decaying_frames
    stats = norm.compute_normalization_stats(iq, depth_axis=1, channel_axis=-1, sample_size=10)
    assert stats.n_frames_sampled == 10
    assert stats.scalar_rms > 0
    assert stats.per_depth_rms_profile.shape == (iq.shape[1],)
    d = stats.to_manifest_dict()
    assert set(d) == {"scalar_rms", "per_depth_rms_profile", "n_frames_sampled"}
    assert isinstance(d["per_depth_rms_profile"], list)  # JSON-serializable


def test_complex_magnitude_rejects_wrong_channel_size():
    with pytest.raises(ValueError, match="expected size-2"):
        norm.complex_magnitude(np.zeros((3, 4, 3)), channel_axis=-1)
