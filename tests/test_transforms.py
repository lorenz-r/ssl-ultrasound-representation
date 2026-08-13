"""K-ladder reduction + occupancy normalization (PLAN.md WI-5).

The property that must never break silently: every K still produces *an*
image, so a wrong operation order doesn't crash — it just quietly corrupts
the deep-K numbers (PLAN.md 5.3, risk #2). These tests exist so that can't
happen without a red test.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.data import geometry as geo
from ssl_us.data import transforms as tf


def _direct_sector_sums(element_values: np.ndarray, sector_id: np.ndarray, k: int) -> np.ndarray:
    """Sum ``element_values`` (n_el,) into K bins by ``sector_id`` (..., n_el),
    independent of `reduce_ladder` — the ground truth these tests check against.
    """
    out = np.zeros((*sector_id.shape[:-1], k), dtype=np.float64)
    for b in range(k):
        out[..., b] = np.where(sector_id == b, element_values, 0.0).sum(axis=-1)
    return out


@pytest.fixture(scope="module")
def toy_scene():
    """A small real grid + a synthetic per-element signal, so raw sector sums
    are physically meaningful (not just random numbers)."""
    z_mm = np.linspace(1.0, 30.0, 12)
    x_mm = np.linspace(-15.0, 15.0, 10)
    rng = np.random.default_rng(0)
    element_values = rng.normal(size=geo.N_ELEMENTS)

    geom16 = sec.build_sector_geometry(k=16, f_number=1.0, z_mm=z_mm, x_mm=x_mm)
    sums16 = _direct_sector_sums(element_values, geom16.sector_id, 16)
    return geom16, sums16, element_values


def test_reduce_ladder_matches_direct_binning_at_every_k(toy_scene):
    """Reducing raw K=16 sums pairwise must equal a direct K-way bin, for
    every K on the ladder (PLAN.md WI-5 done-condition #1)."""
    geom16, sums16, element_values = toy_scene
    for k in (8, 4, 2, 1):
        reduced = tf.reduce_ladder(sums16, target_k=k)
        geom_k = sec.build_sector_geometry(k=k, f_number=1.0, z_mm=geom16.z_mm, x_mm=geom16.x_mm)
        direct = _direct_sector_sums(element_values, geom_k.sector_id, k)
        np.testing.assert_allclose(reduced, direct, rtol=1e-10, atol=1e-10)


def test_reduce_ladder_target_equal_to_k_is_identity():
    x = np.arange(16, dtype=np.float64).reshape(1, 16)
    np.testing.assert_array_equal(tf.reduce_ladder(x, target_k=16), x)


@pytest.mark.parametrize("k", [3, 5, 6, 15])
def test_reduce_ladder_rejects_non_power_of_two_k(k):
    with pytest.raises(ValueError, match="power of two"):
        tf.reduce_ladder(np.ones((1, k)), target_k=1)


def test_reduce_ladder_rejects_target_not_dividing_k():
    with pytest.raises(ValueError, match="power of two"):
        tf.reduce_ladder(np.ones((1, 16)), target_k=3)


def test_reduce_ladder_rejects_target_larger_than_k():
    with pytest.raises(ValueError, match="cannot exceed"):
        tf.reduce_ladder(np.ones((1, 4)), target_k=8)


def test_normalize_mean_mode():
    sums = np.array([[10.0, 20.0, 0.0]])
    occ = np.array([[2, 5, 0]])
    out = tf.normalize(sums, occ, mode="mean")
    np.testing.assert_allclose(out, [[5.0, 4.0, 0.0]])


def test_normalize_rms_mode():
    sums = np.array([[10.0, 20.0]])
    occ = np.array([[4, 16]])
    out = tf.normalize(sums, occ, mode="rms")
    np.testing.assert_allclose(out, [[5.0, 5.0]])


def test_normalize_empty_sector_is_zero_not_nan():
    sums = np.array([[7.0, 0.0]])
    occ = np.array([[0, 0]])
    out = tf.normalize(sums, occ, mode="mean")
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [[0.0, 0.0]])


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown normalization mode"):
        tf.normalize(np.ones((1, 2)), np.ones((1, 2)), mode="bogus")


def test_reduce_then_normalize_is_the_correct_order(toy_scene):
    """`reduce_and_normalize` must equal ground-truth direct-binned sums,
    normalized once by the direct-binned occupancy — the property the ladder
    exists to preserve."""
    geom16, sums16, element_values = toy_scene
    k = 4
    got = tf.reduce_and_normalize(sums16, geom16.occupancy, target_k=k, mode="mean")

    geom_k = sec.build_sector_geometry(k=k, f_number=1.0, z_mm=geom16.z_mm, x_mm=geom16.x_mm)
    direct_sums = _direct_sector_sums(element_values, geom_k.sector_id, k)
    expected = tf.normalize(direct_sums, geom_k.occupancy, mode="mean")
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_normalize_then_reduce_diverges_from_the_correct_order():
    """The bug this module exists to prevent: normalizing per-sector *before*
    merging sectors does not equal merging raw sums and normalizing once,
    whenever the two merged sectors have unequal occupancy.
    """
    sums = np.array([[10.0, 10.0, 10.0, 10.0]])  # K=4, equal raw sums
    occ = np.array([[1, 4, 2, 2]])  # unequal occupancy -> the two orders must diverge

    correct = tf.reduce_and_normalize(sums, occ, target_k=2, mode="mean")

    # naive wrong order: normalize at K=4 first, then pairwise-sum the *normalized* values
    normalized_first = tf.normalize(sums, occ, mode="mean")
    wrong = normalized_first.reshape(1, 2, 2).sum(axis=-1)

    assert not np.allclose(correct, wrong)
    # ground truth for the first merged pair (sectors 0+1): sums 10+10=20 over occ 1+4=5 -> 4.0
    assert correct[0, 0] == pytest.approx(20.0 / 5.0)
    # the wrong order instead gives 10/1 + 10/4 = 12.5 for that same pair
    assert wrong[0, 0] == pytest.approx(10.0 / 1.0 + 10.0 / 4.0)


def test_empty_sectors_round_trip_through_reduce_and_normalize_without_nan():
    """Shallow depths have many empty K=16 sectors (PLAN.md 3.2); the runtime
    transform must never produce NaN/inf when reducing and normalizing them."""
    z_mm = np.array([1.0, 2.0])  # very shallow -> sparse/empty sectors at K=16
    x_mm = np.linspace(-15.0, 15.0, 8)
    geom = sec.build_sector_geometry(k=16, f_number=1.0, z_mm=z_mm, x_mm=x_mm)
    rng = np.random.default_rng(1)
    element_values = rng.normal(size=geo.N_ELEMENTS)
    sums = _direct_sector_sums(element_values, geom.sector_id, 16)

    for k in (16, 8, 4, 2, 1):
        out = tf.reduce_and_normalize(sums, geom.occupancy, target_k=k, mode="mean")
        assert np.all(np.isfinite(out)), f"NaN/inf at K={k}"


# ── numpy / torch equivalence ────────────────────────────────────────────────


def test_torch_ladder_matches_numpy_at_every_level():
    """`reduce_and_normalize_torch` exists so the GPU training path avoids a
    device round-trip, which makes it a second implementation of a rule that
    must not drift. This is the guard: the NumPy version (used by the store's
    read path) and the torch version must agree at every rung of the ladder.
    """
    import torch

    rng = np.random.default_rng(0)
    sums = rng.normal(size=(3, 2, 8, 9, 16)).astype(np.float32)
    occ = rng.integers(0, 10, size=(3, 8, 9, 16)).astype(np.float32)
    for k in (16, 8, 4, 2, 1):
        want = tf.reduce_and_normalize(sums, occ[:, None], target_k=k, axis=-1)
        got = tf.reduce_and_normalize_torch(
            torch.from_numpy(sums), torch.from_numpy(occ), target_k=k
        ).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_torch_ladder_zeroes_empty_sectors_like_numpy():
    """0/0 must be exactly 0, not NaN — an empty sector had nothing summed in."""
    import torch

    sums = np.zeros((1, 2, 2, 2, 4), dtype=np.float32)
    occ = np.zeros((1, 2, 2, 4), dtype=np.float32)
    got = tf.reduce_and_normalize_torch(
        torch.from_numpy(sums), torch.from_numpy(occ), target_k=2
    ).numpy()
    assert np.isfinite(got).all() and not got.any()


def test_torch_ladder_rejects_a_non_dividing_target():
    import torch

    with pytest.raises(ValueError, match="must divide"):
        tf.reduce_and_normalize_torch(torch.zeros(1, 2, 2, 2, 16), torch.zeros(1, 2, 2, 16), 3)
