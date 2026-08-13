"""Probe protocol: grouped folds, capacity control, frame pooling, honest CIs."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.eval.probes import CAPACITY_DIMS, capacity_control, pool_to_frames, probe


@pytest.fixture
def signal_data():
    """Crops whose features linearly encode a frame-level target."""
    rng = np.random.default_rng(0)
    n_frames, per_frame = 40, 6
    y_frame = rng.normal(size=n_frames)
    fid = np.repeat([f"f{i:03d}" for i in range(n_frames)], per_frame)
    y = np.repeat(y_frame, per_frame)
    x = np.concatenate(
        [y[:, None] * 2.0 + rng.normal(scale=0.3, size=(len(y), 1)),
         rng.normal(size=(len(y), 15))],
        axis=1,
    )
    return x, y, fid


def test_capacity_control_caps_dimensionality():
    x = np.random.default_rng(0).normal(size=(200, 512))
    assert capacity_control(x).shape == (200, CAPACITY_DIMS)


def test_capacity_control_passes_narrow_features_through():
    x = np.random.default_rng(0).normal(size=(200, 8))
    assert capacity_control(x).shape == (200, 8)


def test_pool_to_frames_collapses_crops():
    x = np.arange(12, dtype=float).reshape(6, 2)
    fid = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    xp, yp, uniq = pool_to_frames(x, fid, y)
    assert xp.shape == (3, 2)
    assert list(uniq) == ["a", "b", "c"]
    np.testing.assert_allclose(yp, [1.0, 2.0, 3.0])


def test_frame_level_probe_recovers_signal(signal_data):
    x, y, fid = signal_data
    r = probe(x, y, fid, "sos_mean_frame", stage="c0_const", level="frame")
    assert r.score > 0.7
    assert r.n_eff == 40  # frames, not crops
    assert r.score_lo < r.score < r.score_hi


def test_probe_on_noise_is_near_zero():
    rng = np.random.default_rng(1)
    n_frames, per_frame = 40, 6
    fid = np.repeat([f"f{i:03d}" for i in range(n_frames)], per_frame)
    x = rng.normal(size=(n_frames * per_frame, 16))
    y = np.repeat(rng.normal(size=n_frames), per_frame)
    r = probe(x, y, fid, "sos_mean_frame", stage="c0_const", level="frame")
    assert r.score < 0.3, f"noise probe scored {r.score}"


def test_n_eff_is_frames_for_frame_level_targets(signal_data):
    """The single most important protocol property: 923 effective, not 144k."""
    x, y, fid = signal_data
    frame = probe(x, y, fid, "sos_mean_frame", stage="c0_const", level="frame")
    crop = probe(x, y, fid, "sos_mean", stage="c0_const", level="crop")
    assert frame.n_eff == len(np.unique(fid))
    assert crop.n_eff == len(fid)
    assert frame.n_eff < crop.n_eff


def test_logistic_probe_reports_balanced_accuracy():
    rng = np.random.default_rng(2)
    n_frames, per_frame = 40, 6
    fid = np.repeat([f"f{i:03d}" for i in range(n_frames)], per_frame)
    y = (rng.random(n_frames * per_frame) < 0.4).astype(int)
    x = y[:, None] * 1.5 + rng.normal(scale=0.5, size=(len(y), 6))
    r = probe(x, y, fid, "contains_cyst", stage="c0_const", level="crop", kind="logistic")
    assert r.metric == "balanced_accuracy"
    assert r.score > 0.7
    assert "auc" in r.secondary


def test_grouped_folds_do_not_split_a_frame():
    """A frame duplicated across the fold boundary would inflate every score."""
    rng = np.random.default_rng(3)
    n_frames, per_frame = 20, 8
    fid = np.repeat([f"f{i:03d}" for i in range(n_frames)], per_frame)
    # per-frame constant noise: only leakage could make this predictable
    nuisance = np.repeat(rng.normal(size=n_frames), per_frame)
    x = nuisance[:, None] + rng.normal(scale=1e-6, size=(len(fid), 4))
    y = np.repeat(rng.normal(size=n_frames), per_frame)
    r = probe(x, y, fid, "sos_mean_frame", stage="c0_const", level="frame")
    assert r.score < 0.5, "grouped CV should not recover an unrelated per-frame code"
