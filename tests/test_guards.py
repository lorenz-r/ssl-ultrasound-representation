"""The guards must fail loudly. These tests exist to prove they do.

PLAN.md WI-2 done-condition: "both guards are proven live by tests that expect a
hard failure". A guard that is never observed to fire is indistinguishable from a
guard that does not work.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.eval.guards import (
    FrameLeakageError,
    LabelLeakageError,
    assert_no_frame_leakage,
    assert_no_label_leak,
    filter_targets,
)


def test_disjoint_splits_pass():
    assert_no_frame_leakage({"train": ["a", "b"], "test": ["c"]})


def test_shared_frame_id_raises():
    with pytest.raises(FrameLeakageError, match="appears in both"):
        assert_no_frame_leakage({"train": ["a", "b"], "test": ["b", "c"]})


def test_leak_via_crops_of_one_frame_raises():
    """The realistic mode: crops, not frames, land on both sides."""
    frame_ids = np.array(["f0"] * 4 + ["f1"] * 4)
    train_crops, test_crops = frame_ids[:6], frame_ids[6:]  # f1 straddles the cut
    with pytest.raises(FrameLeakageError):
        assert_no_frame_leakage(
            {"train": np.unique(train_crops).tolist(), "test": np.unique(test_crops).tolist()}
        )


@pytest.mark.parametrize("target", ["sos_mean", "sos_std", "phase_error", "sos_mean_frame"])
def test_gt_sos_stage_blocks_physics_targets(target):
    with pytest.raises(LabelLeakageError, match="consumes the ground-truth"):
        assert_no_label_leak("gt_sos", target)


@pytest.mark.parametrize("target", ["contains_cyst", "cyst_fraction", "z_centre"])
def test_gt_sos_stage_permits_tissue_and_geometry(target):
    assert_no_label_leak("gt_sos", target)


@pytest.mark.parametrize(
    "target", ["sos_mean", "phase_error", "contains_cyst", "z_centre", "t0_group"]
)
def test_c0_const_permits_everything(target):
    assert_no_label_leak("c0_const", target)


def test_unknown_stage_is_an_error_not_a_pass():
    """An untagged feature array must not silently probe anything."""
    with pytest.raises(ValueError, match="unknown beamform stage"):
        assert_no_label_leak("bmode_variant_2_untagged", "sos_mean")


def test_filter_targets_drops_only_leaked():
    targets = ["contains_cyst", "sos_mean", "z_centre", "phase_error"]
    assert filter_targets("c0_const", targets) == targets
    assert filter_targets("gt_sos", targets) == ["contains_cyst", "z_centre"]


def test_probe_refuses_leaked_target_before_fitting():
    """The guard is in the probe path, not only in documentation."""
    from ssl_us.eval.probes import probe

    x = np.random.default_rng(0).normal(size=(40, 8))
    y = np.random.default_rng(1).normal(size=40)
    fid = np.array([f"f{i // 4}" for i in range(40)])
    with pytest.raises(LabelLeakageError):
        probe(x, y, fid, "sos_mean", stage="gt_sos", level="frame")
