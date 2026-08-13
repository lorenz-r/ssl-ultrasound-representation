"""Sanity checks on the checked-in WI-4 reference-validation results.

This does NOT assert the PLAN.md WI-4 gate (NCC >= 0.98, median |dB| <= 1.5)
— that gate is not yet cleared (see the measured numbers and the diagnosis in
`scripts/wi4_reference_validation.py`'s module docstring). These tests only
guard the artefact's structure and the one property that *is* solid: the
negative control (phase rotation disabled) must score worse than the best
real variant, which is the qualitative signature the negative control exists
to demonstrate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

RESULTS = os.environ.get("ZEA_WI4_DATA", "tests/data/wi4_results.json")
pytestmark = pytest.mark.skipif(
    not Path(RESULTS).exists(),
    reason=f"{RESULTS} not present; run scripts/wi4_reference_validation.py",
)


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    data = json.loads(Path(RESULTS).read_text())
    assert data, "no frames in results file"
    return data


def test_all_three_aperture_variants_ran(frames):
    for f in frames:
        assert set(f["variants"]) == {"unlimited", "f1.5", "f1.0"}
        for v in f["variants"].values():
            assert v["n_pixels"] > 0
            assert v["ncc"] == v["ncc"]  # not NaN


def test_best_variant_is_reported_and_consistent(frames):
    for f in frames:
        assert f["best_variant"] in f["variants"]
        best_ncc = max(v["ncc"] for v in f["variants"].values())
        assert f["variants"][f["best_variant"]]["ncc"] == pytest.approx(best_ncc)


def test_negative_control_scores_below_best_real_variant(frames):
    """The one property that must hold regardless of absolute calibration:
    disabling phase rotation must not score *better* than the real signal."""
    for f in frames:
        best_ncc = f["variants"][f["best_variant"]]["ncc"]
        neg_ncc = f["negative_control_no_phase_rotation"]["ncc"]
        assert neg_ncc <= best_ncc


def test_gate_status_is_recorded_honestly(frames):
    """`gate_pass` must reflect the actual PLAN.md WI-4 thresholds, not be
    hardcoded True. Thresholds duplicated from scripts/wi4_reference_validation.py
    (NCC_THRESHOLD, MEDIAN_DB_THRESHOLD) rather than imported, since `scripts`
    is a standalone entry point, not an importable package dependency of tests.
    """
    ncc_threshold, median_db_threshold = 0.98, 1.5
    for f in frames:
        best = f["variants"][f["best_variant"]]
        expected = best["ncc"] >= ncc_threshold and best["median_abs_db"] <= median_db_threshold
        assert f["gate_pass"] == expected
