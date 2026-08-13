"""Sanity checks on the checked-in WI-4 reference-validation results.

The gate these guard is the **rev-5** one (PLAN.md WI-4): reference-relative
gCNR and speckle statistics. The original pixel-wise NCC >= 0.98 gate was
withdrawn as unachievable by any beamformer — NVIDIA's own two shipped
reconstructions of the same frame score mean NCC 0.197 against each other
(`wi4_reference_selfconsistency.json`) — so `ncc_pixelwise` is still recorded
in the results, but as a diagnostic only, and nothing here asserts on it.

These tests guard the artefact's structure and that `gate_pass` is derived
from the thresholds rather than hardcoded.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

RESULTS = os.environ.get("ZEA_WI4_DATA", "tests/data/wi4_results.json")
SELFCONSISTENCY = "tests/data/wi4_reference_selfconsistency.json"

pytestmark = pytest.mark.skipif(
    not Path(RESULTS).exists(),
    reason=f"{RESULTS} not present; run scripts/wi4_reference_validation.py",
)

# Duplicated from scripts/wi4_reference_validation.py rather than imported,
# since `scripts` is a standalone entry point, not an importable package.
GCNR_VS_REF_TOLERANCE = 0.10
SPECKLE_SNR_RANGE = (1.60, 2.15)
SPECKLE_SNR_VS_REF_TOLERANCE = 0.25
NEGATIVE_CONTROL_MIN_REF_GCNR = 0.30


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
            assert math.isfinite(v["speckle_snr"]), "speckle SNR is always measurable"


def test_gcnr_present_exactly_when_frame_has_an_inclusion(frames):
    """Some frames ship an empty segmentation map; those run speckle checks only."""
    for f in frames:
        has = f["has_inclusion"]
        assert has == (f["n_lesion_px"] > 0)
        for v in f["variants"].values():
            assert math.isfinite(v["gcnr"]) == has


def test_best_variant_is_reported_and_consistent(frames):
    """Ranked by gCNR where there is contrast, else by closeness of speckle
    statistics to the reference — matching the script's own selection rule."""
    for f in frames:
        assert f["best_variant"] in f["variants"]
        if f["has_inclusion"]:
            best = max(v["gcnr"] for v in f["variants"].values())
            assert f["variants"][f["best_variant"]]["gcnr"] == pytest.approx(best)
        else:
            ref = f["reference_speckle_snr"]
            best = min(abs(v["speckle_snr"] - ref) for v in f["variants"].values())
            chosen = abs(f["variants"][f["best_variant"]]["speckle_snr"] - ref)
            assert chosen == pytest.approx(best)


def test_negative_control_degrades_contrast(frames):
    """The point of WI-4: disabling baseband phase rotation must destroy
    contrast. Only asserted where the reference itself resolves the inclusion —
    below that there is no contrast to destroy and the comparison is noise."""
    checked = 0
    for f in frames:
        if not f["has_inclusion"] or f["reference_gcnr"] < NEGATIVE_CONTROL_MIN_REF_GCNR:
            continue
        best = f["variants"][f["best_variant"]]["gcnr"]
        neg = f["negative_control_no_phase_rotation"]["gcnr"]
        assert neg < best, f"frame {f['idx']}: negative control {neg} !< {best}"
        checked += 1
    assert checked > 0, "no frame exercised the negative control"


def test_gate_status_is_recorded_honestly(frames):
    """`gate_pass` must follow from the recorded numbers, not be hardcoded."""
    snr_lo, snr_hi = SPECKLE_SNR_RANGE
    for f in frames:
        best = f["variants"][f["best_variant"]]
        expected = (
            snr_lo <= best["speckle_snr"] <= snr_hi
            and abs(best["speckle_snr"] - f["reference_speckle_snr"])
            <= SPECKLE_SNR_VS_REF_TOLERANCE
        )
        if f["has_inclusion"]:
            expected = expected and (
                abs(best["gcnr"] - f["reference_gcnr"]) <= GCNR_VS_REF_TOLERANCE
            )
        if f["negative_control_applicable"]:
            expected = expected and (
                f["negative_control_no_phase_rotation"]["gcnr"] < best["gcnr"]
            )
        assert f["gate_pass"] == expected, f"frame {f['idx']}"


@pytest.mark.skipif(
    not Path(SELFCONSISTENCY).exists(), reason=f"{SELFCONSISTENCY} not present"
)
def test_reference_selfconsistency_evidence_still_supports_withdrawing_old_gate():
    """Guards the evidence base for the rev-5 gate rewrite: if NVIDIA's own two
    reconstructions ever did agree at 0.98, the old gate would deserve
    revisiting."""
    data = json.loads(Path(SELFCONSISTENCY).read_text())
    assert data["n_frames"] >= 5
    assert data["max_ncc"] < 0.98, (
        "reference reconstructions now agree at the old gate's threshold — "
        "re-read PLAN.md WI-4 before trusting the rev-5 gate's justification"
    )
