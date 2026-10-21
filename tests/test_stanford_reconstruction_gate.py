"""Sanity checks on the checked-in WI-16 reference-validation results — the
Stanford analogue of `test_validate_nvidia_reconstruction.py`.

Skipped until `scripts/beamform/validate_stanford_reconstruction.py` has actually been run
(it needs the real ~9 GB local Stanford store and non-trivial beamforming
time — see CLUSTER_HANDOFF.md for why that has to happen on `12g`/`24g`, not
in a unit test). What's guarded here is the artefact's structure and that
`gate_pass` is derived from the recorded checks rather than hardcoded — the
same split WI-4's own results-sanity test makes.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

RESULTS = os.environ.get("ZEA_WI16_DATA", "results/gates/stanford_reconstruction.json")

pytestmark = pytest.mark.skipif(
    not Path(RESULTS).exists(),
    reason=f"{RESULTS} not present; run scripts/beamform/validate_stanford_reconstruction.py",
)


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    data = json.loads(Path(RESULTS).read_text())
    assert data, "no frames in results file"
    return data


def test_both_grid_variants_ran(frames):
    for f in frames:
        assert set(f["variants"]) == {"mm_matched", "lambda_matched"}
        for variant_scores in f["variants"].values():
            assert set(variant_scores) == {"unlimited", "f1.5", "f1.0"}
            for s in variant_scores.values():
                assert math.isfinite(s["speckle_snr"]), "speckle SNR is always measurable"


def test_negative_control_recorded_for_both_variants(frames):
    for f in frames:
        for name in ("mm_matched", "lambda_matched"):
            assert "negative_control" in f[name]
            assert math.isfinite(f[name]["negative_control"]["speckle_snr"])


def test_cross_variant_structural_ncc_is_recorded_and_finite(frames):
    """The check specific to WI-16 (module docstring of the validation
    script): agreement between two independently-gridded reconstructions of
    the same raw data stands in for WI-4's reference comparison."""
    for f in frames:
        assert math.isfinite(f["cross_variant_structural_ncc"]), f["idx"]


def test_gate_status_is_recorded_honestly(frames):
    """`gate_pass` must follow from the recorded per-check booleans, not be
    hardcoded — mirrors WI-4's own guard."""
    for f in frames:
        all_checks = list(f["checks"].values())
        for name in ("mm_matched", "lambda_matched"):
            all_checks += list(f[name]["checks"].values())
        assert f["gate_pass"] == all(all_checks), f["idx"]


def test_negative_control_is_graded_on_structure_not_speckle_snr():
    """The withdrawn check, pinned so it cannot come back.

    Disabling phase rotation makes the aperture sum a circular complex
    Gaussian, whose envelope is *exactly* Rayleigh — so the broken
    reconstruction sits closer to the Rayleigh ideal than the correct one, and
    a check on "distance from 1.913" reports a working beamformer as broken.
    Measured over 12 frames it did exactly that, passing on 1 of them. What is
    gated instead is loss of structural agreement.
    """
    from pathlib import Path as _Path

    src = _Path("scripts/beamform/validate_stanford_reconstruction.py").read_text()
    assert "NEGATIVE_CONTROL_MIN_NCC_DROP" in src
    # The negative-control verdict must not be a speckle-SNR comparison again.
    verdict = src.split('"negative_control_degrades"')[1].split("\n\n")[0]
    assert "RAYLEIGH_SNR" not in verdict
    assert "drop" in verdict


def test_negative_control_records_the_structural_numbers(frames):
    for f in frames:
        for name in ("mm_matched", "lambda_matched"):
            neg = f[name]["negative_control"]
            assert math.isfinite(neg["structural_ncc_vs_best"]), f["idx"]
            assert math.isfinite(neg["structural_ncc_drop_vs_cross_variant"]), f["idx"]
            # The corroborating signature is recorded even though it is not gated.
            assert math.isfinite(neg["db_contrast_drop"]), f["idx"]


def test_the_negative_control_check_follows_from_the_recorded_drop(frames):
    """The pass/fail must be derivable from the numbers in the same file — a
    check whose inputs are not recorded cannot be re-audited later."""
    from pathlib import Path as _Path

    src = _Path("scripts/beamform/validate_stanford_reconstruction.py").read_text()
    threshold = float(
        src.split("NEGATIVE_CONTROL_MIN_NCC_DROP = ")[1].split("\n")[0]
    )
    for f in frames:
        for name in ("mm_matched", "lambda_matched"):
            drop = f[name]["negative_control"]["structural_ncc_drop_vs_cross_variant"]
            assert f[name]["checks"]["negative_control_degrades"] == (
                drop >= threshold
            ), (f["idx"], name, drop)


def test_a_grid_change_perturbs_structure_far_less_than_losing_phase_rotation(frames):
    """The claim the whole gate rests on: two correct reconstructions of one
    scene on different grids agree much more than a correct and a
    phase-broken reconstruction on the *same* grid do. If that ordering ever
    inverts, the cross-variant check has stopped standing in for WI-4's
    reference comparison."""
    for f in frames:
        cross = f["cross_variant_structural_ncc"]
        for name in ("mm_matched", "lambda_matched"):
            assert cross > f[name]["negative_control"]["structural_ncc_vs_best"], (
                f["idx"], name
            )
