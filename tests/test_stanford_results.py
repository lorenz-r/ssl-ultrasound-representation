"""Sanity checks on the checked-in WI-16 transfer matrix — the Stanford
analogue of `test_nvidia_matrix.py`'s guards over WI-13's own artefacts.

These do not re-derive the science; they guard the properties that make the
tables *readable*, each of which has already failed once or would fail
silently:

- every column records which amplitude scale produced it (§14.3 — the same
  numbers mean different things matched vs raw);
- the unrunnable axis carries its reason rather than being one fewer row;
- crop centres lie inside the Stanford grid, not NVIDIA's 0-40 mm one;
- the headline finding (Policy B carries no lateral position) holds in every
  cell, which is what makes it a finding rather than one lucky column.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path("results/stanford")
# Matrix columns only. The glob deliberately excludes the qualitative and
# RankMe artefacts, which share the `<variant>_k<k>` prefix but have a
# different schema — matching them here made five unrelated tests fail on a
# missing key rather than on anything real.
_OTHER_KINDS = ("_qual", "_rankme")
COLUMNS = sorted(
    p for p in DATA.glob("*matched_k*.json")
    if not any(kind in p.name for kind in _OTHER_KINDS)
)

pytestmark = pytest.mark.skipif(
    not COLUMNS, reason="no Stanford matrix columns; run scripts/eval/stanford_matrix.py"
)

# PLAN.md §3.6 rev-6: the corpus's real signal extent.
STANFORD_DEPTH_MAX_MM = 19.9


@pytest.fixture(scope="module")
def columns() -> list[dict]:
    return [json.loads(p.read_text()) for p in COLUMNS]


def test_every_column_records_its_amplitude_scale(columns):
    """§14.3: matched and raw are different experiments. A column that does not
    say which it is cannot be compared to any other."""
    for c in columns:
        assert c["intensity_scale_mode"] in {"matched", "raw"}, c["variant"]
        assert c["intensity_scale"] > 0
        if c["intensity_scale_mode"] == "raw":
            assert c["intensity_scale"] == 1.0


def test_the_unrunnable_axis_carries_its_reason(columns):
    for c in columns:
        assert "source_file" in c["not_run"]
        assert "session_id" in c["not_run"]["source_file"]


def test_deep_k_columns_say_why_they_have_no_full_band(columns):
    for c in columns:
        if c["k"] >= 8:
            assert list(c["bands"]) == ["restricted"]
            assert "full-depth band" in c["not_run"]
        else:
            assert set(c["bands"]) == {"full", "restricted"}


def test_no_stanford_column_was_probed_on_a_split(columns):
    """Every Stanford frame is held out by construction; a column that took a
    split would be reporting a smaller n_eff for no reason."""
    for c in columns:
        assert c["n_frames"] == 117, c["variant"]
        assert "held out" in c["split"]


def test_grid_is_stanfords_own_not_nvidias(columns):
    """The `StoreGrid` fix. NVIDIA's bmode grid is 507x456 over 0-40 mm; every
    Stanford column must be on one of the two §3.6 variants instead."""
    for c in columns:
        assert (c["grid"]["nz"], c["grid"]["nx"]) in {(253, 484), (304, 582)}
        assert c["grid"]["z_max_mm"] == STANFORD_DEPTH_MAX_MM


def test_policy_b_carries_no_lateral_position_anywhere(columns):
    """docs/MEMORY.md §14.4 / WI-13 §1: Policy B's `x_centre` R^2 is -0.000 at every
    K on NVIDIA, and this is the claim that it replicates on real tissue. One
    cell above ~0.05 would mean the finding is a property of simulation."""
    seen = 0
    for c in columns:
        if c["intensity_scale_mode"] != "matched":
            continue
        for r in c["results"]:
            if r["encoder"] == "vicreg_B" and r["target"] == "x_centre":
                assert abs(r["score"]) < 0.05, (c["variant"], c["k"], r["band"], r["score"])
                seen += 1
    assert seen >= 6, f"only {seen} Policy B x_centre cells found"


# Encoders whose features are scale-equivariant by construction: an untrained
# ResNet (default BN running stats, bias-free convs, positively-homogeneous
# ReLU), a PCA of the raw crops, and mean/std of the envelope. Scaling their
# input scales their features by the same constant, which the probe's
# `StandardScaler` then removes — so the amplitude match cannot move them.
SCALE_EQUIVARIANT = ("random_resnet18", "pca_raw", "trivial")
# Not *exactly* zero: float32 rounding differs between a x1 and a x0.0076 input
# path, and at n_eff=117 one borderline frame changing class moves balanced
# accuracy by ~0.009. Measured max over 24 cells per encoder: 0.0084 / 0.0083 /
# 0.0111. The trained encoders move by mean 0.099-0.158 over the same cells.
SCALE_NOISE_MAX = 0.02
TRAINED_SHIFT_MIN_MEAN = 0.05


def _matched_raw_pairs(columns, encoder):
    by = {}
    for c in columns:
        for r in c["results"]:
            if r["encoder"] == encoder:
                by[(c["k"], c["variant"], r["band"], r["target"],
                    c["intensity_scale_mode"])] = r["score"]
    out = []
    for (k, v, band, target, mode), score in by.items():
        if mode != "matched":
            continue
        other = by.get((k, v, band, target, "raw"))
        if other is not None:
            out.append((k, v, band, target, abs(score - other)))
    return out


def test_the_amplitude_match_does_not_move_the_scale_equivariant_baselines(columns):
    """§14.3's internal consistency check, and the reason the matched column is
    a correction rather than a general inflation: the baselines that *cannot*
    respond to a change of units do not."""
    for encoder in SCALE_EQUIVARIANT:
        pairs = _matched_raw_pairs(columns, encoder)
        assert len(pairs) >= 6, f"only {len(pairs)} pairs for {encoder}"
        worst = max(pairs, key=lambda t: t[-1])
        assert worst[-1] < SCALE_NOISE_MAX, (encoder, worst)


def test_the_amplitude_match_does_move_the_trained_encoders(columns):
    """The other half of the same check — if the trained encoders were also
    unmoved, the match would be doing nothing and §14.3's whole argument
    (that a frozen `.eval()` backbone cannot cross a 600x scale gap) would be
    unsupported."""
    for encoder in ("vicreg_A", "vicreg_B"):
        pairs = _matched_raw_pairs(columns, encoder)
        assert len(pairs) >= 6, f"only {len(pairs)} pairs for {encoder}"
        mean_shift = sum(p[-1] for p in pairs) / len(pairs)
        assert mean_shift > TRAINED_SHIFT_MIN_MEAN, (encoder, mean_shift)


# ── the collapse finding (docs/MEMORY.md §14.10) ─────────────────────────────────

RANKME = sorted(DATA.glob("*_rankme*.json"))


@pytest.fixture(scope="module")
def rankme_columns() -> list[dict]:
    if not RANKME:
        pytest.skip("no RankMe columns; run scripts/eval/rankme.py")
    return [json.loads(p.read_text()) for p in RANKME]


def test_both_corpora_are_measured_on_the_same_n(rankme_columns):
    """RankMe is sensitive to sample count, so the NVIDIA and Stanford columns
    are only comparable if both used 117 frames. An unmatched N would make the
    ratio a measure of corpus size."""
    for c in rankme_columns:
        assert c["nvidia_frames"] == 117, c["variant"]


def test_policy_b_collapses_on_stanford_and_policy_a_does_not(rankme_columns):
    """§14.10's finding, and the reason Policy B's high `echogenicity` score
    must not be read as a rich representation. If this ever inverts, §14.5 and
    §14.10 both need rewriting rather than quietly disagreeing with the data."""
    for c in rankme_columns:
        enc = c["encoders"]
        if "vicreg_A" not in enc or "vicreg_B" not in enc:
            continue
        a = enc["vicreg_A"]["stanford"]["rankme"] / enc["vicreg_A"]["nvidia"]["rankme"]
        b = enc["vicreg_B"]["stanford"]["rankme"] / enc["vicreg_B"]["nvidia"]["rankme"]
        assert b < a, (c["variant"], c["k"], a, b)
        # Policy B was the highest-rank encoder of all on NVIDIA — the collapse
        # is a property of the transfer, not of a weak starting point.
        assert enc["vicreg_B"]["nvidia"]["rankme"] > 250, c["variant"]


def test_the_amplitude_match_protects_rank_too(rankme_columns):
    """§14.10's third argument for §14.3: skipping the match does not merely
    lower probe scores, it partially collapses Policy A's representation."""
    by = {(c["k"], c["variant"], c["intensity_scale_mode"]): c for c in rankme_columns}
    matched = by.get((4, "mm_matched", "matched"))
    raw = by.get((4, "mm_matched", "raw"))
    if matched is None or raw is None:
        pytest.skip("need both the matched and raw K=4 mm-matched RankMe columns")
    assert (
        matched["encoders"]["vicreg_A"]["stanford"]["rankme"]
        > raw["encoders"]["vicreg_A"]["stanford"]["rankme"]
    )


# ── the qualitative pass ────────────────────────────────────────────────────

QUAL = sorted(DATA.glob("*_qual.json"))


@pytest.fixture(scope="module")
def qual_columns() -> list[dict]:
    if not QUAL:
        pytest.skip("no qualitative columns; run scripts/eval/stanford_qualitative.py")
    return [json.loads(p.read_text()) for p in QUAL]


def test_figures_are_captioned_from_the_matching_probe_column(qual_columns):
    """A figure whose R² came from a different band, grid or scale would be
    mislabelled in a way no reader could detect."""
    for c in qual_columns:
        expected = (
            f"outputs/stanford/{c['variant']}_k{c['k']}"
            f"{'' if c['intensity_scale_mode'] == 'matched' else '_rawscale'}/results.json"
        )
        assert c["probe_results"] == expected, c["variant"]


def test_every_encoder_is_clustered_not_just_the_reference(qual_columns):
    """The gap this pass exists to close: `stanford_matrix.py` clusters only its
    reference encoder, so a per-encoder comparison had no unsupervised half."""
    for c in qual_columns:
        assert set(c["clusters"]) == set(c["encoders"]), c["variant"]
        assert len(c["encoders"]) >= 5
