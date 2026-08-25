"""WI-10: splits over the real corpus, and the per-run audit figure.

`test_guards.py` already covers the leakage guard itself (including the
realistic "crops of one frame straddle the cut" case, which is WI-10's second
done-condition). What is checked here is the rest of WI-10: that the split
built over the *real* 923-frame store is well-formed and that the
distributional-matching claim PLAN.md §8 leans on actually holds, plus that
the audit figure is emitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssl_us.data.splits import inner_probe_split, outer_split, split_audit
from ssl_us.eval import viz

SPLITS_JSON = Path("results/splits/nvidia_splits.json")


# ── the real artefact ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real():
    if not SPLITS_JSON.exists():
        pytest.skip(f"{SPLITS_JSON} not present; run scripts/data/build_splits.py")
    return json.loads(SPLITS_JSON.read_text())


def test_outer_split_is_nvidias_published_sizes(real):
    # PLAN.md §2: 830 train + 93 validation = 923.
    assert real["audit"]["outer"] == {"train": 830, "validation": 93}
    assert real["audit"]["n_frames_total"] == 923


def test_no_frame_appears_in_both_outer_splits(real):
    outer = real["splits"]["outer"]
    assert not set(outer["train"]) & set(outer["validation"])


def test_inner_probe_split_is_carved_from_train_only(real):
    outer, inner = real["splits"]["outer"], real["splits"]["inner"]
    train = set(outer["train"])
    assert set(inner["probe_train"]) | set(inner["probe_test"]) <= train
    assert not set(inner["probe_train"]) & set(inner["probe_test"])
    assert len(inner["probe_train"]) + len(inner["probe_test"]) == len(train)


def test_train_and_validation_are_distributionally_matched(real):
    """PLAN.md §8's basis for adopting NVIDIA's split wholesale. If this ever
    fails, the probes are comparing two populations, not measuring transfer."""
    for key, c in real["distribution_checks"].items():
        assert c["matched"], f"{key}: Welch t={c['welch_t']:.2f}, |t| >= 2"
        assert c["agrees_with_plan"], f"{key} drifted from PLAN.md §8's measurement"


def test_measured_means_reproduce_plan_section_8(real):
    """The planning-time numbers were measured from the source dataset; these
    come from the built store, so agreeing is a real cross-check on WI-8's
    frame ordering, not a tautology."""
    sos = real["distribution_checks"]["sos_mean"]
    phase = real["distribution_checks"]["phase_error"]
    assert sos["train_mean"] == pytest.approx(1474.2, abs=1.0)
    assert sos["validation_mean"] == pytest.approx(1471.1, abs=1.0)
    assert phase["train_mean"] == pytest.approx(1.826, abs=0.01)
    assert phase["validation_mean"] == pytest.approx(1.827, abs=0.01)


def test_empty_segmentation_counts_match_plan_section_2(real):
    # PLAN.md §2: "63 train, 12 validation, measured".
    assert real["empty_segmentation_per_split"] == {"train": 63, "validation": 12}


# ── split construction, synthetic ────────────────────────────────────────────


def test_outer_split_keys_off_the_frame_id_prefix():
    ids = np.array(["train/0", "train/1", "validation/0"])
    outer = outer_split(ids)
    assert outer == {"train": ["train/0", "train/1"], "validation": ["validation/0"]}


def test_inner_split_is_deterministic_for_a_seed():
    ids = [f"train/{i}" for i in range(50)]
    a = inner_probe_split(ids, seed=7)
    b = inner_probe_split(ids, seed=7)
    assert a == b
    assert inner_probe_split(ids, seed=8) != a


def test_split_audit_rejects_inner_ids_outside_train():
    ids = np.array([f"train/{i}" for i in range(10)] + ["validation/0"])
    outer = outer_split(ids)
    bad = {"probe_train": ["train/0"], "probe_test": ["validation/0"]}
    with pytest.raises(ValueError, match="not in the outer train split"):
        split_audit(ids, outer, bad)


# ── the audit figure (WI-10 done-condition 3) ────────────────────────────────


def test_plot_split_audit_writes_a_figure(tmp_path):
    outer = {"train": ["train/0", "train/1", "train/2"], "validation": ["validation/0"]}
    inner = {"probe_train": ["train/0", "train/1"], "probe_test": ["train/2"]}
    out = viz.plot_split_audit(outer, tmp_path / "audit.png", inner=inner)
    assert out.exists() and out.stat().st_size > 0


def test_bootstrap_loader_exposes_labels_keyed_by_frame_id(tmp_path):
    """`matrix.run` feeds these to the audit figure, which looks labels up per
    split member rather than per row — so the by-id view has to exist and line
    up with `frame_id`."""
    from ssl_us.data import geometry as geo
    from ssl_us.eval.matrix import load_bootstrap

    n = 4
    path = tmp_path / "boot.npz"
    np.savez(
        path,
        bmode=np.zeros((n, 8, 8), dtype=np.float32),
        seg=np.zeros((n, 8, 8), dtype=np.uint8),
        sos=np.full((n, 4, 4), 1500.0, dtype=np.float32),
        frame_id=np.array([f"train/{i}" for i in range(n)]),
        phase_error=np.linspace(1.5, 2.0, n).astype(np.float32),
        t0=np.full(n, geo.T0_LEVELS_S[0], dtype=np.float64),
    )
    data = load_bootstrap(path)
    by_id = data["frame_labels_by_id"]
    assert set(by_id) == set(data["frame_id"])
    for i, fid in enumerate(data["frame_id"]):
        assert by_id[fid]["phase_error"] == pytest.approx(data["frame_labels"]["phase_error"][i])
        assert by_id[fid]["sos_mean_frame"] == pytest.approx(1500.0)


def test_plot_split_audit_includes_label_panels_when_given_labels(tmp_path):
    outer = {"train": [f"train/{i}" for i in range(20)], "validation": ["validation/0"]}
    labels = {f: {"sos_mean": 1470.0 + i, "phase_error": 1.8} for i, f in enumerate(outer["train"])}
    labels["validation/0"] = {"sos_mean": 1475.0, "phase_error": 1.9}
    small = viz.plot_split_audit(outer, tmp_path / "a.png")
    big = viz.plot_split_audit(outer, tmp_path / "b.png", frame_labels=labels)
    # More panels -> a wider figure; this is the cheap way to assert the label
    # panels were actually drawn without parsing the PNG.
    assert big.stat().st_size > small.stat().st_size
