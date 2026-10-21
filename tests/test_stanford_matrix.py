"""WI-16's transfer-matrix driver — the decisions it makes on its own.

The probes are covered by `test_probes.py`, the crop/grid path by
`test_store_grid.py`, and the target definitions by `test_stanford_targets.py`.
What is left here is the driver's own logic, all of which fails quietly rather
than loudly:

- the depth-band split, which decides what the headline transfer row *is*;
- the target names it probes matching the names the sampler actually emits (a
  rename on either side shows up only as a silently missing row);
- that the encoders come from the same module WI-13 uses, since two copies of
  checkpoint discovery is how two tables come to differ for a reason that is
  not the data.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "stanford_matrix.py"


def _load_script():
    """Import scripts/eval/stanford_matrix.py by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("stanford_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


def test_headline_band_is_the_full_range_at_low_k(mod):
    """PLAN.md WI-16's done-condition: the headline number is the full
    0-19.9 mm range at K <= 4, with the restricted row reported alongside."""
    bands = mod.depth_bands(4, arm_z_min_mm=0.0)
    assert list(bands) == ["full", "restricted"]
    assert bands["full"] == 0.0
    assert bands["restricted"] == mod.DEPTH_FLOOR_MM == 8.0


def test_deep_k_has_only_the_restricted_band(mod):
    """At K >= 8 the arm already starts at the occupancy floor. Emitting a
    `full` band there would probe the encoder on 0-8 mm crops it never trained
    on and report the result as the headline transfer number."""
    bands = mod.depth_bands(16, arm_z_min_mm=8.0)
    assert list(bands) == ["restricted"]
    assert bands["restricted"] == 8.0


def test_missing_full_band_is_reported_not_dropped(mod):
    """`not_run` is the done-condition's no-silent-truncation instrument, so
    the K >= 8 case must have a reason attached rather than one fewer row."""
    source = SCRIPT.read_text()
    assert 'not_run["full-depth band"]' in source
    assert '"full" not in bands' in source


def test_targets_are_exactly_the_axes_stanford_supports(mod):
    from ssl_us.eval import stanford_targets as st

    names = [t for t, _, _ in mod.TARGETS]
    assert names == [t for t, _, _ in st.STANFORD_CROP_TARGETS] + [
        t for t, _, _ in st.STANFORD_FRAME_TARGETS
    ]
    # The unrunnable axis is carried into the report rather than forgotten.
    assert set(st.UNRUNNABLE) <= set(mod.UNRUNNABLE)


def test_frame_targets_are_probed_at_frame_level(mod):
    """Session/date/kind are constant within a frame. Probed at crop level they
    would inflate n_eff from 117 frames to 1872 crops and shrink every
    confidence interval by ~4x, which is a wrong number rather than a crash."""
    for target, level, kind in mod.TARGETS:
        if target in {"session_id", "date_id", "kind_fast"}:
            assert level == "frame", target
            assert kind == "logistic", target
        else:
            assert level == "crop", target


def test_encoders_come_from_the_shared_module(mod):
    from ssl_us.eval import encoders as shared

    assert mod.build_encoders is shared.build_encoders
    assert mod.find_checkpoint is shared.find_checkpoint


def test_uses_the_same_shared_encoders():
    from ssl_us.eval import encoders as shared

    spec = importlib.util.spec_from_file_location(
        "nvidia_matrix", SCRIPT.parent / "nvidia_matrix.py"
    )
    wi13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wi13)
    assert wi13.build_encoders is shared.build_encoders
    assert wi13.find_checkpoint is shared.find_checkpoint


# ── the qualitative pass (scripts/eval/stanford_qualitative.py) ──────────────────────


def _load_qual():
    spec = importlib.util.spec_from_file_location(
        "stanford_qualitative", SCRIPT.parent / "stanford_qualitative.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qual():
    return _load_qual()


def test_probe_captions_are_filtered_to_one_band(qual, tmp_path):
    """A Stanford column holds both depth bands, so an unfiltered read would
    caption a full-band figure with the restricted band's R^2 — a wrong number
    printed on a picture, which is worse than no number. WI-13's version has no
    band axis and needs no filter, so this is the one place the two diverge."""
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"results": [
        {"band": "full", "encoder": "vicreg_A", "target": "x_centre",
         "metric": "r2", "score": 0.195},
        {"band": "restricted", "encoder": "vicreg_A", "target": "x_centre",
         "metric": "r2", "score": 0.315},
    ]}))
    full = qual.probe_notes(results, "full", ["vicreg_A"], ["x_centre"])
    assert full["vicreg_A"]["x_centre"] == "R²=+0.195"
    restricted = qual.probe_notes(results, "restricted", ["vicreg_A"], ["x_centre"])
    assert restricted["vicreg_A"]["x_centre"] == "R²=+0.315"


def test_missing_probe_results_degrade_to_no_caption(qual, tmp_path):
    """Never invent an R^2: a figure with no number is honest, a figure with a
    fabricated one is not."""
    assert qual.probe_notes(tmp_path / "absent.json", "full", ["vicreg_A"], ["x_centre"]) == {}


def test_qualitative_draws_the_same_crops_the_table_was_computed_on(qual, mod):
    """Figure and table must come from one spec — `band_spec` — or the R^2
    printed on a panel describes different crops than the panel shows."""
    assert qual._load_matrix().band_spec is not None
    import argparse

    args = argparse.Namespace(k=4, crop_size=128, crops_per_frame=16, seed=0)
    spec = mod.band_spec(args, z_min_mm=0.0, intensity_scale=0.5)
    assert spec.derive_echogenicity and spec.intensity_scale == 0.5
    assert (spec.k, spec.crop_size, spec.crops_per_frame) == (4, 128, 16)
