"""WI-13's results-matrix driver — target list and the cyst-probe exclusion.

The probes themselves are covered by `test_probes.py` and the crop/label path
by `test_store_features.py`. What is tested here is the two decisions the
driver makes on its own, both of which fail quietly:

- excluding crops from empty-`segmentation_map` frames from the cyst probes
  (PLAN.md §2.6 — an empty mask is an unknown mix of true negatives and
  unlabelled edge-touching cysts, so including them injects that mix into the
  primary tissue axis as zeros);
- the names it probes matching the names `sample_store_crops` emits, since a
  rename on either side would show up only as a silently missing row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "nvidia_matrix.py"


def _load_script():
    """Import scripts/eval/nvidia_matrix.py by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("nvidia_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


def test_cyst_targets_drop_empty_segmentation_crops(mod):
    targets = mod.TARGETS + mod.CROP_ANATOMY_TARGETS
    empty = np.array([False, False, True, True, False])
    subsets = mod.target_subsets(targets, {"segmentation_empty": empty}, len(empty))

    for t in mod.CYST_TARGETS:
        assert subsets[t].tolist() == [True, True, False, False, True]
    # every other target keeps the full set — the exclusion is about the cyst
    # label being unreliable there, not about the crops being bad
    for t, _, _ in targets:
        if t not in mod.CYST_TARGETS:
            assert subsets[t].all()


def test_sos_crop_target_is_not_excluded(mod):
    """`sound_speed_map` is dense everywhere; only the *segmentation* is empty
    on those 75 frames, so dropping them from the SoS probe would throw away
    good labels."""
    assert "sos_mean_crop" not in mod.CYST_TARGETS
    targets = mod.CROP_ANATOMY_TARGETS
    subsets = mod.target_subsets(targets, {"segmentation_empty": np.array([True, False])}, 2)
    assert subsets["sos_mean_crop"].all()


def test_without_label_maps_every_target_keeps_every_crop(mod):
    subsets = mod.target_subsets(mod.TARGETS, {}, 4)
    assert all(m.all() for m in subsets.values())
    assert set(subsets) == {t for t, _, _ in mod.TARGETS}


def test_crop_anatomy_target_names_match_what_the_sampler_emits(mod):
    """A rename on either side would drop the row rather than raise."""
    from ssl_us.eval import store_features

    source = Path(store_features.__file__).read_text()
    for target, _, _ in mod.CROP_ANATOMY_TARGETS:
        assert f'labels["{target}"]' in source, target


def test_cyst_targets_are_a_subset_of_the_anatomy_targets(mod):
    names = {t for t, _, _ in mod.CROP_ANATOMY_TARGETS}
    assert mod.CYST_TARGETS <= names


def test_frame_level_cyst_fraction_keeps_its_name_and_meaning(mod):
    """Every K column measured so far reports `cyst_fraction` as the *frame*
    quantity from `frame_metadata.json`. The crop-level one is suffixed so the
    old and new numbers are not silently compared."""
    frame_targets = {(t, level) for t, level, _ in mod.TARGETS}
    assert ("cyst_fraction", "frame") in frame_targets
    assert ("cyst_fraction_crop", "crop") in {
        (t, level) for t, level, _ in mod.CROP_ANATOMY_TARGETS
    }


def test_unavailable_reason_covers_every_anatomy_target(mod):
    """If the maps are absent, each target must still be reported by name in
    `not_run` — the done-condition's 'no silent truncation'."""
    assert {t for t, _, _ in mod.CROP_ANATOMY_TARGETS} <= set(mod.UNAVAILABLE_REASON)


def test_contains_cyst_is_probed_as_a_classifier(mod):
    kinds = {t: kind for t, _, kind in mod.CROP_ANATOMY_TARGETS}
    assert kinds["contains_cyst"] == "logistic"
    assert kinds["cyst_fraction_crop"] == "ridge"
    assert kinds["sos_mean_crop"] == "ridge"


def test_legacy_checkpoint_template_does_not_answer_for_other_k(mod, tmp_path):
    """The wi12_* fallback carries no `{k}`, so it matches every K — and WI-12
    only trained K=4.

    Ungated it reports a trained pair for K=8, which does not exist, and the
    availability probe then drops "K=8 column" from `not_run` — the silent
    truncation the done-condition forbids. K=4 must still resolve through it,
    because that is the directory WI-12 actually wrote.
    """
    for policy in ("A", "B"):
        legacy = tmp_path / f"wi12_vicreg_policy{policy}"
        legacy.mkdir()
        (legacy / "last.pt").touch()

    assert mod.find_checkpoint(tmp_path, 4, "A") is not None, "K=4 is WI-12's own run"
    for k in (16, 8, 1):
        assert mod.find_checkpoint(tmp_path, k, "A") is None, (
            f"K={k} resolved to WI-12's K=4 checkpoint via the k-less fallback"
        )


def test_per_k_checkpoint_directory_wins_over_the_legacy_name(mod, tmp_path):
    """An explicit wi13_vicreg_k{k}_* directory is the authoritative one."""
    for policy in ("A", "B"):
        for name in (f"wi12_vicreg_policy{policy}", f"wi13_vicreg_k16_policy{policy}"):
            d = tmp_path / name
            d.mkdir()
            (d / "last.pt").touch()

    found = mod.find_checkpoint(tmp_path, 16, "A")
    assert found is not None and found.parent.name == "wi13_vicreg_k16_policyA"
