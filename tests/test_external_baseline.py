"""PLAN.md §4.4 baseline 5 — the external frozen natural-image control.

The model download needs the network, so the encoder itself is not exercised
here. What is tested is the B-mode conversion it depends on, where two of the
three documented traps are pure arithmetic and fail *silently*:

- per-image dB normalisation would make `echogenicity` a constant;
- a mismatched K=1 batch would encode different pixels than the labels describe.

Plus the recorded artefacts, so a run that quietly fell back to DINOv2 (or
skipped the separate reporting block §4.4 requires) is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssl_us.eval.external import (
    DYNAMIC_RANGE_DB,
    assert_same_crops,
    bmode_from_crops,
)
from ssl_us.eval.store_features import StoreCropBatch

DATA = Path("results/stanford")
# Matrix columns only — `*_rankme_external.json` matches the same
# glob but carries the RankMe schema, which has no `bands`/`results` keys.
EXTERNAL = sorted(
    p for p in DATA.glob("*_external.json") if "_rankme" not in p.name
)


def _crops(scale: float = 1.0, n: int = 8, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    # Crop i is i+1 times brighter than crop 0 — a real echogenicity spread.
    amp = (np.arange(n, dtype=np.float32) + 1.0)[:, None, None, None]
    return (scale * amp * rng.normal(size=(n, 2, size, size))).astype(np.float32)


def test_brightness_differences_between_crops_survive_log_compression():
    """The trap: display B-mode is normalised per image, and doing that here
    would make every crop equally bright and turn `echogenicity` — a probe
    target — into a constant. One corpus-level reference keeps the spread."""
    img, ref = bmode_from_crops(_crops())
    per_crop_mean = img.reshape(len(img), -1).mean(axis=1)
    assert per_crop_mean[-1] > per_crop_mean[0] + 0.05, per_crop_mean
    # Monotone in the crop's own amplitude, which is what the probe reads.
    assert np.all(np.diff(per_crop_mean) > 0), per_crop_mean
    assert ref > 0


def test_a_corpus_scale_change_is_absorbed_by_the_shared_reference():
    """Scaling the whole corpus is a change of display units and must not
    change the image — unlike a per-crop normalisation, which would also
    absorb the *between*-crop differences the previous test pins."""
    a, ref_a = bmode_from_crops(_crops(scale=1.0))
    b, ref_b = bmode_from_crops(_crops(scale=50.0))
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert ref_b == pytest.approx(50.0 * ref_a, rel=1e-6)


def test_output_is_a_bounded_image_over_the_named_dynamic_range():
    img, _ = bmode_from_crops(_crops())
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.dtype == np.float32
    # A sample exactly `DYNAMIC_RANGE_DB` below the reference lands at 0.
    quiet = np.zeros((1, 2, 4, 4), dtype=np.float32)
    quiet[0, 0] = 10 ** (-DYNAMIC_RANGE_DB / 20)
    img_q, _ = bmode_from_crops(quiet, reference=1.0)
    assert img_q.max() == pytest.approx(0.0, abs=1e-5)


def test_a_degenerate_reference_is_refused_not_silently_used():
    with pytest.raises(ValueError, match="reference level"):
        bmode_from_crops(_crops(), reference=0.0)


def _batch(ids, origins):
    return StoreCropBatch(
        images=np.empty((0,), dtype=np.float32),
        frame_id=np.asarray(ids, dtype=object),
        origins=np.asarray(origins, dtype=int),
    )


def test_misaligned_external_batch_is_refused():
    """The external baseline is encoded from a K=1 batch but probed against the
    main batch's labels. If the two ever stop lining up, every one of its
    scores describes different pixels than its labels do — and nothing about
    the resulting number would look wrong."""
    a = _batch(["0", "0", "1"], [(0, 0), (1, 1), (2, 2)])
    assert_same_crops(a, _batch(["0", "0", "1"], [(0, 0), (1, 1), (2, 2)]))

    with pytest.raises(ValueError, match="crop counts"):
        assert_same_crops(a, _batch(["0", "0"], [(0, 0), (1, 1)]))
    with pytest.raises(ValueError, match="frame_id order"):
        assert_same_crops(a, _batch(["0", "1", "0"], [(0, 0), (1, 1), (2, 2)]))
    with pytest.raises(ValueError, match="origins differ"):
        assert_same_crops(a, _batch(["0", "0", "1"], [(0, 0), (1, 1), (9, 9)]))


# ── the recorded runs ───────────────────────────────────────────────────────

pytestmark_external = pytest.mark.skipif(
    not EXTERNAL, reason="no external columns; run stanford_matrix.py --external"
)


@pytest.fixture(scope="module")
def columns() -> list[dict]:
    if not EXTERNAL:
        pytest.skip("no external columns")
    return [json.loads(p.read_text()) for p in EXTERNAL]


def test_every_run_records_which_model_produced_it(columns):
    """§4.4 requires reporting which model was used, because the DINOv3 gate
    means it is not always the one the plan names."""
    for c in columns:
        ext = c["external_baseline"]
        assert ext is not None and ext["model_id"]
        assert "ceiling" in ext["read_as"]


def test_the_runs_used_real_dinov3_not_the_fallback(columns):
    """If the ungated `timm` route ever breaks, the encoder silently falls back
    to DINOv2 and the table still looks fine — this is what makes that visible."""
    for c in columns:
        assert "dinov3" in c["external_baseline"]["model_id"].lower(), (
            c["external_baseline"]["model_id"]
        )


def test_the_external_row_is_probed_on_the_same_targets_as_the_arms(columns):
    for c in columns:
        band = next(iter(c["bands"]))
        ext = {r["target"] for r in c["results"]
               if r["encoder"] == "dino_bmode" and r["band"] == band}
        arm = {r["target"] for r in c["results"]
               if r["encoder"] == "vicreg_A" and r["band"] == band}
        assert ext == arm, c["variant"]
