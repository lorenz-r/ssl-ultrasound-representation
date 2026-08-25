"""WI-16 — Stanford probe targets (`ssl_us.eval.stanford_targets`).

The point of these is that Stanford's sidecar and NVIDIA's are *different
schemas*, and the failure mode is quiet: a label function that reads the wrong
keys either raises far downstream or, worse, produces codes that are stable
within one call and shuffled between calls, which turns every session probe
into noise without anything erroring.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.eval.stanford_targets import (
    STANFORD_CROP_TARGETS,
    STANFORD_FRAME_TARGETS,
    UNRUNNABLE,
    stanford_frame_labels,
)
from ssl_us.eval.store_features import frame_labels_from_metadata


def _metadata(n: int = 6) -> tuple[dict, list[str]]:
    """Six frames over three sessions on two days, one file each."""
    frame_ids = [str(i) for i in range(n)]
    meta = {}
    for i, fid in enumerate(frame_ids):
        day = "20190318" if i < n // 2 else "20190319"
        meta[fid] = {
            "source_file": f"DATA_FSA_{day}_{100000 + i}.mat",
            "kind": "FSA" if i % 2 == 0 else "FSA_Fast",
            "date": day,
            "session": f"{day}_s{i // 2:02d}",
        }
    return meta, frame_ids


def test_frame_labels_cover_the_axes_the_sidecar_supports():
    meta, frame_ids = _metadata()
    labels = stanford_frame_labels(meta, frame_ids)

    for target, _level, _kind in STANFORD_FRAME_TARGETS:
        assert target in labels, target
        assert len(labels[target]) == len(frame_ids)
    # Codes must be integers — `probe(kind="logistic")` casts with `.astype(int)`
    # and a float/str column would either round silently or blow up in sklearn.
    assert labels["session_id"].dtype == np.int64
    assert labels["date_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert labels["kind_fast"].tolist() == [0, 1, 0, 1, 0, 1]
    # The readable strings survive alongside the codes.
    assert labels["session"][0] == "20190318_s00"


def test_codes_do_not_depend_on_the_order_frames_are_passed_in():
    """A code table built from first-seen order would renumber under a
    reordering, so the same session would carry different ids in two runs and
    the probes would silently disagree about what they were classifying."""
    meta, frame_ids = _metadata()
    forward = stanford_frame_labels(meta, frame_ids)
    reversed_ids = list(reversed(frame_ids))
    backward = stanford_frame_labels(meta, reversed_ids)

    by_session_fwd = dict(zip(forward["session"], forward["session_id"], strict=True))
    by_session_bwd = dict(zip(backward["session"], backward["session_id"], strict=True))
    assert by_session_fwd == by_session_bwd


def test_stanford_labels_are_a_drop_in_for_the_nvidia_ones():
    """`sample_store_crops` takes either through one `frame_label_fn` argument,
    so they must agree on signature and on returning frame-aligned vectors."""
    meta, frame_ids = _metadata()
    labels = stanford_frame_labels(meta, frame_ids)
    assert all(len(v) == len(frame_ids) for v in labels.values())

    # And they must NOT be interchangeable in the other direction: NVIDIA's
    # reader on a Stanford sidecar has to fail loudly, not invent zeros.
    with pytest.raises(KeyError):
        frame_labels_from_metadata(meta, frame_ids)


def test_source_file_is_declared_unrunnable_with_its_reason():
    """PLAN.md WI-16 names `file` as a probe axis; this corpus cannot support
    it (one frame per file). The done-condition forbids dropping a cell
    silently, so the reason has to be carried in the code that knows it."""
    assert "source_file" in UNRUNNABLE
    assert "session_id" in UNRUNNABLE["source_file"]
    assert not any(t == "source_file" for t, _, _ in STANFORD_FRAME_TARGETS)


def test_crop_targets_name_the_axes_the_sampler_actually_emits():
    from ssl_us.eval.store_features import StoreCropSpec

    assert StoreCropSpec(derive_echogenicity=True).derive_echogenicity
    names = {t for t, _, _ in STANFORD_CROP_TARGETS}
    assert names == {"z_centre", "x_centre", "echogenicity"}
