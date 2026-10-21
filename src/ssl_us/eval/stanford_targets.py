"""Frame-level probe targets for the Stanford transfer corpus (WI-16).

`eval/store_features.py`'s :func:`frame_labels_from_metadata` reads NVIDIA's
*simulator ground truth* — ``phase_error``, ``sos_mean``, ``sos_std``,
``cyst_fraction``, ``t0``. **Stanford has none of it.** It is real in-vivo rat
tissue with no segmentation, no sound-speed field and no aberration record
(PLAN.md §3.6), so its sidecar (written by
:func:`ssl_us.beamform.stanford_frame.frame_labels`) carries provenance
instead: ``session``, ``source_file``, ``kind``, ``date``.

That difference is the whole point of the transfer arm and not a gap to paper
over: what WI-16 asks is whether a representation pre-trained on simulation
organises *real* tissue at all, and the axes available to ask it with are the
ones PLAN.md WI-16 names — **session, file, depth, echogenicity**. Depth and
echogenicity are per-crop and come from `store_features` (`z_centre`,
`echogenicity`); the two provenance axes are here.

**`source_file` is not runnable on this corpus, and that is a property of the
corpus, not an omission.** The published FSA set is 117 files and the builder
takes one frame from each (`scripts/data/fetch_stanford_fsa.py`), so
``source_file`` is in exact bijection with ``frame_id``: 117 classes with one
frame apiece. At frame level that leaves every class with a single sample; at
crop level the frame-leakage guard (PLAN.md §4.2, non-negotiable) puts all of a
file's crops in one CV group, so every test fold contains only classes absent
from its training half. Either way the cell is chance-by-construction rather
than informative, so it is reported in ``not_run`` with this reason.
:data:`SESSION_TARGET` is the runnable coarsening the corpus does support.

**`session` is an inferred cluster, not an animal id.** figshare flattened the
per-rat directory tree, so sessions are acquisitions grouped by timestamp — 30
of them over the 117 files. Probing it asks whether the representation encodes
acquisition identity (a nuisance/domain factor), which is worth knowing in both
directions: strong session structure in a frozen feature space is a warning
that the transfer numbers are partly reading the machine, not the tissue.
"""

from __future__ import annotations

import numpy as np

# (target, level, kind) for the axes the Stanford sidecar can support.
# `date` is `session` coarsened to acquisition day (10 days, 9-12 frames each)
# and is the better-conditioned of the two; both are reported because a
# representation can encode the day (probe/preset drift) without resolving the
# individual session.
SESSION_TARGET = ("session_id", "frame", "logistic")
STANFORD_FRAME_TARGETS = [
    SESSION_TARGET,
    ("date_id", "frame", "logistic"),
    ("kind_fast", "frame", "logistic"),
]

# Per-crop axes, supplied by `store_features.sample_store_crops` itself rather
# than by the sidecar. `echogenicity` needs `StoreCropSpec.derive_echogenicity`.
STANFORD_CROP_TARGETS = [
    ("z_centre", "crop", "ridge"),
    ("x_centre", "crop", "ridge"),
    ("echogenicity", "crop", "ridge"),
]

UNRUNNABLE = {
    "source_file": (
        "one frame per file on this corpus (117 files, 117 frames), so the "
        "target is in bijection with frame_id: 117 single-sample classes at "
        "frame level, and at crop level the frame-leakage guard puts every "
        "class wholly inside one CV fold. `session_id` is the runnable "
        "coarsening — see ssl_us.eval.stanford_targets"
    ),
}


def _codes(values: list[str]) -> np.ndarray:
    """Stable integer codes for a categorical column, sorted by label.

    Sorted rather than first-seen so the code for a given session is the same
    whichever frame order or split the caller happens to pass.
    """
    levels = sorted(set(values))
    index = {v: i for i, v in enumerate(levels)}
    return np.array([index[v] for v in values], dtype=np.int64)


def stanford_frame_labels(metadata: dict, frame_ids: list[str]) -> dict[str, np.ndarray]:
    """Frame-level label vectors aligned to `frame_ids`.

    Drop-in for `store_features.frame_labels_from_metadata` — same signature,
    same contract — over the Stanford sidecar's keys.

    Emits both the integer codes the probes fit on (``*_id``) and the raw
    strings (``session``, ``source_file``), because the strings are what a
    reader needs to interpret a confusion or a group split and the codes alone
    are unreadable.
    """
    sessions = [metadata[f]["session"] for f in frame_ids]
    dates = [metadata[f]["date"] for f in frame_ids]
    kinds = [metadata[f]["kind"] for f in frame_ids]
    return {
        "session_id": _codes(sessions),
        "date_id": _codes(dates),
        # FSA vs FSA_Fast — the two acquisition presets in the published set.
        "kind_fast": np.array([k.endswith("_Fast") for k in kinds], dtype=np.int64),
        "session": np.array(sessions, dtype=object),
        "source_file": np.array([metadata[f]["source_file"] for f in frame_ids], dtype=object),
    }
