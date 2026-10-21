"""Split construction at frame level, with the leakage guard wired in.

NVIDIA's own train/validation split is adopted as the outer split: it is
distributionally matched (measured — `sos_mean` 1474.2+-47.0 vs 1471.1+-47.8;
`phase_error` 1.826+-0.277 vs 1.827+-0.277) and carries no near-duplicate phantoms
across the boundary (PLAN.md 8).

The inner probe split is carved from the train frames **by frame id**. Every crop
inherits its frame's id, so grouping folds by that id is what stops overlapping
128x128 crops from one frame landing on both sides — the residual leakage mode once
frame-adjacency has been ruled out.
"""

from __future__ import annotations

import numpy as np

from ..eval.guards import assert_no_frame_leakage


def outer_split(frame_ids: np.ndarray) -> dict[str, list[str]]:
    """NVIDIA's published train/validation split, keyed by frame id prefix."""
    ids = frame_ids.astype(str)
    splits = {
        "train": sorted(i for i in ids if i.startswith("train/")),
        "validation": sorted(i for i in ids if i.startswith("validation/")),
    }
    assert_no_frame_leakage(splits)
    return splits


def inner_probe_split(
    train_ids: list[str], ratio: float = 0.8, seed: int = 0
) -> dict[str, list[str]]:
    """Carve a probe train/test split out of the train frames, at frame level."""
    rng = np.random.default_rng(seed)
    ids = sorted(train_ids)
    rng.shuffle(ids)
    cut = int(round(len(ids) * ratio))
    splits = {"probe_train": sorted(ids[:cut]), "probe_test": sorted(ids[cut:])}
    assert_no_frame_leakage(splits)
    return splits


def split_audit(
    frame_ids: np.ndarray,
    outer: dict[str, list[str]],
    inner: dict[str, list[str]] | None = None,
) -> dict:
    """Counts per split plus an explicit re-check, for the per-run audit figure.

    The two levels are audited **separately**. They are nested, not sibling: the
    inner probe split is carved out of ``outer["train"]``, so every inner id is by
    construction also an outer-train id. Checking them in one flat dict would fire
    the guard on that nesting and mask real leakage behind a false positive.
    """
    assert_no_frame_leakage(outer)
    audit = {
        "n_frames_total": int(len(np.unique(frame_ids.astype(str)))),
        "outer": {k: len(v) for k, v in outer.items()},
    }
    if inner is not None:
        assert_no_frame_leakage(inner)
        train = set(outer["train"])
        stray = sorted(set().union(*inner.values()) - train)
        if stray:
            raise ValueError(
                f"{len(stray)} inner-split ids are not in the outer train split "
                f"(e.g. {stray[:3]}); the probe split must be carved from train only."
            )
        audit["inner"] = {k: len(v) for k, v in inner.items()}
    return audit
