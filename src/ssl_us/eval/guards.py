"""The two hard-failure guards that stand between this project and meaningless numbers.

Both are called unconditionally in the probe path, not only from tests:

* :func:`assert_no_frame_leakage` — no ``frame_id`` may appear on both sides of a
  split. Every crop inherits its frame's id, so this is what stops overlapping
  128x128 crops from one frame landing in both probe train and probe test.
  (PLAN.md 4.2)

* :func:`assert_no_label_leak` — data beamformed with the ground-truth sound-speed
  map (``gt_sos``) may not be probed for sound speed or phase error, because that
  beamformer *consumes the label*. A probe recovering it measures the beamformer,
  not the representation. (PLAN.md 4.4b)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Beamforming stages. `c0_const` uses a constant speed of sound and is valid for
# every probe target; `gt_sos` consumes `sound_speed_map` when computing delays.
BEAMFORM_STAGES = ("c0_const", "gt_sos")

# Targets whose ground truth is an *input* to the `gt_sos` beamformer. `phase_error`
# is included because 58.5% of its variance is a linear function of the sound-speed
# summary statistics (PLAN.md 4.1), so an SoS leak is substantially a phase-error
# leak as well.
LEAKED_BY_GT_SOS = frozenset(
    {"sos_mean", "sos_std", "sos_grad", "sos_mean_frame", "sos_std_frame", "phase_error"}
)

STAGE_LEAKS: dict[str, frozenset[str]] = {
    "c0_const": frozenset(),
    "gt_sos": LEAKED_BY_GT_SOS,
}


class FrameLeakageError(AssertionError):
    """A frame id appeared on both sides of a split."""


class LabelLeakageError(AssertionError):
    """A probe target was requested against a beamforming stage that consumed it."""


def assert_no_frame_leakage(splits: Mapping[str, Iterable[str]]) -> None:
    """Raise if any frame id appears in more than one split.

    Args:
        splits: mapping of split name -> iterable of frame ids.

    Raises:
        FrameLeakageError: naming the offending id and both splits it appeared in.
    """
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for fid in ids:
            if fid in seen and seen[fid] != name:
                raise FrameLeakageError(
                    f"frame_id {fid!r} appears in both {seen[fid]!r} and {name!r}. "
                    "Splits must be disjoint at frame level — every crop from a "
                    "frame carries that frame's id, so a shared frame puts "
                    "near-duplicate crops on both sides."
                )
            seen[fid] = name


def assert_no_label_leak(stage: str, target: str) -> None:
    """Raise if ``target`` is contaminated by the beamforming ``stage``.

    Args:
        stage: one of :data:`BEAMFORM_STAGES`.
        target: probe target name.

    Raises:
        ValueError: unknown stage (fail loudly rather than silently permitting).
        LabelLeakageError: the stage consumed this target's ground truth.
    """
    if stage not in STAGE_LEAKS:
        raise ValueError(
            f"unknown beamform stage {stage!r}; expected one of {BEAMFORM_STAGES}. "
            "Every feature array must carry a stage tag — an untagged array cannot "
            "be checked and must not be probed."
        )
    if target in STAGE_LEAKS[stage]:
        raise LabelLeakageError(
            f"target {target!r} cannot be probed on stage {stage!r}: that "
            "beamformer consumes the ground-truth sound-speed map, so the label is "
            "imprinted in the data. Use stage 'c0_const' for physics probes; "
            "'gt_sos' is an oracle ceiling for tissue axes only."
        )


def filter_targets(stage: str, targets: Iterable[str]) -> list[str]:
    """Targets from ``targets`` that are legal on ``stage``, preserving order."""
    if stage not in STAGE_LEAKS:
        raise ValueError(f"unknown beamform stage {stage!r}")
    leaked = STAGE_LEAKS[stage]
    return [t for t in targets if t not in leaked]
