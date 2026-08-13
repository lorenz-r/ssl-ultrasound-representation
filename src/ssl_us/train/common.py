"""Device/AMP selection, worker-count clamping, and the LR schedule shared by
every training loop (VICReg, MAE, cross-aperture). Ported verbatim from the old
`train.py` (PLAN.md 1.2) — these pieces are architecture- and data-agnostic.
"""

from __future__ import annotations

import math
import os

import torch


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_beneficial(device: str) -> bool:
    """Whether mixed precision actually speeds things up on this device.

    fp16 is only a win on GPUs with Tensor Cores (Volta sm_70+), which also
    have the flash-attention kernels. On older cards (e.g. Pascal TITAN Xp,
    sm_61) fp16 runs many times slower than fp32, so autocast would be a large
    slowdown. Evaluated at runtime so a heterogeneous cluster picks the right
    setting per job — AMP off on old GPUs, on automatically on newer ones.
    """
    if device != "cuda":
        return False
    return torch.cuda.get_device_capability()[0] >= 7


def clamp_num_workers(requested: int) -> int:
    """Cap DataLoader workers to the cgroup's CPU allotment.

    `os.cpu_count()` reports the host's CPUs and over-counts inside a
    container (e.g. 12 host CPUs vs a 4-CPU cgroup); over-counting invites the
    cgroup OOM killer, since each forked worker duplicates bookkeeping on top
    of a capped memory budget. `sched_getaffinity` is cgroup-aware; it is not
    available on macOS, so fall back to `os.cpu_count()` there.
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    return min(requested, cpu_count)


def warmup_cosine_lr(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup -> cosine decay, evaluated per optimizer step.

    Returns a multiplier for `LambdaLR`, not an absolute LR. Unlike the old
    MAE loop this has no `overfit_single_sample` special case — that was a
    debug affordance specific to the deleted training path.
    """
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
):
    """AdamW + the shared warmup/cosine schedule, stepped once per training step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_lr(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )
    return optimizer, scheduler
