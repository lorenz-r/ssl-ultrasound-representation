"""VICReg training loop (PLAN.md §5.1, WI-12).

Generic over the data source: takes any iterable of ``(view1, view2)`` batch
pairs, so the same loop runs unmodified against a real Policy-A/Policy-B
`DataLoader` on the cluster or a synthetic one in a CPU smoke test here. The
loop scaffolding (device/AMP selection, LR schedule) is ported from the old
MAE loop via `train/common.py` (PLAN.md §1.2); the body — VICReg forward,
loss, instrumentation — is new.

Every instrument in PLAN.md §4.7 Tier 1 runs every epoch, unconditionally:
the three loss terms separately, the backbone singular-value spectrum,
RankMe, and (if an online probe is supplied) a gradient-isolated probe step.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.vicreg import VICReg, VICRegLossWeights, vicreg_loss
from . import common
from . import instrument as instr


class DimensionalCollapse(RuntimeError):
    """Raised when RankMe falls while the loss falls (PLAN.md §4.7 row 1).

    WI-12's done-when makes this a *halt* condition, not a warning: a run in
    dimensional collapse is not salvageable by letting it finish, and an
    unattended overnight run that keeps burning GPU on a collapsed encoder
    costs the whole window. The exception carries the history so the caller
    can still write out what happened before re-launching with a different
    covariance weight.
    """

    def __init__(self, message: str, history: dict[str, list[float]]):
        super().__init__(message)
        self.history = history


@dataclass
class VICRegTrainConfig:
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-6
    warmup_epochs: int = 0
    device: str | None = None
    amp: bool | None = None
    loss_weights: VICRegLossWeights = VICRegLossWeights()
    max_rankme_samples: int = 2048  # cap on features accumulated for the epoch's SVD
    # ── unattended-run machinery ────────────────────────────────────────────
    # A 5–7 h run that dies at hour 4 with nothing on disk wastes the window,
    # so checkpointing is on whenever a directory is given, and resume is the
    # default: re-launching the same command continues rather than restarts.
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 5
    resume: bool = True
    # Collapse halt. Disabled at 0.
    #
    # `collapse_grace_epochs` is deliberately generous. RankMe starts near its
    # *maximum* at random init (an unorganised backbone has a near-uniform
    # spectrum) and falls for the first stretch of training as features
    # organise — with the loss falling too. That is the healthy trajectory and
    # it is indistinguishable, locally, from the collapse signature. Halting a
    # good overnight run at epoch 21 costs the whole window, whereas a genuine
    # collapse caught one monitoring interval later costs minutes, so the
    # asymmetry is priced in here: long grace, long patience.
    collapse_patience: int = 0
    collapse_grace_epochs: int = 50
    # Step heartbeat: prints throughput mid-epoch so a stalled loader is
    # visible without waiting for the epoch boundary.
    log_every_steps: int = 0


def save_checkpoint(
    path: str | Path,
    *,
    model: VICReg,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    history: dict[str, list[float]],
) -> None:
    """Write a resumable checkpoint atomically.

    Atomic because the alternative — a torch.save interrupted by the job
    being killed — leaves a truncated file that resume then refuses to load,
    which is exactly the moment the checkpoint was supposed to help. Written
    to a sibling temp file and renamed, since rename within a directory is
    atomic on POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": {k: list(v) for k, v in history.items()},
        },
        tmp,
    )
    tmp.replace(path)


def _maybe_resume(path: Path, model, optimizer, scheduler, scaler, history) -> int:
    """Restore state from `path` if it exists; return the epoch to start from."""
    if not path.exists():
        return 1
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    for key, values in state.get("history", {}).items():
        history[key] = list(values)
    return int(state["epoch"]) + 1


def _is_collapsing(history: dict[str, list[float]], patience: int) -> bool:
    """RankMe monotonically down and loss monotonically down over `patience`
    epochs — PLAN.md §4.7's dimensional-collapse signature.

    Both halves are required. RankMe drifting down while the loss *rises* is
    an unstable run, not a collapse, and gets a different remedy; the plan is
    specific that the diagnostic is the conjunction.
    """
    rank, loss = history.get("rankme", []), history.get("loss", [])
    if patience <= 0 or len(rank) <= patience or len(loss) <= patience:
        return False
    recent_rank = rank[-(patience + 1) :]
    recent_loss = loss[-(patience + 1) :]
    # Pairing a sequence with its own tail is intentionally ragged — n values
    # give n-1 consecutive comparisons — so `strict=True` is wrong here and
    # raises `ValueError` on every call that gets past the guard above.
    rank_down = all(b < a for a, b in zip(recent_rank, recent_rank[1:], strict=False))
    loss_down = all(b < a for a, b in zip(recent_loss, recent_loss[1:], strict=False))
    return rank_down and loss_down


def train_vicreg(
    model: VICReg,
    train_loader,
    config: VICRegTrainConfig | None = None,
    monitor=None,
    probe: instr.OnlineProbe | None = None,
    probe_label_fn=None,
) -> dict[str, list[float]]:
    """Train one VICReg arm. Returns per-epoch history, including RankMe.

    Args:
        model: a :class:`~ssl_us.models.vicreg.VICReg`.
        train_loader: iterable of ``(view1, view2)`` tensor pairs, each
            ``(B, C, H, W)``.
        config: hyperparameters.
        monitor: optional `TrainingMonitor` (PLAN.md §5.1 utils/monitoring.py);
            no-op if ``None``.
        probe: optional :class:`~ssl_us.train.instrument.OnlineProbe`. If
            given, ``probe_label_fn`` must also be given.
        probe_label_fn: ``(view1_batch) -> label_tensor``, called once per
            step to produce the online probe's targets (e.g. from a parallel
            label loader the caller manages) — kept as a callback so this
            loop stays agnostic to where labels come from.

    Returns:
        dict with keys ``loss``, ``invariance``, ``variance``,
        ``covariance``, ``rankme`` (and ``probe_loss`` if a probe was given),
        one entry per epoch.
    """
    config = config or VICRegTrainConfig()
    device = config.device or common.default_device()
    model.to(device)
    use_amp = config.amp if config.amp is not None else common.amp_beneficial(device)
    device_type = torch.device(device).type

    steps_per_epoch = max(len(train_loader), 1)
    optimizer, scheduler = common.build_optimizer_and_scheduler(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.warmup_epochs,
    )
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    history: dict[str, list[float]] = defaultdict(list)

    ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    last_ckpt = ckpt_dir / "last.pt" if ckpt_dir else None
    start_epoch = 1
    if last_ckpt is not None and config.resume:
        start_epoch = _maybe_resume(last_ckpt, model, optimizer, scheduler, scaler, history)
        if start_epoch > 1:
            print(f"[resume] continuing from epoch {start_epoch} ({last_ckpt})", flush=True)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_terms = defaultdict(list)
        rankme_features: list[torch.Tensor] = []
        n_rankme = 0
        last_z1: torch.Tensor | None = None
        epoch_start = time.monotonic()
        n_pairs = 0

        for step, (x1, x2) in enumerate(train_loader, start=1):
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                h1, z1 = model(x1)
                _h2, z2 = model(x2)
                loss, terms = vicreg_loss(z1, z2, config.loss_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_terms["loss"].append(loss.item())
            for name, value in terms.items():
                epoch_terms[name].append(value.item())

            if n_rankme < config.max_rankme_samples:
                rankme_features.append(h1.detach().cpu())
                n_rankme += h1.shape[0]

            last_z1 = z1.detach()
            n_pairs += x1.shape[0]

            if probe is not None and probe_label_fn is not None:
                labels = probe_label_fn(x1).to(device)
                epoch_terms["probe_loss"].append(probe.step(h1.detach(), labels))

            if config.log_every_steps and step % config.log_every_steps == 0:
                rate = n_pairs / max(time.monotonic() - epoch_start, 1e-9)
                print(
                    f"[e{epoch} s{step}] loss={loss.item():.3f} {rate:.0f} pairs/s",
                    flush=True,
                )

        for name, values in epoch_terms.items():
            history[name].append(float(np.mean(values)))

        feats = torch.cat(rankme_features, dim=0)[: config.max_rankme_samples]
        spectrum = instr.singular_value_spectrum(feats)
        history["rankme"].append(instr.rankme(spectrum))

        # Tier-1 projector diagnostic (train/README.md): the variance hinge acts
        # on these per-dimension stds, so the fraction sitting *below* the hinge
        # margin is the direct read on whether the term is still doing work.
        elapsed = time.monotonic() - epoch_start
        history["epoch_seconds"].append(float(elapsed))
        history["pairs_per_second"].append(float(n_pairs / max(elapsed, 1e-9)))
        if last_z1 is not None:
            stds = instr.per_dimension_std(last_z1.float())
            history["projector_std_mean"].append(float(stds.mean()))
            history["projector_std_frac_below_margin"].append(
                float((stds < config.loss_weights.variance_margin).mean())
            )

        if monitor is not None:
            monitor.log_epoch(epoch, {f"train/{k}": v[-1] for k, v in history.items()})
            if last_z1 is not None:
                monitor.log_histogram(epoch, "train/projector_std", stds)
            monitor.log_histogram(epoch, "train/backbone_singular_values", spectrum)

        if ckpt_dir is not None:
            save_checkpoint(
                ckpt_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                history=history,
            )
            (ckpt_dir / "history.json").write_text(json.dumps(dict(history), indent=2))
            if config.checkpoint_every and epoch % config.checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    history=history,
                )

        if epoch >= config.collapse_grace_epochs and _is_collapsing(
            history, config.collapse_patience
        ):
            raise DimensionalCollapse(
                f"RankMe fell for {config.collapse_patience} consecutive epochs while the "
                f"loss also fell (epoch {epoch}, RankMe {history['rankme'][-1]:.1f}) — "
                "PLAN.md §4.7 dimensional collapse; halting.",
                dict(history),
            )

    return dict(history)
