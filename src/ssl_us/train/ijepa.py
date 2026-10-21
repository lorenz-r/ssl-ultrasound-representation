"""I-JEPA training loop (WI-17).

Same harness as `train/vicreg.py` and `train/mae.py` — same optimizer and
schedule builder, same atomic checkpoint/resume, same step heartbeat, same
per-epoch history keys — so the three arms differ where the *method* differs
and nowhere else. What is genuinely different here:

- **An EMA target encoder is stepped after every optimizer step**, on the
  paper's linear momentum schedule (`models/ijepa.momentum_schedule`). The
  optimizer is built over `model.trainable` — context encoder and predictor
  only — so weight decay can never reach the target weights.
- **The collapse halt is on, and it means something.** MAE could not collapse
  to a constant embedding without its loss rising, so `train/mae.py`
  deliberately ships no detector. I-JEPA can: its loss is minimised exactly
  when the target representation is constant, and nothing in the objective
  forbids that. §4.7's RankMe-and-loss-both-falling signature is therefore back
  in force, and it is joined by a second, faster instrument —
  `min_target_std`, an absolute floor on the across-batch std of the target
  tokens. RankMe needs a trend to fire; a target std of ~0 is collapse on one
  epoch's evidence.
- **RankMe reads `encode()`** — the unmasked target encoder, mean-pooled —
  because that is the representation the probes will read, not the masked
  pretext path. Same reasoning as WI-14's, one arm over.

⚠ §1.4's lesson applies to reading the history: RankMe starts near its
*maximum* at random init and a fall is not automatically collapse. Both MAE
arms peaked at epoch 4 and decayed for 219 more; that is why the halt requires
the loss to be falling *too*, and why `collapse_grace_epochs` is long.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.ijepa import IJEPA, ijepa_loss, momentum_schedule
from . import common
from . import instrument as instr
from .vicreg import DimensionalCollapse, _is_collapsing, save_checkpoint


@dataclass
class IJEPATrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    device: str | None = None
    amp: bool | None = None
    ema_base: float = 0.996
    ema_final: float = 1.0
    max_rankme_samples: int = 2048
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 25
    resume: bool = True
    log_every_steps: int = 0
    # Collapse instruments. Both disabled at 0, both switched on by
    # `scripts/train/ijepa.py`; see the module docstring for why this arm gets a
    # halt where the MAE arm does not.
    collapse_patience: int = 0
    collapse_grace_epochs: int = 50
    min_target_std: float = 0.0


def _maybe_resume(path: Path, model, optimizer, scheduler, scaler, history) -> int:
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


def train_ijepa(
    model: IJEPA,
    train_loader,
    config: IJEPATrainConfig | None = None,
    monitor=None,
) -> dict[str, list[float]]:
    """Train the I-JEPA arm. Returns per-epoch history including RankMe.

    Args:
        model: an :class:`~ssl_us.models.ijepa.IJEPA`.
        train_loader: iterable of single crop batches, ``(B, C, H, W)``. A
            loader yielding ``(view1, view2)`` tuples is accepted and its first
            element used, so the WI-12 loader can be reused unchanged — but
            note WI-17 trains on **un-augmented** crops by design (the arm
            exists partly to remove the Policy A/B knob), which is what
            `scripts/train/ijepa.py` feeds it.
        config: hyperparameters.
        monitor: optional `TrainingMonitor`; no-op if ``None``.

    Raises:
        DimensionalCollapse: if either collapse instrument fires.
    """
    config = config or IJEPATrainConfig()
    device = config.device or common.default_device()
    device_type = torch.device(device).type
    use_amp = config.amp if config.amp is not None else common.amp_beneficial(device)
    model.to(device)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(config.epochs * steps_per_epoch, 1)
    optimizer, scheduler = common.build_optimizer_and_scheduler(
        model.trainable,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.warmup_epochs,
    )
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    history: dict[str, list[float]] = defaultdict(list)

    ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    start_epoch = 1
    if ckpt_dir is not None and config.resume:
        start_epoch = _maybe_resume(
            ckpt_dir / "last.pt", model, optimizer, scheduler, scaler, history
        )
        if start_epoch > 1:
            print(f"[resume] continuing from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        losses: list[float] = []
        target_stds: list[float] = []
        rankme_features: list[torch.Tensor] = []
        n_rankme = 0
        momentum = config.ema_final
        epoch_start = time.monotonic()
        n_crops = 0

        for step, batch in enumerate(train_loader, start=1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                predictions, targets = model(x)
                loss = ijepa_loss(predictions, targets, model.loss_kind)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # After the optimizer step, so the target tracks the weights that
            # were just written rather than the ones that produced the loss.
            global_step = (epoch - 1) * steps_per_epoch + step
            momentum = momentum_schedule(
                global_step, total_steps, config.ema_base, config.ema_final
            )
            model.update_target(momentum)

            losses.append(loss.item())
            # Across-batch std of the target features: the fast collapse read.
            # A healthy target is O(0.1-1) here; a constant one is 0.
            target_stds.append(float(targets.float().std(dim=0).mean()))
            n_crops += x.shape[0]

            if n_rankme < config.max_rankme_samples:
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    h = model.encode(x)
                rankme_features.append(h.detach().cpu())
                n_rankme += h.shape[0]

            if config.log_every_steps and step % config.log_every_steps == 0:
                rate = n_crops / max(time.monotonic() - epoch_start, 1e-9)
                print(
                    f"[e{epoch} s{step}] loss={loss.item():.4f} "
                    f"tgt_std={target_stds[-1]:.4f} m={momentum:.4f} {rate:.0f} crops/s",
                    flush=True,
                )

        if not losses:
            # Same failure and same cause as the MAE loop's: most often
            # frames x crops_per_frame < batch_size, so every batch is a
            # dropped partial. Say so rather than dying in torch.cat below.
            raise ValueError(
                f"epoch {epoch} saw 0 batches: the loader yielded nothing. "
                "Check that n_frames * crops_per_frame >= batch_size."
            )

        history["loss"].append(float(np.mean(losses)))
        history["target_std"].append(float(np.mean(target_stds)))
        history["ema_momentum"].append(float(momentum))
        feats = torch.cat(rankme_features, dim=0)[: config.max_rankme_samples]
        spectrum = instr.singular_value_spectrum(feats)
        history["rankme"].append(instr.rankme(spectrum))
        elapsed = time.monotonic() - epoch_start
        history["epoch_seconds"].append(float(elapsed))
        history["crops_per_second"].append(float(n_crops / max(elapsed, 1e-9)))

        if monitor is not None:
            monitor.log_epoch(epoch, {f"train/{k}": v[-1] for k, v in history.items()})
            monitor.log_histogram(epoch, "train/encoder_singular_values", spectrum)

        if ckpt_dir is not None:
            save_checkpoint(
                ckpt_dir / "last.pt", model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, epoch=epoch, history=history,
            )
            (ckpt_dir / "history.json").write_text(json.dumps(dict(history), indent=2))
            if config.checkpoint_every and epoch % config.checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:04d}.pt", model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, epoch=epoch, history=history,
                )

        # Written to disk *before* either halt fires, so a collapsed run still
        # leaves the history that diagnoses it.
        if config.min_target_std > 0 and history["target_std"][-1] < config.min_target_std:
            raise DimensionalCollapse(
                f"target representation std {history['target_std'][-1]:.2e} fell below "
                f"{config.min_target_std:.2e} at epoch {epoch} — the EMA target is "
                "constant across the batch, which is I-JEPA's degenerate solution; halting.",
                dict(history),
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
