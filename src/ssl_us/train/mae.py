"""MAE training loop (PLAN.md WI-14) — the baseline the pivot is argued against.

WI-14 exists to *measure* the claim the project rests on: that masked
autoencoding underperforms joint embedding on linear probing. So this loop is
deliberately the same harness as `train/vicreg.py` — same optimizer and
schedule builder, same atomic checkpoint/resume, same step heartbeat, same
per-epoch history keys — and differs only where MAE genuinely differs:

- **One view, not two.** MAE's pretext task is reconstruction under masking,
  so the loader hands over a single crop per sample rather than a view pair.
  `scripts/wi14_train.py` feeds the *un-augmented* crop, which is what "same
  crops, same eval" in WI-14 means.
- **RankMe reads `encode()`, not `forward()`.** `forward` masks 75 % of the
  tokens; the representation being compared against VICReg's backbone must see
  the whole crop. Tracking RankMe on the masked path would measure the pretext
  task rather than the representation, and the two are not the same thing.
- **No collapse halt.** §4.7's signature is RankMe falling while the loss falls,
  which is a joint-embedding failure mode — MAE has a reconstruction target and
  cannot collapse to a constant embedding without the loss rising. Rather than
  port a detector that cannot fire correctly here, RankMe is recorded every
  epoch and left for the human reading `history.json`.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..models.mae2d import MAE2D, mae_loss
from . import common
from . import instrument as instr
from .vicreg import save_checkpoint


@dataclass
class MAETrainConfig:
    epochs: int = 1
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    device: str = "cuda"
    amp: bool | None = None
    max_rankme_samples: int = 2048
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 25
    resume: bool = True
    log_every_steps: int = 0


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


def train_mae(
    model: MAE2D,
    train_loader,
    config: MAETrainConfig | None = None,
    monitor=None,
) -> dict[str, list[float]]:
    """Train the MAE baseline. Returns per-epoch history including RankMe.

    Args:
        model: a :class:`~ssl_us.models.mae2d.MAE2D`.
        train_loader: iterable of single crop batches, ``(B, C, H, W)``. A
            loader that yields ``(view1, view2)`` tuples is accepted and its
            first element used, so the VICReg loader can be reused unchanged.
        config: hyperparameters.
        monitor: optional `TrainingMonitor`; no-op if ``None``.
    """
    config = config or MAETrainConfig()
    device = torch.device(config.device)
    device_type = device.type
    use_amp = config.amp if config.amp is not None else common.amp_beneficial(device)
    model = model.to(device)

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
        rankme_features: list[torch.Tensor] = []
        n_rankme = 0
        epoch_start = time.monotonic()
        n_crops = 0

        for step, batch in enumerate(train_loader, start=1):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred, target, mask = model(x)
                loss = mae_loss(pred, target, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            n_crops += x.shape[0]

            # Unmasked encode, detached — the probe-facing representation, not
            # the pretext one. Costs one extra forward on a subset of batches.
            if n_rankme < config.max_rankme_samples:
                with torch.no_grad(), torch.autocast(device_type=device_type, enabled=use_amp):
                    h = model.encode(x)
                rankme_features.append(h.detach().cpu())
                n_rankme += h.shape[0]

            if config.log_every_steps and step % config.log_every_steps == 0:
                rate = n_crops / max(time.monotonic() - epoch_start, 1e-9)
                print(f"[e{epoch} s{step}] loss={loss.item():.4f} {rate:.0f} crops/s", flush=True)

        if not losses:
            # A loader that yields nothing is always a config error — most
            # often frames x crops_per_frame < batch_size, so every batch is a
            # dropped partial. Say that, rather than dying in torch.cat below
            # with "expected a non-empty list of Tensors".
            raise ValueError(
                f"epoch {epoch} saw 0 batches: the loader yielded nothing. "
                "Check that n_frames * crops_per_frame >= batch_size."
            )

        history["loss"].append(float(np.mean(losses)))
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

    return dict(history)
