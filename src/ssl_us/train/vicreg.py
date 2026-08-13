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

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from ..models.vicreg import VICReg, VICRegLossWeights, vicreg_loss
from . import common
from . import instrument as instr


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

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_terms = defaultdict(list)
        rankme_features: list[torch.Tensor] = []
        n_rankme = 0

        for x1, x2 in train_loader:
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

            if probe is not None and probe_label_fn is not None:
                labels = probe_label_fn(x1).to(device)
                epoch_terms["probe_loss"].append(probe.step(h1.detach(), labels))

        for name, values in epoch_terms.items():
            history[name].append(float(np.mean(values)))

        feats = torch.cat(rankme_features, dim=0)[: config.max_rankme_samples]
        spectrum = instr.singular_value_spectrum(feats)
        history["rankme"].append(instr.rankme(spectrum))

        if monitor is not None:
            monitor.log_epoch(epoch, {f"train/{k}": v[-1] for k, v in history.items()})

    return dict(history)
