"""Physics-aware losses for the RF/IQ conv autoencoder (spec §4).

Dense counterpart of `mae/model/loss.py`, which works in the transformer's
patch space `(B, L, patch_dim)`. Here everything is `(B, 2, n_fast, n_lat)` with
a per-sample boolean mask, and every term is computed on the complex signal
`z = I + jQ` rather than on the two real channels independently.

Why not plain MSE: under uncertainty its optimum is the conditional mean, and
the conditional mean of speckle phase is zero. A model that outputs a smooth
envelope with no carrier scores well on MSE and scores ~0 on
`complex_corr_coeff` — the failure this whole architecture exists to fix. The
phase term is scale- and mean-invariant, so hedging toward zero earns nothing.

Terms (all reduced over masked positions by default):
    complex_mse   |z_pred - z_gt|^2
    log_envelope  L1 on log|z|, matching B-mode's log compression
    phase         1 - Re(rho) per trace — the direct surrogate for
                  `eval.metrics.complex_correlation`
    spectral      the same magnitude/phase pair in the fast-time Fourier domain,
                  which is what `eval.metrics.spectral_coherence` measures

`Re(rho)` is used rather than `|rho|` deliberately: `|rho|` is invariant to a
per-trace phase rotation, so a model could score perfectly while every trace
carried its own phase offset.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

_EPS = 1e-8


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """`(B, 2, n_fast, n_lat)` -> complex `(B, n_fast, n_lat)`.

    Always in float32: complex fp16 is not numerically stable and these terms
    may be called inside `torch.autocast`.
    """
    x = x.float()
    return torch.complex(x[:, 0], x[:, 1])


def from_complex(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_complex`."""
    return torch.stack([z.real, z.imag], dim=1)


def _masked_mean(per_pos: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (per_pos * m).sum() / m.sum().clamp_min(1.0)


def _corr_real(
    zp: torch.Tensor, zg: torch.Tensor, m: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """Per-trace `Re(rho)` over the weighted positions, reduced along `dim`.

    Mean-removed exactly like `eval.metrics.complex_correlation`. Computing it
    per trace rather than once globally gives ~n_lat times more terms and stops
    a model from getting the aggregate right while individual traces are wrong.
    Traces with fewer than two masked samples are dropped — rho is undefined
    there, not zero.
    """
    n = m.sum(dim=dim)
    denom = n.clamp_min(1.0)
    dg = (zg - (zg * m).sum(dim=dim, keepdim=True) / denom.unsqueeze(dim)) * m
    dp = (zp - (zp * m).sum(dim=dim, keepdim=True) / denom.unsqueeze(dim)) * m
    re = (dg.conj() * dp).real.sum(dim=dim)
    ng = dg.abs().pow(2).sum(dim=dim)
    npw = dp.abs().pow(2).sum(dim=dim)
    rho = re / (ng.clamp_min(_EPS) * npw.clamp_min(_EPS)).sqrt()
    valid = n >= 2
    if not bool(valid.any()):
        return torch.zeros((), device=zp.device)
    return rho[valid].mean()


# ── individual terms ────────────────────────────────────────────────────────
# Each takes complex `zp`, `zg` and a float weight map `m` (1 = count this
# position). Passing all-ones reduces over the full signal instead of the masked
# subset, which is how the "_all" diagnostics are produced.


def complex_mse_term(zp, zg, m, eps: float = 1e-6):
    """`mean |z_pred - z_gt|^2`."""
    return _masked_mean((zp - zg).abs().pow(2), m)


def log_envelope_term(zp, zg, m, eps: float = 1e-6):
    """L1 on `log|z|` — mirrors the log compression a B-mode display applies."""
    return _masked_mean(
        (torch.log(zp.abs() + eps) - torch.log(zg.abs() + eps)).abs(), m
    )


def envelope_mse_term(zp, zg, m, eps: float = 1e-6):
    """MSE on the linear envelope. Phase-blind by construction — it is the term
    that looked healthy while `complex_corr_coeff` sat at zero, so it is worth
    logging next to `phase` as the contrast."""
    return _masked_mean((zp.abs() - zg.abs()).pow(2), m)


def phase_term(zp, zg, m, eps: float = 1e-6):
    """`1 - Re(rho)` per trace — the direct surrogate for `complex_corr_coeff`."""
    return 1.0 - _corr_real(zp, zg, m, dim=1)


def spectral_term(zp, zg, m, eps: float = 1e-6):
    """Fast-time Fourier magnitude + phase agreement, tied to `spectral_coherence`.

    Runs on the composite trace the model would actually emit — ground truth
    where visible, prediction where masked — so it measures the spectral damage
    the fill-in does rather than re-scoring positions the model was handed. With
    an all-ones `m` the composite is simply the prediction.
    """
    z_comp = torch.where(m.bool(), zp, zg)
    Zp = torch.fft.fft(z_comp, dim=1)
    Zg = torch.fft.fft(zg, dim=1)
    mag = (torch.log(Zp.abs() + eps) - torch.log(Zg.abs() + eps)).abs().mean()
    corr = _corr_real(Zp, Zg, torch.ones_like(Zg.real), dim=1)
    return mag + (1.0 - corr)


_TERMS = {
    "complex_mse": complex_mse_term,
    "log_envelope": log_envelope_term,
    "envelope_mse": envelope_mse_term,
    "phase": phase_term,
    "spectral": spectral_term,
}
RF_METRICS = tuple(_TERMS)


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    names: "list[str] | tuple[str, ...]",
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Diagnostic reconstruction metrics — for monitoring, not optimisation.

    Dense counterpart of `mae/model/loss.compute_metrics`, and the same
    convention: each name is reduced twice, over the masked positions
    ("<name>_masked", what the model is actually asked to predict) and over every
    position ("<name>_all"). Spec §4 asks for exactly this — the terms are
    optimised on masked positions but reported on the full signal too, because
    with a masked objective the visible region receives no gradient and a
    whole-frame reconstruction can be far worse than the loss suggests.

    Complements `eval.metrics` (physics fidelity on whole frames, every
    `vis_every` epochs); this is the loss-family breakdown, every batch.
    """
    zp, zg = to_complex(pred), to_complex(target)
    m = mask.float()
    if m.ndim == 4:
        m = m[:, 0]
    ones = torch.ones_like(m)
    out: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in _TERMS:
            raise ValueError(f"unknown metric {name!r}; valid: {RF_METRICS}")
        fn = _TERMS[name]
        out[f"{name}_masked"] = fn(zp, zg, m, eps).detach()
        out[f"{name}_all"] = fn(zp, zg, ones, eps).detach()
    return out


@dataclass
class LossWeights:
    """Spec §4: start with the complex and phase terms large relative to envelope.

    If `complex_corr_coeff` stays near zero in training, this is one of the two
    things to check (the other is the fast-time receptive field).
    """

    w_cpx: float = 1.0
    w_env: float = 0.1
    w_phase: float = 1.0
    w_spec: float = 0.3
    w_full: float = 0.1  # small full-signal complex MSE, incl. visible positions


class RFMaeLoss(nn.Module):
    """Weighted sum of the physics terms, reduced on masked positions.

    Returns `(total, parts)` where `parts` holds every detached component plus
    `complex_corr_masked` — the quantity model selection is gated on, logged
    every step rather than only on the periodic full-frame physics pass.
    """

    def __init__(self, weights: LossWeights | None = None, eps: float = 1e-6):
        super().__init__()
        self.w = weights or LossWeights()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zp, zg = to_complex(pred), to_complex(target)
        m = mask.float()
        if m.ndim == 4:
            m = m[:, 0]

        complex_mse = complex_mse_term(zp, zg, m, self.eps)
        log_env = log_envelope_term(zp, zg, m, self.eps)
        corr = _corr_real(zp, zg, m, dim=1)
        spectral = spectral_term(zp, zg, m, self.eps)
        full_mse = (zp - zg).abs().pow(2).mean()

        total = (
            self.w.w_cpx * complex_mse
            + self.w.w_env * log_env
            + self.w.w_phase * (1.0 - corr)
            + self.w.w_spec * spectral
            + self.w.w_full * full_mse
        )
        parts = {
            "complex_mse": complex_mse.detach(),
            "log_envelope": log_env.detach(),
            "phase": (1.0 - corr).detach(),
            "spectral": spectral.detach(),
            "full_mse": full_mse.detach(),
            "complex_corr_masked": corr.detach(),
            "mask_frac": m.mean().detach(),
        }
        return total, parts

    def config(self) -> dict:
        return asdict(self.w)
