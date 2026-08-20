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

        complex_mse = _masked_mean((zp - zg).abs().pow(2), m)
        log_env = _masked_mean(
            (torch.log(zp.abs() + self.eps) - torch.log(zg.abs() + self.eps)).abs(), m
        )
        corr = _corr_real(zp, zg, m, dim=1)

        # Spectral terms run on the composite trace the model would actually
        # emit — ground truth where visible, prediction where masked — so they
        # measure the spectral damage the fill-in does rather than re-scoring
        # positions the model was handed.
        mb = m.bool()
        z_comp = torch.where(mb, zp, zg)
        Zp = torch.fft.fft(z_comp, dim=1)
        Zg = torch.fft.fft(zg, dim=1)
        ones = torch.ones_like(Zg.real)
        spec_mag = (
            torch.log(Zp.abs() + self.eps) - torch.log(Zg.abs() + self.eps)
        ).abs().mean()
        spec_corr = _corr_real(Zp, Zg, ones, dim=1)
        spectral = spec_mag + (1.0 - spec_corr)

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
            "spectral_mag": spec_mag.detach(),
            "spectral_corr": spec_corr.detach(),
            "full_mse": full_mse.detach(),
            "complex_corr_masked": corr.detach(),
            "mask_frac": m.mean().detach(),
        }
        return total, parts

    def config(self) -> dict:
        return asdict(self.w)
