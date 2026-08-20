"""Physics-aware losses and diagnostics for the RF/IQ conv autoencoder (spec §4).

Dense counterpart of `mae/model/loss.py`, which works in the transformer's patch
space `(B, L, patch_dim)`. Here everything is `(B, 2, n_fast, n_lat)` with a
per-sample mask, and every term is computed on the complex signal `z = I + jQ`
rather than on the two real channels independently.

Why not plain MSE: under uncertainty its optimum is the conditional mean, and the
conditional mean of speckle phase is zero. A model that outputs a smooth envelope
with no carrier scores well on MSE and ~0 on `complex_corr_coeff` — the failure
this architecture exists to fix. `correlation` is scale- and mean-invariant, so
hedging toward zero earns nothing.

Naming
------
The metric names and formulas below **match `mae/model/loss.py` exactly**
(`mse`, `envelope`, `log_envelope`, `phase`, `correlation`, `spectral`), so a
number logged here means the same thing as the number logged by `train.py` and
can be compared against the existing runs. Two caveats on that comparison, both
about grouping rather than definition:

* `correlation` is reduced per *trace* here and per *patch* there, and `spectral`
  transforms the full 215-sample trace here versus a 43-sample depth window there.
  Same quantities, different support.

The three terms the loss actually optimises follow the spec rather than the repo
and so carry distinct names — `complex_mse` (= 2x `mse`), `log_envelope_l1`
(L1 on log-amplitude, not MSE on log-power) and `spectral_coh` (log-magnitude
plus spectral correlation, which is what `eval.metrics.spectral_coherence`
measures). They are in the registry too, so you can log exactly what is optimised
alongside the repo-comparable versions.

`Re(rho)` is used rather than `|rho|` deliberately: `|rho|` is invariant to a
per-trace phase rotation, so a model could score perfectly while every trace
carried its own phase offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

_EPS = 1e-8


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """`(B, 2, n_fast, n_lat)` -> complex `(B, n_fast, n_lat)`.

    Always float32: complex fp16 is not numerically stable and these terms may be
    called inside `torch.autocast`.
    """
    x = x.float()
    return torch.complex(x[:, 0], x[:, 1])


def from_complex(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_complex`."""
    return torch.stack([z.real, z.imag], dim=1)


def _masked_mean(per_pos: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (per_pos * m).sum() / m.sum().clamp_min(1.0)


def _trace_weights(m: torch.Tensor) -> torch.Tensor:
    """Per-trace weights `(B, n_lat)`: 1 where a trace holds any masked sample.

    The spectral terms transform a whole trace, so "reduce over masked positions"
    has no meaning at sample granularity — the nearest honest equivalent is to
    score only the traces the model actually had to fill in.
    """
    return (m.sum(dim=1) > 0).float()


def _corr_real(
    zp: torch.Tensor, zg: torch.Tensor, m: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """Per-trace `Re(rho)` over the weighted positions, reduced along `dim`.

    Mean-removed exactly like `eval.metrics.complex_correlation`. Computed per
    trace rather than once globally: that gives ~n_lat times more terms and stops
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


# ── terms ───────────────────────────────────────────────────────────────────
# Signature: (zp, zg, m, eps) -> scalar, where `m` is a float weight map
# (1 = count this position). Passing all-ones reduces over the full signal
# instead of the masked subset, which is how the "_all" diagnostics are made.


def mse_term(zp, zg, m, eps: float = 1e-6):
    """Squared IQ error, averaged over I and Q — `loss.py`'s `mse`.

    Equals `mean|z_pred - z_gt|^2 / 2`; the factor of two is the only difference
    from `complex_mse`, and it is absorbed into the loss weight.
    """
    return _masked_mean((zp - zg).abs().pow(2) * 0.5, m)


def complex_mse_term(zp, zg, m, eps: float = 1e-6):
    """`mean |z_pred - z_gt|^2` — spec §4 term 1 (= 2x :func:`mse_term`)."""
    return _masked_mean((zp - zg).abs().pow(2), m)


def envelope_term(zp, zg, m, eps: float = 1e-6):
    """MSE on the linear envelope `|z|` — `loss.py`'s `envelope`.

    Phase-invariant by construction, so it is the contrast to watch: envelope
    falling while `correlation` stays flat is exactly the carrier-ignoring
    failure mode.
    """
    return _masked_mean((zp.abs() - zg.abs()).pow(2), m)


def log_envelope_term(zp, zg, m, eps: float = 1e-6):
    """MSE on log-compressed power `log(I^2+Q^2+eps)` — `loss.py`'s `log_envelope`.

    Mirrors the log compression a B-mode display applies, equalising near- and
    far-field contributions that plain MSE lets the shallow signal dominate.
    """
    return _masked_mean(
        (torch.log(zp.abs().pow(2) + eps) - torch.log(zg.abs().pow(2) + eps)).pow(2), m
    )


def log_envelope_l1_term(zp, zg, m, eps: float = 1e-6):
    """L1 on log-amplitude `log(|z|+eps)` — spec §4 term 2, optimised by the loss."""
    return _masked_mean(
        (torch.log(zp.abs() + eps) - torch.log(zg.abs() + eps)).abs(), m
    )


def phase_term(zp, zg, m, eps: float = 1e-6):
    """Mean `1 - cos(dphi)` per sample — `loss.py`'s `phase`.

    Each IQ sample treated as a unit phasor: pure phase agreement, with amplitude
    divided out. Distinct from `correlation`, which is a normalised inner product
    over a whole trace.
    """
    denom = (zp.abs() * zg.abs()).clamp_min(1e-8)
    cos_d = (zp * zg.conj()).real / denom
    return _masked_mean(1.0 - cos_d, m)


def correlation_term(zp, zg, m, eps: float = 1e-6):
    """`1 - Re(rho)` per trace — `loss.py`'s `correlation`, and spec §4 term 3.

    The direct surrogate for `eval.metrics.complex_corr_coeff`, the metric model
    selection is gated on. Range [0, 2]; 0 is perfect, 1.0 is uncorrelated.
    """
    return 1.0 - _corr_real(zp, zg, m, dim=1)


def spectral_term(zp, zg, m, eps: float = 1e-6):
    """MSE on the fast-time magnitude spectrum — `loss.py`'s `spectral`.

    Penalises differences in `|Z(f)|` only, so it constrains centre frequency and
    bandwidth but not spectral phase. Scored on the traces that contain masked
    samples (see :func:`_trace_weights`).
    """
    Zp = torch.fft.fft(zp, dim=1)
    Zg = torch.fft.fft(zg, dim=1)
    per_trace = (Zp.abs() - Zg.abs()).pow(2).mean(dim=1)
    w = _trace_weights(m)
    return (per_trace * w).sum() / w.sum().clamp_min(1.0)


def spectral_coh_term(zp, zg, m, eps: float = 1e-6):
    """Spectral log-magnitude L1 + spectral correlation — spec §4 term 4.

    Unlike `spectral` this also constrains spectral *phase*, which is what
    `eval.metrics.spectral_coherence` measures. Computed on the composite trace
    the model would actually emit — ground truth where visible, prediction where
    masked — so it scores the damage the fill-in does rather than re-scoring
    positions the model was handed. With an all-ones `m` the composite is simply
    the prediction.
    """
    z_comp = torch.where(m.bool(), zp, zg)
    Zp = torch.fft.fft(z_comp, dim=1)
    Zg = torch.fft.fft(zg, dim=1)
    mag = (torch.log(Zp.abs() + eps) - torch.log(Zg.abs() + eps)).abs().mean()
    corr = _corr_real(Zp, Zg, torch.ones_like(Zg.real), dim=1)
    return mag + (1.0 - corr)


# Names and formulas on the left of each pair match `mae/model/loss.py`; the
# three below them are the spec's variants that this loss optimises.
_TERMS = {
    "mse": mse_term,
    "envelope": envelope_term,
    "log_envelope": log_envelope_term,
    "phase": phase_term,
    "correlation": correlation_term,
    "spectral": spectral_term,
    "complex_mse": complex_mse_term,
    "log_envelope_l1": log_envelope_l1_term,
    "spectral_coh": spectral_coh_term,
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

    Same convention as `mae/model/loss.compute_metrics`: each name is reduced
    twice, over the masked positions ("<name>_masked", what the model is actually
    asked to predict) and over every position ("<name>_all"). Spec §4 asks for
    both, and it matters here because the visible region receives no gradient, so
    a whole-frame reconstruction can be far worse than the optimised loss implies.

    Complements `eval.metrics` (physics fidelity on whole frames, every
    `vis_every` epochs); this is the loss-family breakdown, every batch.

    Returns a dict of detached 0-d tensors.
    """
    zp, zg = to_complex(pred), to_complex(target)
    m = mask.float()
    if m.ndim == 4:
        m = m[:, 0]
    ones = torch.ones_like(m)
    out: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in _TERMS:
            raise ValueError(f"unknown metric {name!r}; valid: {list(RF_METRICS)}")
        fn = _TERMS[name]
        out[f"{name}_masked"] = fn(zp, zg, m, eps).detach()
        out[f"{name}_all"] = fn(zp, zg, ones, eps).detach()
    return out


@dataclass
class LossWeights:
    """Spec §4: start with the complex and correlation terms large relative to
    envelope.

    If `complex_corr_coeff` stays near zero in training, this is one of the two
    things to check (the other is the fast-time receptive field).
    """

    w_cpx: float = 1.0
    w_env: float = 0.1
    w_phase: float = 1.0  # weight on the correlation term (spec §4 term 3)
    w_spec: float = 0.3
    w_full: float = 0.1  # small full-signal complex MSE, incl. visible positions


class RFMaeLoss(nn.Module):
    """Weighted sum of the physics terms, reduced on masked positions.

    Returns `(total, parts)`. `parts` holds every detached component under its
    own name — `complex_mse`, `log_envelope_l1`, `correlation`, `spectral_coh`
    — plus `complex_corr_masked`, the `Re(rho)` that model selection is gated on,
    logged every step rather than only on the periodic full-frame physics pass.
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
        log_env = log_envelope_l1_term(zp, zg, m, self.eps)
        corr = _corr_real(zp, zg, m, dim=1)
        spectral = spectral_coh_term(zp, zg, m, self.eps)
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
            "log_envelope_l1": log_env.detach(),
            "correlation": (1.0 - corr).detach(),
            "spectral_coh": spectral.detach(),
            "full_mse": full_mse.detach(),
            "complex_corr_masked": corr.detach(),
            "mask_frac": m.mean().detach(),
        }
        return total, parts

    def config(self) -> dict:
        return asdict(self.w)
