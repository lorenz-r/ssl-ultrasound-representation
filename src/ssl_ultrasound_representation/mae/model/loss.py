"""Patch-space reconstruction losses for RF ultrasound IQ data.

Patch layout (flat): (B, L, patch_dim) where patch_dim = 2 * tw * w.
The first half of dim-2 is the I channel, second half is Q, each laid out
as (transducer_window, depth_window) row-major — matching patchify().
"""

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_iq(x):
    """(B, L, 2*P) -> I, Q each (B, L, P)."""
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


# ---------------------------------------------------------------------------
# Per-patch losses — all return (B, L)
# ---------------------------------------------------------------------------


def _mse_per_patch(pred, target):
    """Squared complex error |z_pred - z_target|^2 averaged over the patch."""
    return (pred - target).pow(2).mean(dim=-1)


def _envelope_mse_per_patch(pred, target, eps: float = 1e-6):
    """MSE on the instantaneous envelope √(I²+Q²).

    Phase-invariant: two patches with identical amplitude but shifted phase
    score zero. Good for B-mode quality without penalising receive-phase
    errors (which are irrelevant after envelope detection).
    """
    I_p, Q_p = _split_iq(pred)
    I_t, Q_t = _split_iq(target)
    env_p = (I_p.pow(2) + Q_p.pow(2) + eps).sqrt()
    env_t = (I_t.pow(2) + Q_t.pow(2) + eps).sqrt()
    return (env_p - env_t).pow(2).mean(dim=-1)


def _log_envelope_mse_per_patch(pred, target, eps: float = 1e-6):
    """MSE on log-compressed envelope log(I²+Q²+ε).

    Mirrors the log compression applied in B-mode display: equalises the
    contribution of near-field (high SNR) and far-field (low SNR) patches
    that would otherwise be dominated by the shallow signal in plain MSE.
    Note: the gradient w.r.t. very small amplitudes grows as 1/amplitude,
    so use a slightly larger eps or pair with mse if training is unstable.
    """
    I_p, Q_p = _split_iq(pred)
    I_t, Q_t = _split_iq(target)
    log_env_p = torch.log(I_p.pow(2) + Q_p.pow(2) + eps)
    log_env_t = torch.log(I_t.pow(2) + Q_t.pow(2) + eps)
    return (log_env_p - log_env_t).pow(2).mean(dim=-1)


def _phase_per_patch(pred, target):
    """Mean (1 − cos Δφ) over patch samples — 0 when phases match exactly.

    Treats each IQ sample as a unit phasor and measures phase coherence.
    This directly optimises the receive-phase structure that coherence-based
    imaging methods (SLSC, coherence factor) depend on, without conflating
    amplitude errors.
    """
    I_p, Q_p = _split_iq(pred)
    I_t, Q_t = _split_iq(target)
    env_p = (I_p.pow(2) + Q_p.pow(2)).sqrt().clamp(min=1e-8)
    env_t = (I_t.pow(2) + Q_t.pow(2)).sqrt().clamp(min=1e-8)
    cos_diff = (I_p * I_t + Q_p * Q_t) / (env_p * env_t)
    return (1.0 - cos_diff).mean(dim=-1)


def _spectral_per_patch(pred, target, transducer_window: int, depth_window: int):
    """MSE on the complex magnitude spectrum along the depth axis.

    Treats each patch as an analytic signal (z = I + jQ), FFTs along depth,
    and penalises differences in |Z(f)|. This encourages the model to
    preserve the RF centre frequency and bandwidth — tissue-specific spectral
    signatures invisible to amplitude-only or point-wise losses.

    Casting to float32 before FFT is intentional: complex fp16 is not
    numerically stable, and this function may be called inside torch.autocast.
    """
    B, L = pred.shape[:2]
    I_p, Q_p = _split_iq(pred)  # (B, L, tw*w)
    I_t, Q_t = _split_iq(target)

    # (B*L, tw, w) analytic signal — float32 for FFT stability
    z_p = (I_p.float() + 1j * Q_p.float()).view(B * L, transducer_window, depth_window)
    z_t = (I_t.float() + 1j * Q_t.float()).view(B * L, transducer_window, depth_window)

    Z_p = torch.fft.fft(z_p, dim=-1)  # (B*L, tw, w)
    Z_t = torch.fft.fft(z_t, dim=-1)

    per_token = (Z_p.abs() - Z_t.abs()).pow(2).mean(dim=(-2, -1))
    return per_token.view(B, L)


# ---------------------------------------------------------------------------
# Registry and aggregation
# ---------------------------------------------------------------------------

_REGISTRY = {
    "mse": _mse_per_patch,
    "envelope": _envelope_mse_per_patch,
    "log_envelope": _log_envelope_mse_per_patch,
    "phase": _phase_per_patch,
}
_SPECTRAL = "spectral"

LOSS_TYPES = sorted(_REGISTRY) + [_SPECTRAL]


def _reduce(per_patch, mask, loss_on: str) -> torch.Tensor:
    if loss_on == "all" or mask.sum() == 0:
        return per_patch.mean()
    return (per_patch * mask).sum() / mask.sum()


def _per_patch(
    name: str,
    pred,
    target,
    transducer_window: int | None,
    depth_window: int | None,
):
    if name == _SPECTRAL:
        if depth_window is None or transducer_window is None:
            raise ValueError(
                "'spectral' loss requires depth_window and transducer_window"
            )
        return _spectral_per_patch(pred, target, transducer_window, depth_window)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown loss_type {name!r}. Valid types: {LOSS_TYPES}. "
            "Combine two with '+' (e.g. 'mse+envelope')."
        )
    return _REGISTRY[name](pred, target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_loss(
    pred,
    target,
    mask,
    loss_on: str = "all",
    loss_type: str = "mse",
    alpha: float = 0.5,
    depth_window: int | None = None,
    transducer_window: int | None = None,
) -> torch.Tensor:
    """Reconstruction loss in patch space for RF ultrasound IQ data.

    Args:
        pred, target:    (B, L, patch_dim) — predicted / ground-truth patches.
        mask:            (B, L) — 1 where masked, 0 where visible.
        loss_on:         "all"    — mean over every patch (default; preserves
                                    visible-token supervision with scarce data),
                         "masked" — classic MAE, masked patches only.
        loss_type:       One name or two '+'-joined names from
                           "mse"          — squared IQ error (default)
                           "envelope"     — envelope MSE, phase-invariant
                           "log_envelope" — log-compressed envelope MSE
                           "phase"        — cosine phase-distance
                           "spectral"     — magnitude-spectrum MSE along depth
                         Compound example: "mse+envelope" →
                           (1-alpha)*mse + alpha*envelope.
        alpha:           Weight of the second term for compound loss_type.
        depth_window, transducer_window:
                         Required only when loss_type includes "spectral";
                         must match the values used to build the model.
    """
    terms = [t.strip() for t in loss_type.split("+")]
    if len(terms) > 2:
        raise ValueError(
            f"loss_type supports at most two '+'-separated terms; got {loss_type!r}"
        )

    # Compute the loss in fp32 even under torch.autocast. `pred` is fp16 there
    # while `target` (a view of the fp32 input) is fp32, so MSE's `pred - target`
    # is promoted to fp32 and stays safe — but the amplitude losses square the
    # raw fp16 `pred` first (`I_p**2`), which overflows fp16 (max 65504) as soon
    # as a sample exceeds 256. Per-depth RMS normalization lets a bright specular
    # reflector reach ~sqrt(N), so this produces inf/NaN losses on real data.
    pred = pred.float()
    target = target.float()

    pp0 = _per_patch(terms[0], pred, target, transducer_window, depth_window)
    l0 = _reduce(pp0, mask, loss_on)

    if len(terms) == 1:
        return l0

    pp1 = _per_patch(terms[1], pred, target, transducer_window, depth_window)
    l1 = _reduce(pp1, mask, loss_on)
    return (1.0 - alpha) * l0 + alpha * l1
