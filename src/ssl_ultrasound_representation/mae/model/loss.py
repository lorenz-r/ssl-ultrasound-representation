import torch


def _mse_per_patch(pred, target):
    """Plain per-patch MSE — (B, L). On the 2 IQ channels this is the squared
    complex error |z_pred - z_target|^2 averaged over the patch."""
    return (pred - target).pow(2).mean(dim=-1)


def _iq_per_patch(
    pred,
    target,
    spatial_shape=None,
    env_weight=1.0,
    phase_weight=1.0,
    spectral_weight=0.0,
    eps=1e-6,
):
    """IQ-aware per-patch loss — (B, L).

    Patches are laid out (iq, tw, w) flattened, so the first half of the last
    axis is I and the second half is Q. The loss has up to three terms:

      * log-envelope MSE   — `|z| = sqrt(I^2 + Q^2)` compared in log space, which
        matches how the B-mode is viewed and tames the heavy IQ dynamic range
        far better than raw MSE (which is owned by the few strong reflectors).
      * amplitude-weighted phase coherence — `1 - cos(Δphase)`, weighted by the
        target envelope so phase is enforced where it carries signal and ignored
        in near-zero regions. Plain MSE barely penalises phase at low amplitude,
        yet phase coherence is exactly what beamforming (and tissue structure)
        depends on.
      * optional spectral term — MSE on the 2-D envelope FFT magnitude, which
        pushes the model to reproduce high-frequency speckle texture instead of
        the smooth conditional mean MSE collapses to. Needs `spatial_shape`.

    Computed in float32 (log / sqrt / fft are unstable in fp16 under autocast).
    """
    pred = pred.float()
    target = target.float()
    half = pred.shape[-1] // 2
    Ip, Qp = pred[..., :half], pred[..., half:]
    It, Qt = target[..., :half], target[..., half:]

    env_p = torch.sqrt(Ip * Ip + Qp * Qp + eps)
    env_t = torch.sqrt(It * It + Qt * Qt + eps)

    # log-envelope MSE
    env_loss = (torch.log(env_p) - torch.log(env_t)).pow(2).mean(dim=-1)  # (B, L)

    # amplitude-weighted phase coherence (cos clamped for numerical safety)
    cos = ((Ip * It + Qp * Qt) / (env_p * env_t)).clamp(-1.0, 1.0)
    w = env_t
    phase_loss = (w * (1.0 - cos)).sum(dim=-1) / (w.sum(dim=-1) + eps)  # (B, L)

    per_patch = env_weight * env_loss + phase_weight * phase_loss

    if spectral_weight > 0:
        if spatial_shape is None:
            raise ValueError("spectral_weight > 0 requires spatial_shape=(tw, w)")
        tw, ww = spatial_shape
        e_p = env_p.reshape(*env_p.shape[:-1], tw, ww)
        e_t = env_t.reshape(*env_t.shape[:-1], tw, ww)
        spec_p = torch.fft.rfft2(e_p).abs()
        spec_t = torch.fft.rfft2(e_t).abs()
        spec_loss = (spec_p - spec_t).pow(2).mean(dim=(-1, -2))  # (B, L)
        per_patch = per_patch + spectral_weight * spec_loss

    return per_patch


def calculate_loss(
    pred,
    target,
    mask,
    loss_on="all",
    loss_type="mse",
    spatial_shape=None,
    env_weight=1.0,
    phase_weight=1.0,
    spectral_weight=0.0,
):
    """
    Reconstruction loss in patch space.

    Args:
        pred:    (B, L, patch_dim) — predicted patches.
        target:  (B, L, patch_dim) — ground-truth patches.
        mask:    (B, L) — 1 where the token was masked, 0 where visible.
        loss_on: "all"    — average over every patch (default; with little
                            training data we don't want to discard the
                            supervision signal from visible patches),
                 "masked" — classic MAE loss over masked patches only.
        loss_type: "mse"  — squared complex error on the IQ channels (default),
                   "iq"   — IQ-aware: log-envelope + amplitude-weighted phase
                            coherence (+ optional spectral), see `_iq_per_patch`.
        spatial_shape: (transducer_window, depth_window); only needed for the
                       spectral term of the "iq" loss.
        env_weight / phase_weight / spectral_weight: term weights for "iq".
    """
    if loss_type == "iq":
        per_patch = _iq_per_patch(
            pred,
            target,
            spatial_shape=spatial_shape,
            env_weight=env_weight,
            phase_weight=phase_weight,
            spectral_weight=spectral_weight,
        )
    else:
        per_patch = _mse_per_patch(pred, target)

    if loss_on == "all" or mask.sum() == 0:
        return per_patch.mean()
    return (per_patch * mask).sum() / mask.sum()
