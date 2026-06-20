import torch


def _mse_per_patch(pred, target):
    """Plain per-patch MSE — (B, L). On the 2 IQ channels this is the squared
    complex error |z_pred - z_target|^2 averaged over the patch."""
    return (pred - target).pow(2).mean(dim=-1)


def calculate_loss(pred, target, mask, loss_on="all"):
    """
    Reconstruction loss in patch space (squared complex error on the IQ channels).

    Args:
        pred:    (B, L, patch_dim) — predicted patches.
        target:  (B, L, patch_dim) — ground-truth patches.
        mask:    (B, L) — 1 where the token was masked, 0 where visible.
        loss_on: "all"    — average over every patch (default; with little
                            training data we don't want to discard the
                            supervision signal from visible patches),
                 "masked" — classic MAE loss over masked patches only.
    """
    per_patch = _mse_per_patch(pred, target)

    if loss_on == "all" or mask.sum() == 0:
        return per_patch.mean()
    return (per_patch * mask).sum() / mask.sum()
