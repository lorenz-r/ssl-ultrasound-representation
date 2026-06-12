import torch


def calculate_loss(pred, target, mask, loss_on="all"):
    """
    MSE reconstruction loss in patch space.

    Args:
        pred:    (B, L, patch_dim) — predicted patches.
        target:  (B, L, patch_dim) — ground-truth patches.
        mask:    (B, L) — 1 where the token was masked, 0 where visible.
        loss_on: "all"    — average over every patch (default; with little
                            training data we don't want to discard the
                            supervision signal from visible patches),
                 "masked" — classic MAE loss over masked patches only.
    """
    if loss_on == "all" or mask.sum() == 0:
        return torch.nn.functional.mse_loss(pred, target)

    per_patch = (pred - target).pow(2).mean(dim=-1)  # (B, L)
    return (per_patch * mask).sum() / mask.sum()
