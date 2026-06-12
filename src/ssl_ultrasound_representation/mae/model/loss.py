import torch


def calculate_loss(pred_x, target_x, ids_restore, mask_ratio):
    """
    Compute MSE loss over masked patches in pixel space.

    The model's patchify splits along S with patch_size[0] covering the full depth
    (S dimension), and along C with patch_size[1]=1.  Each patch therefore
    corresponds to one element of the C (transducer) dimension in the
    unpatchified (B, S, C, N) output, so masked patch indices map directly to
    C-axis indices.

    Args:
        pred_x:   (B, S, C, N) — model reconstruction (unpatchified).
        target_x: (B, S, C, N) — original input.
        ids_restore: (B, P) — restore indices from spatial_tube_masking.
        mask_ratio: fraction of patches that were masked.

    Returns:
        MSE over all spatial/temporal positions of the masked patches only.
    """
    B, S, C, N = pred_x.shape
    P = ids_restore.shape[1]
    len_keep = int(P * (1 - mask_ratio))

    ids_masked = ids_restore[:, len_keep:]  # (B, num_masked)

    # Each patch p maps to C-dimension index p
    index = ids_masked[:, None, :, None].expand(-1, S, -1, N)  # (B, S, num_masked, N)

    pred_masked = torch.gather(pred_x, dim=2, index=index)
    target_masked = torch.gather(target_x, dim=2, index=index)

    loss = torch.nn.functional.mse_loss(pred_masked, target_masked)
    return loss
