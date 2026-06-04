import torch

def calculate_loss(pred_x, target_x, ids_restore, mask_ratio):
    """
    Args:
        pred_x: Tensor of shape (Batch, Num_Patches, Sequence, d_model) - the output from the model for masked patches
        target_x: Tensor of shape (Batch, Num_Patches, Sequence, d_model) - the original input for masked patches
        ids_restore: Tensor of shape (Batch, Num_Patches) containing indices to restore original order
        mask_ratio: Float between 0 and 1 indicating the fraction of patches that were masked

    Returns:
        loss: Scalar tensor representing the MSE loss between predicted and target masked patches
    """
    B, P, N, D = pred_x.shape
    len_keep = int(P * (1 - mask_ratio))
    
    ids_masked = ids_restore[:, len_keep:] # [B, P - len_keep]
    
    index = ids_masked.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
    
    pred_masked = torch.gather(pred_x, dim=1, index=index)
    target_masked = torch.gather(target_x, dim=1, index=index)
    
    loss = torch.nn.functional.mse_loss(pred_masked, target_masked)
    return loss