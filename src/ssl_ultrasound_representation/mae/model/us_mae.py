import torch
import torch.nn as nn

from .pos_encoding import get_sinusoidal_positional_encoding

class SignalTransformerAutoencoder(nn.Module):
    def __init__(self, original_channels=847, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, patch_size=(847, 1), mask_ratio=0.75):
        """
        Args:
            original_channels (int): The original signal depth (847 in diagram).
            d_model (int): The embedding dimension (128 in diagram).
            nhead (int): Number of attention heads (must divide d_model evenly).
            num_layers (int): Number of stacked Transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network inside the Transformer.
            mask_ratio (float): The fraction of patches to mask during training.
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 1. Input Linear Projection (847 => 128)
        # We use a 1D convolution here as a clean way to project channels across a sequence
        self.input_projection = nn.Sequential(
            nn.Linear(in_features=original_channels, out_features=d_model),
            nn.GELU() # Non-linearity after projection
        )

        # 2. Transformer Block
        # Setting batch_first=True expects inputs of shape (Batch, Sequence, Features)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu", # GELU is standard for modern transformers
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Linear Projection (128 => 847)
        self.output_projection = nn.Linear(in_features=d_model, out_features=original_channels)

    def forward(self, x, pos_embed_spatial, pos_embed_temporal):
        """
        Args:
            x: Tensor of shape (Batch, 847, 192, 576)
            pos_embed_spatial: Tensor of shape (192, 128)
            pos_embed_temporal: Tensor of shape (576, 128)

        Returns:
            Reconstructed Tensor of shape (Batch, 847, 192, 576)
        """
        # x is currently (Batch, Sample=847, Channels=192, Sequence=576)
        B, S, C, N = x.shape
        
        # Project down to embedding dimension: 847 -> 128
        x = self.patchify(x, patch_size=self.patch_size) # Shape becomes (Batch, Num_Patches, Sequence, Flatten_Patch_Size)
        x = self.input_projection(x) # Shape becomes (Batch, Num_Patches, Sequence, d_model=128)

        B, P, N, D = x.shape

        # positional embedding (broadcasted across the batch)
        pos_embed_spatial = pos_embed_spatial.view(1, P, 1, D).to(x.device)
        pos_embed_temporal = pos_embed_temporal.view(1, 1, N, D).to(x.device)
        pe = pos_embed_spatial + pos_embed_temporal # Shape becomes (1, Num_Patches, Sequence, d_model)

        # Masking
        x_masked, ids_restore, len_keep = self.spatial_tube_masking(x, self.mask_ratio)

        # Currently no encoder transformer, so skip and do unshuffle directly
        x = self.spatial_tube_unshuffle(x_masked, ids_restore, len_keep)
        x = x + pe # Add positional encoding to the input embeddings

        x = x.view(B, P * N, D)
        
        # Pass through the Transformer block
        x = self.transformer(x) # Shape remains (Batch, 110592, 128)
        
        # Permute back to channel-first for the final projection
        x = x.view(B, P, N, D) # Shape becomes (Batch, Num_Patches, Sequence, d_model)
        
        # Project back to original dimension: 128 -> 847
        x = self.output_projection(x) # Shape becomes (Batch, Num_Patches, Sequence, 847)
        x = self.unpatchify(x, S, C, patch_size=self.patch_size)
        
        return x, ids_restore, len_keep

    def patchify(self, x, patch_size=(847, 1)):
        """
        Args:
            x: Tensor of shape (Batch, Sample=847, Channels=192, Sequence=576)
            patch_size: Tuple indicating the size of each patch (Sample, Channels)
        
        Returns:
            Tensor of shape (Batch, Num_Patches, Patch_Size)
        """
        B, S, C, N = x.shape
        patch_S, patch_C = patch_size # (847, 1) for simple patching along the sample dimension

        assert C % patch_C == 0, "Channels must be divisible by patch channel size"
        # Ensure the input dimensions are divisible by the patch size, padding 0 otherswise
        if S % patch_S != 0:
            pad_S = patch_S - (S % patch_S)
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad_S)) # Pad along the sample dimension
            S += pad_S

        num_S = S // patch_S
        num_C = C // patch_C

        x = x.view(B, num_S, patch_S, num_C, patch_C, N)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, num_S * num_C, N, patch_S * patch_C)
        
        return x

    def unpatchify(self, x, original_S, original_C, patch_size=(847, 1)):
        """
        Args:
            x: Tensor of shape (Batch, Num_Patches, Sequence, Patch_Size)
            original_S: Original sample dimension (847)
            original_C: Original channel dimension (128)
            patch_size: Tuple indicating the size of each patch (Sample, Channels)
        
        Returns:
            Tensor of shape (Batch, Sample=847, Channels=128, Sequence=576)
        """
        B, num_patches, N, patch_dim = x.shape
        patch_S, patch_C = patch_size

        pad_S = 0 if original_S % patch_S == 0 else patch_S - (original_S % patch_S)
        padded_S = original_S + pad_S
        
        num_S = padded_S // patch_S
        num_C = original_C // patch_C

        x = x.view(B, num_S, num_C, N, patch_S, patch_C)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, padded_S, original_C, N)
        
        if pad_S > 0:
            x = x[:, :original_S, :, :]

        return x

    def spatial_tube_masking(self, x, mask_ratio):
        """
        Args:
            x: Tensor of shape (Batch, Num_Patches, Sequence, d_model)
            mask_ratio: Float between 0 and 1 indicating the fraction of patches to mask

        Returns:
            x_masked: Tensor of shape (Batch, Num_Kept_Patches, Sequence, d_model)
            ids_restore: Tensor of shape (Batch, Num_Patches) containing indices to restore original order
            len_keep: Integer indicating the number of patches kept (unmasked)
        """
        B, P, N, D = x.shape
        len_keep = int(P * (1 - mask_ratio))
        
        noise = torch.rand(B, P, device=x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        
        expanded_ids_keep = ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
        
        x_masked = torch.gather(x, dim=1, index=expanded_ids_keep)
        
        return x_masked, ids_restore, len_keep

    def spatial_tube_unshuffle(self, x_encoded, ids_restore, len_keep):
        """
        Args:
            x_encoded: Tensor of shape (Batch, Num_Kept_Patches, Sequence, d_model)
            ids_restore: Tensor of shape (Batch, Num_Patches) containing indices to restore original order
            len_keep: Integer indicating the number of patches kept (unmasked)

        Returns:
            x_restored: Tensor of shape (Batch, Num_Patches, Sequence, d_model) with masked positions filled with mask tokens
        """
        B, _, N, D = x_encoded.shape
        P = ids_restore.shape[1] # Total number of patches (including masked ones)

        mask_tokens = self.mask_token.expand(B, P - len_keep, N, D)
        
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)
        
        expanded_ids_restore = ids_restore.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
        
        x_restored = torch.gather(x_full, dim=1, index=expanded_ids_restore)
        
        return x_restored
