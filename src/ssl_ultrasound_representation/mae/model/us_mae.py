import torch
import torch.nn as nn

class FAIRSignalMAE(nn.Module):
    def __init__(self, 
                 original_channels=847, 
                 embed_dim=128, 
                 encoder_depth=4, 
                 encoder_heads=8,
                 decoder_embed_dim=64, 
                 decoder_depth=2, 
                 decoder_heads=4,
                 dim_feedforward=512, 
                 patch_size=(847, 1), 
                 mask_ratio=0.75,
                 grid_spatial=192,
                 grid_temporal=576):
        """
        Authentic asymmetric Masked Autoencoder architecture completely aligned with FAIR's design.
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # -------------------------------------------------------------------------
        # 1. MAE ENCODER MODULES
        # -------------------------------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.Linear(in_features=original_channels, out_features=embed_dim),
            nn.GELU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Fix 3: Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)  # Fix 1 & 6: Final encoder norm

        # -------------------------------------------------------------------------
        # 2. MAE DECODER MODULES (Asymmetric & Lightweight)
        # -------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Fix 5: Mask token initialized to zeros per FAIR paper
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_embed_dim))
        
        # Fix 4: Decoupled Decoder Positional Embeddings
        self.decoder_pos_embed_spatial = nn.Parameter(torch.randn(1, grid_spatial, 1, decoder_embed_dim) * 0.02)
        self.decoder_pos_embed_temporal = nn.Parameter(torch.randn(1, 1, grid_temporal, decoder_embed_dim) * 0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Fix 3: Pre-LN for decoder blocks
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)  # Fix 2 & 6: Final decoder norm
        
        self.output_projection = nn.Linear(in_features=decoder_embed_dim, out_features=original_channels)

        # Initialize mask token and learned embeddings cleanly
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize mask token to zero, matching the FAIR MAE paper specification."""
        torch.nn.init.constant_(self.mask_token, 0.)

    def forward(self, x, pos_embed_spatial, pos_embed_temporal):
        """
        Args:
            x: Tensor of shape (Batch, 847, 192, 576)
            pos_embed_spatial: Encoder Spatial PE Tensor of shape (192, 128)
            pos_embed_temporal: Encoder Temporal PE Tensor of shape (576, 128)
        """
        B, S, C, N = x.shape
        
        # --- 1. Patchify & Embed ---
        x = self.patchify(x, patch_size=self.patch_size) 
        x = self.input_projection(x) 

        B, P, N, D = x.shape

        # --- 2. Calculate Encoder Positional Embeddings ---
        pos_embed_spatial = pos_embed_spatial.view(1, P, 1, D).to(x.device)
        pos_embed_temporal = pos_embed_temporal.view(1, 1, N, D).to(x.device)
        pe_encoder = pos_embed_spatial + pos_embed_temporal 

        # Add positional embedding to the full sequence before slicing out visible ones
        x = x + pe_encoder

        # --- 3. Masking (Keep only a fraction of the patches) ---
        x_masked, ids_restore, len_keep = self.spatial_tube_masking(x, self.mask_ratio)

        # --- 4. Pass through ENCODER ---
        x_masked = x_masked.view(B, len_keep * N, D)
        x_encoded = self.encoder(x_masked) 
        x_encoded = self.encoder_norm(x_encoded)  # Fix 1 & 6: Added final layer norm
        x_encoded = x_encoded.view(B, len_keep, N, D)

        # --- 5. Project to Decoder Space ---
        x_decoder_input = self.decoder_embed(x_encoded) 

        # --- 6. Unshuffle & Append Mask Tokens ---
        x_full = self.spatial_tube_unshuffle(x_decoder_input, ids_restore, len_keep) 

        # --- 7. Add Decoupled Decoder Positional Embeddings ---
        # Fix 4: No longer projecting encoder PE, using dedicated decoder parameters
        pe_decoder = self.decoder_pos_embed_spatial + self.decoder_pos_embed_temporal
        x_full = x_full + pe_decoder

        # --- 8. Pass through DECODER ---
        x_full = x_full.view(B, P * N, self.decoder_embed_dim)
        x_decoded = self.decoder(x_full)
        x_decoded = self.decoder_norm(x_decoded)  # Fix 2 & 6: Added final layer norm
        x_decoded = x_decoded.view(B, P, N, self.decoder_embed_dim)
        
        # --- 9. Output Projection & Unpatchify ---
        x_out = self.output_projection(x_decoded) 
        x_final = self.unpatchify(x_out, S, C, patch_size=self.patch_size)
        
        return x_final, ids_restore, len_keep

    def patchify(self, x, patch_size=(847, 1)):
        """
        Split signal into non-overlapping patches along spatial and channel dimensions.

        Input (B, S, C, N) is reshaped into (B, num_S * num_C, N, patch_S * patch_C)
        where each patch groups a (patch_S, patch_C) block. Pads S if needed.
        """
        B, S, C, N = x.shape
        patch_S, patch_C = patch_size
        assert C % patch_C == 0, "Channels must be divisible by patch channel size"
        if S % patch_S != 0:
            pad_S = patch_S - (S % patch_S)
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, pad_S))
            S += pad_S
        num_S = S // patch_S
        num_C = C // patch_C
        x = x.view(B, num_S, patch_S, num_C, patch_C, N)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, num_S * num_C, N, patch_S * patch_C)
        return x

    def unpatchify(self, x, original_S, original_C, patch_size=(847, 1)):
        """
        Reverse of patchify: reassemble patches back into the original signal layout.

        Input (B, num_patches, N, patch_dim) is reshaped back to (B, original_S, original_C, N).
        Any padding added during patchify is removed to restore the exact original dimensions.
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
        Randomly mask entire spatial tubes across all temporal frames.

        For each sample, shuffles patches along the spatial dimension P, keeps
        the first len_keep patches, and discards the rest. Returns the kept
        patches, restore indices to undo the shuffle, and the number kept.
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
        Restore the full patch sequence by inserting learned mask tokens at masked positions.

        Concatenates encoded visible patches with mask tokens, then uses ids_restore
        to permute them back to the original patch ordering. The decoder processes
        this full-length sequence.
        """
        B, _, N, D = x_encoded.shape
        P = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, P - len_keep, N, D)
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)
        expanded_ids_restore = ids_restore.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
        x_restored = torch.gather(x_full, dim=1, index=expanded_ids_restore)
        return x_restored