import torch
import torch.nn as nn


class SignalMAE(nn.Module):
    """
    Masked Autoencoder over raw IQ channel data (single focal zone).

    Input is one frame with the focal-zone axis already selected:
        (B, transducer, scanline, iq, sample)
    tokenized into a (scanline, transducer_group, depth_window) grid: each
    token holds a sub-aperture of `transducer_window` elements for one
    lateral position and one axial window — patch shape
    (iq, transducer_window, depth_window). Masking hides lateral, aperture
    and axial blocks, so reconstruction requires cross-scanline (anatomy),
    cross-aperture (receive coherence) and cross-depth structure.

    Patches are embedded with a small CNN over (transducer, depth) instead
    of a flat linear layer to keep parameters reasonable.
    """

    def __init__(self,
                 n_transducers=192,
                 transducer_window=64,
                 n_scanlines=192,
                 n_samples=215,
                 depth_window=43,
                 embed_dim=128,
                 encoder_depth=4,
                 encoder_heads=8,
                 decoder_embed_dim=64,
                 decoder_depth=2,
                 decoder_heads=4,
                 dim_feedforward=512,
                 mask_ratio=0.0):
        super().__init__()
        assert n_samples % depth_window == 0, (
            f"n_samples ({n_samples}) must be divisible by depth_window ({depth_window})"
        )
        assert n_transducers % transducer_window == 0, (
            f"n_transducers ({n_transducers}) must be divisible by "
            f"transducer_window ({transducer_window})"
        )

        self.n_transducers = n_transducers
        self.transducer_window = transducer_window
        self.n_transducer_groups = n_transducers // transducer_window
        self.n_scanlines = n_scanlines
        self.n_samples = n_samples
        self.depth_window = depth_window
        self.n_depth_windows = n_samples // depth_window
        self.num_tokens = (
            n_scanlines * self.n_transducer_groups * self.n_depth_windows
        )
        self.patch_dim = 2 * transducer_window * depth_window

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio

        # -------------------------------------------------------------------------
        # 1. MAE ENCODER MODULES
        # -------------------------------------------------------------------------
        # Conv patch embedder over (transducer_window, depth_window), IQ as channels.
        self.patch_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            feat = self.patch_conv(torch.zeros(1, 2, transducer_window, depth_window))
        self.patch_proj = nn.Linear(feat.flatten(1).shape[1], embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # -------------------------------------------------------------------------
        # 2. MAE DECODER MODULES (Asymmetric & Lightweight)
        # -------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, decoder_embed_dim) * 0.02
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_heads,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.output_projection = nn.Linear(decoder_embed_dim, self.patch_dim)

    def forward(self, x, pos_embed):
        """
        Args:
            x: frame batch (B, transducer, scanline, iq, sample),
               single focal zone already selected.
            pos_embed: encoder positional encoding (num_tokens, embed_dim),
               token order (scanline, transducer_group, depth_window) row-major.

        Returns:
            pred:   (B, num_tokens, patch_dim) reconstructed patches
            target: (B, num_tokens, patch_dim) ground-truth patches
            mask:   (B, num_tokens) — 1 where the token was masked, 0 visible
        """
        patches = self.patchify(x)  # (B, L, 2, tw, w)
        B, L = patches.shape[:2]
        target = patches.flatten(2)

        # --- 1. Patch embedding (conv) + positional encoding ---
        tok = self.patch_conv(
            patches.reshape(B * L, 2, self.transducer_window, self.depth_window)
        )
        tok = self.patch_proj(tok.flatten(1)).view(B, L, self.embed_dim)
        tok = tok + pos_embed.unsqueeze(0).to(tok.dtype)

        # --- 2. Random masking over the token grid ---
        tok, mask, ids_restore, len_keep = self.random_masking(tok, self.mask_ratio)

        # --- 3. ENCODER (visible tokens only) ---
        z = self.encoder_norm(self.encoder(tok))

        # --- 4. Project to decoder space, re-insert mask tokens ---
        y = self.decoder_embed(z)
        if len_keep < L:
            mask_tokens = self.mask_token.expand(B, L - len_keep, -1)
            y = torch.cat([y, mask_tokens], dim=1)
            y = torch.gather(
                y, 1,
                ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
            )
        y = y + self.decoder_pos_embed

        # --- 5. DECODER + output projection ---
        y = self.decoder_norm(self.decoder(y))
        pred = self.output_projection(y)

        return pred, target, mask

    def patchify(self, x):
        """
        (B, T, Sc, IQ, D) -> (B, Sc*G*W, IQ, tw, w)
        with T = G * tw (transducer groups) and D = W * w (depth windows).

        Token order is (scanline, transducer_group, depth_window) row-major,
        matching the positional-encoding grid.
        """
        B, T, Sc, IQ, D = x.shape
        G, tw = self.n_transducer_groups, self.transducer_window
        W, w = self.n_depth_windows, self.depth_window
        x = x.view(B, G, tw, Sc, IQ, W, w)
        x = x.permute(0, 3, 1, 5, 4, 2, 6).contiguous()
        return x.view(B, Sc * G * W, IQ, tw, w)

    def unpatchify(self, pred):
        """(B, L, patch_dim) -> (B, T, Sc, IQ, D); inverse of patchify."""
        B = pred.shape[0]
        Sc = self.n_scanlines
        G, tw = self.n_transducer_groups, self.transducer_window
        W, w = self.n_depth_windows, self.depth_window
        x = pred.view(B, Sc, G, W, 2, tw, w)
        x = x.permute(0, 2, 5, 1, 4, 3, 6).contiguous()
        return x.view(B, G * tw, Sc, 2, W * w)

    def random_masking(self, x, mask_ratio):
        """
        Standard MAE per-token random masking.

        Returns the kept tokens, a (B, L) binary mask (1 = masked), restore
        indices, and the number of kept tokens. With mask_ratio <= 0 the
        sequence passes through untouched (plain autoencoder).
        """
        B, L, D = x.shape
        if mask_ratio <= 0:
            mask = torch.zeros(B, L, device=x.device)
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            return x, mask, ids_restore, L

        len_keep = max(1, int(L * (1 - mask_ratio)))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore, len_keep
