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

    def __init__(
        self,
        n_transducers=192,
        transducer_window=64,
        n_scanlines=192,
        n_samples=215,
        depth_window=43,
        embed_dim=256,
        encoder_depth=4,
        encoder_heads=8,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_heads=4,
        dim_feedforward=512,
        mask_ratio=0.0,
        recon_head="linear",
        patch_embed="conv",
        dropout=0.1,
    ):
        super().__init__()
        assert recon_head in ("linear", "conv"), (
            f"recon_head must be 'linear' or 'conv', got {recon_head!r}"
        )
        assert patch_embed in ("conv", "linear"), (
            f"patch_embed must be 'conv' or 'linear', got {patch_embed!r}"
        )
        assert recon_head != "conv" or patch_embed == "conv", (
            "recon_head='conv' needs the conv patch embedder's bottleneck "
            "feature shape to build its inverse transpose-conv stack; use "
            "patch_embed='conv' (the default) or switch recon_head='linear'."
        )
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
        self.num_tokens = n_scanlines * self.n_transducer_groups * self.n_depth_windows
        self.patch_dim = 2 * transducer_window * depth_window

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.recon_head_type = recon_head
        self.patch_embed_type = patch_embed

        # -------------------------------------------------------------------------
        # 1. MAE ENCODER MODULES
        # -------------------------------------------------------------------------
        # Patch embedding. Two options:
        #  * "conv": small CNN over (transducer_window, depth_window), IQ as
        #    channels, downsampled 8x (3 stride-2 convs) before the linear
        #    projection to embed_dim. Cheap, but the IQ carrier here is
        #    demodulated near fc≈fs (~11 cycles per depth_window, normalized
        #    freq ≈0.26 cycles/sample) — a single stride-2 step already pushes
        #    that past the new grid's Nyquist, so this downsampling risks
        #    *aliasing* the axial carrier (destroying it, not just compressing
        #    it) before the transformer or any embed width ever sees it. Default,
        #    kept for parameter efficiency.
        #  * "linear": flat Linear(patch_dim, embed_dim) — no spatial striding,
        #    so no aliasing risk. Used to A/B-test whether the conv embedder's
        #    downsampling is what caps single-sample-overfit loss.
        if patch_embed == "conv":
            self.patch_conv = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            # Record the spatial size after each conv so the decoder's transpose
            # stack can invert the exact down-sampling (output_padding picked per
            # axis), independent of the chosen window sizes.
            with torch.no_grad():
                sizes = [(transducer_window, depth_window)]
                t = torch.zeros(1, 2, transducer_window, depth_window)
                for m in self.patch_conv:
                    t = m(t)
                    if isinstance(m, nn.Conv2d):
                        sizes.append(tuple(t.shape[-2:]))
                feat = t
            self._feat_chw = tuple(feat.shape[1:])  # (C, H, W) at the bottleneck
            self._conv_sizes = sizes  # [(tw, w), s1, s2, s3]
            self.patch_proj = nn.Linear(feat.flatten(1).shape[1], embed_dim)
        else:
            self.patch_embed_linear = nn.Linear(self.patch_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=encoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
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
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head. Two options:
        #  * "linear": a single Linear(decoder_dim -> patch_dim). Strictly the
        #    most expressive per-patch map — it can emit an arbitrary value at
        #    every IQ sample, so it is *not* band-limited and can reconstruct the
        #    near-Nyquist axial carrier (the IQ here is demodulated at fc≈fs, so
        #    each depth_window holds ~11 oscillation cycles). This is the default.
        #  * "conv": project each decoder token to the bottleneck feature map and
        #    up-sample with a transpose-conv stack. Cheaper params, but the coarse
        #    (8x6) bottleneck low-passes the reconstruction — it cannot represent
        #    the axial carrier, which floors even a single-sample overfit around
        #    the low-frequency energy fraction (~0.7). Kept for comparison.
        if recon_head == "linear":
            self.recon_head = nn.Linear(decoder_embed_dim, self.patch_dim)
        else:
            self.decoder_unproj = nn.Linear(
                decoder_embed_dim, feat.flatten(1).shape[1]
            )
            # Up-sampling target sizes mirror patch_conv in reverse; channels
            # reverse patch_conv (64 -> 32 -> 16 -> 2 IQ).
            up_sizes = [self._conv_sizes[-(i + 2)] for i in range(3)]
            chans = [(64, 32), (32, 16), (16, 2)]

            deconv_layers = []
            for i, ((c_in, c_out), (k, s, p)) in enumerate(
                zip(chans, [(3, 2, 1), (3, 2, 1), (5, 2, 2)])
            ):
                in_hw = self._conv_sizes[-(i + 1)]
                out_hw = up_sizes[i]
                out_pad = tuple(
                    out_hw[d] - ((in_hw[d] - 1) * s - 2 * p + k) for d in range(2)
                )
                deconv_layers.append(
                    nn.ConvTranspose2d(c_in, c_out, k, s, p, output_padding=out_pad)
                )
                if i < len(chans) - 1:
                    deconv_layers.append(nn.GELU())
            self.patch_deconv = nn.Sequential(*deconv_layers)

    def _embed_patches(self, patches):
        """(B, L, 2, tw, w) -> (B, L, embed_dim), per the `patch_embed` type."""
        B, L = patches.shape[:2]
        if self.patch_embed_type == "conv":
            tok = self.patch_conv(
                patches.reshape(B * L, 2, self.transducer_window, self.depth_window)
            )
            return self.patch_proj(tok.flatten(1)).view(B, L, self.embed_dim)
        return self.patch_embed_linear(patches.flatten(2))

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

        # --- 1. Patch embedding + positional encoding ---
        tok = self._embed_patches(patches)
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
                y,
                1,
                ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
            )
        y = y + self.decoder_pos_embed

        # --- 5. DECODER + reconstruction head ---
        y = self.decoder_norm(self.decoder(y))
        if self.recon_head_type == "linear":
            pred = self.recon_head(y)  # (B, L, patch_dim)
        else:
            y = self.decoder_unproj(y)  # (B, L, C*H*W)
            y = y.reshape(B * L, *self._feat_chw)
            y = self.patch_deconv(y)  # (B*L, 2, tw, w)
            pred = y.reshape(B, L, self.patch_dim)

        return pred, target, mask

    @torch.no_grad()
    def encode(self, x, pos_embed, pool: str = "mean"):
        """Extract encoder latents for representation analysis.

        Runs the embedding + encoder only — no masking (every token is kept,
        regardless of ``self.mask_ratio``) and no decoder — so the result is the
        full, deterministic representation of a frame.

        Args:
            x: frame batch (B, transducer, scanline, iq, sample), single focal
               zone already selected (same layout as ``forward``).
            pos_embed: (num_tokens, embed_dim) encoder positional encoding.
            pool: "mean" -> (B, embed_dim) frame-level latent (mean over tokens);
                  "none" -> (B, num_tokens, embed_dim) per-token latents.

        Returns:
            Latent tensor as described by ``pool``.
        """
        patches = self.patchify(x)
        tok = self._embed_patches(patches)
        tok = tok + pos_embed.unsqueeze(0).to(tok.dtype)
        z = self.encoder_norm(self.encoder(tok))  # (B, L, embed_dim)
        if pool == "mean":
            return z.mean(dim=1)
        if pool == "none":
            return z
        raise ValueError(f"unknown pool: {pool!r} (use 'mean' or 'none')")

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
