"""Masking for the RF/IQ conv autoencoder (`masked_rf_autoencoder_spec.md` §2).

Random single-sample masking is useless on RF: neighbouring samples inside one
wavelength are near-duplicates, so the gap is filled by local interpolation and
the model never learns to predict the waveform. Two mask types are used instead:

* **fast-time segments** — contiguous runs of `L ~ U(l_min, l_max)` samples along
  the axial axis. Long enough that the middle of a run is not reachable by
  interpolation, which is also why the model's fast-time receptive field has to
  exceed `l_max` (see `rf_conv_mae.assert_receptive_field`).
* **whole channels** — entire lateral columns (receive elements, or scanlines
  depending on the fold), dropped with probability `p_chan`. Only cross-aperture
  structure can recover these, so they are what force the lateral axis to be used
  at all.

By default a fast-time run covers *every* lateral position at the same depths
(`share_lateral=True`). Per-column runs would be much easier than they look:
adjacent elements see the same wavefront, so a gap in column j is readable
straight off column j±1. Shared depth bands force genuine axial extrapolation,
and whole-channel masking is left to test the lateral direction.

Note on `wavelength_samples`: this dataset is IQ-demodulated and decimated to
`fs = fc = 7.8125 MHz`, i.e. **one sample per carrier period**, so the spec's
"L_min >= one wavelength" is satisfied by any `l_min >= 1` and is not a useful
floor. The meaningful axial correlation length is the pulse length (inverse
bandwidth), a handful of samples — hence the `l_min=8` default on a 215-sample
trace.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RFMasker(nn.Module):
    """Produce `(masked_input, mask)` for the conv autoencoder.

    Args:
        mask_ratio: target fraction of masked fast-time samples (spec suggests
            0.5-0.6, below image-MAE's 0.75 because RF is harder and data scarce).
        l_min, l_max: fast-time run length bounds, in samples. `l_max` must stay
            below the model's fast-time receptive field.
        p_chan: probability of dropping a whole lateral column.
        wavelength_samples: sanity floor on `l_min` (see module docstring).
        share_lateral: True -> runs span all lateral positions (depth bands).
        seed: fix for a frozen mask pattern — takes mask sampling out of the
            picture when debugging (e.g. a single-batch overfit that will not fit).

    Forward:
        x: (B, 2, n_fast, n_lat) -> masked_input (B, 3, n_fast, n_lat),
                                    mask (B, 1, n_fast, n_lat) bool, True = masked
    """

    def __init__(
        self,
        mask_ratio: float = 0.55,
        l_min: int = 8,
        l_max: int = 32,
        p_chan: float = 0.1,
        wavelength_samples: int = 1,
        share_lateral: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1); got {mask_ratio}")
        if l_min < 1 or l_max < l_min:
            raise ValueError(f"need 1 <= l_min <= l_max; got {l_min}, {l_max}")
        if l_min < wavelength_samples:
            raise ValueError(
                f"l_min ({l_min}) < wavelength_samples ({wavelength_samples}): runs "
                "shorter than a wavelength are defeated by local interpolation"
            )
        self.mask_ratio = mask_ratio
        self.l_min, self.l_max = l_min, l_max
        self.p_chan = p_chan
        self.share_lateral = share_lateral
        self.seed = seed
        # Learned replacement for masked samples (spec: a length-2 vector, not
        # raw zeros — zeros are a valid IQ value and would be ambiguous).
        self.mask_token = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def _generator(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self.seed)
        return g

    def _n_runs(self, n_fast: int) -> int:
        """Runs needed to reach `mask_ratio` once overlap is accounted for.

        Each run covers a fraction `p = E[L]/n_fast` of the trace, and runs land
        independently, so `k` of them leave `(1-p)^k` uncovered. Solving for the
        target ratio beats laying down `ratio*n_fast/E[L]` runs, which
        systematically undershoots because runs overlap.
        """
        if self.mask_ratio <= 0.0:
            return 0
        p = min(((self.l_min + self.l_max) / 2.0) / n_fast, 0.99)
        return max(1, math.ceil(math.log1p(-self.mask_ratio) / math.log1p(-p)))

    def sample_mask(self, shape: tuple[int, int, int], device) -> torch.Tensor:
        """Sample a `(B, 1, n_fast, n_lat)` boolean mask. Exposed for tests/plots."""
        b, n_fast, n_lat = shape
        g = self._generator(device)
        mask = torch.zeros(b, n_fast, n_lat, dtype=torch.bool, device=device)
        idx = torch.arange(n_fast, device=device).view(1, n_fast, 1)
        run_lat = 1 if self.share_lateral else n_lat

        for _ in range(self._n_runs(n_fast)):
            length = torch.randint(
                self.l_min, self.l_max + 1, (b, run_lat), device=device, generator=g
            )
            # Start uniformly in [0, n_fast - L] so every run is full length
            # instead of being truncated at the deep end.
            span = (n_fast - length + 1).clamp_min(1)
            start = (torch.rand(b, run_lat, device=device, generator=g) * span).long()
            lo, hi = start.unsqueeze(1), (start + length).unsqueeze(1)
            mask |= (idx >= lo) & (idx < hi)

        if self.p_chan > 0.0:
            chan = torch.rand(b, 1, n_lat, device=device, generator=g) < self.p_chan
            mask |= chan

        return mask.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"expected (B, 2, n_fast, n_lat); got {tuple(x.shape)}")
        b, _, n_fast, n_lat = x.shape
        mask = self.sample_mask((b, n_fast, n_lat), x.device)
        masked = torch.where(mask, self.mask_token.to(x.dtype), x)
        # Channel 2 tells the encoder *which* positions are missing — without it
        # the mask token has to be inferred from its own value.
        return torch.cat([masked, mask.to(x.dtype)], dim=1), mask
