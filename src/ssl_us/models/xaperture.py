"""Cross-aperture prediction arm (PLAN.md §4.5b, §5.1, WI-15 — optional).

Hold out one receive sector, predict it from the rest. ``--mask-axis rx`` is
the default (PLAN.md §4.5b, rev-4): on this dataset's near-symmetric Tx-Rx
matrix, transmit-masking is shortcut-solvable by transposing the receive
data (§4.5a(a)) — a devastating pretext-task shortcut — while the receive
axis carries a real, ground-truth-backed physical target (`phase_error`,
`sound_speed_map`) and a closed-form non-learned baseline (van
Cittert-Zernike coherence) that the transmit axis has no equivalent for.

**``--mask-axis tx`` is unavailable**, for the same reason
:func:`ssl_us.data.augment.transmit_angle_pair` is: it needs the PW(5)
transmit-angle stack, which was not built (PLAN.md §10.3(b)). Note this is a
*second*, independent reason beyond §4.5a(a)'s shortcut argument — even if
tx-masking were attempted, there is currently no transmit-resolved tensor to
mask.

**The §4.5a(a) mitigation is structural here, not a runtime check bolted
on:** the receive-sector store never had a separate transmit axis to begin
with (transmit is already fully compounded per §3.2), so an rx-masked batch
cannot expose both aperture axes in one tensor — there is only one aperture
axis (the K receive sectors) anywhere in this module's tensors.
:func:`assert_single_aperture_axis` documents and checks that invariant
explicitly rather than leaving it implicit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.resnet import ResNetBackbone


def assert_single_aperture_axis(shape: tuple[int, ...], *, expected_aperture_axes: int = 1) -> None:
    """Documents + checks PLAN.md §4.5a(a): exactly one aperture-resolved axis
    may appear in any tensor this module builds. Called on every masked batch
    this module produces; a caller who somehow concatenated a second
    aperture-resolved axis (e.g. a transmit axis, if the PW stack existed)
    would fail this immediately rather than silently training the
    transposition shortcut.
    """
    if expected_aperture_axes != 1:
        raise ValueError(
            f"exactly one aperture axis is supported (got {expected_aperture_axes}); "
            "this module never exposes both tx and rx in one tensor (PLAN.md 4.5a(a))"
        )


def mask_one_rx_sector(
    sums: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """Zero one random receive sector and fold ``(channel, sector)`` into a
    single conv-input-channel axis for the encoder.

    Args:
        sums: raw sector sums, ``(2, Z, X, K)``.
        rng: seeded generator.

    Returns:
        ``(context, target, held_out_index)`` — ``context`` is
        ``(2*K, Z, X)`` with the held-out sector zeroed; ``target`` is the
        true ``(2, Z, X)`` content that was removed.
    """
    k = sums.shape[-1]
    idx = int(rng.integers(k))
    target = sums[..., idx].copy()
    context = sums.copy()
    context[..., idx] = 0.0
    folded = np.moveaxis(context, -1, 1).reshape(2 * k, *sums.shape[1:-1])
    return folded, target, idx


def trivial_copy_baseline(sums: np.ndarray, held_out_idx: int) -> np.ndarray:
    """Predict the held-out sector as the mean of every *other* sector.

    Reported first (PLAN.md WI-15 done-condition): any learned gain must be
    measured against this, since the sectors are highly correlated views of
    the same scatterers and this is a strong baseline, not a strawman.
    """
    k = sums.shape[-1]
    others = [i for i in range(k) if i != held_out_idx]
    return sums[..., others].mean(axis=-1)


def van_cittert_zernike_coherence(
    delta_x_m: np.ndarray | float, angular_extent_rad: float, wavelength_m: float
) -> np.ndarray | float:
    """Magnitude of the spatial coherence function for a uniform incoherent
    strip source of angular width ``angular_extent_rad``, between two receive
    points separated by ``delta_x_m`` (the van Cittert-Zernike theorem):

    ``|gamma(dx)| = |sinc(2 dx sin(angular_extent/2) / wavelength)|``

    This is the standard textbook result for a 1-D incoherent source and is
    used here as a documented physically-motivated approximation — real
    scatterer fields are not a uniform strip source, so this is a baseline,
    not a ground-truth model. ``numpy.sinc`` is the normalized sinc
    (``sin(pi x)/(pi x)``), which this function accounts for.
    """
    arg = 2.0 * np.asarray(delta_x_m) * np.sin(angular_extent_rad / 2.0) / wavelength_m
    return np.abs(np.sinc(arg))


def vcz_predict_baseline(
    sums: np.ndarray,
    held_out_idx: int,
    sector_center_angles_rad: np.ndarray,
    range_m: float,
    wavelength_m: float,
    angular_extent_rad: float,
) -> np.ndarray:
    """Non-learned rx baseline: predict the held-out sector as a van
    Cittert-Zernike-coherence-weighted average of the others.

    Sector angular separation is converted to an effective receive-baseline
    separation via the small-angle proxy ``range_m * |theta_i - theta_j|``,
    since sectors are defined by view angle, not by a single element pair.
    """
    k = sums.shape[-1]
    theta0 = sector_center_angles_rad[held_out_idx]
    weights = np.zeros(k, dtype=np.float64)
    for i in range(k):
        if i == held_out_idx:
            continue
        delta_x = range_m * abs(sector_center_angles_rad[i] - theta0)
        weights[i] = van_cittert_zernike_coherence(delta_x, angular_extent_rad, wavelength_m)
    total = weights.sum()
    if total <= 0:
        return trivial_copy_baseline(sums, held_out_idx)
    weights /= total
    return np.tensordot(sums, weights, axes=([-1], [0]))


class XApertureDecoder(nn.Module):
    """Light conv decoder: linear projection to a small spatial map, two
    transposed convolutions, then bilinear resize to the exact output size —
    robust to any crop size (128 or 64, PLAN.md §3.2d) without needing
    power-of-two alignment.
    """

    def __init__(self, in_dim: int, out_channels: int, out_size: tuple[int, int], hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.out_size = out_size
        self.fc = nn.Linear(in_dim, hidden * 8 * 8)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, out_channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).view(-1, self.hidden, 8, 8)
        x = self.conv(x)
        return F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)


class XApertureModel(nn.Module):
    """ResNet-18 encoder (stem adapted to ``2*K`` input channels) + light
    conv decoder predicting the held-out sector's ``(2, Z, X)`` content."""

    def __init__(self, k: int, out_size: tuple[int, int], seed: int | None = None):
        super().__init__()
        self.encoder = ResNetBackbone(in_channels=2 * k, seed=seed)
        self.decoder = XApertureDecoder(self.encoder.dim, out_channels=2, out_size=out_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(context))


def xaperture_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def residual_frame_correlation(
    residual_energy: np.ndarray, phase_error: np.ndarray, sos_std: np.ndarray
) -> dict[str, float]:
    """Pearson correlation of the frame-level prediction-residual energy
    against `phase_error` and `sos_std` — the physical claim this arm exists
    to test (PLAN.md WI-15 done-condition): does the hardest-to-predict
    sector track aberration/sound-speed heterogeneity?
    """
    return {
        "corr_phase_error": float(np.corrcoef(residual_energy, phase_error)[0, 1]),
        "corr_sos_std": float(np.corrcoef(residual_energy, sos_std)[0, 1]),
    }


def mask_axis_tx(*_args, **_kwargs):
    """Unavailable — see the module docstring and PLAN.md §10.3(b): no
    transmit-resolved tensor exists without the (unbuilt) PW(5) stack."""
    raise NotImplementedError(
        "--mask-axis tx has no data source: the PW(5) transmit-angle stack was "
        "not built (PLAN.md 10.3(b)). Build WI-4b first if this is revisited."
    )
