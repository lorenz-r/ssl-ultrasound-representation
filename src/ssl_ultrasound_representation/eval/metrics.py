"""
Physics-aware reconstruction metrics for the ultrasound MAE.

Standard image metrics (MSE / PSNR) say little about whether an IQ
reconstruction is still *usable* for beamforming: they ignore phase, spectral
content and — crucially — the **spatial coherence across the receive aperture**
that every delay-and-sum / adaptive beamformer relies on. The metrics here
target exactly those properties, on the un-beamformed IQ channel data the model
sees.

Data layout (one frame, matching the rest of the pipeline, e.g.
``utils/beamform.py``)::

    frame : (transducer, scanline, 2, sample)   float
            └ aperture    └ beam    │  └ fast-time / depth
                                    └ IQ:  [...,0,...]=I (real), [...,1,...]=Q (imag)

so the complex channel signal is ``I + jQ``. The transducer axis is the receive
aperture (per-element IQ *before* beamforming), which is what makes the spatial
coherence metrics (SLSC, coherence factor) meaningful.

Metrics
-------
* ``complex_correlation_coefficient`` — global complex Pearson correlation |ρ|
  between ground-truth and reconstructed IQ (phase-sensitive; 1 = perfect).
* ``center_frequency_shift``          — shift of the (baseband) spectral centroid
  along fast-time, in Hz. Detects demodulation / spectral-mean drift.
* ``spectral_coherence``              — mean magnitude-squared coherence between
  the two signals' fast-time spectra over the element×scanline ensemble ∈ [0,1].
* ``slsc_error``                      — mean |ΔSLSC|, the error in Short-Lag
  Spatial Coherence (Lediju et al., IEEE TUFFC 2011) across the aperture.
* ``coherence_factor_error``          — mean |ΔCF|, the error in the aperture
  Coherence Factor (Mallart & Fink, JASA 1994).

All functions accept NumPy arrays or torch tensors and return plain floats
(``compute_all_metrics`` bundles them into a dict for an eval loop).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Acquisition defaults for this dataset (see configs/beamform_params.json). The
# data is IQ-demodulated, so its fast-time spectrum sits at baseband (around 0);
# ``fs`` converts spectral moments to Hz and ``fc`` recovers the absolute RF
# centre frequency when reporting.
DEFAULT_FS = 7.8125e6  # sampling frequency of the IQ data [Hz]
DEFAULT_FC = 7.813e6   # RF centre (demodulation) frequency [Hz]

_EPS = 1e-12


# ── helpers ─────────────────────────────────────────────────────────────────
def _as_numpy(x: Any) -> np.ndarray:
    """Accept a torch tensor or array-like; return a detached NumPy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_complex(x: Any, iq_axis: int = -2) -> np.ndarray:
    """Collapse the length-2 IQ axis into a complex array.

    ``(..., 2, ...) -> (...)`` with ``I + jQ``. If the input is already complex
    it is returned unchanged (so callers can pass either representation).
    """
    arr = _as_numpy(x)
    if np.iscomplexobj(arr):
        return arr
    arr = np.moveaxis(arr, iq_axis, 0)
    if arr.shape[0] != 2:
        raise ValueError(
            f"iq_axis={iq_axis} must have length 2 (I, Q); got {arr.shape[0]}"
        )
    return (arr[0] + 1j * arr[1]).astype(np.complex128)


def _boxcar_valid(a: np.ndarray, k: int) -> np.ndarray:
    """Sliding-window (boxcar) sum over the last axis, 'valid' mode.

    Uses a cumulative-sum difference so it is O(N) and works for complex input.
    Output length is ``a.shape[-1] - k + 1``. ``k <= 1`` is a no-op.
    """
    if k <= 1:
        return a
    csum = np.cumsum(a, axis=-1)
    csum = np.concatenate(
        [np.zeros(a.shape[:-1] + (1,), dtype=csum.dtype), csum], axis=-1
    )
    return csum[..., k:] - csum[..., :-k]


# ── 1. complex correlation coefficient ──────────────────────────────────────
def complex_correlation(gt: Any, pred: Any, iq_axis: int = -2) -> complex:
    """Complex Pearson correlation ρ between two IQ signals (mean-removed).

    ``ρ = <Δg, Δp> / sqrt(<|Δg|²> <|Δp|²>)`` over every complex sample. The
    magnitude |ρ| ∈ [0, 1] measures waveform agreement; ``angle(ρ)`` is the bulk
    phase (e.g. demodulation) offset between the two.
    """
    g = to_complex(gt, iq_axis).ravel()
    p = to_complex(pred, iq_axis).ravel()
    g = g - g.mean()
    p = p - p.mean()
    num = np.vdot(g, p)  # = Σ conj(g) * p
    den = np.sqrt(np.vdot(g, g).real * np.vdot(p, p).real) + _EPS
    return complex(num / den)


def complex_correlation_coefficient(
    gt: Any, pred: Any, iq_axis: int = -2
) -> float:
    """Magnitude |ρ| of the complex correlation (headline scalar, 1 = perfect)."""
    return float(np.abs(complex_correlation(gt, pred, iq_axis)))


# ── 2. centre-frequency shift ───────────────────────────────────────────────
def center_frequency(
    x: Any, fs: float = DEFAULT_FS, iq_axis: int = -2, sample_axis: int = -1
) -> float:
    """Power-weighted spectral centroid along fast-time (baseband), in Hz.

    The IQ data is complex-baseband, so its two-sided spectrum is centred near
    0; the returned value is the mean frequency *relative to* ``fc``. Add ``fc``
    for the absolute RF centre frequency.
    """
    z = to_complex(x, iq_axis)
    # `sample_axis` referenced the original (real) layout; after to_complex the
    # IQ axis is gone. Normalise to a positive index, then drop one if the IQ
    # axis sat before it so it keeps pointing at fast-time.
    ndim0 = _as_numpy(x).ndim
    s_ax = sample_axis % ndim0
    iq_ax = iq_axis % ndim0
    if not np.iscomplexobj(_as_numpy(x)) and iq_ax < s_ax:
        s_ax -= 1

    n = z.shape[s_ax]
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spec = np.fft.fft(z, axis=s_ax)
    power = np.abs(spec) ** 2
    # Sum power over every axis except fast-time, then take the centroid.
    power = np.moveaxis(power, s_ax, -1).reshape(-1, n).sum(axis=0)
    return float((freqs * power).sum() / (power.sum() + _EPS))


def center_frequency_shift(
    gt: Any,
    pred: Any,
    fs: float = DEFAULT_FS,
    iq_axis: int = -2,
    sample_axis: int = -1,
) -> float:
    """Centre-frequency error ``fc(pred) - fc(gt)`` along fast-time, in Hz.

    A non-zero shift means the reconstruction moved spectral energy up/down the
    band — e.g. lost high frequencies (axial-resolution loss) or introduced a
    demodulation offset. Sign: positive = reconstruction is higher-frequency.
    """
    return center_frequency(pred, fs, iq_axis, sample_axis) - center_frequency(
        gt, fs, iq_axis, sample_axis
    )


# ── 3. spectral coherence ───────────────────────────────────────────────────
def spectral_coherence(
    gt: Any, pred: Any, iq_axis: int = -2, sample_axis: int = -1
) -> float:
    """Mean magnitude-squared coherence between the two fast-time spectra ∈ [0,1].

    Treats every (element, scanline) trace as one realisation and estimates, per
    frequency bin, ``|E[X·conj(Y)]|² / (E|X|²·E|Y|²)`` — the textbook coherence
    — then averages over frequency. 1 ⇒ the reconstruction preserves the spectral
    structure (per-bin amplitude *and* phase relationship) of the ground truth.
    """
    g = to_complex(gt, iq_axis)
    p = to_complex(pred, iq_axis)
    ndim0 = _as_numpy(gt).ndim
    s_ax = sample_axis % ndim0
    iq_ax = iq_axis % ndim0
    if not np.iscomplexobj(_as_numpy(gt)) and iq_ax < s_ax:
        s_ax -= 1

    n = g.shape[s_ax]
    X = np.moveaxis(np.fft.fft(g, axis=s_ax), s_ax, -1).reshape(-1, n)
    Y = np.moveaxis(np.fft.fft(p, axis=s_ax), s_ax, -1).reshape(-1, n)

    Sxy = np.mean(X * np.conj(Y), axis=0)
    Sxx = np.mean(np.abs(X) ** 2, axis=0)
    Syy = np.mean(np.abs(Y) ** 2, axis=0)
    coh = (np.abs(Sxy) ** 2) / (Sxx * Syy + _EPS)
    return float(np.mean(coh))


# ── 4. Short-Lag Spatial Coherence (SLSC) ───────────────────────────────────
def short_lag_spatial_coherence(
    x: Any,
    max_lag: int = 20,
    kernel: int = 8,
    aperture_axis: int = 0,
    scanline_axis: int = 1,
    iq_axis: int = -2,
    sample_axis: int = -1,
) -> np.ndarray:
    """SLSC image from IQ channel data (Lediju et al., IEEE TUFFC 2011).

    For each scanline and depth, the normalised spatial correlation across the
    receive aperture at lag ``m`` is::

        R(m) = mean_i  Re{ Σ_n s_i(n)·conj(s_{i+m}(n)) }
                       / sqrt( Σ_n|s_i(n)|² · Σ_n|s_{i+m}(n)|² )

    where the axial sum ``Σ_n`` runs over a short ``kernel`` of fast-time
    samples, and SLSC accumulates the short lags ``Σ_{m=1..max_lag} R(m)``. High
    SLSC ⇔ spatially coherent (on-axis) signal; it is the property adaptive
    beamformers exploit to suppress clutter.

    Args:
        max_lag: number of short lags M to sum (``M < n_elements``).
        kernel:  axial correlation-window length in samples.

    Returns:
        Real array ``(scanline, sample - kernel + 1)`` of SLSC values.
    """
    z = to_complex(x, iq_axis)  # complex, IQ axis removed
    ndim0 = _as_numpy(x).ndim
    # Remap the aperture/scanline/sample axes onto the post-to_complex array
    # (the IQ axis, if it preceded an axis, shifts that axis down by one).
    is_cplx = np.iscomplexobj(_as_numpy(x))
    iq_ax = iq_axis % ndim0

    def _remap(ax: int) -> int:
        a = ax % ndim0
        return a - 1 if (not is_cplx and iq_ax < a) else a

    z = np.moveaxis(
        z, (_remap(aperture_axis), _remap(scanline_axis), _remap(sample_axis)),
        (0, 1, 2),
    )
    n_elem = z.shape[0]
    max_lag = min(max_lag, n_elem - 1)

    energy = _boxcar_valid(np.abs(z) ** 2, kernel)  # (N, Sc, Dw) real
    slsc = np.zeros(energy.shape[1:], dtype=np.float64)
    for m in range(1, max_lag + 1):
        cross = _boxcar_valid(z[:-m] * np.conj(z[m:]), kernel)  # (N-m, Sc, Dw)
        denom = np.sqrt(energy[:-m] * energy[m:]) + _EPS
        rm = np.mean(cross.real / denom, axis=0)  # (Sc, Dw)
        slsc += rm
    return slsc


def slsc_error(
    gt: Any,
    pred: Any,
    max_lag: int = 20,
    kernel: int = 8,
    aperture_axis: int = 0,
    scanline_axis: int = 1,
    iq_axis: int = -2,
    sample_axis: int = -1,
) -> float:
    """Mean absolute SLSC error between reconstruction and ground truth.

    ``mean |SLSC(pred) - SLSC(gt)|`` over the whole SLSC image; 0 ⇒ the aperture
    spatial-coherence structure is perfectly preserved. Same keyword args as
    :func:`short_lag_spatial_coherence`.
    """
    a = short_lag_spatial_coherence(
        gt, max_lag, kernel, aperture_axis, scanline_axis, iq_axis, sample_axis
    )
    b = short_lag_spatial_coherence(
        pred, max_lag, kernel, aperture_axis, scanline_axis, iq_axis, sample_axis
    )
    return float(np.mean(np.abs(a - b)))


# ── 5. Coherence Factor (CF) ────────────────────────────────────────────────
def coherence_factor(
    x: Any, aperture_axis: int = 0, iq_axis: int = -2
) -> np.ndarray:
    """Aperture Coherence Factor per (scanline, sample) (Mallart & Fink 1994).

    ``CF = |Σ_i s_i|² / (N · Σ_i |s_i|²)`` — the ratio of coherent to total
    energy across the ``N`` receive elements, ∈ [0, 1]. CF≈1 for a focused,
    spatially coherent echo; CF→0 for incoherent clutter/noise.

    Returns a real array with the aperture axis reduced away.
    """
    z = to_complex(x, iq_axis)
    ndim0 = _as_numpy(x).ndim
    ap = aperture_axis % ndim0
    if not np.iscomplexobj(_as_numpy(x)) and (iq_axis % ndim0) < ap:
        ap -= 1
    n = z.shape[ap]
    coherent = np.abs(z.sum(axis=ap)) ** 2
    incoherent = (np.abs(z) ** 2).sum(axis=ap)
    return coherent / (n * incoherent + _EPS)


def coherence_factor_error(
    gt: Any, pred: Any, aperture_axis: int = 0, iq_axis: int = -2
) -> float:
    """Mean absolute Coherence-Factor error ``mean |CF(pred) - CF(gt)|`` ∈ [0,1]."""
    a = coherence_factor(gt, aperture_axis, iq_axis)
    b = coherence_factor(pred, aperture_axis, iq_axis)
    return float(np.mean(np.abs(a - b)))


# ── aggregate ───────────────────────────────────────────────────────────────
def compute_all_metrics(
    gt: Any,
    pred: Any,
    fs: float = DEFAULT_FS,
    *,
    slsc_max_lag: int = 20,
    slsc_kernel: int = 8,
) -> dict[str, float]:
    """Compute every metric for one frame pair ``(transducer, scanline, 2, sample)``.

    Returns a flat ``{metric_name: float}`` dict, ready to average over a split
    or log. ``ccc_phase_rad`` is the bulk IQ phase offset (see
    :func:`complex_correlation`).
    """
    rho = complex_correlation(gt, pred)
    return {
        "complex_corr_coeff": float(np.abs(rho)),
        "ccc_phase_rad": float(np.angle(rho)),
        "center_freq_shift_hz": center_frequency_shift(gt, pred, fs),
        "spectral_coherence": spectral_coherence(gt, pred),
        "slsc_error": slsc_error(
            gt, pred, max_lag=slsc_max_lag, kernel=slsc_kernel
        ),
        "coherence_factor_error": coherence_factor_error(gt, pred),
    }
