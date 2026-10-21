"""Per-frame beamforming for the WI-8 build: NVIDIA sample -> raw sector sums.

This is the `beamform_frame` callback PLAN.md WI-8 calls for — the piece that
wires :class:`~ssl_us.beamform.sector_sum.AngularSectorSum`, the near-field
mute, and the acquisition parameters together into one function from a streamed
HuggingFace sample to a ``(2, Z, X, K)`` array ready for the store.

**Every constant here was fixed by WI-4's validation, not chosen freshly.** The
gate run (`scripts/beamform/validate_nvidia_reconstruction.py`, results in
`results/gates/nvidia_reconstruction.json`) swept them and these won:

- ``f_number = 1.5`` — best on 7 of 8 frames.
- ``t_peak = 0`` — *not* zea's ``1/center_frequency`` default, which the
  OpenH-RF FSA scan dict never sets. Worth NCC 0.076 -> 0.257, the single
  largest parameter effect measured.
- ``initial_times = +t0`` as stored (the sign convention; -t0 scores 0.078).
- ``MUTE_SAMPLES = 32`` near-field direct-coupling mute (PLAN.md §10.4).
- Raw axis 0 treated as transmit — measured immaterial (0.2587 vs 0.2573),
  as PLAN.md §10.2 predicted from reciprocity.

Changing any of them invalidates WI-4's gate, so they live here as named
module constants rather than inline literals, and the values are recorded into
the store's attrs by the build so a reader can always tell what produced it.
"""

from __future__ import annotations

import keras
import numpy as np
from zea import Parameters
from zea.ops.pipeline import Beamform

from ..data import geometry as geo
from . import sector_sum  # noqa: F401 — registers "angular_sector_sum"

# WI-4-validated beamforming constants. See module docstring before changing.
F_NUMBER = 1.5
MUTE_SAMPLES = 32
T_PEAK = 0.0
DEFAULT_K = 16


def build_parameters(
    sample: dict,
    k: int = DEFAULT_K,
    f_number: float = F_NUMBER,
    nz: int = geo.BMODE_NZ,
    nx: int = geo.BMODE_NX,
) -> Parameters:
    """FSA scan parameters for one frame, on the dataset's native B-mode grid.

    The grid is (507, 456) — the same grid `bmode`/`segmentation_map`/the store
    all use — so nothing downstream has to resample to line labels up with
    pixels. (WI-4's validation script used a square grid instead, purely to
    compare against a resampled reference; that is not the build's geometry.)
    """
    n_tx = n_el = geo.N_ELEMENTS
    probe_geometry = np.asarray(sample["elpos"], dtype=np.float32).T.copy()
    return Parameters(
        probe_geometry=probe_geometry,
        sound_speed=float(sample["c0"]),
        sampling_frequency=float(sample["fs"]),
        center_frequency=float(sample["fc"]),
        demodulation_frequency=float(sample["fd"]),
        n_el=n_el,
        n_tx=n_tx,
        n_ax=geo.N_FAST_TIME,
        n_ch=2,
        xlims=(geo.BMODE_X_MIN_MM * 1e-3, geo.BMODE_X_MAX_MM * 1e-3),
        zlims=(geo.BMODE_Z_MIN_MM * 1e-3, geo.BMODE_Z_MAX_MM * 1e-3),
        grid_size_x=nx,
        grid_size_z=nz,
        t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
        tx_apodizations=np.eye(n_tx, n_el, dtype=np.float32),
        focus_distances=np.zeros(n_tx, dtype=np.float32),
        polar_angles=np.zeros(n_tx, dtype=np.float32),
        transmit_origins=probe_geometry,
        initial_times=np.full(n_tx, float(sample["t0"]), dtype=np.float32),
        f_number=f_number,
    )


def load_raw_iq(sample: dict, mute_samples: int = MUTE_SAMPLES) -> np.ndarray:
    """``(n_tx, n_ax, n_el, 2)`` muted-near-field IQ from a streamed sample."""
    re = np.asarray(sample["iq_real"], dtype=np.float32)
    im = np.asarray(sample["iq_imag"], dtype=np.float32)
    raw = np.stack([re, im], axis=-1).transpose(0, 2, 1, 3)
    if mute_samples:
        raw[:, :mute_samples, :, :] = 0.0
    return raw


def beamform_sample(
    sample: dict,
    k: int = DEFAULT_K,
    f_number: float = F_NUMBER,
    num_patches: int | None = None,
    nz: int = geo.BMODE_NZ,
    nx: int = geo.BMODE_NX,
) -> np.ndarray:
    """Beamform one streamed sample to the store's ``(2, Z, X, K)`` layout.

    Returns **raw** sector sums — never normalized by occupancy, per PLAN.md
    §3.2 / WI-5. The channel axis is moved to the front to match the store's
    ``(2, Z, X, K)`` spec (§3.7); the beamformer itself emits ``(Z, X, 2, K)``.
    """
    params = build_parameters(sample, k=k, f_number=f_number, nz=nz, nx=nx)
    raw = load_raw_iq(sample)
    pipeline = Beamform(
        beamformer="angular_sector_sum",
        k=k,
        f_number=f_number,
        num_patches=num_patches if num_patches is not None else max(4, nz // 2),
        with_batch_dim=False,
    )
    inputs = pipeline.prepare_parameters(
        params,
        data=raw,
        f_number=f_number,
        t_peak=np.full(params.n_tx, T_PEAK, dtype=np.float32),
    )
    out = keras.ops.convert_to_numpy(pipeline(**inputs)["data"])  # (Z, X, 2, K)
    return np.ascontiguousarray(np.moveaxis(out, 2, 0), dtype=np.float32)


def frame_labels(sample: dict) -> dict:
    """Per-frame scalars the probes need (PLAN.md §4.1), for the sidecar.

    Kept next to the beamforming rather than in the eval code because they are
    read off the *same* streamed sample — re-streaming 265 MB later just to
    recover four scalars would be absurd.
    """
    sos = np.asarray(sample["sound_speed_map"], dtype=np.float32)
    seg = np.asarray(sample["segmentation_map"])
    return {
        "t0": float(sample["t0"]),
        "phase_error": float(sample["phase_error"]),
        "sos_mean": float(sos.mean()),
        "sos_std": float(sos.std()),
        "cyst_fraction": float((seg == 1).mean()),
        # 75 of 923 frames ship an empty mask, and empty does NOT reliably mean
        # "no cyst" (PLAN.md §2.x, NVIDIA known issue #3) — recorded so the
        # probes can exclude them rather than silently treat them as negatives.
        "segmentation_empty": bool(not (seg == 1).any()),
    }
