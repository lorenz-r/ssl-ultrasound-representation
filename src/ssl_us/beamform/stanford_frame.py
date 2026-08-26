"""Per-frame beamforming for WI-16: Stanford FSA sample -> raw sector sums.

The Stanford analogue of :mod:`ssl_us.beamform.frame` (NVIDIA/WI-8) — wires
the same :class:`~ssl_us.beamform.sector_sum.AngularSectorSum` op to the
Stanford FSA store's ``(n_tx, n_rx, n_ax)`` layout, one grid variant
(:mod:`ssl_us.data.stanford_geometry`) at a time.

**Three constants differ from NVIDIA's, all for physical reasons — see
`ssl_us.data.stanford_fsa`'s module docstring before changing any of them:**

- ``initial_times`` is set to the store's ``t0_s`` attribute (a corpus-wide
  constant here, unlike NVIDIA's three-level per-frame ``t0``), and it is
  **negative** — the Verasonics time origin already carries the transducer
  lens correction and pulse-peak offset. Forcing it to 0 (NVIDIA's own
  convention) would misregister every pixel by ~1.4 mm.
- ``T_PEAK`` is still forced to 0, but for the opposite reason NVIDIA's is:
  there, OpenH-RF's scan dict never sets it, so 0 overrides zea's own
  ``1/center_frequency`` default. Here, the pulse-peak term is *already*
  folded into ``t0_s`` above, so a nonzero ``t_peak`` would double-count it.
- The near-field mute is not a fixed sample count (NVIDIA's ``MUTE_SAMPLES =
  32`` is specific to its direct-coupling artifact). Instead,
  :func:`load_raw_iq` mutes every sample whose depth is < 0 mm — physically,
  anything before the array face — via the store's own
  :meth:`~ssl_us.data.stanford_fsa.StanfordFsaStore.depth_axis_m`.

``f_number = 1.5`` is kept as the WI-4-validated default for the receive
aperture: it is a property of the sector-sum optics, not of which array
supplied the elements, so there is no a priori reason to expect Stanford's
in-vivo tissue to prefer a different one. ``scripts/beamform/validate_stanford_reconstruction.py``
re-sweeps it anyway, the same way WI-4 did, rather than assuming.
"""

from __future__ import annotations

import keras
import numpy as np
from zea import Parameters
from zea.ops.pipeline import Beamform

from ..data.stanford_fsa import StanfordFsaStore
from ..data.stanford_geometry import DEPTH_MIN_MM, GridVariant
from . import sector_sum  # noqa: F401 — registers "angular_sector_sum"

F_NUMBER = 1.5
T_PEAK = 0.0
DEFAULT_K = 16


def build_parameters(
    store: StanfordFsaStore, grid: GridVariant, k: int = DEFAULT_K, f_number: float = F_NUMBER
) -> Parameters:
    """FSA scan parameters for one grid variant. Same transmit scheme as
    NVIDIA's — ``tx i fires element i`` (`stanford_fsa` store attrs' own
    ``layout`` string) — so `tx_apodizations`/`transmit_origins` are identical
    in form to `frame.build_parameters`, just built from Stanford's 192
    physical element positions instead of a streamed ``elpos``.
    """
    n_tx = n_el = int(store.attrs["n_elem"])
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = store.element_x_m()
    t0 = float(store.attrs["t0_s"])
    return Parameters(
        probe_geometry=probe_geometry,
        sound_speed=float(store.attrs["c_m_s"]),
        sampling_frequency=float(store.attrs["fs_hz"]),
        center_frequency=float(store.attrs["fc_hz"]),
        demodulation_frequency=float(store.attrs["fd_hz"]),
        n_el=n_el,
        n_tx=n_tx,
        n_ax=int(store.shape[-1]),
        n_ch=2,
        xlims=(float(grid.x_mm[0]) * 1e-3, float(grid.x_mm[-1]) * 1e-3),
        zlims=(float(grid.z_mm[0]) * 1e-3, float(grid.z_mm[-1]) * 1e-3),
        grid_size_x=len(grid.x_mm),
        grid_size_z=len(grid.z_mm),
        t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
        tx_apodizations=np.eye(n_tx, n_el, dtype=np.float32),
        focus_distances=np.zeros(n_tx, dtype=np.float32),
        polar_angles=np.zeros(n_tx, dtype=np.float32),
        transmit_origins=probe_geometry,
        initial_times=np.full(n_tx, t0, dtype=np.float32),
        f_number=f_number,
    )


def load_raw_iq(
    store: StanfordFsaStore, index: int, mute_before_mm: float = DEPTH_MIN_MM
) -> np.ndarray:
    """``(n_tx, n_ax, n_el, 2)`` muted-near-field IQ for one frame.

    The store's own layout is ``(n_tx, n_rx, n_ax)`` complex — different axis
    order from NVIDIA's streamed samples, so this is a transpose, not the
    same code as `frame.load_raw_iq`.
    """
    iq = store.iq(index)  # (n_tx, n_rx, n_ax) complex64
    raw = np.stack([iq.real, iq.imag], axis=-1).astype(np.float32)  # (n_tx, n_rx, n_ax, 2)
    raw = np.moveaxis(raw, 1, 2)  # (n_tx, n_ax, n_rx, 2)
    if mute_before_mm is not None:
        depth_mm = store.depth_axis_m() * 1e3
        mute = int(np.searchsorted(depth_mm, mute_before_mm))
        raw[:, :mute, :, :] = 0.0
    return raw


def beamform_frame(
    store: StanfordFsaStore,
    index: int,
    grid: GridVariant,
    k: int = DEFAULT_K,
    f_number: float = F_NUMBER,
    num_patches: int | None = None,
    demodulation_frequency: float | None = None,
) -> np.ndarray:
    """Beamform one Stanford frame to the shared store's ``(2, Z, X, K)`` layout.

    Returns **raw** sector sums, never normalized (PLAN.md §3.2/WI-5), exactly
    like `frame.beamform_sample`. ``demodulation_frequency`` is exposed only
    for the phase-rotation negative control (pass ``0.0`` to disable it).
    """
    params = build_parameters(store, grid, k=k, f_number=f_number)
    raw = load_raw_iq(store, index)
    pipeline = Beamform(
        beamformer="angular_sector_sum",
        k=k,
        f_number=f_number,
        num_patches=num_patches if num_patches is not None else max(4, len(grid.z_mm) // 2),
        with_batch_dim=False,
    )
    overrides = {"f_number": f_number, "t_peak": np.full(params.n_tx, T_PEAK, dtype=np.float32)}
    if demodulation_frequency is not None:
        overrides["demodulation_frequency"] = demodulation_frequency
    inputs = pipeline.prepare_parameters(params, data=raw, **overrides)
    out = keras.ops.convert_to_numpy(pipeline(**inputs)["data"])  # (Z, X, 2, K)
    return np.ascontiguousarray(np.moveaxis(out, 2, 0), dtype=np.float32)


def frame_labels(store: StanfordFsaStore, index: int) -> dict:
    """Per-frame provenance for the sidecar — the probe axes WI-16 has:
    session, source file, acquisition kind, date. No physics ground truth
    (sos/cyst/phase_error) exists for Stanford — it is an unlabelled transfer
    corpus (PLAN.md §3.6), unlike NVIDIA's simulation.
    """
    meta = store.frame_metadata[str(index)]
    return {
        "session": meta["session"],
        "source_file": meta["source_file"],
        "kind": meta["kind"],
        "date": meta["date"],
        "t0_s": float(store.attrs["t0_s"]),
    }
