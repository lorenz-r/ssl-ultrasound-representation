"""Stanford Verasonics ``.mat`` reader + HVMux receive-aperture remap
(PLAN.md WI-16).

**HVMux coverage verified first, per PLAN.md WI-16/§3.6's open question:**
`zea.data.convert.verasonics` (0.1.4) has **zero** handling for
aperture-multiplexed receive — a grep of its ~1700 lines for
"aperture"/"hvmux"/"multiplex" returns nothing. Stanford's acquisition reads
each transmit's echo through **two** 128-channel receive apertures (HVMux)
that must be recombined into the physical 192-element array before any
sector/beamforming code can treat it as a normal array. So this module ports
the repo's own hard-won cyclic remap (PLAN.md §1.3's "correct for Stanford,
meaningless for NVIDIA") rather than building on zea's converter for this one
piece — the rest of zea (delays, TOF correction, `AngularSectorSum`, `Refocus`)
still applies once this reader hands back a per-scanline/focal-zone 192-element
complex-IQ signal.

**The remap, verified against ``ProcessDataMultiFocal_L12_3v.m``'s
``[aperture:end, 1:aperture-1]`` reorder:** HVMux maps hardware receive
channels onto physical elements *cyclically*, not linearly — physical element
``(ap0 + i)`` is read by hardware channel ``(ap0 + i) mod n_ch``. For a
128-channel receive aperture starting at element 65 (1-indexed), that means
channels 65-128 read elements 65-128, and channels 1-64 read elements
129-192 (PLAN.md §1.3, commit ``2307e23``). Getting this backwards produces a
plausible-looking but wrong 192-element signal — the same class of silent
error PLAN.md §3.5 warns about for the NVIDIA beamformer.

Each acquisition transmits once and receives through both aperture halves
(two "receive events" per transmit); the two are summed and averaged by
per-element coverage count, since an element read by both halves would
otherwise be double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _scalar(f: Any, ds: Any) -> float:
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def _deref_col(f: Any, refs_ds: Any, idx: int) -> np.ndarray:
    """Dereference one object-reference entry from an (N, 1) dataset."""
    refs = refs_ds[()]
    return np.asarray(f[refs[idx, 0]][()])


@dataclass
class TransmitGroup:
    """One (scanline, focal-zone) transmit, with both its receive-half events."""

    origin_x_m: float
    focus_m: float
    tx_aperture: int
    apod: np.ndarray  # channel-space transmit apodization
    delay_s: np.ndarray  # channel-space transmit delay, seconds
    rx_events: list[tuple[int, int]] = field(default_factory=list)  # (event_idx, rx_aperture)


@dataclass
class StanfordMetadata:
    c_m_s: float
    fc_hz: float
    wavelength_m: float
    pitch_m: float
    elem_x_m: np.ndarray  # (n_elem,) physical element positions
    n_elem: int
    fs_hz: float
    start_depth_m: float
    end_depth_m: float
    t0_s: float  # round-trip time to start_depth
    n_samp: int  # samples per receive event (buffer stride)
    n_frames: int
    groups: list[TransmitGroup]


def load_metadata(filepath: str | Path) -> StanfordMetadata:
    """Read acquisition geometry and group receive events by transmit.

    Ported from the old repo's ``visualize_beamformed_pymust.load_metadata``
    (PLAN.md §1.2/§1.3) — the numerics and group semantics are unchanged,
    only the return type is now a dataclass instead of a loose dict.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as _f:
        f: Any = _f  # h5py's type stubs can't model MATLAB v7.3 object references

        c = _scalar(f, f["Resource"]["Parameters"]["speedOfSound"])
        fc = _scalar(f, f["Trans"]["frequency"]) * 1e6
        lam = c / fc

        elem_x = f["Trans"]["ElementPos"][()][0, :] * lam
        n_elem = elem_x.size
        pitch = float(np.mean(np.diff(elem_x)))

        adc_rate = _scalar(f, f[f["Receive"]["ADCRate"][()][0, 0]]) * 1e6
        decim = _scalar(f, f[f["Receive"]["decimFactor"][()][0, 0]])
        fs = adc_rate / decim

        start_depth = _scalar(f, f["P"]["startDepth"])  # wavelengths
        end_depth = _scalar(f, f["P"]["endDepth"])

        s_refs = f["Receive"]["startSample"][()]
        s0 = int(_scalar(f, f[s_refs[0, 0]]))
        s1 = int(_scalar(f, f[s_refs[1, 0]]))
        n_samp = s1 - s0

        fr_refs = f["Receive"]["framenum"][()]
        n_acq = sum(1 for i in range(len(fr_refs)) if int(_scalar(f, f[fr_refs[i, 0]])) == 1)

        rcv_ref = f["RcvData"][()].flat[0]
        n_frames = int(f[rcv_ref].shape[0])

        apod_refs = f["TX"]["Apod"][()]
        orig_refs = f["TX"]["Origin"][()]
        focus_refs = f["TX"]["focus"][()]
        delay_refs = f["TX"]["Delay"][()]
        txap_refs = f["TX"]["aperture"][()]
        rxap_refs = f["Receive"]["aperture"][()]

        groups: dict[tuple[float, float], TransmitGroup] = {}
        order: list[tuple[float, float]] = []
        for i in range(min(len(apod_refs), n_acq)):
            origin_x = float(_deref_col(f, orig_refs, i).flatten()[0]) * lam
            focus_m = float(_deref_col(f, focus_refs, i).flatten()[0]) * lam
            rx_ap = int(_scalar(f, f[rxap_refs[i, 0]]))

            key = (round(origin_x, 6), round(focus_m, 6))
            if key not in groups:
                apod = _deref_col(f, apod_refs, i).flatten()
                delay_wl = _deref_col(f, delay_refs, i).flatten()
                tx_ap = int(_scalar(f, f[txap_refs[i, 0]]))
                groups[key] = TransmitGroup(
                    origin_x_m=origin_x,
                    focus_m=focus_m,
                    tx_aperture=tx_ap,
                    apod=apod,
                    delay_s=delay_wl / fc,
                )
                order.append(key)
            groups[key].rx_events.append((i, rx_ap))

    return StanfordMetadata(
        c_m_s=c,
        fc_hz=fc,
        wavelength_m=lam,
        pitch_m=pitch,
        elem_x_m=elem_x,
        n_elem=n_elem,
        fs_hz=fs,
        start_depth_m=start_depth * lam,
        end_depth_m=end_depth * lam,
        t0_s=2.0 * start_depth * lam / c,
        n_samp=n_samp,
        n_frames=n_frames,
        groups=[groups[k] for k in order],
    )


def combine_rx_halves(
    group: TransmitGroup, rcv_dataset: h5py.Dataset, frame_idx: int, meta: StanfordMetadata
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble the full ``n_elem``-element complex IQ signal for one
    transmit from its two HVMux receive-half events.

    Reads directly from the open ``RcvData`` HDF5 dataset (a few MB per
    event) rather than materializing the whole ``(n_rx, total_samples)``
    per-frame buffer (~2+ GB float32) — the same streaming approach as the
    old repo's ``_combine_rx_halves_streaming`` (PLAN.md §1.2).

    Returns:
        ``(iq, coverage)`` — ``iq`` is ``(n_samp, n_elem)`` complex64
        (zero where never received); ``coverage`` is ``(n_elem,)`` — number
        of receive halves that covered each element (0, 1, or 2).
    """
    from pymust import rf2iq

    n_samp, n_elem, fs, fc = meta.n_samp, meta.n_elem, meta.fs_hz, meta.fc_hz
    iq_out = np.zeros((n_samp, n_elem), dtype=np.complex64)
    coverage = np.zeros(n_elem, dtype=np.float32)

    for event_idx, rx_ap in group.rx_events:
        s = event_idx * n_samp
        rf = rcv_dataset[frame_idx, :, s : s + n_samp]  # (n_ch, n_samp), int16
        if rf.shape[1] != n_samp:
            continue
        iq = rf2iq(rf.T.astype(np.float64), fs, fc)  # (n_samp, n_ch)
        n_ch = iq.shape[1]
        # Cyclic HVMux remap (module docstring): physical element (ap0+i) is
        # read by hardware channel (ap0+i) mod n_ch.
        phys = (rx_ap - 1) + np.arange(n_ch)
        src = ((rx_ap - 1) + np.arange(n_ch)) % n_ch
        valid = phys < n_elem
        iq_out[:, phys[valid]] += iq[:, src[valid]]
        coverage[phys[valid]] += 1.0

    np.divide(iq_out, coverage[np.newaxis, :], out=iq_out, where=coverage[np.newaxis, :] > 0)
    return iq_out, coverage


def open_rcv_dataset(f: h5py.File) -> h5py.Dataset:
    """The ``(n_frames, n_rx, total_samples)`` raw-channel dataset, streamed."""
    return f[f["RcvData"][()].flat[0]]
