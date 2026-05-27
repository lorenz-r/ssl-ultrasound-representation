"""
Beamformed B-mode visualization of Verasonics MultiFocal .mat ultrasound files.

Implements IQ (analytic-signal) DAS with a virtual-source model at the transmit
focus depth (MultiFocal acquisitions only).

Usage:
    poetry run python scripts/visualize_beamformed.py
    poetry run python scripts/visualize_beamformed.py --data-dir /path --out figures/bmode.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── HDF5 helpers ───────────────────────────────────────────────────────────────


def _scalar(f: Any, ds: Any) -> float:
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def _deref_col(f: Any, refs_ds: Any, idx: int) -> np.ndarray:
    """Dereference one object-reference entry from a (N,1) dataset."""
    refs = refs_ds[()]
    return np.asarray(f[refs[idx, 0]][()])


# ── metadata ───────────────────────────────────────────────────────────────────


def load_metadata(filepath: Path) -> dict:  # read the probes/scanners parameters
    with h5py.File(filepath, "r") as _f:
        f: Any = _f  # h5py stubs can't model MATLAB v7.3 object references
        c = _scalar(
            f, f["Resource"]["Parameters"]["speedOfSound"]
        )  # speed of sound (during scans in tissue)
        freq = (
            _scalar(f, f["Trans"]["frequency"]) * 1e6
        )  # transducer frequency (dominant frequency of emitted sound waves)
        lam = c / freq  # wavelength in metres

        elem_x = (
            f["Trans"]["ElementPos"][()][0, :] * lam
        )  # every transducer element x-position in metres

        adc_rate = (
            _scalar(f, f[f["Receive"]["ADCRate"][()][0, 0]]) * 1e6
        )  # read the probes adc rate and convert to Hz (from MHz)
        decim = _scalar(
            f, f[f["Receive"]["decimFactor"][()][0, 0]]
        )  # how much downsampling to apply (decimation)
        fs = adc_rate / decim  # resulting sampling rate

        # configured depth window of scans in wavelengths
        start_depth = _scalar(f, f["P"]["startDepth"])
        end_depth = _scalar(f, f["P"]["endDepth"])

        s_refs = f["Receive"]["startSample"][
            ()
        ]  # start of recording (acquisition event) at sample index
        s0 = int(_scalar(f, f[s_refs[0, 0]]))
        s1 = int(_scalar(f, f[s_refs[1, 0]]))
        n_samp = s1 - s0  # how many samples belong to one transmit-receive event

        # count acquisitions in frame 1 (for example num of elements)
        fr_refs = f["Receive"]["framenum"][()]
        n_acq = sum(
            1 for i in range(len(fr_refs)) if int(_scalar(f, f[fr_refs[i, 0]])) == 1
        )

        # per event
        apod_refs = f["TX"]["Apod"][
            ()
        ]  # array of transducer elements activation/voltage (apodization)
        orig_refs = f["TX"]["Origin"][
            ()
        ]  # lateral center of transmit element in wavelengths
        focus_refs = f["TX"]["focus"][()]  # focus of the wave-depth in wavelengths
        tx_events = []
        for i in range(min(len(apod_refs), n_acq)):
            ap = _deref_col(f, apod_refs, i).flatten()
            active = np.where(ap > 0)[0]  # did the element transmit?
            orig_x = (
                float(_deref_col(f, orig_refs, i).flatten()[0]) * lam
            )  # origin of the element x in meters
            foc_wav = float(_deref_col(f, focus_refs, i).flatten()[0])
            tx_events.append(
                {
                    "active": active,
                    "origin_x": orig_x,
                    "focus_m": foc_wav * lam,  # focus depth in metres
                }
            )

    return {
        "c": c,
        "freq": freq,
        "lam": lam,
        "elem_x": elem_x,
        "fs": fs,
        "start_depth": start_depth,
        "end_depth": end_depth,
        "n_samp": n_samp,
        "n_acq": n_acq,
        "tx_events": tx_events,
    }


# ── RF frame loading ────────────────────────────────────────────────────────────


def load_rf_frame(filepath: Path, frame: int = 0) -> np.ndarray:
    """Return (n_rx, total_samples) float32 for the requested frame."""
    with h5py.File(filepath, "r") as _f:
        f: Any = _f  # h5py stubs can't model MATLAB v7.3 object references
        rcv_ref = f["RcvData"][()].flat[0]
        rcv = f[rcv_ref]  # (n_frames, n_rx, total_samples)
        return rcv[frame, :, :].astype(np.float32)


# ── signal processing ───────────────────────────────────────────────────────────


def _to_iq(rf: np.ndarray) -> np.ndarray:
    """FFT-based Hilbert transform along axis=1 → complex64 analytic signal."""
    N = rf.shape[1]
    S = np.fft.fft(rf.astype(np.float64), axis=1)
    h = np.zeros(N)
    h[0] = 1
    h[1 : (N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    return np.fft.ifft(S * h[np.newaxis, :], axis=1).astype(np.complex64)


# ── transmit delay model ────────────────────────────────────────────────────────


def _tx_delay_focused(xx, zz, tx_x, tx_z, elem_x_active, c):
    """
    Virtual-source transmit delay following the rtbf convention.

    t0 = max distance from any active TX element to the focus / c.
    This is the time at which the converging wavefront arrives at the focus.
    Using just tx_z/c (depth only) misses the lateral aperture contribution
    and shifts the reconstructed image.

    Above focus (zz < tx_z): converging wavefront, subtract distance from t0.
    Below focus (zz >= tx_z): diverging from virtual source, add distance to t0.
    """
    t0 = float(np.max(np.sqrt((elem_x_active - tx_x) ** 2 + tx_z ** 2))) / c
    pv_z = zz - tx_z
    pv_dist = np.sqrt((xx - tx_x) ** 2 + pv_z ** 2)
    return np.copysign(pv_dist, pv_z) / c + t0


# ── beamformer ──────────────────────────────────────────────────────────────────


def das_beamform(
    rf_raw: np.ndarray,
    meta: dict,
    grid_size: tuple[int, int] = (300, 300),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IQ DAS beamformer for MultiFocal acquisitions.

    Uses a virtual-source delay model with per-scan-line Voronoi lateral masking.

    Returns:
        log_img : (nz, nx) log-compressed envelope in dB  [−50, 0]
        z_grid  : depth axis (m)
        x_grid  : lateral axis (m)
    """
    c = meta["c"]
    fs = meta["fs"]
    lam = meta["lam"]
    elem_x = meta["elem_x"]
    start_depth = meta["start_depth"]
    end_depth = meta["end_depth"]
    n_samp = meta["n_samp"]
    tx_events = meta["tx_events"]

    n_rx, total_samples = rf_raw.shape
    n_elem = len(elem_x)

    # RX element positions: map n_rx channels to centre n_rx elements of the array
    rx_off = (n_elem - n_rx) // 2
    rx_elem_x = elem_x[rx_off : rx_off + n_rx].astype(np.float32)

    # Image grid
    z_min = start_depth * lam
    z_max = end_depth * lam
    nz, nx = grid_size
    z_grid = np.linspace(z_min, z_max, nz, dtype=np.float32)
    x_grid = np.linspace(elem_x[0], elem_x[-1], nx, dtype=np.float32)
    zz, xx = np.meshgrid(z_grid, x_grid, indexing="ij")  # (nz, nx)

    # Receive delays — precomputed once (do not depend on TX event)
    dx_rx = xx[np.newaxis] - rx_elem_x[:, np.newaxis, np.newaxis]  # (n_rx, nz, nx)
    t_rx = np.sqrt(dx_rx**2 + zz[np.newaxis] ** 2) / c

    # Time of first received sample
    t_start = 2.0 * start_depth * lam / c

    # Hanning taper across receive aperture
    rx_apod = np.hanning(n_rx).astype(np.float32)[:, np.newaxis, np.newaxis]

    # Group consecutive TX events with the same virtual-source position (for averaging)
    def _src_key(ev):
        return round(ev["origin_x"] * 1e6)  # µm precision

    groups: list[tuple[dict, list[int]]] = []
    for ev_i, ev in enumerate(tx_events):
        k = _src_key(ev)
        if groups and _src_key(groups[-1][0]) == k:
            groups[-1][1].append(ev_i)
        else:
            groups.append((ev, [ev_i]))

    rx_idx = np.arange(n_rx)[:, np.newaxis, np.newaxis]
    beamformed = np.zeros((nz, nx), dtype=np.complex64)

    # Restrict each scan-line's contribution to the Voronoi cell of its aperture centre
    origins = np.array([g[0]["origin_x"] for g in groups], dtype=np.float32)
    dist_to_origin = np.abs(x_grid[np.newaxis, :] - origins[:, np.newaxis])  # (N_groups, nx)
    nearest_group = np.argmin(dist_to_origin, axis=0)  # (nx,)

    for grp_idx, (ev, ev_list) in enumerate(groups):
        # Average RF over repeated acquisitions for the same TX position
        rf_acq = np.zeros((n_rx, n_samp), dtype=np.float32)
        cnt = 0
        for i in ev_list:
            s, e = i * n_samp, i * n_samp + n_samp
            if e <= total_samples:
                rf_acq += rf_raw[:, s:e]
                cnt += 1
        if cnt == 0:
            continue
        rf_acq /= cnt

        # Convert to analytic (IQ) signal along the depth axis
        iq_acq = _to_iq(rf_acq)  # (n_rx, n_samp) complex64

        # Transmit delay — virtual-source model (rtbf convention)
        tx_x = ev["origin_x"]
        tx_z = ev["focus_m"]
        t_tx = _tx_delay_focused(xx, zz, tx_x, tx_z, elem_x[ev["active"]], c)

        # Float sample indices (enable linear interpolation)
        t_total = t_tx[np.newaxis] + t_rx  # (n_rx, nz, nx)
        si_f = np.clip((t_total - t_start) * fs, 0.0, n_samp - 1.001)
        si_i = si_f.astype(np.int32)
        frac = (si_f - si_i).astype(np.float32)

        # Complex linear interpolation + aperture-weighted coherent sum
        va = iq_acq[rx_idx, si_i]
        vb = iq_acq[rx_idx, np.minimum(si_i + 1, n_samp - 1)]
        values = va + frac * (vb - va)  # (n_rx, nz, nx)
        contribution = (values * rx_apod).sum(axis=0)  # (nz, nx)

        contribution *= nearest_group == grp_idx  # zero columns owned by other scan-lines

        beamformed += contribution

    env = np.abs(beamformed)
    env = np.maximum(env, 1e-9)
    log_img = 20.0 * np.log10(env / env.max())
    return np.clip(log_img, -50, 0), z_grid, x_grid


# ── plotting ────────────────────────────────────────────────────────────────────


def visualize(
    data_dir: Path,
    out: Path,
    grid_size: tuple[int, int] = (300, 300),
    frame: int = 0,
) -> None:
    all_files = sorted(data_dir.glob("DATA_*.mat"))
    mat_files = [f for f in all_files if "multifocal" in f.name.lower()]
    if not mat_files:
        raise FileNotFoundError(f"No MultiFocal DATA_*.mat files in {data_dir}")

    n = len(mat_files)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, fpath in zip(axes, mat_files):
        print(f"  {fpath.name}  [MultiFocal] …", flush=True)
        try:
            meta = load_metadata(fpath)
            rf_raw = load_rf_frame(fpath, frame=frame)
            log_img, z_grid, x_grid = das_beamform(rf_raw, meta, grid_size=grid_size)

            z_mm = z_grid * 1e3
            x_mm = x_grid * 1e3
            ax.imshow(
                log_img,
                cmap="gray",
                vmin=-50,
                vmax=0,
                extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
                aspect="equal",
            )
            ax.set_title(f"MultiFocal — {fpath.stem[-15:]}", fontsize=8)
            ax.set_xlabel("Lateral (mm)", fontsize=7)
            ax.set_ylabel("Depth (mm)", fontsize=7)
        except Exception as exc:
            import traceback

            ax.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
                color="red",
            )
            traceback.print_exc()

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("IQ DAS Beamformed B-mode", fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n→ {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/rats/Rat1/exposed_liver/")
    )
    parser.add_argument("--out", type=Path, default=Path("figures/bmode_all.png"))
    parser.add_argument("--grid", type=int, default=300)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    visualize(
        args.data_dir, args.out, grid_size=(args.grid, args.grid), frame=args.frame
    )
