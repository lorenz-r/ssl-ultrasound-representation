"""
Beamformed B-mode visualization of Verasonics .mat ultrasound files.

Usage:
    poetry run python scripts/visualize_beamformed.py
    poetry run python scripts/visualize_beamformed.py --data-dir /path/to/files --out figures/bmode.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ────────────────────────────────────────────────────────────────────

def _deref_scalar(f: h5py.File, ds) -> float:
    v = ds[()] if isinstance(ds, h5py.Dataset) else ds
    if hasattr(v, "dtype") and v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def _deref_arr(f: h5py.File, ds) -> np.ndarray:
    v = ds[()] if isinstance(ds, h5py.Dataset) else ds
    if hasattr(v, "dtype") and v.dtype == object:
        return np.asarray(f[v.flat[0]][()])
    return np.asarray(v)


# ── metadata ───────────────────────────────────────────────────────────────────

def load_metadata(filepath: Path) -> dict:
    with h5py.File(filepath, "r") as f:
        c    = _deref_scalar(f, f["Resource"]["Parameters"]["speedOfSound"])
        freq = _deref_scalar(f, f["Trans"]["frequency"]) * 1e6          # Hz
        lam  = c / freq                                                   # m

        # All element x-positions in metres
        elem_x = f["Trans"]["ElementPos"][()][0, :] * lam

        # Effective sample rate (ADC rate / decimation)
        adc_refs = f["Receive"]["ADCRate"][()]
        fs_adc   = _deref_scalar(f, f[adc_refs[0, 0]]) * 1e6
        dec_refs = f["Receive"]["decimFactor"][()]
        decim    = _deref_scalar(f, f[dec_refs[0, 0]])
        fs       = fs_adc / decim

        start_depth = _deref_scalar(f, f["P"]["startDepth"])   # wavelengths
        end_depth   = _deref_scalar(f, f["P"]["endDepth"])

        # Samples per acquisition from first two receive events
        s_refs = f["Receive"]["startSample"][()]
        s0 = int(_deref_scalar(f, f[s_refs[0, 0]]))
        s1 = int(_deref_scalar(f, f[s_refs[1, 0]]))
        n_samples_per_acq = s1 - s0

        # Count acquisitions per frame
        fr_refs = f["Receive"]["framenum"][()]
        n_acq_per_frame = sum(
            1 for i in range(len(fr_refs))
            if int(_deref_scalar(f, f[fr_refs[i, 0]])) == 1
        )

        # TX apodisation → identify active element indices per event
        apod_refs = f["TX"]["Apod"][()]
        n_tx      = len(apod_refs)
        tx_apods  = []
        for i in range(min(n_tx, n_acq_per_frame)):        # first frame only
            ap = _deref_arr(f, f[apod_refs[i, 0]]).flatten()
            tx_apods.append(np.where(ap > 0)[0])

        # TX origin (centre of active aperture, in wavelengths)
        orig_refs = f["TX"]["Origin"][()]
        tx_origins = []
        for i in range(min(n_tx, n_acq_per_frame)):
            orig = _deref_arr(f, f[orig_refs[i, 0]]).flatten()
            tx_origins.append(orig[0] * lam)   # x only, convert to metres

    return {
        "c": c, "freq": freq, "lam": lam, "elem_x": elem_x,
        "fs": fs, "start_depth": start_depth, "end_depth": end_depth,
        "n_samples_per_acq": n_samples_per_acq,
        "n_acq_per_frame": n_acq_per_frame,
        "tx_apods": tx_apods,
        "tx_origins_x": tx_origins,
    }


# ── RF data loading ────────────────────────────────────────────────────────────

def load_rf_frame(filepath: Path, frame: int = 0) -> np.ndarray:
    """Return (n_rx, total_samples) float32 for the requested frame."""
    with h5py.File(filepath, "r") as f:
        rcv_ref = f["RcvData"][()].flat[0]
        rcv     = f[rcv_ref]                        # (n_frames, n_rx, total_samples)
        return rcv[frame, :, :].astype(np.float32)


# ── beamforming ────────────────────────────────────────────────────────────────

def _envelope(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """FFT-based Hilbert envelope (no scipy needed)."""
    N   = signal.shape[axis]
    S   = np.fft.fft(signal.astype(np.float64), axis=axis)
    h   = np.zeros(N)
    h[0] = 1
    h[1 : (N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    shape       = [1] * signal.ndim
    shape[axis] = N
    analytic    = np.fft.ifft(S * h.reshape(shape), axis=axis)
    return np.abs(analytic).astype(np.float32)


def das_beamform(
    rf_raw: np.ndarray,
    meta: dict,
    grid_size: tuple[int, int] = (300, 300),
) -> np.ndarray:
    """
    Delay-and-sum beamformer.  Works for FSA (single-element TX) and
    multi-element TX (MultiFocal / Hadamard) using a virtual-source at
    the aperture centre.

    Returns a (nz, nx) log-compressed B-mode image in dB.
    """
    c            = meta["c"]
    fs           = meta["fs"]
    lam          = meta["lam"]
    elem_x       = meta["elem_x"]
    start_depth  = meta["start_depth"]
    end_depth    = meta["end_depth"]
    n_samp       = meta["n_samples_per_acq"]
    tx_apods     = meta["tx_apods"]
    tx_origins_x = meta["tx_origins_x"]

    n_rx, total_samples = rf_raw.shape
    n_elem = len(elem_x)

    # RX positions: map 128 channels to centre n_rx elements of the array
    rx_off   = (n_elem - n_rx) // 2
    rx_elem_x = elem_x[rx_off : rx_off + n_rx].astype(np.float32)

    # Image grid (metres)
    z_min = start_depth * lam
    z_max = end_depth * lam
    nz, nx = grid_size
    z_grid = np.linspace(z_min, z_max, nz, dtype=np.float32)
    x_grid = np.linspace(elem_x[0], elem_x[-1], nx, dtype=np.float32)
    zz, xx = np.meshgrid(z_grid, x_grid, indexing="ij")  # (nz, nx)

    # Precompute receive delays (independent of TX event)
    dx_rx  = xx[np.newaxis] - rx_elem_x[:, np.newaxis, np.newaxis]  # (n_rx, nz, nx)
    t_rx   = np.sqrt(dx_rx**2 + zz[np.newaxis]**2) / c              # (n_rx, nz, nx)
    t_start = 2.0 * start_depth * lam / c

    # Group consecutive TX events that share the same virtual-source x position
    # (handles both FSA "fire same element twice" and MultiFocal sub-apertures)
    groups: list[tuple[float, list[int]]] = []
    for ev_i, (apod, orig_x) in enumerate(zip(tx_apods, tx_origins_x)):
        if len(apod) == 1:
            tx_x = elem_x[apod[0]]          # FSA: exact element position
        else:
            tx_x = orig_x                    # multi-element: aperture centre

        if groups and abs(groups[-1][0] - tx_x) < 1e-9:
            groups[-1][1].append(ev_i)
        else:
            groups.append((tx_x, [ev_i]))

    rx_idx = np.arange(n_rx)[:, np.newaxis, np.newaxis]  # for advanced indexing

    beamformed = np.zeros((nz, nx), dtype=np.float32)

    for tx_x, ev_list in groups:
        # Average RF data over repeated acquisitions for the same TX position
        rf_acq = np.zeros((n_rx, n_samp), dtype=np.float32)
        count  = 0
        for ev in ev_list:
            s, e = ev * n_samp, ev * n_samp + n_samp
            if e <= total_samples:
                rf_acq += rf_raw[:, s:e]
                count += 1
        if count == 0:
            continue
        rf_acq /= count

        # Transmit delay: spherical wave from virtual source
        t_tx = np.sqrt((xx - tx_x)**2 + zz**2) / c        # (nz, nx)

        # Look-up indices
        t_total    = t_tx[np.newaxis] + t_rx               # (n_rx, nz, nx)
        samp_idx   = np.clip(
            (t_total - t_start) * fs, 0, n_samp - 1
        ).astype(np.int32)                                  # (n_rx, nz, nx)

        beamformed += rf_acq[rx_idx, samp_idx].sum(axis=0)

    # Envelope along depth + log-compression
    env = _envelope(beamformed, axis=0)
    env = np.maximum(env, 1e-9)
    log_img = 20.0 * np.log10(env / env.max())
    return np.clip(log_img, -50, 0), z_grid, x_grid


# ── plotting ───────────────────────────────────────────────────────────────────

def _detect_type(name: str) -> str:
    n = name.lower()
    if "fsa" in n:       return "FSA"
    if "hadamard" in n:  return "Hadamard"
    if "multifocal" in n: return "MultiFocal"
    return "Unknown"


def visualize(data_dir: Path, out: Path, grid_size: tuple[int, int] = (300, 300)) -> None:
    mat_files = sorted(data_dir.glob("DATA_*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No DATA_*.mat files found in {data_dir}")

    n = len(mat_files)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, fpath in zip(axes, mat_files):
        dtype = _detect_type(fpath.name)
        print(f"Processing {fpath.name} ({dtype}) …", flush=True)
        try:
            meta   = load_metadata(fpath)
            rf_raw = load_rf_frame(fpath, frame=0)
            log_img, z_grid, x_grid = das_beamform(rf_raw, meta, grid_size=grid_size)

            z_mm = z_grid * 1e3
            x_mm = x_grid * 1e3
            ax.imshow(
                log_img,
                cmap="gray", vmin=-50, vmax=0,
                extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
                aspect="equal",
            )
            ax.set_title(f"{dtype}\n{fpath.stem[-15:]}", fontsize=8)
            ax.set_xlabel("Lateral (mm)", fontsize=7)
            ax.set_ylabel("Depth (mm)", fontsize=7)
        except Exception as exc:
            ax.set_visible(False)
            print(f"  ERROR: {exc}")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("DAS Beamformed B-mode Images", fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize beamformed ultrasound images")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads",
                        help="Directory containing DATA_*.mat files")
    parser.add_argument("--out", type=Path, default=Path("figures/bmode_all.png"),
                        help="Output image path")
    parser.add_argument("--grid", type=int, default=300,
                        help="Grid resolution (NxN pixels per image)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to load (0-indexed)")
    args = parser.parse_args()

    visualize(args.data_dir, args.out, grid_size=(args.grid, args.grid))
