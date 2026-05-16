"""
Beamformed B-mode visualization of Verasonics .mat ultrasound files.

Implements IQ (analytic-signal) DAS with:
  - FSA:        spherical wave from each single active element
  - MultiFocal: virtual-source model at the transmit focus depth
  - Hadamard:   plane-wave approximation (unfocused flash)

Usage:
    poetry run python scripts/visualize_beamformed.py
    poetry run python scripts/visualize_beamformed.py --data-dir /path --out figures/bmode.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── HDF5 helpers ───────────────────────────────────────────────────────────────

def _scalar(f: h5py.File, ds) -> float:
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def _arr(f: h5py.File, ds) -> np.ndarray:
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        return np.asarray(f[v.flat[0]][()])
    return np.asarray(v)


def _deref_col(f: h5py.File, refs_ds, idx: int) -> np.ndarray:
    """Dereference one object-reference entry from a (N,1) dataset."""
    refs = refs_ds[()]
    return np.asarray(f[refs[idx, 0]][()])


# ── metadata ───────────────────────────────────────────────────────────────────

def load_metadata(filepath: Path) -> dict:
    with h5py.File(filepath, "r") as f:
        c    = _scalar(f, f["Resource"]["Parameters"]["speedOfSound"])
        freq = _scalar(f, f["Trans"]["frequency"]) * 1e6
        lam  = c / freq

        elem_x = f["Trans"]["ElementPos"][()][0, :] * lam   # (N_elem,) metres

        adc_rate = _scalar(f, f[f["Receive"]["ADCRate"][()][0, 0]]) * 1e6
        decim    = _scalar(f, f[f["Receive"]["decimFactor"][()][0, 0]])
        fs       = adc_rate / decim

        start_depth = _scalar(f, f["P"]["startDepth"])   # wavelengths
        end_depth   = _scalar(f, f["P"]["endDepth"])

        # Samples per acquisition: spacing between consecutive startSample indices
        s_refs  = f["Receive"]["startSample"][()]
        s0      = int(_scalar(f, f[s_refs[0, 0]]))
        s1      = int(_scalar(f, f[s_refs[1, 0]]))
        n_samp  = s1 - s0

        # Count acquisitions in frame 1
        fr_refs = f["Receive"]["framenum"][()]
        n_acq   = sum(
            1 for i in range(len(fr_refs))
            if int(_scalar(f, f[fr_refs[i, 0]])) == 1
        )

        # Per-event: active element indices, aperture origin x, focus depth
        apod_refs  = f["TX"]["Apod"][()]
        orig_refs  = f["TX"]["Origin"][()]
        focus_refs = f["TX"]["focus"][()]
        tx_events  = []
        for i in range(min(len(apod_refs), n_acq)):
            ap      = _deref_col(f, apod_refs, i).flatten()
            active  = np.where(ap > 0)[0]
            orig_x  = float(_deref_col(f, orig_refs, i).flatten()[0]) * lam
            foc_wav = float(_deref_col(f, focus_refs, i).flatten()[0])
            tx_events.append({
                "active": active,
                "origin_x": orig_x,
                "focus_m": foc_wav * lam,  # focus depth in metres
            })

    return {
        "c": c, "freq": freq, "lam": lam, "elem_x": elem_x,
        "fs": fs, "start_depth": start_depth, "end_depth": end_depth,
        "n_samp": n_samp, "n_acq": n_acq, "tx_events": tx_events,
    }


# ── RF frame loading ────────────────────────────────────────────────────────────

def load_rf_frame(filepath: Path, frame: int = 0) -> np.ndarray:
    """Return (n_rx, total_samples) float32 for the requested frame."""
    with h5py.File(filepath, "r") as f:
        rcv_ref = f["RcvData"][()].flat[0]
        rcv     = f[rcv_ref]              # (n_frames, n_rx, total_samples)
        return rcv[frame, :, :].astype(np.float32)


# ── signal processing ───────────────────────────────────────────────────────────

def _to_iq(rf: np.ndarray) -> np.ndarray:
    """FFT-based Hilbert transform along axis=1 → complex64 analytic signal."""
    N   = rf.shape[1]
    S   = np.fft.fft(rf.astype(np.float64), axis=1)
    h   = np.zeros(N)
    h[0] = 1
    h[1 : (N + 1) // 2] = 2
    if N % 2 == 0:
        h[N // 2] = 1
    return np.fft.ifft(S * h[np.newaxis, :], axis=1).astype(np.complex64)


# ── transmit delay models ───────────────────────────────────────────────────────

def _tx_delay_fsa(xx, zz, tx_x, c):
    """Spherical wave from single element at (tx_x, 0)."""
    return np.sqrt((xx - tx_x) ** 2 + zz ** 2) / c


def _tx_delay_focused(xx, zz, tx_x, tx_z, c):
    """
    Virtual-source model for a focused transmit.

    Above focus (zz < tx_z): converging wavefront, subtract distance.
    Below focus (zz >= tx_z): diverging from virtual source, add distance.
    """
    t_focus = tx_z / c
    dist    = np.sqrt((xx - tx_x) ** 2 + (zz - tx_z) ** 2)
    return t_focus + np.where(zz >= tx_z, dist, -dist) / c


def _tx_delay_plane(zz, c):
    """Unfocused flash (all elements simultaneous) ≈ plane wave along z."""
    return zz / c


# ── beamformer ──────────────────────────────────────────────────────────────────

def das_beamform(
    rf_raw: np.ndarray,
    meta: dict,
    grid_size: tuple[int, int] = (300, 300),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IQ delay-and-sum beamformer with linear interpolation and Hanning apodisation.

    Returns:
        log_img : (nz, nx) log-compressed envelope in dB  [−50, 0]
        z_grid  : depth axis (m)
        x_grid  : lateral axis (m)
    """
    c           = meta["c"]
    fs          = meta["fs"]
    lam         = meta["lam"]
    elem_x      = meta["elem_x"]
    start_depth = meta["start_depth"]
    end_depth   = meta["end_depth"]
    n_samp      = meta["n_samp"]
    tx_events   = meta["tx_events"]

    n_rx, total_samples = rf_raw.shape
    n_elem = len(elem_x)

    # RX element positions: map n_rx channels to centre n_rx elements of the array
    rx_off    = (n_elem - n_rx) // 2
    rx_elem_x = elem_x[rx_off : rx_off + n_rx].astype(np.float32)

    # Image grid
    z_min  = start_depth * lam
    z_max  = end_depth * lam
    nz, nx = grid_size
    z_grid = np.linspace(z_min, z_max, nz, dtype=np.float32)
    x_grid = np.linspace(elem_x[0], elem_x[-1], nx, dtype=np.float32)
    zz, xx = np.meshgrid(z_grid, x_grid, indexing="ij")          # (nz, nx)

    # Receive delays — precomputed once (do not depend on TX event)
    dx_rx = xx[np.newaxis] - rx_elem_x[:, np.newaxis, np.newaxis]  # (n_rx, nz, nx)
    t_rx  = np.sqrt(dx_rx ** 2 + zz[np.newaxis] ** 2) / c

    # Time of first received sample
    t_start = 2.0 * start_depth * lam / c

    # Hanning taper across receive aperture
    rx_apod = np.hanning(n_rx).astype(np.float32)[:, np.newaxis, np.newaxis]

    # Determine transmit type from the first event
    first_ev = tx_events[0]
    n_active  = len(first_ev["active"])
    foc_m     = first_ev["focus_m"]
    is_fsa     = n_active == 1
    is_plane   = (n_active > 1) and (foc_m < 1e-6)   # focus ≈ 0 → plane wave

    # Group consecutive TX events with the same virtual-source position (for averaging)
    def _src_key(ev):
        if is_fsa:
            return round(elem_x[ev["active"][0]] * 1e6)   # µm precision
        return round(ev["origin_x"] * 1e6)

    groups: list[tuple[dict, list[int]]] = []
    for ev_i, ev in enumerate(tx_events):
        k = _src_key(ev)
        if groups and _src_key(groups[-1][0]) == k:
            groups[-1][1].append(ev_i)
        else:
            groups.append((ev, [ev_i]))

    rx_idx   = np.arange(n_rx)[:, np.newaxis, np.newaxis]
    beamformed = np.zeros((nz, nx), dtype=np.complex64)

    # For focused (MultiFocal) acquisitions: restrict each scan-line's contribution
    # to the lateral Voronoi cell of its aperture centre.  FSA and plane-wave
    # compound across the full aperture, so they get no mask.
    if not is_fsa and not is_plane:
        origins = np.array([g[0]["origin_x"] for g in groups], dtype=np.float32)  # (N_groups,)
        # For every x in the image grid, find the nearest scan-line centre
        dist_to_origin = np.abs(x_grid[np.newaxis, :] - origins[:, np.newaxis])   # (N_groups, nx)
        nearest_group  = np.argmin(dist_to_origin, axis=0)                         # (nx,)
    else:
        nearest_group = None

    for grp_idx, (ev, ev_list) in enumerate(groups):
        # Average RF over repeated acquisitions for the same TX position
        rf_acq = np.zeros((n_rx, n_samp), dtype=np.float32)
        cnt    = 0
        for i in ev_list:
            s, e = i * n_samp, i * n_samp + n_samp
            if e <= total_samples:
                rf_acq += rf_raw[:, s:e]
                cnt    += 1
        if cnt == 0:
            continue
        rf_acq /= cnt

        # Convert to analytic (IQ) signal along the depth axis
        iq_acq = _to_iq(rf_acq)                                   # (n_rx, n_samp) complex64

        # Transmit delay
        if is_fsa:
            tx_x  = elem_x[ev["active"][0]]
            t_tx  = _tx_delay_fsa(xx, zz, tx_x, c)
        elif is_plane:
            t_tx  = _tx_delay_plane(zz, c)
        else:
            tx_x  = ev["origin_x"]
            tx_z  = ev["focus_m"]
            t_tx  = _tx_delay_focused(xx, zz, tx_x, tx_z, c)

        # Float sample indices (enable linear interpolation)
        t_total = t_tx[np.newaxis] + t_rx                         # (n_rx, nz, nx)
        si_f    = np.clip((t_total - t_start) * fs, 0.0, n_samp - 1.001)
        si_i    = si_f.astype(np.int32)
        frac    = (si_f - si_i).astype(np.float32)

        # Complex linear interpolation + aperture-weighted coherent sum
        va     = iq_acq[rx_idx, si_i]
        vb     = iq_acq[rx_idx, np.minimum(si_i + 1, n_samp - 1)]
        values = va + frac * (vb - va)                            # (n_rx, nz, nx)
        contribution = (values * rx_apod).sum(axis=0)             # (nz, nx)

        if nearest_group is not None:
            # Zero out lateral positions that belong to a different scan-line
            contribution *= (nearest_group == grp_idx)            # broadcast over nz

        beamformed += contribution

    env     = np.abs(beamformed)
    env     = np.maximum(env, 1e-9)
    log_img = 20.0 * np.log10(env / env.max())
    return np.clip(log_img, -50, 0), z_grid, x_grid


# ── plotting ────────────────────────────────────────────────────────────────────

def _acq_type(name: str) -> str:
    n = name.lower()
    if "fsa" in n:        return "FSA"
    if "hadamard" in n:   return "Hadamard"
    if "multifocal" in n: return "MultiFocal"
    return "Unknown"


def visualize(
    data_dir: Path,
    out: Path,
    grid_size: tuple[int, int] = (300, 300),
    frame: int = 0,
) -> None:
    mat_files = sorted(data_dir.glob("DATA_*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No DATA_*.mat files in {data_dir}")

    n     = len(mat_files)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, fpath in zip(axes, mat_files):
        dtype = _acq_type(fpath.name)
        print(f"  {fpath.name}  [{dtype}] …", flush=True)
        try:
            meta   = load_metadata(fpath)
            rf_raw = load_rf_frame(fpath, frame=frame)
            log_img, z_grid, x_grid = das_beamform(rf_raw, meta, grid_size=grid_size)

            z_mm = z_grid * 1e3
            x_mm = x_grid * 1e3
            ax.imshow(
                log_img,
                cmap="gray", vmin=-50, vmax=0,
                extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]],
                aspect="equal",
            )
            ax.set_title(f"{dtype} — {fpath.stem[-15:]}", fontsize=8)
            ax.set_xlabel("Lateral (mm)", fontsize=7)
            ax.set_ylabel("Depth (mm)", fontsize=7)
        except Exception as exc:
            import traceback
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")
            traceback.print_exc()

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("IQ DAS Beamformed B-mode", fontsize=11, y=1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n→ {out}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--out", type=Path, default=Path("figures/bmode_all.png"))
    parser.add_argument("--grid", type=int, default=300)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    visualize(args.data_dir, args.out, grid_size=(args.grid, args.grid), frame=args.frame)
