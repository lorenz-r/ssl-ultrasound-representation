"""
Usage:
    python scripts/visualize_beamformed_pymust.py
    python scripts/visualize_beamformed_pymust.py --data-dir data/rats --out figures/bmode_pymust.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pymust
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── HDF5 helpers ────────────────────────────────────────────────────────────────


def _scalar(f: Any, ds: Any) -> float:
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def _deref_col(f: Any, refs_ds: Any, idx: int) -> np.ndarray:
    """Dereference one object-reference entry from an (N,1) dataset."""
    refs = refs_ds[()]
    return np.asarray(f[refs[idx, 0]][()])


# ── metadata ────────────────────────────────────────────────────────────────────


def load_metadata(filepath: Path) -> dict:
    """
    Read acquisition geometry and group events into (scan line, focal zone) units.

    Each group bundles the two receive-half events that share one transmit.
    """
    with h5py.File(filepath, "r") as _f:
        f: Any = _f  # h5py stubs can't model MATLAB v7.3 object references

        c = _scalar(f, f["Resource"]["Parameters"]["speedOfSound"])
        fc = _scalar(f, f["Trans"]["frequency"]) * 1e6
        lam = c / fc

        elem_x = f["Trans"]["ElementPos"][()][0, :] * lam  # 192 physical positions (m)
        n_elem = elem_x.size
        pitch = float(np.mean(np.diff(elem_x)))

        adc_rate = _scalar(f, f[f["Receive"]["ADCRate"][()][0, 0]]) * 1e6
        decim = _scalar(f, f[f["Receive"]["decimFactor"][()][0, 0]])
        fs = adc_rate / decim

        start_depth = _scalar(f, f["P"]["startDepth"])  # wavelengths
        end_depth = _scalar(f, f["P"]["endDepth"])

        # samples per receive event (buffer stride)
        s_refs = f["Receive"]["startSample"][()]
        s0 = int(_scalar(f, f[s_refs[0, 0]]))
        s1 = int(_scalar(f, f[s_refs[1, 0]]))
        n_samp = s1 - s0

        # acquisitions belonging to frame 1
        fr_refs = f["Receive"]["framenum"][()]
        n_acq = sum(
            1 for i in range(len(fr_refs)) if int(_scalar(f, f[fr_refs[i, 0]])) == 1
        )

        apod_refs = f["TX"]["Apod"][()]
        orig_refs = f["TX"]["Origin"][()]
        focus_refs = f["TX"]["focus"][()]
        delay_refs = f["TX"]["Delay"][()]
        txap_refs = f["TX"]["aperture"][()]
        rxap_refs = f["Receive"]["aperture"][()]

        groups: dict[tuple, dict] = {}
        order: list[tuple] = []
        for i in range(min(len(apod_refs), n_acq)):
            origin_x = float(_deref_col(f, orig_refs, i).flatten()[0]) * lam
            focus_m = float(_deref_col(f, focus_refs, i).flatten()[0]) * lam
            rx_ap = int(_scalar(f, f[rxap_refs[i, 0]]))

            key = (round(origin_x, 6), round(focus_m, 6))
            if key not in groups:
                # transmit info is identical across the two receive halves
                apod = _deref_col(f, apod_refs, i).flatten()
                delay_wl = _deref_col(f, delay_refs, i).flatten()  # channel space
                tx_ap = int(_scalar(f, f[txap_refs[i, 0]]))
                groups[key] = {
                    "origin_x": origin_x,
                    "focus_m": focus_m,
                    "tx_aperture": tx_ap,
                    "apod": apod,
                    "delay_s": delay_wl / fc,  # seconds, channel space
                    "rx_events": [],  # (buffer_event_index, rx_aperture)
                }
                order.append(key)
            groups[key]["rx_events"].append((i, rx_ap))

    return {
        "c": c,
        "fc": fc,
        "lam": lam,
        "pitch": pitch,
        "elem_x": elem_x,
        "n_elem": n_elem,
        "fs": fs,
        "start_depth_m": start_depth * lam,
        "end_depth_m": end_depth * lam,
        "t0": 2.0 * start_depth * lam / c,  # round-trip time to startDepth
        "n_samp": n_samp,
        "groups": [groups[k] for k in order],
    }


# ── RF frame loading ──────────────────────────────────────────────────────────────


def load_rf_frame(filepath: Path, frame: int = 0) -> np.ndarray:
    """Return (n_rx, total_samples) float32 for the requested frame."""
    with h5py.File(filepath, "r") as _f:
        f: Any = _f
        rcv = f[f["RcvData"][()].flat[0]]  # (n_frames, n_rx, total_samples)
        return rcv[frame, :, :].astype(np.float32)


# ── receive-half synthesis ──────────────────────────────────────────────────────


def _combine_rx_halves(group: dict, rf_raw: np.ndarray, meta: dict) -> np.ndarray:
    """
    Synthesise a full 192-element I/Q signal from the two receive sub-apertures.

    Returns (n_samp, n_elem) complex I/Q, zero in any element never received.
    """
    n_samp, n_elem, fs, fc = meta["n_samp"], meta["n_elem"], meta["fs"], meta["fc"]
    iq192 = np.zeros((n_samp, n_elem), dtype=np.complex64)
    count = np.zeros(n_elem, dtype=np.float32)

    for ev_i, rx_ap in group["rx_events"]:
        s = ev_i * n_samp
        rf = rf_raw[:, s : s + n_samp]  # (n_ch, n_samp)
        if rf.shape[1] != n_samp:
            continue
        iq = pymust.rf2iq(rf.T.astype(np.float64), fs, fc)  # (n_samp, n_ch)
        n_ch = iq.shape[1]
        # HVMux maps channels onto physical elements cyclically (cf.
        # [aperture:end, 1:aperture-1] in ProcessDataMultiFocal_L12_3v.m):
        # element (ap0+i) is read by channel (ap0+i) mod n_ch.
        phys = (rx_ap - 1) + np.arange(n_ch)          # destination physical element
        src = ((rx_ap - 1) + np.arange(n_ch)) % n_ch  # source hardware channel
        valid = phys < n_elem
        iq192[:, phys[valid]] += iq[:, src[valid]]
        count[phys[valid]] += 1.0

    np.divide(iq192, count[np.newaxis, :], out=iq192, where=count[np.newaxis, :] > 0)
    return iq192


def _tx_delays_192(group: dict, meta: dict) -> np.ndarray:
    """Map channel-space TX delays onto the 192-element array; NaN = inactive."""
    n_elem = meta["n_elem"]
    delays = np.full(n_elem, np.nan, dtype=np.float64)
    apod = group["apod"]
    active = np.where(apod > 0)[0]
    phys = (group["tx_aperture"] - 1) + active
    valid = phys < n_elem
    delays[phys[valid]] = group["delay_s"][active[valid]]
    return delays


# ── beamformer ──────────────────────────────────────────────────────────────────


def beamform_per_zone(
    meta: dict,
    rf_raw: np.ndarray,
    nx: int = 256,
    nz: int = 384,
    strip_halfwidth_m: float = 1.5e-3,
    fnumber: float = 1.5,
    scanline_step: int = 1,
    max_depth_m: float | None = 25e-3,
) -> tuple[list[float], dict[float, np.ndarray], np.ndarray, np.ndarray]:
    """
    DAS-beamform each focal zone into its own full-depth complex composite.

    This is the expensive step. Each focal zone is reconstructed across the
    entire depth grid (it is only well-focused near its own focal depth), so the
    per-zone composites can later be stitched in any combination.

    This acquisition records no echo below ~20 mm, so max_depth_m caps the
    reconstruction depth (saves compute and crops the empty region).

    Returns (focal_depths_sorted, {focus_m: complex (nz,nx)}, x_grid_m, z_grid_m).
    """
    elem_x = meta["elem_x"]
    z_end = (
        meta["end_depth_m"]
        if max_depth_m is None
        else min(meta["end_depth_m"], max_depth_m)
    )
    x_grid = np.linspace(elem_x[0], elem_x[-1], nx)
    z_grid = np.linspace(meta["start_depth_m"], z_end, nz)

    # PyMUST parameters (uniform 192-element array centred at x=0)
    param = pymust.utils.Param()
    param.fs = meta["fs"]
    param.fc = meta["fc"]
    param.c = meta["c"]
    param.pitch = meta["pitch"]
    param.Nelements = meta["n_elem"]
    param.t0 = np.array([[meta["t0"]]])
    param.fnumber = fnumber

    n_samp = meta["n_samp"]
    iq_shape = 1j * np.array(
        [n_samp, meta["n_elem"]]
    )  # PyMUST IQ shape convention: imaginary flag signals complex input

    # focal zones (sorted by depth) for the axial cross-fade
    focal_depths = sorted({round(g["focus_m"], 6) for g in meta["groups"]})
    zone_imgs = {fz: np.zeros((nz, nx), dtype=np.complex64) for fz in focal_depths}
    zone_wts = {fz: np.zeros((nz, nx), dtype=np.float32) for fz in focal_depths}

    groups = meta["groups"][::scanline_step]
    for gi, group in enumerate(groups):
        fz = round(group["focus_m"], 6)
        ox = group["origin_x"]

        # lateral strip of columns around this scan line
        cols = np.where(np.abs(x_grid - ox) <= strip_halfwidth_m)[0]
        if cols.size == 0:
            continue
        xs = x_grid[cols]

        iq192 = _combine_rx_halves(group, rf_raw, meta)
        delays = _tx_delays_192(group, meta)

        X, Z = np.meshgrid(xs, z_grid)  # (nz, n_strip)
        M = pymust.dasmtx(iq_shape, X, Z, delays, param)
        bf = (M @ iq192.flatten(order="F")).reshape(X.shape, order="F")

        # raised-cosine lateral apodisation centred on the scan line (smooth compounding)
        w = (0.5 * (1.0 + np.cos(np.pi * (xs - ox) / strip_halfwidth_m))).astype(
            np.float32
        )
        zone_imgs[fz][:, cols] += bf * w[np.newaxis, :]
        zone_wts[fz][:, cols] += w[np.newaxis, :]

    # normalise each focal-zone composite
    for fz in focal_depths:
        wt = zone_wts[fz]
        np.divide(zone_imgs[fz], wt, out=zone_imgs[fz], where=wt > 0)

    return focal_depths, zone_imgs, x_grid, z_grid


def stitch_zones(
    zone_imgs: dict[float, np.ndarray],
    focal_depths: list[float],
    z_grid: np.ndarray,
) -> np.ndarray:
    """
    Axially combine focal-zone composites with a linear cross-fade.

    `focal_depths` may be any non-empty subset of the available zones (sorted by
    depth): shallower than the first zone uses that zone, deeper than the last
    uses that one, and depths in between are linearly blended between neighbours.

    Returns the combined complex image (nz, nx).
    """
    fds = np.array(focal_depths)
    nz = z_grid.size
    n_fz = len(focal_depths)
    zone_stack = np.stack([zone_imgs[fz] for fz in focal_depths])  # (n_fz, nz, nx)
    w = np.zeros((n_fz, nz), dtype=np.float32)

    below = z_grid <= fds[0]
    above = z_grid >= fds[-1]
    w[0, below] = 1.0
    w[-1, above] = 1.0

    mid = ~(below | above)
    j = np.searchsorted(fds, z_grid[mid]) - 1
    t = (z_grid[mid] - fds[j]) / (fds[j + 1] - fds[j])
    np.add.at(w, (j, np.where(mid)[0]), 1.0 - t)
    np.add.at(w, (j + 1, np.where(mid)[0]), t)

    return np.einsum("fz,fzx->zx", w, zone_stack)


def to_db(final: np.ndarray, ref_max: float | None = None) -> np.ndarray:
    """
    Time-gain compensate, log-compress and clip a complex image to [-50, 0] dB.

    Pass `ref_max` (a shared 0 dB reference envelope) to make several images
    brightness-comparable; otherwise each image is normalised to its own peak.
    """
    env = np.abs(final)
    env, _ = pymust.tgc(env)
    env = np.maximum(env, 1e-9)
    ref = env.max() if ref_max is None else ref_max
    log_img = 20.0 * np.log10(env / ref)
    return np.clip(log_img, -50, 0)


def beamform_multifocal(
    meta: dict,
    rf_raw: np.ndarray,
    nx: int = 256,
    nz: int = 384,
    strip_halfwidth_m: float = 1.5e-3,
    fnumber: float = 1.5,
    scanline_step: int = 1,
    max_depth_m: float | None = 25e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct a log-compressed B-mode image with PyMUST.

    Beamforms every focal zone and stitches them into the full multi-focal image.

    Returns (log_img_dB[-50..0], x_grid_m, z_grid_m).
    """
    focal_depths, zone_imgs, x_grid, z_grid = beamform_per_zone(
        meta,
        rf_raw,
        nx=nx,
        nz=nz,
        strip_halfwidth_m=strip_halfwidth_m,
        fnumber=fnumber,
        scanline_step=scanline_step,
        max_depth_m=max_depth_m,
    )
    final = stitch_zones(zone_imgs, focal_depths, z_grid)
    return to_db(final), x_grid, z_grid


# ── visualisation ────────────────────────────────────────────────────────────────


def visualize(
    data_dir: Path,
    out: Path,
    nx: int = 256,
    nz: int = 384,
    frame: int = 0,
    scanline_step: int = 1,
    max_depth_m: float | None = 25e-3,
) -> None:
    all_files = sorted(data_dir.rglob("DATA_*.mat"))
    mat_files = [f for f in all_files if "multifocal" in f.name.lower()]
    if not mat_files:
        raise FileNotFoundError(f"No MultiFocal DATA_*.mat files under {data_dir}")

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
            log_img, x_grid, z_grid = beamform_multifocal(
                meta,
                rf_raw,
                nx=nx,
                nz=nz,
                scanline_step=scanline_step,
                max_depth_m=max_depth_m,
            )
            ax.imshow(
                log_img,
                cmap="gray",
                vmin=-50,
                vmax=0,
                extent=[
                    x_grid[0] * 1e3,
                    x_grid[-1] * 1e3,
                    z_grid[-1] * 1e3,
                    z_grid[0] * 1e3,
                ],
                aspect="equal",
            )
            ax.set_title(f"{fpath.parent.name} — {fpath.stem[-15:]}", fontsize=8)
            ax.set_xlabel("Lateral (mm)", fontsize=7)
            ax.set_ylabel("Depth (mm)", fontsize=7)
        except Exception as exc:
            import traceback

            traceback.print_exc()
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

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "PyMUST DAS Beamformed B-mode (multi-focal, synthetic RX aperture)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n→ {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/rats/"))
    parser.add_argument("--out", type=Path, default=Path("figures/bmode_pymust.png"))
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nz", type=int, default=384)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--scanline-step",
        type=int,
        default=1,
        help="Use every Nth scan line (>1 speeds up at the cost of lateral sampling).",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=25.0,
        help="Reconstruction depth cap in mm (this data has no echo below ~20mm). "
        "Pass 0 to use the full configured depth.",
    )
    args = parser.parse_args()
    max_depth_m = None if args.max_depth_mm == 0 else args.max_depth_mm * 1e-3
    visualize(
        args.data_dir,
        args.out,
        nx=args.nx,
        nz=args.nz,
        frame=args.frame,
        scanline_step=args.scanline_step,
        max_depth_m=max_depth_m,
    )
