"""
Utility functions for loading raw Verasonics ultrasound data from HDF5 (.mat) files.

Provides low-level helpers for dereferencing and scalar/array extraction,
as well as higher-level loaders for metadata and RF frame data.
"""

from __future__ import annotations
import gc
from pathlib import Path

import h5py
import numpy as np


def scalar(f: h5py.File, ds) -> float:
    """Extract a scalar value from an HDF5 dataset or reference."""
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        v = f[v.flat[0]][()]
    return float(np.asarray(v).flat[0])


def arr(f: h5py.File, ds) -> np.ndarray:
    """Extract a numpy array from an HDF5 dataset or reference."""
    v = ds[()] if isinstance(ds, h5py.Dataset) else np.asarray(ds)
    if v.dtype == object:
        return np.asarray(f[v.flat[0]][()])
    return np.asarray(v)


def deref_col(f: h5py.File, refs_ds, idx: int) -> np.ndarray:
    """Dereference one object-reference entry from a (N, 1) dataset."""
    refs = refs_ds[()]
    return np.asarray(f[refs[idx, 0]][()])


def load_metadata(filepath: str | Path) -> dict:
    """
    Load acquisition metadata from a Verasonics .mat (HDF5) file.

    Returns a dict with keys: c, freq, lam, elem_x, fs, start_depth,
    end_depth, stride, active_samples, n_acq, tx_events.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        c    = scalar(f, f["Resource"]["Parameters"]["speedOfSound"])
        freq = scalar(f, f["Trans"]["frequency"]) * 1e6
        lam  = c / freq

        elem_x = f["Trans"]["ElementPos"][()][0, :] * lam   # (N_elem,) metres

        adc_rate = scalar(f, f[f["Receive"]["ADCRate"][()][0, 0]]) * 1e6
        decim    = scalar(f, f[f["Receive"]["decimFactor"][()][0, 0]])
        fs       = adc_rate / decim

        start_depth = scalar(f, f["P"]["startDepth"])   # wavelengths
        end_depth   = scalar(f, f["P"]["endDepth"])



        # 1. TIME DOMAIN: Calculate Stride vs Active Samples
        s_refs  = f["Receive"]["startSample"][()]
        e_refs  = f["Receive"]["endSample"][()]

        s0 = int(scalar(f, f[s_refs[0, 0]]))
        s1 = int(scalar(f, f[s_refs[1, 0]]))
        e0 = int(scalar(f, f[e_refs[0, 0]]))

        # Stride: The total memory block size allocated per acquisition
        stride = s1 - s0
        
        # Active: The actual usable acoustic data 
        # (MATLAB indices are inclusive, so we add 1)
        active_samples = e0 - s0 + 1

        # Count acquisitions in frame 1
        fr_refs = f["Receive"]["framenum"][()]
        n_acq   = sum(
            1 for i in range(len(fr_refs))
            if int(scalar(f, f[fr_refs[i, 0]])) == 1
        )


        num_rays = int(scalar(f, f["P"]["numRays"]))
        
        # Safely extract txFoci, or infer it mathematically if the file omitted it
        if "txFoci" in f["P"]:
            tx_foci = arr(f, f["P"]["txFoci"])
            num_foci = tx_foci.size 
        else:
            # Fallback: Total acquisitions divided by total scanlines
            num_foci = n_acq // num_rays

        # 2. SPATIAL DOMAIN: Calculate Active Elements
        apod_refs  = f["TX"]["Apod"][()]
        orig_refs  = f["TX"]["Origin"][()]
        focus_refs = f["TX"]["focus"][()]
        tx_events  = []
        
        n_tx_structs = len(apod_refs)

        # Loop over total acquisitions, safely handling if Verasonics 
        # condensed a static sequence into a single TX struct.
        for i in range(n_acq):
            tx_idx = min(i, n_tx_structs - 1)
            
            ap = deref_col(f, apod_refs, tx_idx).flatten()
            
            # Use != 0 to safely catch both positive and negative apodization weights
            active_elements = np.where(ap != 0)[0] 
            
            orig_x  = float(deref_col(f, orig_refs, tx_idx).flatten()[0]) * lam
            foc_wav = float(deref_col(f, focus_refs, tx_idx).flatten()[0])
            
            tx_events.append({
                "active_elements": active_elements,
                "origin_x": orig_x,
                "focus_m": foc_wav * lam,
            })

  
    return {
        "c": c, "freq": freq, "lam": lam, "elem_x": elem_x,
        "fs": fs, "start_depth": start_depth, "end_depth": end_depth,
        "stride": stride, "active_samples": active_samples, 
        "n_acq": n_acq, "tx_events": tx_events, "num_rays": num_rays,
        "num_foci": num_foci,
    }
  


def load_rf_frame(filepath: str | Path, frame: int = 0) -> np.ndarray:
    """Return (n_rx, total_samples) float32 RF data for the requested frame."""
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        rcv_ref = f["RcvData"][()].flat[0]
        rcv     = f[rcv_ref]
        return rcv[frame, :, :].astype(np.float32)


def load_3d_rf_frame(filepath: str | Path, frame: int = 0) -> np.ndarray:
    """
    Load and reconstruct a 3D RF data array for a specific frame.
    """
    filepath = Path(filepath) 

    rf = load_rf_frame(filepath, frame)
    meta = load_metadata(filepath)

    n_acq  = meta["n_acq"]
    stride = meta["stride"]
    active = meta["active_samples"]
    n_rx   = rf.shape[0]

    # --- THE FIX ---
    # Calculate the exact number of padded samples needed for all acquisitions
    total_frame_samples = n_acq * stride
    
    # Slice off the trailing dead space from the Verasonics memory buffer
    rf_exact = rf[:, :total_frame_samples]

    # STEP 1: RESHAPE USING STRIDE
    # Shape becomes: (n_rx, n_acq, stride)
    rf_reshaped = rf_exact.reshape(n_rx, n_acq, stride)

    # STEP 2: TRANSPOSE
    # Shape becomes: (n_acq, n_rx, stride)
    rf_3d = np.transpose(rf_reshaped, (1, 0, 2))

    # STEP 3: SLICE USING ACTIVE SAMPLES
    # Discard the hardware padding at the end of the time dimension.
    rf_final = rf_3d[:, :, :active].copy()

    # Clean up large intermediate arrays from RAM
    del rf, rf_exact, rf_reshaped, rf_3d
    gc.collect()

    return rf_final


def load_multifocal_rf_frame(filepath: str | Path, frame: int = 0) -> np.ndarray:
    """
    Load and reconstruct a multifocal RF data array for a specific frame.
    Dynamically adapts to standard multifocal (4D) or multi-transmit (5D) sequences.
    """
    filepath = Path(filepath) 

    # 1. Load the flat 2D frame (n_rx, total_samples) and the metadata
    rf = load_rf_frame(filepath, frame)
    meta = load_metadata(filepath)

    n_rx     = rf.shape[0]
    n_acq    = meta["n_acq"]
    stride   = meta["stride"]
    active   = meta["active_samples"]
    num_rays = meta["num_rays"]
    num_foci = meta["num_foci"]

    # --- MEMORY ALIGNMENT ---
    # Calculate the exact number of padded samples needed for all acquisitions
    total_frame_samples = n_acq * stride
    
    # Slice off the trailing dead space from the Verasonics memory buffer
    rf_exact = rf[:, :total_frame_samples]

    # --- 3D RECONSTRUCTION ---
    # Reshape using stride to respect hardware memory blocks
    # Shape becomes: (n_rx, n_acq, stride)
    rf_reshaped = rf_exact.reshape(n_rx, n_acq, stride)

    # Transpose to standard order
    # Shape becomes: (n_acq, n_rx, stride)
    rf_3d = np.transpose(rf_reshaped, (1, 0, 2))

    # Slice off the hardware padding using active samples
    # Shape becomes: (n_acq, n_rx, active)
    rf_sliced = rf_3d[:, :, :active].copy()

    # --- DYNAMIC MULTIFOCAL SEPARATION ---
    # Detect if there are multiple transmits per focal zone (e.g. beam pairs)
    tx_per_focus = n_acq // (num_rays * num_foci)

    if tx_per_focus == 1:
        # Standard 4D Sequence
        # Shape: (Scanlines, Focal_Zones, Channels, Time)
        rf_nd = rf_sliced.reshape(num_rays, num_foci, n_rx, active)
    else:
        # Multi-Transmit / Beam-Pair 5D Sequence
        # Shape: (Scanlines, Focal_Zones, Transmits, Channels, Time)
        rf_nd = rf_sliced.reshape(num_rays, num_foci, tx_per_focus, n_rx, active)

    # Clean up large intermediate arrays from RAM
    del rf, rf_exact, rf_reshaped, rf_3d, rf_sliced
    gc.collect()

    return rf_nd