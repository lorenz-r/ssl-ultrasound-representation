"""Acquisition geometry constants for NV-Raw2Insights-US, all measured, not quoted.

Every value here was read from the dataset itself (all 923 frames for the ones
marked constant). Units are in the names. See PLAN.md 2.3.
"""

from __future__ import annotations

import numpy as np

# ── constant across all 923 frames (verified) ────────────────────────────────
FS_HZ = 13_330_000.000153637
FC_HZ = 6_500_000.0
FD_HZ = 6_500_000.0
C0_M_S = 1540.0
N_ELEMENTS = 180
PITCH_M = 201.0e-6
N_FAST_TIME = 1024

# `t0` takes exactly three values across the corpus (435 / 116 / 372 frames).
T0_LEVELS_S = (-1.7620129710849142e-07, -1.6641233457903581e-07, -1.5662337204958020e-07)

# B-mode / segmentation grid. bmode_extent is [x_min, x_max, z_max, z_min] in mm
# (note the z ordering, and note the unit differs from sound_speed_extent).
BMODE_NZ, BMODE_NX = 507, 456
BMODE_X_MIN_MM, BMODE_X_MAX_MM = -17.989500045776367, 17.989500045776367
BMODE_Z_MIN_MM, BMODE_Z_MAX_MM = 0.0, 40.0

# Sound-speed map grid. sound_speed_extent is in METRES (unit differs from bmode).
SOS_NZ, SOS_NX = 32, 32
SOS_X_MIN_MM, SOS_X_MAX_MM = -17.989499494433403, 17.989499494433403
SOS_Z_MIN_MM, SOS_Z_MAX_MM = -0.2887499867938459, 40.65600037574768

WAVELENGTH_M = C0_M_S / FC_HZ  # 236.9 um


def bmode_dz_mm() -> float:
    return (BMODE_Z_MAX_MM - BMODE_Z_MIN_MM) / (BMODE_NZ - 1)


def bmode_dx_mm() -> float:
    return (BMODE_X_MAX_MM - BMODE_X_MIN_MM) / (BMODE_NX - 1)


def bmode_z_axis_mm() -> np.ndarray:
    return np.linspace(BMODE_Z_MIN_MM, BMODE_Z_MAX_MM, BMODE_NZ)


def bmode_x_axis_mm() -> np.ndarray:
    return np.linspace(BMODE_X_MIN_MM, BMODE_X_MAX_MM, BMODE_NX)


def row_to_z_mm(row: np.ndarray | float) -> np.ndarray | float:
    return BMODE_Z_MIN_MM + np.asarray(row) * bmode_dz_mm()


def col_to_x_mm(col: np.ndarray | float) -> np.ndarray | float:
    return BMODE_X_MIN_MM + np.asarray(col) * bmode_dx_mm()


def z_mm_to_row(z_mm: float) -> int:
    return int(round((z_mm - BMODE_Z_MIN_MM) / bmode_dz_mm()))


def resample_sos_to_bmode_grid(sos_32x32: np.ndarray) -> np.ndarray:
    """Bilinearly resample a (32, 32) sound-speed map onto the (507, 456) B-mode grid.

    The two grids have *different* extents (the SoS map starts 0.29 mm above the
    B-mode's z=0 and ends 0.66 mm deeper), so this is a genuine coordinate
    transform, not a plain resize. Values outside the SoS extent are clamped to
    the edge rather than extrapolated.
    """
    zs = np.linspace(SOS_Z_MIN_MM, SOS_Z_MAX_MM, SOS_NZ)
    xs = np.linspace(SOS_X_MIN_MM, SOS_X_MAX_MM, SOS_NX)
    zt, xt = bmode_z_axis_mm(), bmode_x_axis_mm()

    # fractional source index for each target sample, clamped to the source range
    zi = np.clip(np.interp(zt, zs, np.arange(SOS_NZ)), 0, SOS_NZ - 1)
    xi = np.clip(np.interp(xt, xs, np.arange(SOS_NX)), 0, SOS_NX - 1)

    z0 = np.floor(zi).astype(int)
    z1 = np.minimum(z0 + 1, SOS_NZ - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.minimum(x0 + 1, SOS_NX - 1)
    wz = (zi - z0)[:, None]
    wx = (xi - x0)[None, :]

    top = sos_32x32[np.ix_(z0, x0)] * (1 - wx) + sos_32x32[np.ix_(z0, x1)] * wx
    bot = sos_32x32[np.ix_(z1, x0)] * (1 - wx) + sos_32x32[np.ix_(z1, x1)] * wx
    return (top * (1 - wz) + bot * wz).astype(np.float32)


def t0_group(t0_s: float, tol: float = 1e-12) -> int:
    """Map a frame's ``t0`` onto its categorical level (0, 1, 2)."""
    for i, level in enumerate(T0_LEVELS_S):
        if abs(t0_s - level) < tol:
            return i
    raise ValueError(f"t0={t0_s!r} matches none of the three measured levels")
