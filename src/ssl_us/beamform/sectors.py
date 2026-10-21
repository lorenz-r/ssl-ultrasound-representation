"""Angular receive-sector geometry (PLAN.md WI-3 / §3.2).

Sub-apertures are ANGULAR SECTORS, not element groups: for each output pixel,
each receive element's view angle to that pixel is binned into K uniform
sectors spanning ±theta_max, where theta_max is set by a receive f-number
(theta_max = atan(1 / (2 f#)) — the standard f-number half-angle). Angular
sectors mean the same thing on any array, unlike element index, which is why
this — not element grouping — is the quantity a model can consume without its
weights silently changing meaning between probes (PLAN.md 3.2, 3.6).

For a linear array, view angle at a fixed pixel is *monotonic* in element
index (elements are ordered by increasing x, and arctan2(dx, z) is monotonic
in dx for z > 0), so the valid aperture and every sector within it are
contiguous ranges of element index. This is what makes the beamformer's inner
loop (WI-4) a plain scatter-add rather than a K-fold search.

Everything here is **frame-invariant** — it depends only on array geometry,
the output grid, K, and the f-number, never on the IQ data — so it is computed
once and reused for every frame (PLAN.md 3.2: 41.6 MB sector map + 7.4 MB
occupancy map, for the whole dataset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import geometry as geo


def element_positions_m(
    n_elements: int = geo.N_ELEMENTS, pitch_m: float = geo.PITCH_M
) -> np.ndarray:
    """(n_el, 3) element positions ``[x, y, z]``, centred at x=0, on y=0, z=0.

    Matches the measured geometry (PLAN.md 2.3): all elements have y=z=0, and
    ``zea``'s own ``probe_geometry`` convention is ``(n_el, 3)`` in the same
    ``[x, y, z]`` order, so this array is directly usable as one.
    """
    idx = np.arange(n_elements, dtype=np.float64)
    x = (idx - (n_elements - 1) / 2) * pitch_m
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def aperture_half_angle_rad(f_number: float | None) -> float:
    """theta_max = atan(1 / (2 f#)) — the standard f-number half-angle.

    ``f_number`` of ``None`` or ``0`` means an unlimited receive aperture
    (every element valid at every depth): returned as ``pi/2``.

    Verified against PLAN.md 3.2's three named spans: f#=1.5 -> 18.43° (theirs:
    18.4°), f#=1.0 -> 26.57° (26.6°), f#=0.75 -> 33.69° (33.7°).
    """
    if not f_number:
        return np.pi / 2
    return float(np.arctan(1.0 / (2.0 * f_number)))


def view_angles_rad(
    z_mm: np.ndarray, x_mm: np.ndarray, elem_x_m: np.ndarray
) -> np.ndarray:
    """View angle from every (z, x) pixel to every element, shape (nz, nx, n_el).

    Angle is measured from the array normal (broadside, +z): 0 for an element
    directly beneath the pixel, increasing towards +x. Monotonic in element
    index at fixed pixel — the property the whole module exploits.
    """
    z_m = np.asarray(z_mm, dtype=np.float64) * 1e-3
    x_m = np.asarray(x_mm, dtype=np.float64) * 1e-3
    dx = elem_x_m[None, None, :] - x_m[None, :, None]  # (1, nx, n_el)
    return np.arctan2(dx, z_m[:, None, None])  # (nz, nx, n_el), broadcasts to (nz, nx, n_el)


def build_sector_id(theta: np.ndarray, k: int, f_number: float | None) -> np.ndarray:
    """Bin view angles into K uniform sectors spanning [-theta_max, theta_max].

    Returns int8 array, same shape as ``theta``: sector index in [0, K) for
    elements inside the aperture, -1 for elements outside it (excluded, not
    merely down-weighted — PLAN.md 3.2 stores raw sums over the *valid*
    elements only, with the count itself carried in the occupancy map).
    """
    theta_max = aperture_half_angle_rad(f_number)
    valid = np.abs(theta) <= theta_max
    frac = (theta + theta_max) / (2 * theta_max)
    idx = np.clip(np.floor(frac * k), 0, k - 1).astype(np.int8)
    return np.where(valid, idx, np.int8(-1))


def build_occupancy(sector_id: np.ndarray, k: int) -> np.ndarray:
    """Count of valid elements per sector, shape (..., K), uint16."""
    occ = np.zeros((*sector_id.shape[:-1], k), dtype=np.uint16)
    for b in range(k):
        occ[..., b] = (sector_id == b).sum(axis=-1)
    return occ


def assert_sectors_contiguous(sector_id: np.ndarray) -> None:
    """Assert every pixel's valid elements form one contiguous run in element
    index, and sector id is non-decreasing across it (WI-3 done-condition).

    This is the property the AngularSectorSum beamformer (WI-4) depends on to
    do a scatter-add instead of a K-way search: if it ever fails, angle is no
    longer monotonic in element index for this geometry/grid and the fast path
    is unsound.
    """
    flat = sector_id.reshape(-1, sector_id.shape[-1])
    for row in flat:
        valid_idx = np.flatnonzero(row >= 0)
        if len(valid_idx) == 0:
            continue
        lo, hi = valid_idx[0], valid_idx[-1]
        if not np.array_equal(valid_idx, np.arange(lo, hi + 1)):
            raise AssertionError("valid elements are not a contiguous index range")
        vals = row[lo : hi + 1]
        if np.any(np.diff(vals.astype(np.int16)) < 0):
            raise AssertionError("sector id is not monotonically non-decreasing")


def expected_active_elements(
    z_mm: float, x_mm: float, f_number: float | None, n_elements: int, pitch_m: float
) -> float:
    """Closed-form active-element count at one pixel, independent of the
    view-angle/binning code above — used to cross-check it (PLAN.md 3.2 table).

    Elements within the f-number aperture satisfy |x_e - x_p| <= z * tan(theta_max);
    counting lattice points in that interval gives the continuous approximation
    ``2 z tan(theta_max) / pitch`` used in the plan's own table, plus the centre
    element. Clamped to the physical array width.
    """
    theta_max = aperture_half_angle_rad(f_number)
    half_width_m = (z_mm * 1e-3) * np.tan(theta_max)
    n = 2 * half_width_m / pitch_m
    return float(min(n, n_elements))


@dataclass(frozen=True)
class SectorGeometry:
    k: int
    f_number: float | None
    theta_max_rad: float
    z_mm: np.ndarray  # (nz,)
    x_mm: np.ndarray  # (nx,)
    sector_id: np.ndarray  # (nz, nx, n_el) int8
    occupancy: np.ndarray  # (nz, nx, k) uint16


def build_sector_geometry(
    k: int,
    f_number: float | None,
    z_mm: np.ndarray | None = None,
    x_mm: np.ndarray | None = None,
    n_elements: int = geo.N_ELEMENTS,
    pitch_m: float = geo.PITCH_M,
) -> SectorGeometry:
    """The full frame-invariant sector + occupancy geometry for one (K, f#) config.

    Defaults to the dataset's own (507, 456) B-mode grid (PLAN.md 3.3); callers
    that only need a few rows (tests, the occupancy-vs-depth figure) can pass a
    smaller ``z_mm``/``x_mm``.
    """
    if z_mm is None:
        z_mm = geo.bmode_z_axis_mm()
    if x_mm is None:
        x_mm = geo.bmode_x_axis_mm()
    elem_x = element_positions_m(n_elements, pitch_m)[:, 0]
    theta = view_angles_rad(z_mm, x_mm, elem_x)
    sector_id = build_sector_id(theta, k, f_number)
    occupancy = build_occupancy(sector_id, k)
    return SectorGeometry(
        k=k,
        f_number=f_number,
        theta_max_rad=aperture_half_angle_rad(f_number),
        z_mm=np.asarray(z_mm),
        x_mm=np.asarray(x_mm),
        sector_id=sector_id,
        occupancy=occupancy,
    )
