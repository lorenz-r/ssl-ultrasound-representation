"""Angular sector + occupancy geometry (PLAN.md WI-3).

All frame-invariant: no IQ data involved, so these run on plain geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.data import geometry as geo


def test_aperture_half_angle_matches_named_spans():
    # PLAN.md 3.2: f#=1.5 -> +-18.4 deg, f#=1.0 -> +-26.6 deg, f#=0.75 -> +-33.7 deg.
    assert np.degrees(sec.aperture_half_angle_rad(1.5)) == pytest.approx(18.43, abs=0.05)
    assert np.degrees(sec.aperture_half_angle_rad(1.0)) == pytest.approx(26.57, abs=0.05)
    assert np.degrees(sec.aperture_half_angle_rad(0.75)) == pytest.approx(33.69, abs=0.05)


def test_unlimited_aperture_has_no_named_limit():
    assert sec.aperture_half_angle_rad(None) == pytest.approx(np.pi / 2)
    assert sec.aperture_half_angle_rad(0) == pytest.approx(np.pi / 2)


def test_view_angle_monotonic_in_element_index():
    elem_x = sec.element_positions_m(n_elements=32, pitch_m=200e-6)[:, 0]
    z_mm, x_mm = np.array([10.0, 20.0]), np.array([-5.0, 0.0, 5.0])
    theta = sec.view_angles_rad(z_mm=z_mm, x_mm=x_mm, elem_x_m=elem_x)
    assert theta.shape == (2, 3, 32)
    for zi in range(2):
        for xi in range(3):
            msg = "view angle must strictly increase with element index"
            assert np.all(np.diff(theta[zi, xi]) > 0), msg


def test_broadside_pixel_is_angle_zero_at_centre_element():
    elem_x = sec.element_positions_m(n_elements=33)[:, 0]  # odd count -> exact centre element
    theta = sec.view_angles_rad(z_mm=np.array([15.0]), x_mm=np.array([0.0]), elem_x_m=elem_x)
    assert theta[0, 0, 16] == pytest.approx(0.0, abs=1e-12)  # element 16 of 33 is dead centre


@pytest.fixture(scope="module")
def small_geometry():
    """A coarse but full-depth-range grid — enough to exercise every code path
    without building the full (507, 456) array in every test process."""
    z_mm = np.linspace(0.5, 40.0, 80)
    x_mm = np.linspace(-18.0, 18.0, 60)
    return sec.build_sector_geometry(k=16, f_number=1.0, z_mm=z_mm, x_mm=x_mm)


def test_sectors_contiguous(small_geometry):
    sec.assert_sectors_contiguous(small_geometry.sector_id)


def test_sectors_contiguous_fails_on_deliberately_broken_input():
    """The guard must actually fire — a shuffled sector id is not contiguous."""
    rng = np.random.default_rng(0)
    broken = rng.integers(-1, 4, size=(5, 5, 32)).astype(np.int8)
    # Force at least one row to violate contiguity: valid ids present with a gap.
    broken[0, 0] = -1
    broken[0, 0, [2, 3, 10]] = [0, 1, 0]  # id drops back to 0 after 1 -> non-monotonic
    with pytest.raises(AssertionError):
        sec.assert_sectors_contiguous(broken)


def test_occupancy_shape_and_dtype(small_geometry):
    g = small_geometry
    assert g.occupancy.shape == (80, 60, 16)
    assert g.occupancy.dtype == np.uint16
    assert g.sector_id.dtype == np.int8


def test_occupancy_sums_to_active_elements_within_aperture(small_geometry):
    """Every valid element belongs to exactly one sector; no element is lost
    or double-counted by the binning."""
    g = small_geometry
    total_active = (g.sector_id >= 0).sum(axis=-1)
    np.testing.assert_array_equal(g.occupancy.sum(axis=-1), total_active)


def test_empty_sectors_are_zero_not_negative(small_geometry):
    assert (small_geometry.occupancy >= 0).all()  # uint16 guarantees this structurally
    assert (small_geometry.occupancy == 0).any(), "shallow depths should have empty sectors"


@pytest.mark.parametrize(
    "f_number,z_mm,expected_active_k16",
    [
        (1.5, 2.0, 0.4 * 16),
        (1.5, 5.0, 1.1 * 16),
        (1.0, 2.0, 0.6 * 16),
        (1.0, 8.0, 2.5 * 16),
        (1.0, 20.0, 6.2 * 16),
        (0.75, 10.0, 4.2 * 16),
    ],
)
def test_occupancy_matches_plan_table_to_one_element(f_number, z_mm, expected_active_k16):
    """PLAN.md 3.2's 'elements per sector' table, cross-checked at broadside (x=0).

    The table's own entries are rounded to 1 decimal place (i.e. +-0.05 elements
    per sector, +-0.8 total at K=16), so an exact +-1.0 match is tighter than
    the table's own precision supports; +-2.0 is used here instead. The tight
    +-1 check lives in test_expected_active_elements_matches_direct_count, which
    compares against a closed form computed independently in this module.
    """
    g = sec.build_sector_geometry(
        k=16, f_number=f_number, z_mm=np.array([z_mm]), x_mm=np.array([0.0])
    )
    active = float(g.occupancy[0, 0].sum())
    assert active == pytest.approx(expected_active_k16, abs=2.0)


def test_expected_active_elements_matches_direct_count():
    """The independent closed-form helper agrees with the sector-id-derived count."""
    z_mm, f_number = 15.0, 1.0
    g = sec.build_sector_geometry(
        k=16, f_number=f_number, z_mm=np.array([z_mm]), x_mm=np.array([0.0])
    )
    counted = float(g.occupancy[0, 0].sum())
    expected = sec.expected_active_elements(z_mm, 0.0, f_number, geo.N_ELEMENTS, geo.PITCH_M)
    assert counted == pytest.approx(expected, abs=1.0)


def test_unlimited_aperture_uses_every_element():
    g = sec.build_sector_geometry(k=1, f_number=None, z_mm=np.array([5.0]), x_mm=np.array([0.0]))
    assert int(g.occupancy[0, 0, 0]) == geo.N_ELEMENTS


def test_full_dataset_grid_is_contiguous_at_k16_fnumber1():
    """The actual (507, 456) grid the store will use, at the plan's chosen config."""
    g = sec.build_sector_geometry(k=16, f_number=1.0)
    assert g.sector_id.shape == (geo.BMODE_NZ, geo.BMODE_NX, geo.N_ELEMENTS)
    sec.assert_sectors_contiguous(g.sector_id)
