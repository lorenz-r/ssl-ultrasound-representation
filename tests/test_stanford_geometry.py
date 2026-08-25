"""Shared-grid geometry for the Stanford transfer corpus (WI-16, PLAN.md §3.6).

Frame-invariant, no IQ data or real store needed — a minimal attrs dict with
the two keys `build_grid_variant` reads stands in for a real
`StanfordFsaStore.attrs`.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_us.data import geometry as geo
from ssl_us.data import stanford_geometry as sgeo

# Real measured values (CLUSTER_HANDOFF.md): 192 elements at 200.0 um pitch,
# so the aperture is (192 - 1) * 0.2 mm = 38.2 mm, half = 19.1 mm.
ATTRS = {
    "elem_x_m": np.linspace(-0.0191, 0.0191, 192).tolist(),
    "wavelength_m": 0.00019710738512735185,  # 197.1 um
}


def test_aperture_half_width_matches_measured_geometry():
    assert sgeo.aperture_half_width_mm(ATTRS) == pytest.approx(19.1, abs=0.01)


def test_mm_matched_spacing_is_nvidias_own_bmode_spacing():
    assert sgeo.MM_MATCHED_SPACING_MM == geo.bmode_dz_mm()


def test_mm_matched_variant_covers_the_documented_range():
    v = sgeo.build_grid_variant("mm_matched", ATTRS)
    assert v.z_mm[0] == pytest.approx(0.0)
    assert v.z_mm[-1] == pytest.approx(19.9)
    assert v.x_mm[0] == pytest.approx(-19.1, abs=0.01)
    assert v.x_mm[-1] == pytest.approx(19.1, abs=0.01)
    # realised spacing close to the mm-matched target (endpoints are exact,
    # spacing is what absorbs the rounding to a whole pixel count)
    assert np.diff(v.z_mm).mean() == pytest.approx(sgeo.MM_MATCHED_SPACING_MM, rel=0.02)


def test_lambda_matched_variant_uses_stanfords_own_wavelength_over_three():
    v = sgeo.build_grid_variant("lambda_matched", ATTRS)
    expected_spacing = ATTRS["wavelength_m"] * 1e3 / 3.0
    assert v.spacing_mm == pytest.approx(expected_spacing)
    assert np.diff(v.z_mm).mean() == pytest.approx(expected_spacing, rel=0.02)


def test_lambda_matched_is_finer_than_mm_matched():
    """Stanford's wavelength (197.1 um) is smaller than NVIDIA's (236.9 um),
    so lambda/3 spacing must be finer than the mm-matched NVIDIA spacing —
    otherwise the whole point of the ablation (PLAN.md 3.6 table) is moot."""
    mm = sgeo.build_grid_variant("mm_matched", ATTRS)
    lam = sgeo.build_grid_variant("lambda_matched", ATTRS)
    assert lam.spacing_mm < mm.spacing_mm
    assert len(lam.z_mm) > len(mm.z_mm)


def test_unknown_variant_name_is_rejected():
    with pytest.raises(ValueError, match="unknown grid variant"):
        sgeo.build_grid_variant("nonexistent", ATTRS)


def test_grid_variants_tuple_lists_both_known_names():
    assert set(sgeo.GRID_VARIANTS) == {"mm_matched", "lambda_matched"}
    for name in sgeo.GRID_VARIANTS:
        sgeo.build_grid_variant(name, ATTRS)  # must not raise
