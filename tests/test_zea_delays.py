"""zea's delay round-trip, verified once against a hand-computed value
(PLAN.md WI-3 done-condition: "we trust it, but we check it once").

Not a network test — this only exercises zea's own geometry maths against a
pixel grid and probe geometry we construct ourselves, using the same
coordinate frame as `ssl_us.beamform.sectors` (PLAN.md 2.3: elements on
y=0, z=0). If this ever drifts from a plain Euclidean distance, every
downstream delay (WI-4's AngularSectorSum, WI-16's Stanford beamforming)
would be silently wrong.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.data import geometry as geo


# zea's torch backend moves everything through `convert_to_tensor`, which on
# this machine lands on MPS — and MPS has no float64 support. numpy defaults
# to float64, so every array handed to a zea function here is cast to
# float32 first; tolerances below are float32-appropriate, not float64.
def test_receive_distance_matches_hand_computed_euclidean_distance():
    from zea.beamform.beamformer import compute_receive_distances
    from zea.beamform.pixelgrid import cartesian_pixel_grid

    grid = cartesian_pixel_grid(
        xlims=(-10e-3, 10e-3), zlims=(5e-3, 30e-3), grid_size_x=9, grid_size_z=11
    )  # (nz, nx, 3), [x, y, z]
    flat_grid = grid.reshape(-1, 3).astype(np.float32)
    probe_geometry = sec.element_positions_m(n_elements=32, pitch_m=200e-6).astype(np.float32)

    dist = keras.ops.convert_to_numpy(compute_receive_distances(flat_grid, probe_geometry))

    hand = np.linalg.norm(flat_grid[:, None, :] - probe_geometry[None, :, :], axis=-1)

    assert dist.shape == (flat_grid.shape[0], 32)
    np.testing.assert_allclose(dist, hand, rtol=1e-5, atol=1e-7)


def test_receive_distance_zero_for_element_directly_below_pixel():
    """A pixel at z=0 sitting exactly on an element must have distance 0 to it."""
    from zea.beamform.beamformer import compute_receive_distances

    probe_geometry = sec.element_positions_m(n_elements=8, pitch_m=200e-6).astype(np.float32)
    elem3 = probe_geometry[3]
    grid = elem3[None, :]  # one pixel, coincident with element 3

    dist = keras.ops.convert_to_numpy(compute_receive_distances(grid, probe_geometry))
    assert dist[0, 3] == pytest.approx(0.0, abs=1e-7)
    assert np.all(dist[0, np.arange(8) != 3] > 0.0)


def test_receive_distance_agrees_with_view_angle_geometry_on_the_real_grid():
    """The distance used for delays and the angle used for sector binning must
    describe the same right-triangle geometry: at a fixed pixel,
    tan(theta_e) = dx_e / z, independently derived from `compute_receive_distances`
    (hypotenuse + z) rather than from `sectors.view_angles_rad` itself.
    """
    from zea.beamform.beamformer import compute_receive_distances

    z_mm, x_mm = 15.0, 0.0
    probe_geometry = sec.element_positions_m(
        n_elements=geo.N_ELEMENTS, pitch_m=geo.PITCH_M
    ).astype(np.float32)
    grid = np.array([[x_mm * 1e-3, 0.0, z_mm * 1e-3]], dtype=np.float32)

    dist = keras.ops.convert_to_numpy(compute_receive_distances(grid, probe_geometry))[0]
    dx = probe_geometry[:, 0] - x_mm * 1e-3
    theta_from_distance = np.arcsin(np.clip(dx / dist, -1.0, 1.0))

    theta_direct = sec.view_angles_rad(
        z_mm=np.array([z_mm]),
        x_mm=np.array([x_mm]),
        elem_x_m=probe_geometry[:, 0].astype(np.float64),
    )[0, 0]

    np.testing.assert_allclose(theta_from_distance, theta_direct, rtol=1e-4, atol=1e-5)
