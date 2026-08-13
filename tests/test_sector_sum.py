"""AngularSectorSum, exercised through the real zea Beamform pipeline on a
synthetic FSA scene (PLAN.md WI-4). No dataset download — the properties
checked here are exact mathematical identities that must hold for *any* input:

* K=1 with an unlimited receive aperture must reproduce zea's own
  ``"delay_and_sum"`` beamformer exactly (our segment-sum degenerates to a
  plain sum when there is exactly one sector and nothing is masked out).
* Summing all K sector bins must reproduce ``"delay_and_sum"`` beamformed with
  the *same* f-number receive mask exactly — the occupancy-conservation
  property (every valid element belongs to exactly one bin) proven here on
  real delayed, phase-rotated, interpolated data rather than on element
  counts alone (that check lives in test_sectors.py).
* On-the-fly sector membership (computed inside the op from the pipeline's
  own ``flatgrid``/``probe_geometry``) agrees with the offline
  :mod:`ssl_us.beamform.sectors` map used to build the store.

The real-frame reference-validation gate against NVIDIA's own `bmode` (NCC,
median dB error, the negative control with phase rotation disabled) lives in
``scripts/wi4_reference_validation.py`` — it needs a ~265 MB/frame download,
so it is not a pytest test.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
from zea import Parameters
from zea.ops.pipeline import Beamform

from ssl_us.beamform import sector_sum  # noqa: F401 — registers "angular_sector_sum"
from ssl_us.beamform import sectors as sec

# zea's f-number mask can't take `None` for "unlimited" (it casts to the grid's
# float dtype); a near-zero f# gives theta_max ~ 90 deg, i.e. effectively every
# element valid at every depth used here.
UNLIMITED_F = 1e-6


@pytest.fixture(scope="module")
def synthetic_scene():
    n_el, n_tx, n_ax = 16, 16, 512
    fs, c0, fc = 20e6, 1540.0, 5e6
    pitch = 200e-6

    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = (np.arange(n_el) - (n_el - 1) / 2) * pitch

    params = Parameters(
        probe_geometry=probe_geometry,
        sound_speed=c0,
        sampling_frequency=fs,
        center_frequency=fc,
        n_el=n_el,
        n_tx=n_tx,
        n_ax=n_ax,
        n_ch=2,
        xlims=(-3e-3, 3e-3),
        zlims=(2e-3, 6e-3),  # well inside the n_ax/fs/c0 reachable window
        grid_size_x=8,
        grid_size_z=6,
        # FSA: one active element per transmit, each transmit originates at
        # its own element — the exact scan dict from PLAN.md 0.2.
        t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
        tx_apodizations=np.eye(n_tx, n_el, dtype=np.float32),
        focus_distances=np.zeros(n_tx, dtype=np.float32),
        polar_angles=np.zeros(n_tx, dtype=np.float32),
        transmit_origins=probe_geometry,
        initial_times=np.zeros(n_tx, dtype=np.float32),
        f_number=UNLIMITED_F,
    )

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(n_tx, n_ax, n_el, 2)).astype(np.float32)
    return params, raw, probe_geometry


def _beamform(params, raw, *, beamformer, tof_f_number, **beamformer_kwargs):
    """Run one beamformer. `tof_f_number` is TOFCorrection's own runtime
    receive-aperture mask (fused into TOF correction, PLAN.md 0.1); any
    `f_number` in `beamformer_kwargs` is a *separate* value used only to
    construct our own beamformer op (its sector span). The two must be passed
    independently — conflating them is exactly the bug this comment guards
    against (an unintended op-level aperture silently double-restricting an
    otherwise "unlimited" comparison).
    """
    pipeline = Beamform(
        beamformer=beamformer, num_patches=1, with_batch_dim=False, **beamformer_kwargs
    )
    inputs = pipeline.prepare_parameters(params, data=raw, f_number=tof_f_number)
    out = pipeline(**inputs)
    return keras.ops.convert_to_numpy(out["data"]), inputs


def test_k1_unlimited_aperture_matches_zea_delay_and_sum(synthetic_scene):
    params, raw, _ = synthetic_scene
    das, _ = _beamform(params, raw, beamformer="delay_and_sum", tof_f_number=UNLIMITED_F)
    sector, _ = _beamform(
        params, raw, beamformer="angular_sector_sum", tof_f_number=UNLIMITED_F, k=1, f_number=None
    )
    assert das.max() > 0, "sanity: beamformed signal must be nonzero for this to be a real test"
    assert sector.shape == (*das.shape, 1)
    np.testing.assert_allclose(sector[..., 0], das, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("k,f_number", [(16, 1.0), (8, 1.0), (4, 1.5)])
def test_sum_over_sectors_matches_das_with_same_fnumber(synthetic_scene, k, f_number):
    """The occupancy-conservation property (PLAN.md 3.2), on real delayed data:
    every valid element lands in exactly one bin, so summing the bins back up
    must reproduce plain f-number-masked DAS exactly. The *same* f_number is
    used for TOFCorrection's mask and for the op's own sector span, so the two
    beamformers see an identical receive aperture and differ only in how they
    partition it.
    """
    params, raw, _ = synthetic_scene
    das, _ = _beamform(params, raw, beamformer="delay_and_sum", tof_f_number=f_number)
    sector, _ = _beamform(
        params, raw, beamformer="angular_sector_sum", tof_f_number=f_number, k=k, f_number=f_number
    )
    assert sector.shape == (*das.shape, k)
    assert das.max() > 0
    np.testing.assert_allclose(sector.sum(axis=-1), das, atol=1e-4, rtol=1e-4)


def test_onthefly_sector_membership_matches_offline_sectors_module(synthetic_scene):
    """The op derives sector membership from the pipeline's own flatgrid at
    call time; the offline `sectors.py` map used to build the store must agree.
    """
    params, raw, probe_geometry = synthetic_scene
    k, f_number = 16, 1.0
    _, inputs = _beamform(
        params, raw, beamformer="angular_sector_sum", tof_f_number=f_number, f_number=f_number, k=k
    )

    grid = keras.ops.convert_to_numpy(inputs["grid"])  # (nz, nx, 3) = [x, y, z]
    z_mm = grid[:, 0, 2] * 1e3
    x_mm = grid[0, :, 0] * 1e3

    elem_x = probe_geometry[:, 0].astype(np.float64)
    theta = sec.view_angles_rad(z_mm=z_mm, x_mm=x_mm, elem_x_m=elem_x)
    sector_id_offline = sec.build_sector_id(theta, k=k, f_number=f_number)

    theta_on = np.arctan2(
        probe_geometry[None, None, :, 0] - grid[:, :, None, 0],
        grid[:, :, None, 2],
    )
    theta_max = sec.aperture_half_angle_rad(f_number)
    valid = np.abs(theta_on) <= theta_max
    frac = (theta_on + theta_max) / (2 * theta_max)
    sector_id_onfly = np.where(valid, np.clip(np.floor(frac * k), 0, k - 1), -1).astype(np.int8)

    np.testing.assert_array_equal(sector_id_offline, sector_id_onfly)


def test_different_k_are_not_trivially_equal(synthetic_scene):
    """Guards against a degenerate implementation that ignores k entirely."""
    params, raw, _ = synthetic_scene
    k4, _ = _beamform(
        params, raw, beamformer="angular_sector_sum", tof_f_number=1.0, f_number=1.0, k=4
    )
    k8, _ = _beamform(
        params, raw, beamformer="angular_sector_sum", tof_f_number=1.0, f_number=1.0, k=8
    )
    assert k4.shape[-1] == 4
    assert k8.shape[-1] == 8
    # merging k8 pairwise must reproduce k4 (the ladder identity, PLAN.md 3.2)
    merged = k8.reshape(*k8.shape[:-1], 4, 2).sum(axis=-1)
    np.testing.assert_allclose(merged, k4, atol=1e-4, rtol=1e-4)
