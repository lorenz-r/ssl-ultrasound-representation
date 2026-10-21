"""Per-frame Stanford FSA beamforming (WI-16), through the real zea Beamform
pipeline on a synthetic scene — no download, no real store needed, same style
as `test_sector_sum.py`'s NVIDIA-side coverage.

What is specific to Stanford and therefore worth testing here (as opposed to
being covered by `test_sector_sum.py`'s generic op-level checks): the
``(n_tx, n_rx, n_ax)`` -> ``(n_tx, n_ax, n_el, 2)`` transpose, the
depth-negative near-field mute, and that ``t0_s`` (negative) actually reaches
`Parameters.initial_times` unmodified rather than being forced to 0 as
NVIDIA's path does.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from ssl_us.beamform import stanford_frame as sf
from ssl_us.data import stanford_geometry as sgeo
from ssl_us.data.stanford_fsa import StanfordFsaStore


def _synthetic_store(tmp_path, n_elem=16, n_tx=16, n_ax=64, t0_s=-1.0e-6):
    """A minimal in-memory-shaped StanfordFsaStore — real attrs keys, tiny
    array. Bypasses `open_fsa_store`'s manifest check since no build ran."""
    root = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    rng = np.random.default_rng(0)
    iq = (rng.normal(size=(1, n_tx, n_elem, n_ax)) + 1j * rng.normal(size=(1, n_tx, n_elem, n_ax)))
    root.create_dataset("iq", data=iq.astype(np.complex64))
    root.create_dataset("coverage", data=np.ones((1, n_tx, n_elem), dtype=np.uint8))
    pitch_m = 200e-6
    elem_x = (np.arange(n_elem) - (n_elem - 1) / 2) * pitch_m
    attrs = {
        "c_m_s": 1540.0,
        "fc_hz": 7_813_000.0,
        "fd_hz": 7_813_000.0,
        "fs_hz": 10_416_666.67,
        "wavelength_m": 1540.0 / 7_813_000.0,
        "pitch_m": pitch_m,
        "n_elem": n_elem,
        "elem_x_m": elem_x.tolist(),
        "t0_s": t0_s,
    }
    root.attrs.update(attrs)
    frame_metadata = {
        "0": {"session": "s0", "source_file": "f.mat", "kind": "FSA", "date": "20260101"}
    }
    return StanfordFsaStore(
        path=tmp_path / "s.zarr", root=root, attrs=attrs, frame_metadata=frame_metadata
    )


@pytest.fixture
def store(tmp_path):
    return _synthetic_store(tmp_path)


@pytest.fixture
def small_grid(store):
    # A handful of pixels, well inside the n_ax/fs/c0 reachable window
    # (t0_s=-1.0us, n_ax=64, fs~10.4 MHz => two-way time in
    # [-1.0, 5.05] us => depth in [-0.77, 3.89] mm) and entirely at positive
    # depth, since the near-field-mute test below is what checks the muted
    # (negative-depth) samples, not these pixels.
    z_mm = np.linspace(0.5, 3.0, 5)
    x_mm = np.linspace(-2.0, 2.0, 5)
    return sgeo.GridVariant(name="test", z_mm=z_mm, x_mm=x_mm, spacing_mm=1.0)


def test_load_raw_iq_shape_and_axis_order(store):
    raw = sf.load_raw_iq(store, 0, mute_before_mm=None)
    n_tx, n_rx, n_ax = store.shape[1:]
    assert raw.shape == (n_tx, n_ax, n_rx, 2)


def test_negative_depth_samples_are_muted(store):
    """t0_s = -1.0 us at fs ~10.4 MHz means depth crosses 0 partway through the
    fast-time window — everything before that must be zeroed."""
    raw_unmuted = sf.load_raw_iq(store, 0, mute_before_mm=None)
    raw_muted = sf.load_raw_iq(store, 0, mute_before_mm=0.0)
    depth_mm = store.depth_axis_m() * 1e3
    n_negative = int(np.sum(depth_mm < 0.0))
    assert n_negative > 0, "test setup: t0_s must put some samples at negative depth"
    assert not np.any(raw_muted[:, :n_negative, :, :])
    assert np.any(raw_unmuted[:, :n_negative, :, :]), "sanity: the unmuted samples were nonzero"
    # samples from depth 0 onward are untouched
    np.testing.assert_array_equal(raw_muted[:, n_negative:], raw_unmuted[:, n_negative:])


def test_build_parameters_uses_t0_s_unmodified_and_negative(store, small_grid):
    params = sf.build_parameters(store, small_grid)
    assert np.all(params.initial_times == pytest.approx(store.attrs["t0_s"]))
    assert store.attrs["t0_s"] < 0, "sanity: this is the trap the module docstring warns about"


def test_build_parameters_grid_matches_variant(store, small_grid):
    params = sf.build_parameters(store, small_grid)
    assert params.grid_size_z == len(small_grid.z_mm)
    assert params.grid_size_x == len(small_grid.x_mm)


def test_beamform_frame_output_shape(store, small_grid):
    out = sf.beamform_frame(store, 0, small_grid, k=4, f_number=1.5)
    assert out.shape == (2, len(small_grid.z_mm), len(small_grid.x_mm), 4)
    assert np.isfinite(out).all()


def test_phase_rotation_negative_control_changes_output(store, small_grid):
    """Same check WI-4's gate relies on: disabling baseband phase rotation
    must produce a measurably different reconstruction — otherwise the
    negative control in `validate_stanford_reconstruction.py` has nothing to detect."""
    normal = sf.beamform_frame(store, 0, small_grid, k=1, f_number=1.5)
    no_rotation = sf.beamform_frame(
        store, 0, small_grid, k=1, f_number=1.5, demodulation_frequency=0.0
    )
    assert not np.allclose(normal, no_rotation)


def test_frame_labels_carry_provenance_not_physics_ground_truth(store):
    labels = sf.frame_labels(store, 0)
    assert labels["session"] == "s0"
    assert labels["source_file"] == "f.mat"
    assert labels["t0_s"] == store.attrs["t0_s"]
    # Unlike NVIDIA's frame.frame_labels, there is no sos/cyst ground truth.
    assert "sos_mean" not in labels
    assert "cyst_fraction" not in labels
