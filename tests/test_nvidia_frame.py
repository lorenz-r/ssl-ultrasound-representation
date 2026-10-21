"""WI-0: load a real NVIDIA frame end-to-end and assert it against the documentation.

Network test — skipped unless ``ZEA_WI0_DATA`` points at a cached results json
produced by ``scripts/data/probe_nvidia_frames.py`` (each frame is a 265 MB download, so the
measurement is run once and its output checked in).

This is the item everything else is built on. It exists to resolve three
ambiguities that no amount of reading the documentation can settle (PLAN.md WI-0):

1. the IQ axis order, which NVIDIA's card and OpenH-RF's converter disagree about
   and which is invisible in both shape and image;
2. what ``bmode_focused`` actually is;
3. the reciprocity residual, which decides whether the PW stack gets built at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ssl_us.data import geometry as geo

RESULTS = os.environ.get("ZEA_WI0_DATA", "results/gates/nvidia_frame_probe.json")
pytestmark = pytest.mark.skipif(
    not Path(RESULTS).exists(),
    reason=f"{RESULTS} not present; run scripts/data/probe_nvidia_frames.py",
)


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    data = json.loads(Path(RESULTS).read_text())
    assert data, "no frames in results file"
    return data


def test_iq_shape_and_dtype(frames):
    for f in frames:
        assert f["iq_shape"] == [180, 180, 1024]
        assert f["iq_dtype"] == "float32"


def test_element_pitch_matches_documentation(frames):
    for f in frames:
        assert abs(f["pitch_um"] - 201.0) < 0.1
        assert f["elpos_shape"] == [3, 180]


def test_scalar_acquisition_parameters(frames):
    for f in frames:
        assert abs(f["fs"] - geo.FS_HZ) < 1.0
        assert f["fc"] == geo.FC_HZ
        assert f["fd"] == geo.FD_HZ
        assert f["c0"] == geo.C0_M_S
        assert f["frame_idx"] == 0
        geo.t0_group(f["t0"])  # raises if t0 is not one of the three measured levels


def test_grids_and_extent(frames):
    for f in frames:
        assert f["bmode_shape"] == [507, 456]
        assert f["seg_shape"] == [507, 456]
        assert f["sos_shape"] == [32, 32]
        assert f["seg_unique"] in ([0], [1], [0, 1])
        x0, x1, z1, z0 = f["bmode_extent"]
        assert abs(z0 - 0.0) < 1e-6, "depth starts at 0 mm — the 8 mm floor is ours"
        assert abs(z1 - 40.0) < 1e-6
        assert abs(x1 - geo.BMODE_X_MAX_MM) < 1e-3


def test_fast_time_window_covers_the_field_of_view(frames):
    """1024 samples must cover the worst-case two-way path in the FOV."""
    for f in frames:
        assert f["last_sample_above_1e-4"] < 1024


def test_reciprocity_residual_is_small_and_consistent(frames):
    """The measurement that decides WI-4b and forces the 4.5a(a) mitigation."""
    res = np.array([f["recip_residual"] for f in frames])
    assert res.max() < 0.05, f"reciprocity residual larger than expected: {res}"
    assert res.std() < 0.01, f"residual not consistent across frames: {res}"


def test_onset_structure_is_symmetric(frames):
    """Onset carries no tx/rx distinction — the axis order is not recoverable from it."""
    for f in frames:
        assert f["onset_asym_mean_abs"] == pytest.approx(0.0, abs=1e-6)


def test_axis_energy_is_symmetric(frames):
    """Per-index energy along axis 0 and axis 1 agree, so neither axis is 'the' tx axis."""
    for f in frames:
        assert f["axis_energy_corr"] > 0.9
        assert f["axis_energy_reldiff"] < 0.05
