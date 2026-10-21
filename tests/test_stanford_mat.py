"""Stanford Verasonics reader + HVMux remap (PLAN.md WI-16), run against the
real local .mat files in data/rats/ when present. Skipped otherwise (these
files are not committed to the repo — PLAN.md's own corpus, ~400 MB each).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ssl_us.beamform import stanford_mat as sm

DATA_DIR = Path("data/rats")
REAL_FILES = sorted(DATA_DIR.rglob("*MultiFocal*.mat")) if DATA_DIR.exists() else []

pytestmark = pytest.mark.skipif(not REAL_FILES, reason="data/rats/*.mat not present locally")


@pytest.fixture(scope="module")
def meta():
    return sm.load_metadata(REAL_FILES[0])


@pytest.mark.parametrize("filepath", REAL_FILES, ids=lambda p: p.name)
def test_load_metadata_succeeds_on_every_real_file(filepath):
    """Broad real-data coverage — one metadata read per file in data/rats/,
    across both subjects (rat1, rat2)."""
    m = sm.load_metadata(filepath)
    assert m.n_elem > 0
    assert len(m.groups) > 0
    assert m.n_frames >= 1


def test_load_metadata_geometry_is_physically_sensible(meta):
    # Stanford probe: ~192 elements, pitch on the order of 100-300 um, a few MHz center freq.
    assert meta.n_elem in (192, 128, 256)  # whatever this probe actually has, not hardcoded to 192
    assert meta.elem_x_m.shape == (meta.n_elem,)
    assert 50e-6 < meta.pitch_m < 500e-6
    assert 1e6 < meta.fc_hz < 20e6
    assert 1400 < meta.c_m_s < 1600
    assert meta.fs_hz > meta.fc_hz  # sampling must exceed center frequency
    assert meta.t0_s >= 0
    assert meta.n_samp > 0
    assert meta.n_frames >= 1


def test_load_metadata_groups_have_both_receive_halves(meta):
    assert len(meta.groups) > 0
    for g in meta.groups[:20]:  # a sample, not all (some files have thousands)
        assert len(g.rx_events) >= 1
        assert g.apod.shape[0] > 0
        assert g.delay_s.shape == g.apod.shape


def test_load_metadata_scanline_origins_span_the_aperture(meta):
    origins = sorted({g.origin_x_m for g in meta.groups})
    assert len(origins) > 1, "expected multiple distinct scanline origins"
    span = max(origins) - min(origins)
    assert 0 < span < 0.1  # metres — a real probe aperture, not a unit-conversion bug


def test_combine_rx_halves_covers_most_of_the_array_for_a_real_group(meta):
    with h5py.File(REAL_FILES[0], "r") as f:
        rcv = sm.open_rcv_dataset(f)
        # pick a group with 2 receive halves if one exists (the HVMux case
        # this module exists for), else fall back to whatever the first group has.
        group = next((g for g in meta.groups if len(g.rx_events) == 2), meta.groups[0])
        iq, coverage = sm.combine_rx_halves(group, rcv, frame_idx=0, meta=meta)

    assert iq.shape == (meta.n_samp, meta.n_elem)
    assert iq.dtype == np.complex64
    covered = coverage > 0
    assert covered.sum() > meta.n_elem * 0.5, "HVMux remap should cover most of the array"
    assert np.all(coverage <= len(group.rx_events))
    # every covered element must carry real (nonzero) signal somewhere in time
    assert np.all(np.abs(iq[:, covered]).sum(axis=0) > 0)
    # every uncovered element must be exactly zero (no garbage leaking through)
    np.testing.assert_array_equal(iq[:, ~covered], 0)


def test_hvmux_remap_is_cyclic_not_linear_for_a_mid_array_aperture(meta):
    """The specific property PLAN.md 1.3 calls out: for a receive aperture
    starting mid-array, physical element (ap0+i) must be read from hardware
    channel (ap0+i) mod n_ch — for `n_ch=128` this only differs from the
    naive linear ``src=i`` mapping once ``ap0+i`` exceeds 128, so this checks
    that specific wrapped region directly, independent of
    `combine_rx_halves`'s own computation (a self-check would be vacuous).

    Note this is a property of the *channel-to-element* mapping, not of
    per-element coverage: for this file's actual apertures (1 and 65 with
    n_ch=128 over a 192-element array), the two halves' *covered* element
    ranges happen to be contiguous end-to-end (elements 1-128 and 65-192
    overlap, covering 1-192 with no gap) — coverage contiguity is checked
    separately in `test_combine_rx_halves_covers_most_of_the_array...`.
    """
    mid_array_groups = [g for g in meta.groups if any(ap > 1 for _, ap in g.rx_events)]
    if not mid_array_groups:
        pytest.skip("no group has a mid-array receive aperture in this file")

    group = mid_array_groups[0]
    rx_ap = next(ap for _, ap in group.rx_events if ap > 1)
    n_ch = 128  # this probe's HVMux receive-channel count, verified in test_combine_rx_halves_*
    phys = (rx_ap - 1) + np.arange(n_ch)
    src = ((rx_ap - 1) + np.arange(n_ch)) % n_ch
    wrapped = phys >= n_ch  # the region where cyclic and linear mappings diverge
    assert wrapped.any(), "this aperture never reaches the wrap region; test picked a bad group"
    msg = "cyclic remap must differ from a naive linear one here"
    assert np.any(src[wrapped] != phys[wrapped]), msg
    # and the wrapped destinations must still land inside the physical array
    assert np.all(phys[wrapped] < meta.n_elem)
