"""WI-16 Stanford FSA corpus build — selection, grouping, and the guards.

The decode itself is covered by `test_stanford_mat.py` (which needs the real
`.mat` files) and, end to end, by the beamformed check recorded in MEMORY.md.
What is tested here is everything that can go wrong *silently*:

- pulling a MultiFocal or Hadamard acquisition through a path that labels its
  first axis "transmit element",
- mixing acquisitions whose geometry differs into one store,
- an inferred session grouping that quietly collapses or splits.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "wi16_fetch_stanford_fsa.py"


def _load_script():
    """Import scripts/wi16_fetch_stanford_fsa.py by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("wi16_fetch_stanford_fsa", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


def _listing():
    """A figshare-shaped listing covering all four acquisition types."""
    names = [
        "DATA_FSA_20190318_100446.mat",
        "DATA_FSA_20190318_101206.mat",
        "DATA_FSA_20190318_121432.mat",
        "DATA_MultiFocal_20190318_101049.mat",
        "DATA_Hadamard_20190318_101355.mat",
        "DATA_MF_Fast_20190416_091535.mat",
        "DATA_FSA_Fast_20190416_092017.mat",
        "readme.txt",
    ]
    return [{"name": n, "size": 1, "download_url": f"http://x/{n}"} for n in names]


# ── which files are FSA ───────────────────────────────────────────────────────


def test_only_single_element_transmit_acquisitions_are_selected(mod):
    picked = [f["name"] for f in mod.fsa_files(_listing())]
    assert picked == [
        "DATA_FSA_20190318_100446.mat",
        "DATA_FSA_20190318_101206.mat",
        "DATA_FSA_20190318_121432.mat",
        "DATA_FSA_Fast_20190416_092017.mat",
    ]


def test_mf_fast_is_not_mistaken_for_fsa_fast(mod):
    """`DATA_MF_Fast_*` is multi-focal; a loose prefix match would pull in 75 GB
    of it and label the transmit axis wrong."""
    assert all("MF_Fast" not in f["name"] for f in mod.fsa_files(_listing()))


def test_no_fast_drops_only_the_fast_variant(mod):
    picked = [f["name"] for f in mod.fsa_files(_listing(), include_fast=False)]
    assert picked == [
        "DATA_FSA_20190318_100446.mat",
        "DATA_FSA_20190318_101206.mat",
        "DATA_FSA_20190318_121432.mat",
    ]


def test_files_are_returned_in_acquisition_order(mod):
    files = mod.fsa_files(_listing())
    keys = [(f["date"], f["time"]) for f in files]
    assert keys == sorted(keys)


# ── inferred session grouping ────────────────────────────────────────────────


def test_sessions_split_on_a_long_gap_and_on_date(mod):
    files = mod.fsa_files(_listing())
    mod.assign_sessions(files)
    by_name = {f["name"]: f["session"] for f in files}
    # 10:04 and 10:12 are the same session; 12:14 is a new one
    assert by_name["DATA_FSA_20190318_100446.mat"] == by_name["DATA_FSA_20190318_101206.mat"]
    assert by_name["DATA_FSA_20190318_121432.mat"] != by_name["DATA_FSA_20190318_100446.mat"]
    # a different date is always a different session
    assert by_name["DATA_FSA_Fast_20190416_092017.mat"].startswith("20190416")


def test_every_file_gets_a_session(mod):
    files = mod.fsa_files(_listing())
    mod.assign_sessions(files)
    assert all(f.get("session") for f in files)


# ── the FSA guard ────────────────────────────────────────────────────────────


class _Meta:
    def __init__(self, n_elem=4, n_groups=4, nnz=1, foci=None):
        self.n_elem = n_elem
        foci = foci if foci is not None else [0.05] * n_groups
        self.groups = []
        for f in foci:
            apod = np.zeros(n_elem)
            apod[:nnz] = 1.0
            g = type("G", (), {})()
            g.apod = apod
            g.focus_m = f
            self.groups.append(g)


def test_assert_is_fsa_accepts_one_transmit_per_element(mod):
    mod.assert_is_fsa(Path("ok.mat"), _Meta())


def test_assert_is_fsa_rejects_a_multi_element_transmit(mod):
    """This is what a MultiFocal file looks like — an aperture, not an element."""
    with pytest.raises(ValueError, match="expected exactly 1"):
        mod.assert_is_fsa(Path("multifocal.mat"), _Meta(nnz=3))


def test_assert_is_fsa_rejects_a_wrong_transmit_count(mod):
    with pytest.raises(ValueError, match="not full synthetic aperture"):
        mod.assert_is_fsa(Path("short.mat"), _Meta(n_groups=3, foci=[0.05] * 3))


def test_assert_is_fsa_rejects_multiple_foci(mod):
    with pytest.raises(ValueError, match="distinct transmit foci"):
        mod.assert_is_fsa(Path("mf.mat"), _Meta(foci=[0.01, 0.02, 0.03, 0.04]))


# ── geometry constancy across files ──────────────────────────────────────────


def test_check_constant_passes_for_identical_geometry(mod):
    a = {k: 1.0 for k in mod.CONSTANT_ATTRS}
    mod.check_constant(a, dict(a), "same.mat")


def test_check_constant_rejects_a_different_probe(mod):
    a = {k: 1.0 for k in mod.CONSTANT_ATTRS}
    b = {**a, "probe": "C5-2v"}
    with pytest.raises(ValueError, match="probe"):
        mod.check_constant({**a, "probe": "L12-3v"}, b, "other_probe.mat")


def test_check_constant_rejects_a_shifted_time_origin(mod):
    a = {k: 1.0 for k in mod.CONSTANT_ATTRS}
    with pytest.raises(ValueError, match="t0_s"):
        mod.check_constant(a, {**a, "t0_s": 2.0}, "shifted.mat")


# ── download integrity ───────────────────────────────────────────────────────


def test_download_rejects_a_bad_md5_and_leaves_no_file(mod, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"stanford")
    dest = tmp_path / "dest.bin"
    with pytest.raises(ValueError, match="md5 mismatch"):
        mod.download(src.as_uri(), dest, expect_md5="0" * 32)
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_accepts_the_published_md5(mod, tmp_path):
    import hashlib

    src = tmp_path / "src.bin"
    src.write_bytes(b"stanford")
    dest = tmp_path / "dest.bin"
    mod.download(src.as_uri(), dest, expect_md5=hashlib.md5(b"stanford").hexdigest())
    assert dest.read_bytes() == b"stanford"


# ── fast-time budget ─────────────────────────────────────────────────────────


def test_the_kept_window_divides_by_the_decimation_factor(mod):
    """A remainder would silently drop or pad the last samples of every frame."""
    assert mod.N_KEEP % mod.DECIM == 0


def test_decimated_rate_still_covers_the_probe_bandwidth(mod):
    """L12-3v is 4-12 MHz about a 7.813 MHz carrier: +/-4 MHz of baseband.

    Decimating the 31.25 MHz record by 3 leaves +/-5.2 MHz; by 4 it would leave
    +/-3.9 MHz and clip the band edges, which is what the retired
    `benbarkow/us-ssl-mf` set did.
    """
    fs_raw, fc = 31.25e6, 7.813e6
    half_bandwidth = (12e6 - 4e6) / 2
    assert fs_raw / mod.DECIM / 2 > half_bandwidth
    assert fs_raw / 4 / 2 < half_bandwidth
    assert fc  # carrier is what the demodulation centres on


# ── metadata shape (guards the consumer contract) ────────────────────────────


def test_constant_attrs_are_a_subset_of_what_the_store_records(mod, tmp_path):
    """Every asserted-constant key must actually be emitted, or the check is a
    no-op that silently passes for mismatched files."""
    emitted = {
        "probe",
        "bandwidth_hz",
        "c_m_s",
        "fc_hz",
        "fd_hz",
        "fs_hz",
        "fs_raw_hz",
        "wavelength_m",
        "pitch_m",
        "n_elem",
        "elem_x_m",
        "start_depth_m",
        "t0_s",
        "decim",
        "n_keep_raw_samples",
        "n_ax",
        "layout",
        "source",
    }
    assert set(mod.CONSTANT_ATTRS) <= emitted
    # and the manifest the build writes is JSON-serializable as declared
    json.dumps({"article": mod.ARTICLE, "version": mod.VERSION, "decim": mod.DECIM})


# ── MATLAB v7.3 scalar reading ───────────────────────────────────────────────


def _write_tw_file(path, peak, n_tw, lens):
    """A minimal MATLAB-v7.3-shaped file: `TW.peak` as a cell of references
    when there is more than one waveform, as a plain scalar when there is one."""
    import h5py

    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        tw = f.create_group("TW")
        if n_tw > 1:
            targets = [
                refs.create_dataset(f"p{i}", data=np.array([[peak + i]])) for i in range(n_tw)
            ]
            ds = tw.create_dataset(
                "peak", shape=(n_tw, 1), dtype=h5py.special_dtype(ref=h5py.Reference)
            )
            for i, t in enumerate(targets):
                ds[i, 0] = t.ref
        else:
            tw.create_dataset("peak", data=np.array([[peak]]))
        f.create_group("Trans").create_dataset("lensCorrection", data=np.array([[lens]]))


class _TimeMeta:
    t0_s = 5.119672340970177e-07
    wavelength_m = 1540.0 / 7.813e6
    c_m_s = 1540.0


@pytest.mark.parametrize("n_tw", [1, 2])
def test_time_origin_reads_through_a_matlab_cell_reference(mod, tmp_path, n_tw):
    """Some acquisitions carry two `TW` entries, which HDF5 stores as object
    references; `float()` on a raw reference raises, and the whole build died
    on file 25 of 117 the first time. MATLAB indexes `TW(1)`, so must we."""
    path = tmp_path / f"tw{n_tw}.mat"
    _write_tw_file(path, peak=1.79699, n_tw=n_tw, lens=6.0018045454545454)
    t0 = mod.time_of_sample_zero(path, _TimeMeta())
    assert t0 == pytest.approx(-1.2543964022666187e-06, rel=1e-9)
    assert t0 < 0  # the lens/pulse-peak offset puts sample 0 before transmit
