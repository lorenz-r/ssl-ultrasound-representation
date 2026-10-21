"""WI-16 beamform build wiring: frame indexing, store creation, resumability.

The beamforming itself is covered by `test_stanford_frame.py` (through the
real zea pipeline on a synthetic scene) and `test_sector_sum.py`/WI-4's gate.
What's tested here — the same split `test_build_nvidia_store.py` makes for NVIDIA — is
the part specific to this script: that `--limit` selects the right frame ids,
that a frame_id round-trips to the right store index, and that a second run
does not re-beamform anything already recorded. `sf.beamform_frame` is faked
so this stays fast and independent of grid size.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "beamform" / "build_stanford_store.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_stanford_store", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wi16():
    return _load_script()


@pytest.fixture
def fsa_source(tmp_path):
    """A minimal but fully valid `wi16_stanford_fsa.zarr`-shaped store, small
    enough that even the real `build_sector_geometry` call on it is instant."""
    out = tmp_path / "wi16_stanford_fsa.zarr"
    n_frames, n_elem, n_ax = 3, 8, 32
    root = zarr.open_group(str(out), mode="w")
    rng = np.random.default_rng(0)
    iq = rng.normal(size=(n_frames, n_elem, n_elem, n_ax)) + 1j * rng.normal(
        size=(n_frames, n_elem, n_elem, n_ax)
    )
    root.create_dataset("iq", data=iq.astype(np.complex64))
    root.create_dataset("coverage", data=np.ones((n_frames, n_elem, n_elem), dtype=np.uint8))
    pitch_m = 200e-6
    elem_x = (np.arange(n_elem) - (n_elem - 1) / 2) * pitch_m
    root.attrs.update(
        {
            "c_m_s": 1540.0,
            "fc_hz": 7_813_000.0,
            "fd_hz": 7_813_000.0,
            "fs_hz": 10_416_666.67,
            "wavelength_m": 1540.0 / 7_813_000.0,
            "pitch_m": pitch_m,
            "n_elem": n_elem,
            "elem_x_m": elem_x.tolist(),
            "t0_s": -1.0e-6,
        }
    )
    (out / "manifest.json").write_text(
        json.dumps({"files": {f"f{i}.mat": {"start": i, "n_frames": 1} for i in range(n_frames)}})
    )
    (out / "frame_metadata.json").write_text(
        json.dumps(
            {
                str(i): {
                    "session": f"s{i}",
                    "source_file": f"f{i}.mat",
                    "kind": "FSA",
                    "date": "20260101",
                }
                for i in range(n_frames)
            }
        )
    )
    return out


@pytest.fixture
def fake_beamform(monkeypatch, wi16):
    """Fast stand-in for `stanford_frame.beamform_frame`/`frame_labels` — the
    real pipeline is exercised elsewhere; here only the script's own wiring
    is under test. Returns a call-count list the tests inspect."""
    calls: list[str] = []

    def fake_frame(store, index, grid, k=16, f_number=1.5, num_patches=None, **kw):
        calls.append(str(index))
        return np.full((2, len(grid.z_mm), len(grid.x_mm), k), float(index), dtype=np.float32)

    def fake_labels(store, index):
        return {"session": f"s{index}"}

    monkeypatch.setattr(wi16.sf, "beamform_frame", fake_frame)
    monkeypatch.setattr(wi16.sf, "frame_labels", fake_labels)
    return calls


def _run(wi16, monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["build_stanford_store.py", *argv])
    wi16.main()


def test_limit_selects_a_prefix_of_frames(wi16, monkeypatch, fsa_source, fake_beamform, tmp_path):
    out = tmp_path / "out.zarr"
    _run(
        wi16,
        monkeypatch,
        ["--out", str(out), "--variant", "mm_matched", "--source", str(fsa_source), "--limit", "2"],
    )
    assert fake_beamform == ["0", "1"]
    store = wi16.open_store(out)
    assert len(store) == 2


def test_frame_index_round_trips_to_the_right_store_slot(
    wi16, monkeypatch, fsa_source, fake_beamform, tmp_path
):
    out = tmp_path / "out.zarr"
    _run(
        wi16,
        monkeypatch,
        ["--out", str(out), "--variant", "mm_matched", "--source", str(fsa_source), "--limit", "3"],
    )
    store = wi16.open_store(out)
    for i in range(3):
        assert store.read_frame(i)[0, 0, 0, 0] == pytest.approx(float(i))


def test_second_run_does_not_rebeamform_anything(
    wi16, monkeypatch, fsa_source, fake_beamform, tmp_path
):
    out = tmp_path / "out.zarr"
    argv = [
        "--out",
        str(out),
        "--variant",
        "mm_matched",
        "--source",
        str(fsa_source),
        "--limit",
        "2",
    ]
    _run(wi16, monkeypatch, argv)
    assert fake_beamform == ["0", "1"]
    fake_beamform.clear()
    _run(wi16, monkeypatch, argv)
    assert fake_beamform == []


def test_frame_metadata_sidecar_is_written(wi16, monkeypatch, fsa_source, fake_beamform, tmp_path):
    out = tmp_path / "out.zarr"
    _run(
        wi16,
        monkeypatch,
        ["--out", str(out), "--variant", "mm_matched", "--source", str(fsa_source), "--limit", "2"],
    )
    store = wi16.open_store(out)
    meta = store.read_frame_metadata()
    assert meta["0"]["session"] == "s0"
    assert meta["1"]["session"] == "s1"


def test_beamform_params_record_the_grid_variant(
    wi16, monkeypatch, fsa_source, fake_beamform, tmp_path
):
    out = tmp_path / "out.zarr"
    _run(
        wi16,
        monkeypatch,
        [
            "--out",
            str(out),
            "--variant",
            "lambda_matched",
            "--source",
            str(fsa_source),
            "--limit",
            "1",
        ],
    )
    store = wi16.open_store(out)
    assert store.beamform_params["grid_variant"] == "lambda_matched"
    assert store.beamform_params["normalized"] is False


def test_store_z_mm_x_mm_reflect_the_stanford_grid_not_nvidias(
    wi16, monkeypatch, fsa_source, fake_beamform, tmp_path
):
    """Regression guard for the `store.py` bug this WI fixed: a Stanford-grid
    store must not silently carry NVIDIA's (507, 456) bmode axes."""
    out = tmp_path / "out.zarr"
    _run(
        wi16,
        monkeypatch,
        ["--out", str(out), "--variant", "mm_matched", "--source", str(fsa_source), "--limit", "1"],
    )
    store = wi16.open_store(out)
    z_mm = store.root.attrs["z_mm"]
    assert z_mm[0] == pytest.approx(0.0)
    assert z_mm[-1] == pytest.approx(19.9)
