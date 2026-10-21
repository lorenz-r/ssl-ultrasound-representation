"""Reading the Stanford FSA corpus (`ssl_us.data.stanford_fsa`).

The load-bearing property is that a partial build cannot be read: zarr returns
zeros for chunks that were never written, and an all-zero acquisition
beamforms to a black image instead of raising.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from ssl_us.data.stanford_fsa import open_fsa_store

N_TX = N_RX = 4
N_AX = 6


@pytest.fixture
def store(tmp_path):
    root = zarr.open_group(str(tmp_path), mode="w")
    iq = root.create_dataset(
        "iq", shape=(3, N_TX, N_RX, N_AX), chunks=(1, N_TX, N_RX, N_AX), dtype="complex64"
    )
    cov = root.create_dataset(
        "coverage", shape=(3, N_TX, N_RX), chunks=(1, N_TX, N_RX), dtype="uint8"
    )
    rng = np.random.default_rng(0)
    data = (rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape)).astype("complex64")
    iq[:] = data
    cov[:] = 2
    root.attrs.update(
        {
            "probe": "L12-3v",
            "c_m_s": 1540.0,
            "fc_hz": 7.813e6,
            "fd_hz": 7.813e6,
            "fs_hz": 31.25e6 / 3,
            "fs_raw_hz": 31.25e6,
            "wavelength_m": 1540.0 / 7.813e6,
            "pitch_m": 200e-6,
            "n_elem": N_TX,
            "elem_x_m": [-0.3e-3, -0.1e-3, 0.1e-3, 0.3e-3],
            "start_depth_m": 0.394e-3,
            "t0_s": -1.2547e-6,
            "decim": 3,
            "n_keep_raw_samples": 861,
            "n_ax": N_AX,
            "layout": "(frame, n_tx, n_rx, n_ax) complex64; tx i fires element i",
            "source": "test",
        }
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "article": 28291985,
                "version": 1,
                "decim": 3,
                "files": {
                    "DATA_FSA_20190318_100446.mat": {"start": 0, "n_frames": 1},
                    "DATA_FSA_20190318_101206.mat": {"start": 1, "n_frames": 1},
                    "DATA_FSA_Fast_20190416_092017.mat": {"start": 2, "n_frames": 1},
                },
            }
        )
    )
    (tmp_path / "frame_metadata.json").write_text(
        json.dumps(
            {
                "0": {
                    "source_file": "DATA_FSA_20190318_100446.mat",
                    "kind": "FSA",
                    "date": "20190318",
                    "time": "100446",
                    "session": "20190318_s00",
                },
                "1": {
                    "source_file": "DATA_FSA_20190318_101206.mat",
                    "kind": "FSA",
                    "date": "20190318",
                    "time": "101206",
                    "session": "20190318_s00",
                },
                "2": {
                    "source_file": "DATA_FSA_Fast_20190416_092017.mat",
                    "kind": "FSA_Fast",
                    "date": "20190416",
                    "time": "092017",
                    "session": "20190416_s01",
                },
            }
        )
    )
    return tmp_path, data


def test_opens_a_complete_store(store):
    root, data = store
    s = open_fsa_store(root)
    assert len(s) == 3
    assert s.shape == (3, N_TX, N_RX, N_AX)
    np.testing.assert_array_equal(s.iq(1), data[1])
    np.testing.assert_array_equal(s.coverage(0), np.full((N_TX, N_RX), 2, dtype=np.uint8))


def test_refuses_a_partial_build(store):
    root, _ = store
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"].pop("DATA_FSA_Fast_20190416_092017.mat")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holds 2 of 3"):
        open_fsa_store(root)


def test_refuses_metadata_that_does_not_cover_every_frame(store):
    root, _ = store
    meta = json.loads((root / "frame_metadata.json").read_text())
    del meta["2"]
    (root / "frame_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="2 metadata entries for 3 frames"):
        open_fsa_store(root)


def test_missing_manifest_names_the_builder(store):
    root, _ = store
    (root / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="fetch_stanford_fsa"):
        open_fsa_store(root)


def test_time_axis_starts_before_the_transmit(store):
    """`t0_s` carries the lens/pulse-peak offset, so sample 0 is negative.

    Dropping that term — as the NVIDIA path does with `t_peak = 0` — would
    shift every Stanford pixel by ~1.4 mm.
    """
    root, _ = store
    s = open_fsa_store(root)
    t = s.time_axis_s()
    assert t[0] < 0
    assert t[0] == pytest.approx(s.attrs["t0_s"])
    np.testing.assert_allclose(np.diff(t), 1.0 / s.attrs["fs_hz"], rtol=1e-12)
    np.testing.assert_allclose(s.depth_axis_m(), t * s.attrs["c_m_s"] / 2)


def test_sessions_group_frames_and_cover_all_of_them(store):
    root, _ = store
    s = open_fsa_store(root)
    sessions = s.sessions()
    assert sessions == {"20190318_s00": [0, 1], "20190416_s01": [2]}
    assert sorted(i for v in sessions.values() for i in v) == list(range(len(s)))


def test_labels_are_aligned_to_frame_index(store):
    root, _ = store
    s = open_fsa_store(root)
    labels = s.labels()
    assert list(labels["kind"]) == ["FSA", "FSA", "FSA_Fast"]
    assert list(labels["source_file"])[2].startswith("DATA_FSA_Fast")
    assert all(len(v) == len(s) for v in labels.values())


def test_element_positions_match_the_declared_count(store):
    root, _ = store
    s = open_fsa_store(root)
    x = s.element_x_m()
    assert x.size == s.attrs["n_elem"] == s.shape[1]
    assert np.allclose(np.diff(x), s.attrs["pitch_m"])
