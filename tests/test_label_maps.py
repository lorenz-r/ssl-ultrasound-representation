"""Per-pixel label maps: the fetch decoder and the read side.

Two things here can go wrong silently and invalidate every crop-level anatomy
label built on top, so both are asserted rather than trusted:

1. **The decoder.** `nested_to_ndarray` reshapes a flat parquet buffer into
   (n, 507, 456); a transposed or off-by-one reshape still produces a
   plausible-looking mask.
2. **Partial fetches.** The `.npy` files are allocated at full size up front,
   so un-fetched frames read back as zeros — an empty cyst mask and a 0 m/s
   sound speed, neither of which looks like an error at the probe.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ssl_us.data import geometry as geo
from ssl_us.data.label_maps import CONTAINS_CYST_THRESHOLD, LabelMaps, open_label_maps

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "fetch_nvidia_label_maps.py"


def _load_script():
    """Import scripts/data/fetch_nvidia_label_maps.py by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("fetch_nvidia_label_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested(values: np.ndarray) -> pa.ChunkedArray:
    """`values` (n, r, c) as the `list<list<T>>` column parquet actually holds."""
    return pa.chunked_array([pa.array([[list(row) for row in frame] for frame in values])])


# ── the decoder ──────────────────────────────────────────────────────────────


def test_nested_to_ndarray_roundtrips_shape_and_values():
    mod = _load_script()
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2, size=(3, 5, 7)).astype(np.uint8)
    out = mod.nested_to_ndarray(_nested(values), (5, 7), np.uint8)
    assert out.shape == (3, 5, 7)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, values)


def test_nested_to_ndarray_rejects_a_shape_that_does_not_fit():
    mod = _load_script()
    values = np.zeros((2, 5, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        mod.nested_to_ndarray(_nested(values), (5, 8), np.float32)


def test_split_sizes_counts_from_the_store_metadata():
    mod = _load_script()
    meta = {"train/0": {}, "train/1": {}, "validation/0": {}}
    assert mod.split_sizes(meta) == {"train": 2, "validation": 1}


# ── the read side ────────────────────────────────────────────────────────────


@pytest.fixture
def fetched(tmp_path):
    """A miniature but structurally complete label-map directory."""
    n = 4
    rng = np.random.default_rng(1)
    seg = np.zeros((n, geo.BMODE_NZ, geo.BMODE_NX), dtype=np.uint8)
    seg[1, 100:200, 50:150] = 1  # one frame with a cyst, one without, etc.
    seg[3, 0:20, 0:20] = 1
    sos = rng.uniform(1400, 1500, size=(n, geo.SOS_NZ, geo.SOS_NX)).astype(np.float32)
    bmode = rng.random((n, geo.BMODE_NZ, geo.BMODE_NX)).astype(np.float32)
    for name, arr in (("seg", seg), ("sos", sos), ("bmode", bmode)):
        np.save(tmp_path / f"{name}.npy", arr)

    frame_ids = [f"train/{i}" for i in range(3)] + ["validation/0"]
    (tmp_path / "scalars.json").write_text(
        json.dumps(
            {
                "frame_ids": frame_ids,
                "extents": {},
                "per_frame": {
                    fid: {"phase_error": 1.5 + i, "t0": geo.T0_LEVELS_S[i % 3], "frame_idx": 0}
                    for i, fid in enumerate(frame_ids)
                },
            }
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "repo": "test",
                "columns": ["seg", "sos", "bmode"],
                "shards": {"a.parquet": {"split": "train", "n_rows": 4, "start": 0}},
                "verification": {"n_frames": 4},
            }
        )
    )
    return tmp_path, seg, sos


def test_open_refuses_an_unfinished_fetch(fetched):
    root, _, _ = fetched
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["shards"]["a.parquet"]["n_rows"] = 2
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="holds 2 of 4"):
        open_label_maps(root)


def test_open_refuses_an_unverified_fetch(fetched):
    root, _, _ = fetched
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["verification"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not been verified"):
        open_label_maps(root)


def test_index_of_maps_frame_ids_to_rows(fetched):
    root, seg, _ = fetched
    maps = open_label_maps(root)
    assert list(maps.index_of(["validation/0", "train/1"])) == [3, 1]
    np.testing.assert_array_equal(maps.seg[maps.index_of(["train/1"])[0]], seg[1])
    with pytest.raises(KeyError):
        maps.index_of(["train/99"])


def test_sos_is_resampled_onto_the_bmode_grid_and_cached(fetched):
    root, _, sos = fetched
    maps = open_label_maps(root)
    grid = maps.sos_on_grid("train/0")
    assert grid.shape == (geo.BMODE_NZ, geo.BMODE_NX)
    np.testing.assert_allclose(
        grid, geo.resample_sos_to_bmode_grid(sos[0]), rtol=0, atol=0
    )
    assert maps.sos_on_grid("train/0") is grid  # cached, not recomputed per crop


def test_crop_labels_match_a_direct_slice(fetched):
    root, seg, _ = fetched
    maps = open_label_maps(root)
    size = 64
    origins = np.array([[100, 50], [100, 50], [0, 0]])
    ids = ["train/0", "train/1", "train/1"]
    labels = maps.crop_labels(ids, origins, size)

    # frame 1's cyst covers rows 100:200, cols 50:150 -> this crop is all cyst
    assert labels["cyst_fraction"][1] == pytest.approx(1.0)
    assert labels["contains_cyst"][1] == 1
    # frame 0 has no cyst at all
    assert labels["cyst_fraction"][0] == 0.0
    assert labels["contains_cyst"][0] == 0
    # and the top-left crop of frame 1 misses it entirely
    assert labels["cyst_fraction"][2] == 0.0

    expected = maps.sos_on_grid("train/1")[100:164, 50:114]
    assert labels["sos_mean"][1] == pytest.approx(float(expected.mean()), rel=1e-6)
    assert labels["sos_std"][1] == pytest.approx(float(expected.std()), rel=1e-6)


def test_contains_cyst_uses_the_same_threshold_as_the_crop_sampler(fetched):
    """`crops.py` defines the target as `cyst_fraction > 0.01`; a second
    definition here would make the two probes silently incomparable."""
    root, _, _ = fetched
    maps = open_label_maps(root)
    # row 3 is `validation/0`, whose 20x20 cyst inside a 64x64 crop is 9.8 %
    labels = maps.crop_labels(["validation/0"], np.array([[0, 0]]), 64)
    frac = float(labels["cyst_fraction"][0])
    assert labels["contains_cyst"][0] == int(frac > CONTAINS_CYST_THRESHOLD)
    assert CONTAINS_CYST_THRESHOLD == 0.01


def test_crop_labels_reject_a_crop_that_runs_off_the_frame(fetched):
    root, _, _ = fetched
    maps = open_label_maps(root)
    with pytest.raises(ValueError, match="runs off frame"):
        maps.crop_labels(["train/0"], np.array([[geo.BMODE_NZ - 10, 0]]), 64)


def test_to_bootstrap_has_the_shape_the_probe_harness_expects(fetched):
    root, _, sos = fetched
    maps = open_label_maps(root)
    d = maps.to_bootstrap(["train/2", "train/0"])
    assert set(d) >= {"bmode", "seg", "sos", "frame_id", "frame_labels", "frame_labels_by_id"}
    assert d["bmode"].shape == (2, geo.BMODE_NZ, geo.BMODE_NX)
    assert list(d["frame_id"]) == ["train/2", "train/0"]
    np.testing.assert_allclose(d["frame_labels"]["sos_mean_frame"][0], sos[2].mean(), rtol=1e-6)
    assert d["frame_labels"]["t0_group"].tolist() == [2, 0]


def test_to_bootstrap_without_bmode_says_so(fetched):
    root, _, _ = fetched
    maps = LabelMaps(
        root=root,
        frame_ids=open_label_maps(root).frame_ids,
        arrays={"seg": np.load(root / "seg.npy", mmap_mode="r")},
        scalars={},
    )
    with pytest.raises(KeyError, match="without the `bmode` column"):
        maps.to_bootstrap(["train/0"])
