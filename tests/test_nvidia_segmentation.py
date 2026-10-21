"""WI-19's segmentation driver, end to end on a synthetic store.

The probe protocol is covered by `test_dense.py` and the crop/label path by
`test_store_features.py`. What is tested here is the wiring the driver owns and
that nothing else would flag: the figure frames being chosen by a stated rule
rather than by eye, the baselines being ordered ahead of the trained arms, and
the whole thing running from `main()` to a written figure without a real store.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ssl_us.beamform import sectors as sec
from ssl_us.beamform.store import SectorStore, StoreSpec
from ssl_us.data import geometry as geo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "nvidia_segmentation.py"


def _load_script():
    """Import the script by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("nvidia_segmentation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_script()


# ── the driver's own decisions ──────────────────────────────────────────────


def test_baselines_are_ordered_ahead_of_the_trained_arms(mod):
    """Column order is the argument the figure makes. A reader who meets
    `trivial_dense` before `vicreg_A` cannot mistake "cysts are dark" for
    "the representation found the cysts"."""
    got = mod.ordered(["ijepa", "vicreg_A", "trivial_dense", "pca_raw_dense"])
    assert got == ["trivial_dense", "pca_raw_dense", "vicreg_A", "ijepa"]


def test_an_unlisted_encoder_is_appended_not_dropped(mod):
    """A new arm must never vanish from the figure for want of an entry in
    `FIGURE_ORDER` — the project's no-silent-truncation rule."""
    got = mod.ordered(["brand_new_arm", "trivial_dense"])
    assert got == ["trivial_dense", "brand_new_arm"]


def test_figure_frames_are_the_median_and_the_worst(mod):
    dice = {"f0": 0.9, "f1": 0.1, "f2": 0.5, "f3": 0.7, "f4": 0.3}
    picked = mod.pick_figure_frames(dice, n=2)
    assert set(picked) == {"f2", "f1"}  # median 0.5 and worst 0.1
    # Worst last, so the slide ends on the failure case.
    assert picked[-1] == "f1"


def test_figure_frame_picker_survives_fewer_frames_than_asked_for(mod):
    assert mod.pick_figure_frames({"a": 0.4, "b": 0.2}, n=5) == ["a", "b"]
    assert mod.pick_figure_frames({}, n=2) == []


# ── end to end ──────────────────────────────────────────────────────────────

NZ, NX, K_STORE = geo.BMODE_NZ, geo.BMODE_NX, 16
CROP = 32
N_FRAMES = 6


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A small store on the real (507, 456) grid plus a fetched-looking label dir.

    The grid has to be the real one: `LabelMaps` indexes `seg` with the crop
    origins the store's own geometry produces, and `sos_on_grid` resamples onto
    the bmode axes. A toy grid would make the two disagree.
    """
    root = tmp_path_factory.mktemp("wi19")
    frame_ids = [f"train/{i}" for i in range(N_FRAMES - 2)] + [
        f"validation/{i}" for i in range(2)
    ]
    spec = StoreSpec(n_frames=len(frame_ids), nz=NZ, nx=NX, k=K_STORE)
    geometry = sec.build_sector_geometry(
        k=K_STORE, f_number=1.5, z_mm=geo.bmode_z_axis_mm(), x_mm=geo.bmode_x_axis_mm()
    )
    store = SectorStore.create(
        root / "store.zarr", spec, frame_ids=frame_ids,
        beamform_params={"f_number": 1.5, "k": K_STORE, "t_peak": 0.0},
        geometry=geometry,
    )
    rng = np.random.default_rng(0)

    # A cyst is a dark disc, which is what the real anatomy is: it makes the
    # `trivial_dense` baseline behave the way it does on real data, so a run
    # that scores it at chance here would be a bug in the harness.
    seg = np.zeros((len(frame_ids), NZ, NX), dtype=np.uint8)
    zz, xx = np.mgrid[0:NZ, 0:NX]
    for i in range(len(frame_ids)):
        cz, cx = 180 + 30 * i, 200 + 20 * i
        # Big enough that randomly-placed 32 px crops actually land on it — a
        # realistically-sized lesion here would leave most crops all-background
        # and the fixture would test the degenerate path instead of the probe.
        seg[i] = ((zz - cz) ** 2 + (xx - cx) ** 2 < 150**2).astype(np.uint8)
        frame = rng.normal(size=spec.frame_shape).astype(np.float32)
        frame *= np.where(seg[i], 0.05, 1.0)[None, :, :, None]
        store.write_frame(i, frame)

    store.write_frame_metadata({
        fid: {
            "t0": geo.T0_LEVELS_S[i % 3],
            "phase_error": 1.0 + i,
            "sos_mean": 1500.0 + i,
            "sos_std": 10.0 + i,
            "cyst_fraction": float(seg[i].mean()),
            "segmentation_empty": False,
        }
        for i, fid in enumerate(frame_ids)
    })

    maps = root / "label_maps"
    maps.mkdir()
    np.save(maps / "seg.npy", seg)
    np.save(
        maps / "sos.npy",
        np.full((len(frame_ids), geo.SOS_NZ, geo.SOS_NX), 1500.0, dtype=np.float32),
    )
    (maps / "manifest.json").write_text(json.dumps({
        "columns": ["seg", "sos"],
        "shards": {"0": {"n_rows": len(frame_ids)}},
        "verification": {"ok": True},
    }))
    (maps / "scalars.json").write_text(json.dumps({
        "frame_ids": frame_ids,
        "per_frame": {
            fid: {"t0": geo.T0_LEVELS_S[i % 3], "phase_error": 1.0 + i}
            for i, fid in enumerate(frame_ids)
        },
    }))
    return root / "store.zarr", maps, root


def test_main_writes_a_table_and_a_figure(mod, corpus, tmp_path, monkeypatch):
    """The whole driver on CPU, with only the untrained baselines available —
    no checkpoint root exists, so this also asserts the 'registered only when
    trained' path leaves a runnable column rather than an empty one."""
    store_path, maps, root = corpus
    out = tmp_path / "seg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvidia_segmentation.py",
            "--store", str(store_path),
            "--label-maps", str(maps),
            "--checkpoint-root", str(root / "no-checkpoints"),
            "--out", str(out),
            "--k", "4",
            "--crop-size", str(CROP),
            "--z-min-mm", "0.0",
            "--crops-per-frame", "6",
            "--split", "train",
            "--dense-grid", "4",
            "--device", "cpu",
            "--n-boot", "50",
            "--max-fit-cells", "5000",
            "--window-stride", "16",
            "--figure-frames", "2",
            "--reference-encoder", "trivial_dense",
        ],
    )
    assert mod.main() == 0

    record = json.loads((out / "results.json").read_text())
    assert record["wi"] == 19
    assert record["target"] == "cyst_seg"
    assert record["dense_grid"] == 4
    names = [r["encoder"] for r in record["rows"]]
    assert names[:2] == ["trivial_dense", "pca_raw_dense"]
    assert "random_resnet18" in names

    # The brightness baseline must actually find a dark disc. If it does not,
    # the target and the crops are misaligned and every trained column that
    # ever runs through here would be scored against a broken floor.
    trivial = next(r for r in record["rows"] if r["encoder"] == "trivial_dense")
    assert trivial["dice"] > 0.3

    assert (out / "masks.png").exists()
    assert (out / "dice_table.md").exists()
    assert len(record["figure_frames"]) == 2
    assert set(record["whole_frame_dice"]) == set(names)


def test_empty_mask_frames_are_excluded_from_the_dense_target(mod, corpus, tmp_path, monkeypatch):
    """PLAN.md 2.6 applies harder to a dense target than to a scalar one: an
    empty mask that is really an unlabelled edge-touching cyst teaches the head
    to erase real cysts, not merely to mislabel a crop."""
    store_path, maps, root = corpus
    from ssl_us.beamform.store import open_store

    store = open_store(store_path, mode="a")
    metadata = store.read_frame_metadata()
    train_ids = sorted(f for f in metadata if f.startswith("train/"))
    for fid in train_ids[:2]:
        metadata[fid]["segmentation_empty"] = True
    store.write_frame_metadata(metadata)

    out = tmp_path / "seg_excluded"
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvidia_segmentation.py",
            "--store", str(store_path), "--label-maps", str(maps),
            "--checkpoint-root", str(root / "no-checkpoints"), "--out", str(out),
            "--k", "4", "--crop-size", str(CROP), "--z-min-mm", "0.0",
            "--crops-per-frame", "4", "--split", "train", "--dense-grid", "4",
            "--device", "cpu", "--n-boot", "20", "--max-fit-cells", "2000",
            "--no-figure", "--reference-encoder", "trivial_dense",
        ],
    )
    assert mod.main() == 0
    record = json.loads((out / "results.json").read_text())
    assert record["crops_dropped_empty_mask"] == 2 * 4
    assert record["frames_probed"] == len(train_ids) - 2

    for fid in train_ids[:2]:
        metadata[fid]["segmentation_empty"] = False
    store.write_frame_metadata(metadata)
