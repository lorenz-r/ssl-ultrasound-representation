"""WI-8 build wiring: the stream adapter and the store writer path.

The beamforming itself is covered by `test_sector_sum.py` / WI-4's gate; what
is tested here is the part that is easy to get silently wrong — pulling frames
from a forward-only stream in the order `build_store` asks for them, and
resuming without re-beamforming.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ssl_us.beamform.build import FrameResult, ShardSpec, build_store
from ssl_us.beamform.store import SectorStore, StoreSpec

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "beamform" / "build_nvidia_store.py"


def _load_script():
    """Import scripts/beamform/build_nvidia_store.py by path — `scripts` is not a package."""
    spec = importlib.util.spec_from_file_location("build_nvidia_store", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wi8():
    return _load_script()


def _fake_stream(ids):
    for fid in ids:
        yield fid, {"id": fid}


def test_source_yields_frames_in_order(wi8):
    src = wi8.OrderedFrameSource(_fake_stream(["a", "b", "c"]))
    assert src.take("a")["id"] == "a"
    assert src.take("b")["id"] == "b"


def test_source_skips_forward_without_yielding_skipped_frames(wi8):
    """The resume path: already-built frames are discarded, not beamformed."""
    src = wi8.OrderedFrameSource(_fake_stream(["a", "b", "c", "d"]))
    assert src.take("c")["id"] == "c"
    assert src.take("d")["id"] == "d"


def test_source_raises_on_out_of_order_request(wi8):
    """Must not silently return the wrong frame if ordering assumptions break."""
    src = wi8.OrderedFrameSource(_fake_stream(["a", "b", "c"]))
    src.take("b")
    with pytest.raises(KeyError, match="stream order"):
        src.take("a")


def test_build_writes_into_zarr_store_and_resumes(tmp_path, wi8):
    frame_ids = ["train/0", "train/1", "train/2"]
    spec = StoreSpec(n_frames=3, nz=6, nx=5, k=16)
    store = SectorStore.create(tmp_path / "s.zarr", spec, frame_ids, {"k": 16})
    index = {f: i for i, f in enumerate(frame_ids)}
    manifest_dir = tmp_path / "build"

    calls: list[str] = []

    def beamform_frame(frame_id: str) -> FrameResult:
        calls.append(frame_id)
        value = float(index[frame_id] + 1)
        return FrameResult(
            array=np.full(spec.frame_shape, value, dtype=np.float32), metadata={"v": value}
        )

    def write_frame(_out, frame_id, array):
        store.write_frame(index[frame_id], array)

    shards = [ShardSpec(name="train", frame_ids=frame_ids)]
    build_store(shards, manifest_dir, beamform_frame, {"k": 16}, write_frame=write_frame)

    assert calls == frame_ids
    for fid, i in index.items():
        assert store.read_frame(i) == pytest.approx(float(i + 1)), fid

    # Resuming must not re-beamform anything already recorded.
    calls.clear()
    build_store(shards, manifest_dir, beamform_frame, {"k": 16}, write_frame=write_frame)
    assert calls == []


def test_default_writer_still_used_when_none_passed(tmp_path):
    """The WI-6 `.npy` path must keep working — build_store's writer is optional."""
    from ssl_us.beamform.build import read_frame_chunk

    def beamform_frame(frame_id: str) -> FrameResult:
        return FrameResult(array=np.ones((2, 2), dtype=np.float32), metadata={})

    build_store(
        [ShardSpec(name="s", frame_ids=["f0"])], tmp_path, beamform_frame, {}
    )
    assert read_frame_chunk(tmp_path, "f0") == pytest.approx(np.ones((2, 2)))


def test_beamform_constants_match_reconstruction_gate_validated_values():
    """These are not free parameters — WI-4's gate was measured at them."""
    from ssl_us.beamform import frame as bf

    assert bf.F_NUMBER == 1.5
    assert bf.T_PEAK == 0.0
    assert bf.MUTE_SAMPLES == 32
    assert bf.DEFAULT_K == 16
