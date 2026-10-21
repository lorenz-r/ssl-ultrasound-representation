"""Resumable build pipeline (PLAN.md WI-6).

The central guarantee under test: a crash mid-shard, followed by a restart,
must produce byte-identical output to an uninterrupted run — not just "the
same data" but the same files on disk, because that is the only way to be
sure the resumed frames were not silently re-beamformed differently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ssl_us.beamform.build import (
    BuildManifest,
    FrameResult,
    ShardSpec,
    build_store,
    load_manifest,
    read_frame_chunk,
)


def _deterministic_frame(frame_id: str) -> FrameResult:
    """A pure function of frame_id — same input, same bytes, every call.
    Standing in for the real AngularSectorSum beamformer in these tests."""
    seed = int(hashlib.sha256(frame_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(2, 4, 4, 4)).astype(np.float32)
    return FrameResult(array=array, metadata={"t0": seed % 3, "wall_s": 0.01})


BEAMFORM_PARAMS = {"grid": "507x456", "f_number": 1.0, "k": 16, "code_git_sha": "deadbeef"}


def _shards(n_shards: int, frames_per_shard: int) -> list[ShardSpec]:
    return [
        ShardSpec(
            name=f"shard{s:03d}",
            frame_ids=[f"train/shard{s:03d}/{i:03d}" for i in range(frames_per_shard)],
        )
        for s in range(n_shards)
    ]


def _read_all_frame_bytes(out_dir: Path, shards: list[ShardSpec]) -> dict[str, bytes]:
    out = {}
    for shard in shards:
        for fid in shard.frame_ids:
            safe = fid.replace("/", "__")
            out[fid] = (out_dir / "frames" / f"{safe}.npy").read_bytes()
    return out


def test_fresh_build_writes_every_frame_and_completes_every_shard(tmp_path):
    shards = _shards(n_shards=3, frames_per_shard=4)
    manifest = build_store(shards, tmp_path, _deterministic_frame, BEAMFORM_PARAMS)

    assert manifest.completed_shards == {s.name for s in shards}
    assert set(manifest.frames) == {fid for s in shards for fid in s.frame_ids}
    for shard in shards:
        for fid in shard.frame_ids:
            arr = read_frame_chunk(tmp_path, fid)
            np.testing.assert_array_equal(arr, _deterministic_frame(fid).array)


def test_rerun_on_a_finished_store_is_a_no_op(tmp_path):
    shards = _shards(n_shards=2, frames_per_shard=3)
    build_store(shards, tmp_path, _deterministic_frame, BEAMFORM_PARAMS)
    before = _read_all_frame_bytes(tmp_path, shards)
    before_manifest_bytes = (tmp_path / "manifest.json").read_bytes()

    calls = []

    def tracking(fid):
        calls.append(fid)
        return _deterministic_frame(fid)

    build_store(shards, tmp_path, tracking, BEAMFORM_PARAMS)

    assert calls == [], "a re-run on a fully-completed store must not beamform anything"
    after = _read_all_frame_bytes(tmp_path, shards)
    assert after == before
    assert (tmp_path / "manifest.json").read_bytes() == before_manifest_bytes


def test_manifest_round_trips_beamform_params_and_per_frame_metadata(tmp_path):
    shards = _shards(n_shards=1, frames_per_shard=2)
    build_store(shards, tmp_path, _deterministic_frame, BEAMFORM_PARAMS)

    reloaded = load_manifest(tmp_path)
    assert reloaded.beamform_params == BEAMFORM_PARAMS
    for fid in shards[0].frame_ids:
        assert reloaded.frames[fid] == _deterministic_frame(fid).metadata


def test_load_manifest_returns_none_when_absent(tmp_path):
    assert load_manifest(tmp_path) is None


class _SimulatedCrash(RuntimeError):
    pass


def test_restart_after_mid_shard_crash_is_byte_identical_to_an_uninterrupted_run(tmp_path):
    shards = _shards(n_shards=3, frames_per_shard=4)  # 12 frames total, crash inside shard 1

    reference_dir = tmp_path / "reference"
    build_store(shards, reference_dir, _deterministic_frame, BEAMFORM_PARAMS)
    reference_frames = _read_all_frame_bytes(reference_dir, shards)
    reference_manifest = load_manifest(reference_dir)

    crash_dir = tmp_path / "crash_then_resume"
    call_count = {"n": 0}
    crash_after = 5  # partway through shard 1 (frames 0-3 belong to shard 0)

    def flaky_beamform(frame_id: str) -> FrameResult:
        call_count["n"] += 1
        if call_count["n"] == crash_after:
            raise _SimulatedCrash("simulated SIGKILL mid-shard")
        return _deterministic_frame(frame_id)

    with pytest.raises(_SimulatedCrash):
        build_store(shards, crash_dir, flaky_beamform, BEAMFORM_PARAMS)

    # the crash must have left a real partial manifest behind, mid-shard
    partial = load_manifest(crash_dir)
    assert 0 < len(partial.frames) < 12
    assert partial.completed_shards == {"shard000"}  # shard 0 finished before the crash

    # restart: resume=True (the default) picks up exactly where it left off
    resumed = build_store(shards, crash_dir, _deterministic_frame, BEAMFORM_PARAMS)

    assert resumed.completed_shards == {s.name for s in shards}
    resumed_frames = _read_all_frame_bytes(crash_dir, shards)
    msg = "resumed frame bytes differ from an uninterrupted run"
    assert resumed_frames == reference_frames, msg

    resumed_manifest = load_manifest(crash_dir)
    assert resumed_manifest.frames == reference_manifest.frames
    assert resumed_manifest.completed_shards == reference_manifest.completed_shards


def test_resume_false_forces_a_full_rebuild_even_with_an_existing_manifest(tmp_path):
    shards = _shards(n_shards=1, frames_per_shard=2)
    build_store(shards, tmp_path, _deterministic_frame, BEAMFORM_PARAMS)

    calls = []

    def counting(fid: str) -> FrameResult:
        calls.append(fid)
        return _deterministic_frame(fid)

    build_store(shards, tmp_path, counting, BEAMFORM_PARAMS, resume=False)
    assert calls == shards[0].frame_ids


def test_build_manifest_dict_round_trip():
    m = BuildManifest(
        beamform_params={"k": 16}, completed_shards={"a", "b"}, frames={"f1": {"t0": 1}}
    )
    round_tripped = BuildManifest.from_dict(m.to_dict())
    assert round_tripped.beamform_params == m.beamform_params
    assert round_tripped.completed_shards == m.completed_shards
    assert round_tripped.frames == m.frames
