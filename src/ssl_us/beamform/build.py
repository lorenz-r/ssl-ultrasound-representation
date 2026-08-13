"""Deterministic, resumable, parallel-safe build pipeline (PLAN.md WI-6).

Iterates shards; for each frame, beamforms it (via a caller-supplied
callback — the real WI-8 build passes the `AngularSectorSum` pipeline, tests
pass a fast synthetic stub), writes one frame chunk, and appends to the
manifest with an `fsync`. A restart re-reads the manifest and skips every
frame already recorded, at **frame granularity, not shard granularity** — a
crash mid-shard must not force re-beamforming the frames that shard already
finished, and re-beamforming them would risk producing slightly different
files (if the beamformer has any run-to-run nondeterminism), breaking the
byte-identical-after-restart guarantee this module exists to provide.

**Store format is swappable.** The real WI-8 store is zarr, per PLAN.md
§3.7 — a decision that depends on shapes/dtypes that do not exist until a
real frame has been beamformed. This module writes frames as individual
``.npy`` files instead, so the resumability logic (the actual subject of
WI-6) can be built and tested now; ``write_frame_chunk``/``read_frame_chunk``
are the two functions a zarr-backed store would replace.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ShardSpec:
    name: str
    frame_ids: list[str]


@dataclass
class FrameResult:
    array: np.ndarray
    metadata: dict  # e.g. {"t0": ..., "wall_s": ..., "peak_rss_mb": ...}


@dataclass
class BuildManifest:
    beamform_params: dict
    completed_shards: set[str] = field(default_factory=set)
    frames: dict[str, dict] = field(default_factory=dict)  # frame_id -> metadata

    def to_dict(self) -> dict:
        return {
            "beamform_params": self.beamform_params,
            "completed_shards": sorted(self.completed_shards),
            "frames": self.frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BuildManifest:
        return cls(
            beamform_params=d["beamform_params"],
            completed_shards=set(d["completed_shards"]),
            frames=dict(d["frames"]),
        )


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _write_manifest_atomic(out_dir: Path, manifest: BuildManifest) -> None:
    """Write + fsync, via a temp file + atomic rename, so a crash mid-write
    can never leave a half-written (corrupt) manifest on disk."""
    path = _manifest_path(out_dir)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest.to_dict(), f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest(out_dir: Path) -> BuildManifest | None:
    path = _manifest_path(out_dir)
    if not path.exists():
        return None
    return BuildManifest.from_dict(json.loads(path.read_text()))


def _frame_chunk_path(out_dir: Path, frame_id: str) -> Path:
    safe = frame_id.replace("/", "__")
    return out_dir / "frames" / f"{safe}.npy"


def write_frame_chunk(out_dir: Path, frame_id: str, array: np.ndarray) -> None:
    path = _frame_chunk_path(out_dir, frame_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_frame_chunk(out_dir: Path, frame_id: str) -> np.ndarray:
    return np.load(_frame_chunk_path(out_dir, frame_id))


def build_store(
    shards: list[ShardSpec],
    out_dir: Path,
    beamform_frame: Callable[[str], FrameResult],
    beamform_params: dict,
    resume: bool = True,
    write_frame: Callable[[Path, str, np.ndarray], None] | None = None,
) -> BuildManifest:
    """Build (or resume building) the store at ``out_dir``.

    Args:
        shards: shards to process, in order.
        out_dir: destination directory (created if absent).
        beamform_frame: ``frame_id -> FrameResult``. Called at most once per
            frame_id across the whole build, resumed or not.
        beamform_params: recorded verbatim into the manifest (grid, f-number,
            K, code git SHA, ... — PLAN.md WI-6's "beamforming parameters"
            requirement). Only written on a fresh build; an existing
            manifest's params are kept as the record of what actually
            produced the frames already on disk.
        resume: if ``True`` (default) and a manifest already exists, skip
            every already-completed shard and frame. If ``False``, start
            from empty even if a manifest exists (a full rebuild).
        write_frame: ``(out_dir, frame_id, array) -> None``, defaulting to
            :func:`write_frame_chunk`'s ``.npy`` writer. The real WI-8 build
            passes a zarr writer (:mod:`ssl_us.beamform.store`) here, so that
            the resumability logic this module exists to provide — which is
            tested against the ``.npy`` writer — is the same code path in both
            cases rather than being reimplemented alongside the real store.

    Returns:
        The final :class:`BuildManifest`.
    """
    write_frame = write_frame or write_frame_chunk
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out_dir) if resume else None
    if manifest is None:
        manifest = BuildManifest(beamform_params=beamform_params)
        _write_manifest_atomic(out_dir, manifest)

    for shard in shards:
        if shard.name in manifest.completed_shards:
            continue
        for frame_id in shard.frame_ids:
            if frame_id in manifest.frames:
                continue  # already written by a prior (possibly crashed) run
            result = beamform_frame(frame_id)
            write_frame(out_dir, frame_id, result.array)
            manifest.frames[frame_id] = result.metadata
            _write_manifest_atomic(out_dir, manifest)  # fsync after every frame
        manifest.completed_shards.add(shard.name)
        _write_manifest_atomic(out_dir, manifest)

    return manifest
