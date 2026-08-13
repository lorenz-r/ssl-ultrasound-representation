"""WI-8: build the full K-sector store from NV-Raw2Insights-US (PLAN.md §5.2).

Streams the dataset, beamforms every frame with `AngularSectorSum` at the
settings WI-4's gate validated (see :mod:`ssl_us.beamform.frame` — f#=1.5,
t_peak=0, 32-sample near-field mute), and writes raw ``(2, 507, 456, 16)``
sector sums into the zarr store of PLAN.md §3.7. 923 frames, ~27.3 GB.

Resumable at **frame** granularity via WI-6's manifest (`build.build_store`):
a crash or a preemption re-runs only the frames not already recorded, and
never re-beamforms one that is. That matters here because the job is 6-10 h
wall, dominated by streaming 271 GB rather than by the ~3 s/frame of GPU.

Usage (on a `12g` or `24g` instance, in the zea venv):

    source /opt/venvs/.venv-zea/bin/activate
    KERAS_BACKEND=jax python scripts/wi8_build.py --out /path/to/store.zarr

    # smoke run first — a few frames, same code path end to end:
    KERAS_BACKEND=jax python scripts/wi8_build.py --out /tmp/smoke.zarr --limit 4

Pick `--out` on a filesystem with ~30 GB free that survives the job; note the
NFS home on this cluster is quota'd (see CLUSTER_HANDOFF.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
from datasets import load_dataset, load_dataset_builder  # noqa: E402
from datasets.config import HF_DATASETS_CACHE  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

from ssl_us.beamform import frame as bf  # noqa: E402
from ssl_us.beamform import sectors as sec  # noqa: E402
from ssl_us.beamform.build import BuildManifest, FrameResult, ShardSpec, build_store  # noqa: E402
from ssl_us.beamform.store import SectorStore, StoreSpec, open_store  # noqa: E402
from ssl_us.data import geometry as geo  # noqa: E402

HF_REPO = "nvidia/NV-Raw2Insights-US"
# PLAN.md 2.x: 830 train + 93 validation = 923.
SPLITS = ("train", "validation")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def split_sizes(splits=SPLITS) -> dict[str, int]:
    """Frame count per split, from the dataset's own metadata.

    Read from the builder info rather than by counting the stream: the store
    has to be created with its final ``n_frames`` *before* the first frame is
    written, and walking the stream to count would download all 271 GB twice.
    """
    info = load_dataset_builder(HF_REPO).info
    return {s: int(info.splits[s].num_examples) for s in splits}


def stream_frames(splits=SPLITS, limit: int | None = None):
    """Yield ``(frame_id, sample)`` via HuggingFace's streaming reader.

    Kept for reference and as a fallback, but **~25x slower than
    :func:`shard_frames`** on this cluster: measured ~2.7 MB/s streaming
    against 68-76 MB/s downloading the same shards outright. Streaming is the
    bottleneck, not the network.
    """
    seen = 0
    for split in splits:
        ds = load_dataset(HF_REPO, split=split, streaming=True).with_format("numpy")
        for i, sample in enumerate(ds):
            if limit is not None and seen >= limit:
                return
            yield f"{split}/{i}", sample
            seen += 1


def shard_frames(splits=SPLITS, limit: int | None = None, keep_shards: bool = False):
    """Yield ``(frame_id, sample)`` by downloading one parquet shard at a time.

    ``frame_id`` is ``"{split}/{index}"`` — the convention PLAN.md WI-10's
    leakage assertion keys on, so the split a frame came from is never lost.
    Indices count rows within a split in shard order, which reproduces exactly
    the order the streaming reader yields.

    Why not stream: measured 2.7 MB/s streamed vs ~70 MB/s for a plain shard
    download of the same data, i.e. ~28 h vs ~1 h for the 271 GB corpus. The
    file is decoded through `datasets` all the same, just from local disk, so
    every column arrives with the same shapes and dtypes streaming produced.

    Disk stays bounded: each shard is deleted once its frames are consumed
    (``keep_shards=True`` disables that, for repeated local runs). Peak usage
    is one shard (~5.4 GB for train) plus its arrow cache, not 271 GB.
    """
    seen = 0
    for split in splits:
        shards = sorted(
            f
            for f in list_repo_files(HF_REPO, repo_type="dataset")
            if f.endswith(".parquet") and f"/{split}-" in f"/{f.split('/')[-1]}"
        )
        index = 0
        for shard in shards:
            if limit is not None and seen >= limit:
                return
            local = hf_hub_download(HF_REPO, shard, repo_type="dataset")
            ds = load_dataset("parquet", data_files=local, split="train").with_format("numpy")
            try:
                for row in ds:
                    if limit is not None and seen >= limit:
                        return
                    yield f"{split}/{index}", row
                    index += 1
                    seen += 1
            finally:
                del ds
                if not keep_shards:
                    # Free the shard *and* the arrow cache datasets built from
                    # it; otherwise the 271 GB arrives on disk after all.
                    Path(local).unlink(missing_ok=True)
                    shutil.rmtree(HF_DATASETS_CACHE, ignore_errors=True)


class OrderedFrameSource:
    """Pull-based adapter from the one-pass stream to ``build_store``'s
    ``beamform_frame(frame_id)`` callback.

    ``build_store`` requests frames in the order they appear in the shard list
    and simply *skips* ones already recorded in the manifest. The dataset is a
    forward-only stream, so this advances the iterator to the requested
    ``frame_id``, discarding (without beamforming) anything skipped — which is
    exactly the resume case. Requests must therefore be non-decreasing in
    stream order; anything else means the caller's ordering assumption broke,
    and that raises rather than silently beamforming the wrong frame.
    """

    def __init__(self, iterator):
        self._it = iterator
        self._exhausted = False

    def take(self, frame_id: str) -> dict:
        for fid, sample in self._it:
            if fid == frame_id:
                return sample
            # Skipped: already built in a previous run (or excluded). Dropping
            # the sample here is what makes resume cost only download, not GPU.
        self._exhausted = True
        raise KeyError(
            f"{frame_id!r} not found in the remaining stream — frames must be "
            "requested in stream order"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="destination .zarr directory")
    p.add_argument("--k", type=int, default=bf.DEFAULT_K)
    p.add_argument("--f-number", type=float, default=bf.F_NUMBER)
    p.add_argument("--limit", type=int, default=None, help="stop after N frames (smoke runs)")
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--num-patches",
        type=int,
        default=None,
        help="grid chunking for beamforming; lower it if you hit OOM",
    )
    p.add_argument(
        "--source",
        choices=("shard", "stream"),
        default="shard",
        help="'shard' downloads parquet shards (~70 MB/s here); 'stream' uses "
        "the HF streaming reader (~2.7 MB/s measured) — fallback only",
    )
    p.add_argument(
        "--keep-shards",
        action="store_true",
        help="don't delete each parquet shard after use (needs 271 GB free)",
    )
    args = p.parse_args()

    beamform_params = {
        "beamformer": "angular_sector_sum",
        "k": args.k,
        "f_number": args.f_number,
        "t_peak": bf.T_PEAK,
        "mute_samples": bf.MUTE_SAMPLES,
        "grid_nz": geo.BMODE_NZ,
        "grid_nx": geo.BMODE_NX,
        "sound_speed_m_s": geo.C0_M_S,
        "normalized": False,
        "source_dataset": HF_REPO,
        "splits": list(args.splits),
        "git_sha": _git_sha(),
        "validated_by": "WI-4 gate, tests/data/wi4_results.json (8/8 frames pass)",
    }

    # Single pass over the stream. Frame ids and counts come from the dataset's
    # metadata rather than from walking it, so the store can be allocated at
    # full size up front and each 265 MB sample is downloaded exactly once.
    out_dir = Path(args.out)
    manifest_dir = out_dir.parent / f"{out_dir.name}.build"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    state: dict = {"store": None, "metadata": {}, "index": {}}

    if out_dir.exists() and not args.no_resume:
        state["store"] = open_store(out_dir, mode="r+")
        state["index"] = {fid: i for i, fid in enumerate(state["store"].frame_ids)}
        state["metadata"] = state["store"].read_frame_metadata()
        print(f"resuming into existing store with {len(state['store'])} frame slots")

    sizes = split_sizes(tuple(args.splits))
    frame_ids = [f"{s}/{i}" for s in args.splits for i in range(sizes[s])]
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]
    print(f"{len(frame_ids)} frames to build ({sizes}) -> {out_dir} [source={args.source}]")
    iterator = (
        shard_frames(tuple(args.splits), limit=args.limit, keep_shards=args.keep_shards)
        if args.source == "shard"
        else stream_frames(tuple(args.splits), limit=args.limit)
    )
    source = OrderedFrameSource(iterator)

    if state["store"] is None:
        geometry = sec.build_sector_geometry(k=args.k, f_number=args.f_number)
        spec = StoreSpec(n_frames=len(frame_ids), k=args.k)
        state["store"] = SectorStore.create(
            out_dir, spec, frame_ids, beamform_params, geometry=geometry, overwrite=True
        )
        state["index"] = {fid: i for i, fid in enumerate(frame_ids)}
        print(f"created store: {spec.n_frames} x {spec.frame_shape} float32")

    def beamform_frame(frame_id: str) -> FrameResult:
        t = time.time()
        sample = source.take(frame_id)
        array = bf.beamform_sample(
            sample, k=args.k, f_number=args.f_number, num_patches=args.num_patches
        )
        state["metadata"][frame_id] = bf.frame_labels(sample)
        # Persisted every frame, not once at the end: this job runs for hours
        # and the labels are unrecoverable without re-streaming 265 MB apiece,
        # so they must survive a crash exactly as the frame data does.
        state["store"].write_frame_metadata(state["metadata"])
        wall = round(time.time() - t, 2)
        print(f"  {frame_id}: {array.shape} in {wall}s", flush=True)
        return FrameResult(array=array, metadata={"wall_s": wall, "t0": float(sample["t0"])})

    def write_frame(_out_dir: Path, frame_id: str, array: np.ndarray) -> None:
        state["store"].write_frame(state["index"][frame_id], array)

    manifest: BuildManifest = build_store(
        shards=[ShardSpec(name=s, frame_ids=[f for f in frame_ids if f.startswith(f"{s}/")])
                for s in args.splits],
        out_dir=manifest_dir,
        beamform_frame=beamform_frame,
        beamform_params=beamform_params,
        resume=not args.no_resume,
        write_frame=write_frame,
    )

    state["store"].write_frame_metadata(state["metadata"])
    (manifest_dir / "beamform_params.json").write_text(json.dumps(beamform_params, indent=1))
    built = len(manifest.frames)
    print(f"\n{built}/{len(frame_ids)} frames in manifest")
    print(f"store:    {out_dir}")
    print(f"manifest: {manifest_dir / 'manifest.json'}")
    if built < len(frame_ids):
        print("INCOMPLETE — re-run the same command to resume from the manifest.")

    # Everything above is written and fsync'd. The HuggingFace streaming reader
    # keeps fsspec/aiohttp threads alive that can wedge interpreter shutdown on
    # a long run, so exit without waiting on them rather than leaving a job that
    # looks hung after it has actually finished.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if built == len(frame_ids) else 1)


if __name__ == "__main__":
    main()
