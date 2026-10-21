"""Beamform the Stanford FSA transfer corpus to the shared K-sector
store format, one grid variant at a time (WI-16, PLAN.md §3.6).

Unlike WI-8's NVIDIA build, there is no download or streaming here — the
source (`/mnt/home/benbarkow/stores/wi16_stanford_fsa.zarr`, built by
`scripts/data/fetch_stanford_fsa.py`) is already local and randomly
accessible, so this is a plain per-frame beamform-and-write loop, reusing
WI-6's `build_store` only for its resumability/manifest bookkeeping (not for
the forward-stream adapter WI-8 needed).

Run both grid variants — the PLAN.md 3.6 ablation:

    source /opt/venvs/.venv-zea/bin/activate
    KERAS_BACKEND=jax python scripts/beamform/build_stanford_store.py --variant mm_matched \\
        --out /mnt/home/benbarkow/stores/wi16_mm_matched_k16.zarr
    KERAS_BACKEND=jax python scripts/beamform/build_stanford_store.py --variant lambda_matched \\
        --out /mnt/home/benbarkow/stores/wi16_lambda_matched_k16.zarr

    # smoke run first — a few frames, same code path end to end:
    KERAS_BACKEND=jax python scripts/beamform/build_stanford_store.py --variant mm_matched \\
        --out /tmp/smoke.zarr --limit 4

Grid size (see `ssl_us.data.stanford_geometry`): mm-matched is ~253x484
pixels, lambda-matched ~304x582 — both well under NVIDIA's 507x456, so this
should run faster per frame than the WI-8 build despite having more
transmits (192 vs 180).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402

from ssl_us.beamform import sectors as sec  # noqa: E402
from ssl_us.beamform import stanford_frame as sf  # noqa: E402
from ssl_us.beamform.build import BuildManifest, FrameResult, ShardSpec, build_store  # noqa: E402
from ssl_us.beamform.store import SectorStore, StoreSpec, open_store  # noqa: E402
from ssl_us.data import stanford_geometry as sgeo  # noqa: E402
from ssl_us.data.stanford_fsa import DEFAULT_PATH as STANFORD_FSA_DEFAULT  # noqa: E402
from ssl_us.data.stanford_fsa import open_fsa_store  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="destination .zarr directory")
    p.add_argument("--variant", choices=sgeo.GRID_VARIANTS, required=True)
    p.add_argument("--source", type=Path, default=STANFORD_FSA_DEFAULT)
    p.add_argument("--k", type=int, default=sf.DEFAULT_K)
    p.add_argument("--f-number", type=float, default=sf.F_NUMBER)
    p.add_argument("--limit", type=int, default=None, help="stop after N frames (smoke runs)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--num-patches",
        type=int,
        default=None,
        help="grid chunking for beamforming; lower it if you hit OOM",
    )
    args = p.parse_args()

    fsa_store = open_fsa_store(args.source)
    grid = sgeo.build_grid_variant(args.variant, fsa_store.attrs)
    n_frames = len(fsa_store) if args.limit is None else min(args.limit, len(fsa_store))
    frame_ids = [str(i) for i in range(n_frames)]

    beamform_params = {
        "beamformer": "angular_sector_sum",
        "k": args.k,
        "f_number": args.f_number,
        "t_peak": sf.T_PEAK,
        "grid_variant": args.variant,
        "grid_spacing_mm": grid.spacing_mm,
        "grid_nz": len(grid.z_mm),
        "grid_nx": len(grid.x_mm),
        "z_min_mm": float(grid.z_mm[0]),
        "z_max_mm": float(grid.z_mm[-1]),
        "x_min_mm": float(grid.x_mm[0]),
        "x_max_mm": float(grid.x_mm[-1]),
        "sound_speed_m_s": fsa_store.attrs["c_m_s"],
        "t0_s": fsa_store.attrs["t0_s"],
        "normalized": False,
        "source_store": str(args.source),
        "git_sha": _git_sha(),
    }

    out_dir = Path(args.out)
    manifest_dir = out_dir.parent / f"{out_dir.name}.build"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and not args.no_resume:
        store = open_store(out_dir, mode="r+")
        print(f"resuming into existing store with {len(store)} frame slots")
    else:
        geometry = sec.build_sector_geometry(
            k=args.k,
            f_number=args.f_number,
            z_mm=grid.z_mm,
            x_mm=grid.x_mm,
            n_elements=int(fsa_store.attrs["n_elem"]),
            pitch_m=float(fsa_store.attrs["pitch_m"]),
        )
        spec = StoreSpec(n_frames=n_frames, nz=len(grid.z_mm), nx=len(grid.x_mm), k=args.k)
        store = SectorStore.create(
            out_dir, spec, frame_ids, beamform_params, geometry=geometry, overwrite=True
        )
        print(
            f"created store: {spec.n_frames} x {spec.frame_shape} float32 "
            f"({args.variant}, spacing {grid.spacing_mm:.4f} mm)"
        )

    metadata = store.read_frame_metadata()

    def beamform_one(frame_id: str) -> FrameResult:
        t = time.time()
        index = int(frame_id)
        array = sf.beamform_frame(
            fsa_store, index, grid, k=args.k, f_number=args.f_number, num_patches=args.num_patches
        )
        metadata[frame_id] = sf.frame_labels(fsa_store, index)
        store.write_frame_metadata(metadata)
        wall = round(time.time() - t, 2)
        print(f"  {frame_id}: {array.shape} in {wall}s", flush=True)
        return FrameResult(array=array, metadata={"wall_s": wall})

    def write_frame(_out_dir: Path, frame_id: str, array: np.ndarray) -> None:
        store.write_frame(int(frame_id), array)

    manifest: BuildManifest = build_store(
        shards=[ShardSpec(name="all", frame_ids=frame_ids)],
        out_dir=manifest_dir,
        beamform_frame=beamform_one,
        beamform_params=beamform_params,
        resume=not args.no_resume,
        write_frame=write_frame,
    )

    store.write_frame_metadata(metadata)
    (manifest_dir / "beamform_params.json").write_text(json.dumps(beamform_params, indent=1))
    built = len(manifest.frames)
    print(f"\n{built}/{len(frame_ids)} frames in manifest")
    print(f"store:    {out_dir}")
    print(f"manifest: {manifest_dir / 'manifest.json'}")
    if built < len(frame_ids):
        print("INCOMPLETE — re-run the same command to resume from the manifest.")


if __name__ == "__main__":
    main()
