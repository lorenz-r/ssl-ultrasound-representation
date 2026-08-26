"""Report the NV store's fast-time sampling, and whether the mask lengths fit it.

`--l-min` defaults to 8, which was derived from the MultiFocal rat corpus where
`fs == fc` — exactly one IQ sample per carrier period. NV is a different
acquisition decimated by `--sample-stride`, so that floor does not automatically
carry over. A mask run shorter than the pulse is filled by local interpolation
and the pretext task becomes trivial.

Advisory only: always exits 0, so a warning never kills the job it runs inside.

    python scripts/eval/check_nv_sampling.py "$NV_BF" --l-min 8 --l-max 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("params", help="*.beamform_params.json written by the NV builder")
    p.add_argument("--l-min", type=int, default=8, dest="l_min")
    p.add_argument("--l-max", type=int, default=32, dest="l_max")
    args = p.parse_args()

    path = Path(args.params)
    if not path.is_file():
        print(f"[sampling] no params at {path} — skipping the check")
        return 0
    try:
        cfg = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[sampling] could not read {path} ({exc}) — skipping the check")
        return 0

    fs, fc = cfg.get("fs"), cfg.get("fc")
    if not fs or not fc:
        print(f"[sampling] {path} has no fs/fc — skipping the check")
        return 0

    # The builder already divides fs by sample_stride, so this ratio is samples
    # per carrier period at the rate actually stored in the zarr.
    ratio = fs / fc
    print(
        f"[sampling] fs={fs / 1e6:.3f} MHz  fc={fc / 1e6:.3f} MHz  "
        f"-> {ratio:.2f} samples/carrier period  "
        f"n_samples={cfg.get('n_samples')}  stride={cfg.get('sample_stride', '?')}  "
        f"acquisition={cfg.get('acquisition_type', '?')}"
    )
    print(
        f"[sampling] masks span {args.l_min / ratio:.1f}-{args.l_max / ratio:.1f} "
        f"carrier periods (l_min={args.l_min}, l_max={args.l_max})"
    )
    if ratio > 1.5:
        print(
            f"[sampling] WARNING: at {ratio:.2f} samples/period, l_min={args.l_min} "
            f"covers only {args.l_min / ratio:.1f} periods. Consider "
            f"--l-min {round(args.l_min * ratio)} --l-max {round(args.l_max * ratio)} "
            f"so a run is at least as long as the pulse."
        )
    else:
        print("[sampling] OK: l_min spans at least one carrier period.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
