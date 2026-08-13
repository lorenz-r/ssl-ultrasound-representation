"""WI-8 final done-condition: whole-store integrity + corpus depth-RMS profile.

PLAN.md WI-8 asks for "the depth-RMS profile plotted for the whole corpus".
That profile is not decoration: PLAN.md §1.3 records that the old repo's
`normalize_frame` divided every fast-time sample by its own RMS, which is a
per-depth correction that silently deletes the attenuation signal — exactly
the axis §4.5c exists to probe. So the profile is computed and *stored as
metadata*, never divided out (:mod:`ssl_us.data.normalization` says the same).

This walks every frame once and, in the same pass, checks the store is whole:
all-finite, non-degenerate, and no frame left as zeros by a crashed build.

    python scripts/wi8_profile.py --store <store>.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ssl_us.beamform.store import open_store  # noqa: E402
from ssl_us.data import geometry as geo  # noqa: E402
from ssl_us.data import transforms  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out-fig", type=Path, default=Path("figures/wi8/depth_rms_profile.png"))
    args = p.parse_args()

    store = open_store(args.store)
    occ = store.occupancy[None]
    n = len(store)
    print(f"walking {n} frames in {args.store}")

    z_mm = geo.bmode_z_axis_mm()
    profiles = np.zeros((n, len(z_mm)), dtype=np.float64)
    frame_rms = np.zeros(n, dtype=np.float64)
    bad: list[str] = []

    for i in range(n):
        f = store.read_frame(i)
        fid = store.frame_ids[i]
        if not np.isfinite(f).all():
            bad.append(f"{fid}: non-finite values")
        elif not f.any():
            bad.append(f"{fid}: all zeros (frame never written?)")
        # K=1 envelope: the physically meaningful amplitude at each depth.
        k1 = transforms.reduce_and_normalize(f, occ, target_k=1, axis=-1)
        env = np.sqrt(k1[0, ..., 0] ** 2 + k1[1, ..., 0] ** 2)
        profiles[i] = np.sqrt((env.astype(np.float64) ** 2).mean(axis=1))
        frame_rms[i] = float(np.sqrt((env.astype(np.float64) ** 2).mean()))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    median = np.median(profiles, axis=0)
    p10, p90 = np.percentile(profiles, [10, 90], axis=0)

    # How much amplitude is lost shallow -> deep. PLAN.md 1.3 notes raw IQ
    # decays ~2 orders of magnitude; this is the beamformed equivalent, and the
    # number the depth probes are entitled to see.
    lo = median[(z_mm >= 5) & (z_mm <= 10)].mean()
    hi = median[(z_mm >= 30) & (z_mm <= 38)].mean()
    decay_db = 20 * np.log10(lo / max(hi, 1e-30))
    print(f"\ndepth decay 5-10 mm vs 30-38 mm: {decay_db:.1f} dB")
    print(f"dataset RMS (mean over frames): {frame_rms.mean():.4g}")

    if bad:
        print(f"\nINTEGRITY FAILURES ({len(bad)}):")
        for b in bad[:20]:
            print(f"  {b}")
    else:
        print(f"\nintegrity OK: {n}/{n} frames finite and non-empty")

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].fill_between(z_mm, p10, p90, alpha=0.25, label="10-90th pct")
    axes[0].plot(z_mm, median, lw=1.6, label="median")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("depth z [mm]")
    axes[0].set_ylabel("K=1 envelope RMS (a.u.)")
    axes[0].set_title(
        f"Corpus depth-RMS profile ({n} frames)\n"
        f"{decay_db:.1f} dB decay 5-10 -> 30-38 mm"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(frame_rms, bins=40)
    axes[1].set_xlabel("per-frame RMS (a.u.)")
    axes[1].set_ylabel("frames")
    axes[1].set_title("Per-frame amplitude spread")
    axes[1].grid(alpha=0.3)

    fig.suptitle("WI-8 — measured, NOT divided out (PLAN.md 1.3 / 4.5c)")
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out_fig}")

    out = Path("tests/data/wi8_depth_profile.json")
    out.write_text(
        json.dumps(
            {
                "n_frames": n,
                "z_mm": z_mm.tolist(),
                "median_rms": median.tolist(),
                "p10_rms": p10.tolist(),
                "p90_rms": p90.tolist(),
                "depth_decay_db": float(decay_db),
                "dataset_rms_mean": float(frame_rms.mean()),
                "integrity_failures": bad,
            },
            indent=1,
        )
    )
    print(f"wrote {out}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
