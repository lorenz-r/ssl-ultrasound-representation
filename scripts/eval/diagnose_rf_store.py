"""Measure whether an RF/IQ store is predictable enough for masked reconstruction.

A masked autoencoder can only recover a gap if the signal is correlated across
it. This reports the two things that decide that, before any training:

* **fast-time autocorrelation** — how far along a trace correlation survives. If
  |rho| has decayed to noise by lag `l_min`, the centre of every mask run is
  statistically independent of the nearest visible sample, the conditional mean
  is zero, and an MAE that outputs zero is behaving *correctly*. That looks
  identical to a broken model (mse == 1.0, rho ~ 0) and is not fixable by tuning.
* **per-depth RMS profile** — `train.normalize_frame` divides each depth by its
  RMS across (transducer, scanline, IQ). A depth that is near-silent (simulated
  data before the first arrival, or past the modelled extent) has that dust
  amplified to unit variance, injecting pure noise into a slice of every trace.

Run it on two stores and compare — the corpus that trains and the one that does
not — rather than reading either in isolation.

    python scripts/eval/diagnose_rf_store.py --data-dir "$NV_STORE" \
        --data-metadata "$NV_META" --frames 8 --out results/diag/nv.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssl_ultrasound_representation.dataset.dataset import load_local_dataset


def _complex(frame: np.ndarray) -> np.ndarray:
    """(transducer, scanline, 2, sample) -> complex (transducer, scanline, sample)."""
    return frame[:, :, 0, :] + 1j * frame[:, :, 1, :]


def _autocorr(z: np.ndarray, axis: int, max_lag: int) -> list[float]:
    """Mean |rho(lag)| along `axis`, averaged over every other position.

    Mean-removed per series, so this is correlation rather than a DC artefact.
    """
    z = np.moveaxis(z, axis, -1)
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(axis=-1, keepdims=True)
    denom = np.sqrt((np.abs(z) ** 2).sum(axis=-1))
    keep = denom > 0
    z, denom = z[keep], denom[keep]
    out = []
    for lag in range(1, max_lag + 1):
        if lag >= z.shape[-1]:
            break
        a, b = z[:, :-lag], z[:, lag:]
        num = np.abs((np.conj(a) * b).sum(axis=-1))
        da = np.sqrt((np.abs(a) ** 2).sum(axis=-1))
        db = np.sqrt((np.abs(b) ** 2).sum(axis=-1))
        ok = (da > 0) & (db > 0)
        out.append(float(np.mean(num[ok] / (da[ok] * db[ok]))) if ok.any() else float("nan"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, dest="data_dir",
                   help=".zarr directory or .zarr.zip")
    p.add_argument("--data-metadata", required=True, dest="data_metadata")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--focal-zone", type=int, default=0, dest="focal_zone")
    p.add_argument("--max-lag", type=int, default=40, dest="max_lag")
    p.add_argument("--l-min", type=int, default=8, dest="l_min")
    p.add_argument("--out", default="")
    args = p.parse_args()

    arr, _ = load_local_dataset(args.data_dir, args.data_metadata)
    print(f"[diag] store {tuple(arr.shape)} dtype={arr.dtype}")
    n = min(args.frames, arr.shape[0])

    raw_profiles, norm_profiles = [], []
    fast_ac, lat_ac = [], []
    for i in range(n):
        frame = np.asarray(
            arr[i, :, :, args.focal_zone] if arr.ndim >= 6 else arr[i], dtype=np.float64
        )  # (transducer, scanline, 2, sample)
        # Per-depth RMS over (transducer, scanline, IQ) — exactly normalize_frame.
        rms = np.sqrt((frame ** 2).mean(axis=(0, 1, 2)))
        raw_profiles.append(rms)
        normed = frame / (rms[None, None, None, :] + 1e-8)
        norm_profiles.append(np.sqrt((normed ** 2).mean(axis=(0, 1, 2))))
        z = _complex(normed)
        fast_ac.append(_autocorr(z, axis=2, max_lag=args.max_lag))
        lat_ac.append(_autocorr(z, axis=0, max_lag=min(args.max_lag, z.shape[0] - 1)))

    rms = np.concatenate([r[None] for r in raw_profiles]).mean(axis=0)
    med = float(np.median(rms))
    dead = int((rms < med * 1e-3).sum())
    print(f"\n[diag] per-depth RMS over {len(rms)} depths (raw):")
    for q in (0, 1, 5, 25, 50, 75, 100):
        print(f"    p{q:<3d} {np.percentile(rms, q):.3e}")
    print(f"    depths below 1e-3 x median: {dead} / {len(rms)}"
          + ("  <-- normalization amplifies these to unit variance" if dead else ""))

    fast = np.nanmean(np.array(fast_ac), axis=0)
    lat = np.nanmean(np.array(lat_ac), axis=0)
    print("\n[diag] fast-time |rho| vs lag (after normalization):")
    for lag in (1, 2, 4, 8, 16, 32):
        if lag <= len(fast):
            print(f"    lag {lag:<3d} {fast[lag - 1]:.3f}")
    print("\n[diag] lateral |rho| vs lag:")
    for lag in (1, 2, 4, 8, 16):
        if lag <= len(lat):
            print(f"    lag {lag:<3d} {lat[lag - 1]:.3f}")

    at_lmin = float(fast[args.l_min - 1]) if args.l_min <= len(fast) else float("nan")
    half = float(fast[max(args.l_min // 2 - 1, 0)]) if fast.size else float("nan")
    print(
        f"\n[diag] VERDICT: at the shortest mask (l_min={args.l_min}) the gap centre sits "
        f"{args.l_min // 2} samples from visible signal, where fast-time |rho| = {half:.3f}; "
        f"at a full l_min gap |rho| = {at_lmin:.3f}."
    )
    if half < 0.1:
        print("    Fast-time correlation is gone well inside the shortest mask. Nothing in "
              "the trace predicts the gap, so the conditional mean is ~0 and mse~1.0 is the "
              "CORRECT output. Shorten the masks or mask a different axis — tuning will not help.")
    elif half < 0.3:
        print("    Weak. Expect a low correlation ceiling; try a smaller --l-min.")
    else:
        print("    There is exploitable fast-time structure at this mask length.")
    if lat.size and lat[0] > max(fast[0], 1e-9):
        print(f"    Lateral |rho| at lag 1 ({lat[0]:.3f}) exceeds fast-time ({fast[0]:.3f}) — "
              "cross-aperture masking (--per-column-masks) is the better-posed task here.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "store": str(Path(args.data_dir).resolve()),
            "shape": list(arr.shape), "frames_sampled": n,
            "depth_rms_raw": rms.tolist(),
            "depths_below_1e-3_median": dead,
            "fast_time_autocorr": fast.tolist(),
            "lateral_autocorr": lat.tolist(),
            "l_min": args.l_min,
            "rho_at_half_lmin": half, "rho_at_lmin": at_lmin,
        }, indent=1))
        print(f"\n[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
