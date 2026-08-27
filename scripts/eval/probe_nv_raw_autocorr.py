"""Does the ×4 decimation whiten NV, or is the raw IQ already white?

The converted store shows fast-time |rho| at the 1/sqrt(N) noise floor for every
lag, so masked fast-time reconstruction cannot work on it. That has two very
different causes with very different fixes:

* the raw 1024-sample IQ is band-limited and `--sample-stride 4` aliased it
  -> rebuild at stride 1 or 2 (122-245 GB, hours)
* the raw IQ is already white along fast-time
  -> no rebuild helps; fast-time masking is the wrong pretext for this corpus

This reads a handful of raw rows straight from the Parquet and measures the
autocorrelation at several strides, so the 245 GB decision costs a few minutes.
Traces are subsampled (`--max-traces`) because the estimate needs a few thousand
series, not 32400 of them.

    python scripts/eval/probe_nv_raw_autocorr.py --nv-root "$NV_RAW_ROOT" --frames 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssl_ultrasound_representation.dataset.nv_raw2insights import (
    iter_nv_raw2insights_samples,
)


def autocorr(z: np.ndarray, max_lag: int) -> list[float]:
    """Mean |rho(lag)| along the last axis over mean-removed series."""
    z = z - z.mean(axis=-1, keepdims=True)
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
    p.add_argument("--nv-root", required=True, dest="nv_root")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--max-traces", type=int, default=2000, dest="max_traces")
    p.add_argument("--max-lag", type=int, default=16, dest="max_lag")
    p.add_argument("--out", default="")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    per_stride: dict[int, list[list[float]]] = {s: [] for s in args.strides}
    n_raw = None

    it = iter_nv_raw2insights_samples(
        args.nv_root, split=args.split, columns=("iq_real", "iq_imag")
    )
    for i, sample in enumerate(it):
        if i >= args.frames:
            break
        re = np.asarray(sample["iq_real"], dtype=np.float64)
        im = np.asarray(sample["iq_imag"], dtype=np.float64)
        z = (re + 1j * im).reshape(-1, re.shape[-1])  # (tx*rx, sample)
        n_raw = z.shape[-1]
        if z.shape[0] > args.max_traces:
            z = z[rng.choice(z.shape[0], args.max_traces, replace=False)]
        for s in args.strides:
            per_stride[s].append(autocorr(z[:, ::s], args.max_lag))
        print(f"  frame {i}: {z.shape[0]} traces x {n_raw} raw samples", flush=True)

    print(f"\n[raw] fast-time |rho| vs lag, by decimation stride "
          f"(raw n_fast={n_raw}):")
    summary = {}
    for s in args.strides:
        ac = np.nanmean(np.array(per_stride[s]), axis=0)
        n_eff = (n_raw or 0) // s
        floor = 1.0 / np.sqrt(max(n_eff, 1))
        summary[s] = {"autocorr": ac.tolist(), "n_fast": n_eff, "noise_floor": float(floor)}
        lags = [1, 2, 4, 8]
        cells = "  ".join(
            f"lag{l}={ac[l - 1]:.3f}" for l in lags if l <= len(ac)
        )
        verdict = "WHITE (at the noise floor)" if ac[0] < 2 * floor else "correlated"
        print(f"  stride {s} (n={n_eff:5d}, floor={floor:.3f}):  {cells}   -> {verdict}")

    s_lo, s_hi = min(args.strides), max(args.strides)
    lo, hi = summary[s_lo], summary[s_hi]
    print()
    if lo["autocorr"][0] > 2 * lo["noise_floor"] >= 0 and hi["autocorr"][0] < 2 * hi["noise_floor"]:
        print(f"[raw] VERDICT: stride {s_lo} is correlated but stride {s_hi} is not — the "
              f"decimation is what whitened the store. Rebuild with --sample-stride {s_lo}.")
    elif lo["autocorr"][0] < 2 * lo["noise_floor"]:
        print(f"[raw] VERDICT: even at stride {s_lo} (no decimation) the raw IQ is at its "
              f"noise floor along fast-time. A rebuild will NOT help — fast-time masking is "
              f"the wrong pretext for this corpus. Mask across the aperture instead.")
    else:
        print("[raw] VERDICT: inconclusive — inspect the per-lag numbers above.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"nv_root": str(args.nv_root), "split": args.split,
             "frames": args.frames, "raw_n_fast": n_raw,
             "by_stride": {str(k): v for k, v in summary.items()}}, indent=1))
        print(f"[raw] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
