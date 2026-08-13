"""WI-8 acceptance check: re-run WI-4's gate on frames read back *from the store*.

PLAN.md WI-8's done-condition is not just "923 files exist" — it is that
WI-4's validation still passes on frames pulled out of the finished store.
That is a different claim from WI-4's own run: it exercises the store's
layout, the raw-sums contract, and the K-ladder reduction, any of which could
silently corrupt an image that was correct when it left the beamformer.

So this reduces each stored frame from K=16 to K=1 via the *documented* read
path (`transforms.reduce_and_normalize` — reduce, then normalize, never the
reverse) and scores the result against NVIDIA's `bmode` with the same
speckle-invariant metrics WI-4's gate uses.

    KERAS_BACKEND=jax python scripts/wi8_verify.py --store /tmp/store.zarr --n 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402

from ssl_us.beamform import image_quality as iqm  # noqa: E402
from ssl_us.beamform.store import open_store  # noqa: E402
from ssl_us.data import transforms  # noqa: E402

GCNR_VS_REF_TOLERANCE = 0.10
SPECKLE_SNR_RANGE = (1.60, 2.15)
SPECKLE_BAND_MM = (12.0, 30.0)
NEGATIVE_CONTROL_MIN_REF_GCNR = 0.30


def to_db(x: np.ndarray) -> np.ndarray:
    db = 20 * np.log10(np.maximum(np.asarray(x, dtype=np.float32), 1e-9))
    return db - db.max()


def grow(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(n):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def k1_log_envelope(store, index: int) -> np.ndarray:
    """Store -> K=1 log-envelope, via the documented read path."""
    raw = store.read_frame(index)  # (2, Z, X, K) raw sums
    k1 = transforms.reduce_and_normalize(raw, store.occupancy[None], target_k=1, axis=-1)
    env = np.sqrt(k1[0, ..., 0] ** 2 + k1[1, ..., 0] ** 2)
    return to_db(env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--n", type=int, default=10, help="frames to check")
    p.add_argument("--split", default="validation")
    args = p.parse_args()

    store = open_store(args.store)
    ids = [f for f in store.frame_ids if f.startswith(f"{args.split}/")][: args.n]
    if not ids:
        raise SystemExit(f"no {args.split} frames in {args.store}")
    wanted = {int(f.split("/")[1]): f for f in ids}
    print(f"checking {len(wanted)} frames from {args.store}")

    ds = load_dataset(
        "nvidia/NV-Raw2Insights-US", split=args.split, streaming=True
    ).with_format("numpy")

    results, n_pass = [], 0
    for i, sample in enumerate(ds):
        if not wanted:
            break
        if i not in wanted:
            continue
        frame_id = wanted.pop(i)

        ours_db = k1_log_envelope(store, store.index_of(frame_id))
        ref_db = to_db(sample["bmode"])
        assert ours_db.shape == ref_db.shape, f"{ours_db.shape} vs {ref_db.shape}"

        lesion = np.asarray(sample["segmentation_map"]) == 1
        has_lesion = bool(lesion.any())

        z_mm = np.linspace(0.0, 40.0, ours_db.shape[0])
        band = np.zeros_like(lesion)
        band[(z_mm >= SPECKLE_BAND_MM[0]) & (z_mm <= SPECKLE_BAND_MM[1]), :] = True
        speckle = band & ~grow(lesion, 10)

        row: dict = {"frame_id": frame_id, "has_inclusion": has_lesion}
        checks = {}

        env = 10 ** (ours_db / 20.0)
        row["speckle_snr"] = iqm.speckle_snr_patched(env, speckle, patch=32)
        ref_env = np.asarray(sample["bmode"], dtype=np.float32)
        row["reference_speckle_snr"] = iqm.speckle_snr_patched(ref_env, speckle, patch=32)
        lo, hi = SPECKLE_SNR_RANGE
        checks["speckle_snr_rayleigh"] = lo <= row["speckle_snr"] <= hi

        if has_lesion:
            rows_ = np.where(lesion.any(axis=1))[0]
            bg_band = np.zeros_like(lesion)
            bg_band[rows_.min() : rows_.max() + 1, :] = True
            background = bg_band & ~grow(lesion, 6)
            row["gcnr"] = iqm.gcnr(ours_db[lesion], ours_db[background])
            row["reference_gcnr"] = iqm.gcnr(ref_db[lesion], ref_db[background])
            checks["gcnr_tracks_reference"] = (
                abs(row["gcnr"] - row["reference_gcnr"]) <= GCNR_VS_REF_TOLERANCE
            )

        row["checks"] = {k: bool(v) for k, v in checks.items()}
        row["pass"] = bool(all(checks.values()))
        n_pass += row["pass"]
        results.append(row)

        gtxt = (
            f"gCNR={row['gcnr']:.4f} (ref {row['reference_gcnr']:.4f})"
            if has_lesion
            else "no inclusion"
        )
        print(
            f"  {frame_id}: {gtxt} speckleSNR={row['speckle_snr']:.3f} "
            f"(ref {row['reference_speckle_snr']:.3f}) pass={row['pass']}"
        )

    print(f"\n{n_pass}/{len(results)} frames pass WI-4's gate when read back from the store")
    out = Path("tests/data/wi8_store_verification.json")
    out.write_text(json.dumps({"n_pass": n_pass, "n": len(results), "frames": results}, indent=1))
    print(f"wrote {out}")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
