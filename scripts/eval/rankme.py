"""RankMe collapse diagnostic — does a frozen encoder's representation
*collapse* on real tissue?

PLAN.md §4.7 makes RankMe the project's primary collapse diagnostic precisely
because it needs no labels: a healthy 512-d backbone scores near 512, a
representation whose variance has concentrated into a handful of directions
scores near 1. §4.7 uses it *during* training to catch dimensional collapse.
This runs the same measurement on **frozen** encoders, over each corpus in
turn, to ask a question the transfer table cannot:

    is a high Stanford probe score evidence of a rich representation,
    or of a collapsed one that happens to align with the probed axis?

The question is not hypothetical. `outputs/stanford/mm_matched_k4_qual/grid_pca.png`
shows Policy B's Stanford embedding lying along a narrow, curved, near-1D
manifold, and Policy B scores **R² 0.964 on echogenicity** — a suspiciously
high number that a 1-D brightness curve would also produce. RankMe separates
those two readings; the eye cannot.

    python scripts/eval/rankme.py --k 4 --variant mm_matched

Both corpora are measured through the identical crop/encoder path (the same
`band_spec` the probe column used, and the same amplitude match), so a
difference between the two columns is the corpus and nothing else.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.eval.external import assert_same_crops
from ssl_us.eval.stanford_targets import stanford_frame_labels
from ssl_us.eval.store_features import (
    StoreCropSpec,
    frame_labels_from_metadata,
    sample_store_crops,
)
from ssl_us.train.instrument import rankme, singular_value_spectrum

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_matrix():
    spec = importlib.util.spec_from_file_location(
        "stanford_matrix", SCRIPT_DIR / "stanford_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spectrum_stats(feats: np.ndarray) -> dict:
    s = singular_value_spectrum(feats)
    var = s**2
    var = var / var.sum()
    return {
        "rankme": round(rankme(s), 2),
        "dim": int(feats.shape[1]),
        # Complementary and easier to state in a sentence than RankMe: how much
        # of the variance one direction holds, and how many directions it takes
        # to reach 90 %.
        "pc1_variance_frac": round(float(var[0]), 4),
        "dims_for_90pct": int(np.searchsorted(np.cumsum(var), 0.90) + 1),
    }


def main() -> int:
    matrix = _load_matrix()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=matrix.VARIANTS, default="mm_matched")
    p.add_argument("--store", default=None)
    p.add_argument("--nvidia-store", default=matrix.NVIDIA_STORE, dest="nvidia_store")
    p.add_argument("--checkpoint-root", default=matrix.CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--nvidia-frames", type=int, default=117, dest="nvidia_frames",
                   help="match Stanford's frame count so N is not a confound")
    p.add_argument("--intensity-scale", choices=["matched", "raw"], default="matched",
                   dest="intensity_scale")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external", action="store_true",
                   help="also measure PLAN.md §4.4 baseline 5 (frozen DINOv3 on B-mode)")
    p.add_argument("--external-model", default=None, dest="external_model")
    args = p.parse_args()

    if args.store is None:
        args.store = matrix.STORE_TEMPLATE.format(variant=args.variant)
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    bands = matrix.depth_bands(args.k, arm.z_min_mm)
    band = next(iter(bands))
    out_dir = Path(args.out or f"outputs/stanford/{args.variant}_k{args.k}_rankme")
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = matrix.resolve_intensity_scale(args)
    spec = matrix.band_spec(args, bands[band], scale)
    encoders = matrix.build_encoders(args, spec.in_channels, tag="stanford-rankme")

    # NVIDIA is measured on the *same* number of frames and crops, and on its
    # own native scale (intensity_scale=1.0 — it is the reference corpus, so
    # matching it to itself is the identity). RankMe is sensitive to N, so an
    # unmatched frame count would confound the comparison it exists to make.
    nvidia_spec = StoreCropSpec(
        k=args.k, crop_size=args.crop_size, z_min_mm=bands[band],
        crops_per_frame=args.crops_per_frame, seed=args.seed,
        derive_echogenicity=True,
    )

    corpora = [
        ("stanford", args.store, spec, stanford_frame_labels, None),
        ("nvidia", args.nvidia_store, nvidia_spec, frame_labels_from_metadata,
         args.nvidia_frames),
    ]

    out: dict[str, dict] = {}
    for corpus, path, crop_spec, label_fn, limit in corpora:
        ids = list(open_store(path).frame_ids)
        if limit:
            ids = ids[:limit]
        batch = sample_store_crops(path, ids, crop_spec, frame_label_fn=label_fn)
        print(f"\n[rankme] {corpus}: {len(batch)} crops from {len(ids)} frames", flush=True)
        encoder_items = list(encoders.items())
        if args.external:
            # Baseline 5 reads the K=1 log-envelope stage, so it needs its own
            # sampling pass — same origins, different K (see `eval.external`).
            ext_batch = sample_store_crops(
                path, ids, replace(crop_spec, k=1), frame_label_fn=label_fn
            )
            assert_same_crops(batch, ext_batch)
            ext = matrix.external_encoder(args)
            # A fresh dB reference per corpus: B-mode display is normalised
            # within a corpus, and reusing Stanford's level for NVIDIA would
            # push it out of the displayed range entirely.
            ext.reference = None
            encoder_items.append((matrix.EXTERNAL_NAME, ext))
        for name, fn in encoder_items:
            source = ext_batch if name == matrix.EXTERNAL_NAME else batch
            stats = _spectrum_stats(fn(source.images))
            out.setdefault(name, {})[corpus] = stats
            print(
                f"  {name:<18} RankMe {stats['rankme']:>7.2f} / {stats['dim']}   "
                f"PC1 {stats['pc1_variance_frac'] * 100:5.1f}%   "
                f"dims for 90% var: {stats['dims_for_90pct']}",
                flush=True,
            )
        batch.images = np.empty((0,), dtype=np.float32)

    print(f"\n{'encoder':<18} {'RankMe nvidia':>14} {'RankMe stanford':>16} {'ratio':>8}")
    for name, d in out.items():
        if "nvidia" in d and "stanford" in d:
            n, s = d["nvidia"]["rankme"], d["stanford"]["rankme"]
            print(f"{name:<18} {n:>14.2f} {s:>16.2f} {s / n if n else float('nan'):>8.2f}")

    payload = {
        "wi": 16, "kind": "rankme", "variant": args.variant, "k": args.k,
        "band": band, "crop_size": args.crop_size,
        "crops_per_frame": args.crops_per_frame,
        "nvidia_frames": args.nvidia_frames,
        "intensity_scale_mode": args.intensity_scale, "intensity_scale": scale,
        "encoders": out,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"\n[rankme] wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
