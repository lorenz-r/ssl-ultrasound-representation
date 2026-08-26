"""Frozen-feature probes for an `RFConvMAE` checkpoint, on the NV corpus.

Runs the same protocol as `scripts/eval/nvidia_matrix.py` — PCA capacity control
to 64 dims, GroupKFold by frame, inner-CV-selected penalty, bootstrap CI — by
importing `ssl_us.eval.probes` rather than reimplementing it. What differs is the
input: that matrix crops a beamformed K-sector store, this one encodes
pre-beamform channel-data frames, which is what RFConvMAE was trained on.

    # once: stream the label columns out of the raw Parquet (minutes, no IQ decode)
    python scripts/eval/rf_conv_mae_probe.py --build-labels \
        --nv-root "$NV_RAW_ROOT" --labels labels/nv_frame_labels.json

    # then: probe a checkpoint
    python scripts/eval/rf_conv_mae_probe.py \
        --checkpoint checkpoints/best_rf_conv_mae.pt \
        --data-dir "$NV_WORK/nv_raw2insights_full.zarr" \
        --data-metadata "$NV_WORK/nv_raw2insights_full.metadata.json" \
        --labels labels/nv_frame_labels.json --split validation \
        --out outputs/probe/rf_conv_mae

`--split validation` is the default and the honest one: those 93 frames never
shaped the weights. `--split train` runs the in-sample version for contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssl_ultrasound_representation.dataset import build_split_datasets
from ssl_ultrasound_representation.eval.probe_adapter import (
    NV_FRAME_TARGETS,
    RFConvMAEEncoder,
    build_frame_labels,
    extract_frame_features,
    labels_to_arrays,
    run_probe_matrix,
    write_report,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-labels", action="store_true", dest="build_labels")
    p.add_argument("--nv-root", dest="nv_root")
    p.add_argument("--labels", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--data-metadata", dest="data_metadata")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--out", default="outputs/probe/rf_conv_mae")
    p.add_argument("--device", default="cuda")
    p.add_argument("--focal-zone", type=int, default=0, dest="focal_zone")
    p.add_argument("--row-batch", type=int, default=16, dest="row_batch")
    p.add_argument("--limit-frames", type=int, default=0, dest="limit_frames",
                   help="smoke testing only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument(
        "--no-baseline", action="store_true", dest="no_baseline",
        help="skip the untrained-twin floor. Without it a score has nothing to "
             "clear, and cannot distinguish learning from architecture.",
    )
    args = p.parse_args()

    labels_path = Path(args.labels)
    if args.build_labels:
        if not args.nv_root:
            raise SystemExit("--build-labels needs --nv-root")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(json.dumps(build_frame_labels(args.nv_root), indent=1))
        print(f"[probe] wrote {labels_path}")
        if not args.checkpoint:
            return 0

    for flag, val in (("--checkpoint", args.checkpoint), ("--data-dir", args.data_dir),
                      ("--data-metadata", args.data_metadata)):
        if not val:
            raise SystemExit(f"{flag} is required to probe")

    all_labels = json.loads(labels_path.read_text())
    if args.split not in all_labels:
        raise SystemExit(
            f"labels file has no {args.split!r} split (has {sorted(all_labels)})")
    labels = labels_to_arrays(all_labels[args.split])

    # The NV builder writes `subject` == split name, so this reproduces the
    # publisher's own division rather than a ratio split (which would put every
    # frame in train and leave val empty).
    splits, _ = build_split_datasets(
        local_zip=args.data_dir,
        local_metadata=args.data_metadata,
        split_strategy="subject",
        train_subjects=["train"],
        val_subjects=["validation"],
        focal_zone=args.focal_zone,
        preload=False,
    )
    ds = splits["train" if args.split == "train" else "val"]
    n = len(ds)
    if len(labels["frame_index"]) != n:
        raise SystemExit(
            f"labels have {len(labels['frame_index'])} {args.split} frames but the "
            f"store has {n}. They must line up positionally — rebuild the labels "
            f"from the same NV root the store was converted from."
        )
    keep = n if not args.limit_frames else min(args.limit_frames, n)

    class _Sub:
        def __len__(self): return keep
        def __getitem__(self, i): return ds[i]

    arms = [("rf_conv_mae", False)]
    if not args.no_baseline:
        arms.append(("random_rf_conv_mae", True))

    labels = {k: v[:keep] for k, v in labels.items()}
    rows: list[dict] = []
    not_run: dict[str, str] = {}
    feature_dim = None
    for name, rand in arms:
        encoder = RFConvMAEEncoder(
            args.checkpoint, device=args.device, row_batch=args.row_batch,
            random_init=rand, seed=args.seed,
        )
        print(f"\n[probe] {name} | epoch {encoder.epoch} | dim {encoder.dim} "
              f"| lateral {encoder.lateral} | frames {keep}", flush=True)
        feats, frame_ids = extract_frame_features(
            _Sub(), encoder, [str(i) for i in range(keep)])
        feature_dim = int(feats.shape[1])
        r, nr = run_probe_matrix(
            feats, labels, frame_ids, encoder_name=name,
            targets=NV_FRAME_TARGETS, stage=args.stage, seed=args.seed,
        )
        rows += r
        not_run.update(nr)
        for x in r:
            print(f"  {x['target']:<18} {x['metric']:<18} {x['score']:+.3f}  "
                  f"ci95 {x['ci95']}  n_eff={x['n_eff']}")
        for cell, why in nr.items():
            print(f"  {cell:<18} NOT RUN ({why})")

    # The comparison the numbers exist for: trained vs the identical untrained net.
    contrast = []
    by = {(r["encoder"], r["target"]): r for r in rows}
    for target, level, _kind in NV_FRAME_TARGETS:
        t, b = by.get(("rf_conv_mae", target)), by.get(("random_rf_conv_mae", target))
        if not t:
            continue
        contrast.append({
            "target": target, "level": level, "metric": t["metric"],
            "trained": t["score"],
            "random_baseline": b["score"] if b else None,
            "delta": None if not b else t["score"] - b["score"],
            "beats_random": None if not b else bool(t["score"] > b["score"]),
            # Overlapping CIs mean this cell does not separate them — a result
            # in itself, and one worth saying rather than ranking anyway.
            "separated": None if not b else bool(
                t["score_lo"] > b["score_hi"] or b["score_lo"] > t["score_hi"]),
        })
    if contrast:
        print("\n[probe] trained vs untrained twin:")
        for c in contrast:
            d = "n/a" if c["delta"] is None else f"{c['delta']:+.3f}"
            print(f"  {c['target']:<18} trained {c['trained']:+.3f}  "
                  f"random {c['random_baseline']:+.3f}  delta {d}  "
                  f"beats_random={c['beats_random']}  separated={c['separated']}")

    out_dir = Path(args.out)
    write_report(rows, not_run, {
        "checkpoint": str(args.checkpoint),
        "contrast": contrast,
        "split": args.split, "n_frames": keep, "stage": args.stage,
        "feature_dim": feature_dim,
        "protocol": "ssl_us.eval.probes (PCA-64, GroupKFold by frame, inner-CV penalty, bootstrap CI)",
    }, out_dir)
    print(f"\n[probe] wrote {out_dir/'results.json'} and {out_dir/'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
