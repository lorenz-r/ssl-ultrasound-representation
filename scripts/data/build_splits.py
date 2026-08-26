"""Build and audit the real train/validation splits over the K-sector store (WI-10).

PLAN.md WI-10 adopts NVIDIA's own train/validation boundary as the outer split
**on the strength of a distributional-matching claim** (§8: `sos_mean`
1474.2±47.0 vs 1471.1±47.8; `phase_error` 1.826±0.277 vs 1.827±0.277). That
claim was measured during planning, from the source dataset. This re-checks it
against the store actually built (WI-8), because the split is only safe to
adopt wholesale if it still holds for the frames that ended up in the corpus —
if train and validation were *not* matched, every probe number would be
comparing two different populations rather than measuring generalisation.

Also emits the per-run split-audit figure (WI-10's third done-condition) and
writes the split id lists so training and eval consume the *same* split rather
than each re-deriving it.

    python scripts/data/build_splits.py --store <store>.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.splits import inner_probe_split, outer_split, split_audit
from ssl_us.eval import viz

# PLAN.md §8's planning-time measurements, restated so a drift is visible.
PLAN_CLAIM = {
    "sos_mean": {"train": (1474.2, 47.0), "validation": (1471.1, 47.8)},
    "phase_error": {"train": (1.826, 0.277), "validation": (1.827, 0.277)},
}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t statistic — unequal variances, unequal n."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/splits"))
    p.add_argument("--ratio", type=float, default=0.8, help="inner probe train fraction")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    store = open_store(args.store)
    labels = store.read_frame_metadata()
    frame_ids = np.array(store.frame_ids, dtype=str)
    if not labels:
        raise SystemExit(f"{args.store} has no frame_metadata.json — rebuild with WI-8")

    outer = outer_split(frame_ids)
    inner = inner_probe_split(outer["train"], ratio=args.ratio, seed=args.seed)
    audit = split_audit(frame_ids, outer, inner)
    print(f"outer: {audit['outer']}   inner: {audit['inner']}   total {audit['n_frames_total']}")

    # ── the claim WI-10 leans on ────────────────────────────────────────────
    print("\ndistributional matching, measured on the built store:")
    header = f"{'label':<14}{'train mean±sd':>22}{'validation mean±sd':>24}{'Welch t':>10}"
    print(f"{header}  vs PLAN.md")
    checks: dict[str, dict] = {}
    for key, claim in PLAN_CLAIM.items():
        vals = {
            split: np.array([labels[f][key] for f in outer[split] if key in labels[f]])
            for split in ("train", "validation")
        }
        t = welch_t(vals["train"], vals["validation"])
        # |t| < 2 is the usual "no detectable difference" rule of thumb at these n.
        matched = abs(t) < 2.0
        agrees = all(
            abs(vals[s].mean() - claim[s][0]) < 3 * claim[s][1] for s in ("train", "validation")
        )
        checks[key] = {
            "train_mean": float(vals["train"].mean()),
            "train_sd": float(vals["train"].std(ddof=1)),
            "validation_mean": float(vals["validation"].mean()),
            "validation_sd": float(vals["validation"].std(ddof=1)),
            "welch_t": t,
            "matched": bool(matched),
            "agrees_with_plan": bool(agrees),
        }
        c = checks[key]
        print(
            f"{key:<14}{c['train_mean']:>13.4g} ± {c['train_sd']:<6.3g}"
            f"{c['validation_mean']:>15.4g} ± {c['validation_sd']:<6.3g}"
            f"{t:>10.2f}  {'agrees' if agrees else 'DRIFTED'}"
            f"{'' if matched else '   ** NOT MATCHED **'}"
        )

    # cyst_fraction is not part of PLAN.md's claim but is the primary tissue
    # axis, so a mismatch there would bite the probes just as hard.
    extra = {}
    for key in ("cyst_fraction", "sos_std"):
        vals = {
            s: np.array([labels[f][key] for f in outer[s] if key in labels[f]])
            for s in ("train", "validation")
        }
        t = welch_t(vals["train"], vals["validation"])
        extra[key] = {"welch_t": t, "matched": bool(abs(t) < 2.0)}
        print(f"{key:<14}{vals['train'].mean():>13.4g} ± {vals['train'].std(ddof=1):<6.3g}"
              f"{vals['validation'].mean():>15.4g} ± {vals['validation'].std(ddof=1):<6.3g}"
              f"{t:>10.2f}  (not a PLAN.md claim)")

    empty = {s: sum(labels[f]["segmentation_empty"] for f in outer[s]) for s in outer}
    print(f"\nframes with empty segmentation_map: {empty} (excluded from cyst probes, PLAN.md §2)")

    fig = viz.plot_split_audit(
        outer, args.out / "split_audit.png", inner=inner, frame_labels=labels
    )
    print(f"wrote {fig}")

    payload = {
        "store": str(args.store),
        "audit": audit,
        "distribution_checks": checks,
        "additional_checks": extra,
        "empty_segmentation_per_split": empty,
        "splits": {"outer": outer, "inner": inner},
        "ratio": args.ratio,
        "seed": args.seed,
    }
    out_json = Path("results/splits/nvidia_splits.json")
    out_json.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_json}")

    failed = [k for k, v in checks.items() if not v["matched"]]
    if failed:
        raise SystemExit(
            f"train/validation are NOT distributionally matched on {failed} — "
            "PLAN.md §8's basis for adopting NVIDIA's split wholesale does not hold"
        )


if __name__ == "__main__":
    main()
