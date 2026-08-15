"""WI-14 — does *any* (patch, mask) configuration let the MAE learn on speckle?

The PLAN.md-spec configuration (16 px patches, 75 % masking) converges to
MSE 1.0 under normalized-pixel targets, which is exactly "predict zero". The
measured reason is scale: this data's speckle autocorrelation falls to half by
**2 px axially and 4 px laterally**, so a 16 px patch is 4-8 correlation
lengths across and a masked patch has no usable evidence anywhere near it.

That indicts the configuration, not masked modelling in general. This sweep
settles which, by shrinking the patch toward the correlation length and
lowering the mask ratio, and asking whether the loss ever drops meaningfully
below 1.0.

Short runs on purpose: the failure signature appears by epoch 4 in every run
so far, so 12 epochs on 120 frames (~2 min each) is enough to separate "learns
something" from "predicts zero" without spending hours per cell.

    python scripts/wi14_mask_sweep.py --out figures/wi14_sweep

Reads final loss as the headline: with `norm_pix_loss` the target has unit
variance per patch, so **variance explained = 1 - loss**, directly comparable
across configurations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# (patch_size, mask_ratio). Tokens on a 128 px crop = (128/patch)^2, and
# attention is O(tokens^2), so patch 4 at low mask ratios is deliberately
# omitted — 1024 tokens with 512 visible is not affordable even as a probe.
CONFIGS = [
    (16, 0.75),  # the PLAN.md spec, as the control
    (16, 0.50),
    (16, 0.25),
    (8, 0.75),
    (8, 0.50),
    (8, 0.25),
    (4, 0.75),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="figures/wi14_sweep")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--crops-per-frame", type=int, default=64, dest="crops_per_frame")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per config")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for patch, mask in CONFIGS:
        tokens = (128 // patch) ** 2
        label = f"p{patch}_m{int(mask * 100)}"
        print(f"\n=== {label}: {tokens} tokens, {int(tokens * (1 - mask))} visible ===",
              flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable, "-u", "scripts/wi14_train.py",
                "--out", tmp, "--k", str(args.k),
                "--limit-frames", str(args.frames),
                "--crops-per-frame", str(args.crops_per_frame),
                "--epochs", str(args.epochs), "--warmup-epochs", "1",
                "--patch-size", str(patch), "--mask-ratio", str(mask),
                "--log-every-steps", "0", "--no-wandb",
            ]
            try:
                proc = subprocess.run(cmd, timeout=args.timeout, capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after {args.timeout}s", flush=True)
                results.append({"patch": patch, "mask": mask, "tokens": tokens,
                                "status": "timeout"})
                continue
            if proc.returncode != 0:
                print(f"  FAILED rc={proc.returncode}: {proc.stderr[-400:]}", flush=True)
                results.append({"patch": patch, "mask": mask, "tokens": tokens,
                                "status": f"failed rc={proc.returncode}"})
                continue
            history = json.loads((Path(tmp) / "history.json").read_text())

        loss = history["loss"]
        row = {
            "patch": patch, "mask": mask, "tokens": tokens,
            "visible_tokens": int(tokens * (1 - mask)),
            "status": "ok",
            "loss_first": loss[0], "loss_final": loss[-1], "loss_min": min(loss),
            # norm_pix_loss => unit-variance target => 1 - loss is the fraction
            # of masked-patch variance the model actually predicts.
            "variance_explained": 1.0 - min(loss),
            "rankme_first": history["rankme"][0], "rankme_final": history["rankme"][-1],
            "epoch_seconds": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        }
        results.append(row)
        print(f"  loss {row['loss_first']:.4f} -> {row['loss_final']:.4f} "
              f"(min {row['loss_min']:.4f}) | variance explained "
              f"{row['variance_explained'] * 100:+.1f}% | RankMe "
              f"{row['rankme_first']:.0f} -> {row['rankme_final']:.0f}", flush=True)

    (out_dir / "sweep.json").write_text(json.dumps(results, indent=1))

    print("\n=== summary (variance explained; >0 means it learned something) ===")
    print(f"{'config':<12}{'tokens':>8}{'visible':>9}{'min loss':>10}{'var expl':>10}{'RankMe':>9}")
    for r in results:
        if r["status"] != "ok":
            print(f"p{r['patch']}/m{int(r['mask'] * 100):<8} {r['status']}")
            continue
        print(f"p{r['patch']}/m{int(r['mask'] * 100):<7}{r['tokens']:>8}{r['visible_tokens']:>9}"
              f"{r['loss_min']:>10.4f}{r['variance_explained'] * 100:>9.1f}%"
              f"{r['rankme_final']:>9.0f}")
    print(f"\nwrote {out_dir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
