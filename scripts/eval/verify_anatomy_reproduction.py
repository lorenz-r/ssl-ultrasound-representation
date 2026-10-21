"""Check the anatomy re-run reproduced the seven committed targets exactly (WI-13).

Adding the three crop-level anatomy targets must not perturb the columns that
were already reported: they draw the same crops (origins are sampled for the
whole corpus up front, from the same seed), extract the same features from the
same checkpoints, and probe them the same way. So every pre-existing
`(encoder, target)` cell has to come back **bitwise identical** to the
committed `results/nvidia/*.json`. If one moved, the re-run is not a superset
of the old column and the two cannot be reported side by side.

    python scripts/eval/verify_anatomy_reproduction.py

Compares `outputs/nvidia/k{K}_{split}_anatomy/results.json` against
`results/nvidia/k{K}_{split}.json` for the six columns, on `score`,
`score_lo`, `score_hi` and `n_eff`. Exits non-zero if any cell differs or any
column is missing. `--rtol` loosens the float comparison for the case where a
difference is genuinely a library/hardware round-off rather than a protocol
change — it defaults to 0, i.e. exact.

⚠ **This is a pre-replacement gate, and it goes vacuous once the anatomy
columns are promoted into `results/nvidia/`** — from then on it compares a file
against its own copy and passes for free. It was run for real on 2026-08-17,
before the promotion, against the columns committed in `1147c19`; to reproduce
that check rather than the trivial one, restore the old baseline first:

    # NOTE: at 1147c19 the committed columns still lived under tests/data/.
    git show 1147c19:tests/data/wi13_k4_train_results.json > /tmp/old.json
    python scripts/eval/verify_anatomy_reproduction.py --committed /tmp   # after renaming

Result of the real run, recorded because it cannot be re-derived cheaply: all
238 shared `(encoder, target)` cells reproduced `score` and `n_eff` **bitwise**;
the two MAE encoders' `score_lo`/`score_hi` moved by ≤ 3.6e-6 on the two K=4
columns, i.e. five orders of magnitude below the interval width (~0.1), and
`--rtol 1e-5` is what that needs. The cause is not established: the encoders
were checked to be deterministic on repeat, invariant to batch chunking, and
invariant to BLAS thread count, so it is neither of the two usual suspects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COLUMNS = [(k, split) for k in (4, 16, 1) for split in ("validation", "train")]
FIELDS = ("score", "score_lo", "score_hi", "n_eff")
ANATOMY = ("contains_cyst", "cyst_fraction_crop", "sos_mean_crop")


def _cells(path: Path) -> dict:
    rows = json.loads(path.read_text())["results"]
    return {(r["encoder"], r["target"]): r for r in rows}


def compare(old: Path, new: Path, rtol: float) -> tuple[list[str], set, int, int]:
    """(differences, cells that moved, cells compared, anatomy cells added).

    A cell is one `(encoder, target)`; each carries four compared fields, so
    counting differences and counting cells are not the same number and
    reporting one as the other overstates the damage.
    """
    a, b = _cells(old), _cells(new)
    diffs, moved = [], set()
    for key, row in a.items():
        if key not in b:
            diffs.append(f"{key[0]}/{key[1]}: missing from the re-run")
            moved.add(key)
            continue
        for field in FIELDS:
            x, y = row[field], b[key][field]
            same = x == y if rtol == 0 else math.isclose(x, y, rel_tol=rtol, abs_tol=rtol)
            if not same:
                diffs.append(f"{key[0]}/{key[1]}.{field}: {x!r} -> {y!r}  (Δ{abs(x - y):.2e})")
                moved.add(key)
    added = sum(1 for enc, target in b if target in ANATOMY)
    return diffs, moved, len(a), added


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", default="figures")
    p.add_argument("--committed", default="results/nvidia")
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args()

    failed = False
    for k, split in COLUMNS:
        old = Path(args.committed) / f"k{k}_{split}.json"
        new = Path(args.figures) / f"k{k}_{split}_anatomy" / "results.json"
        label = f"K={k:<2} {split:<10}"
        if not new.exists():
            print(f"{label} MISSING  {new}")
            failed = True
            continue
        diffs, moved, n, added = compare(old, new, args.rtol)
        if diffs:
            failed = True
            print(f"{label} DIFFERS  {len(moved)}/{n} cells, {len(diffs)} field values")
            for d in diffs[:10]:
                print(f"    {d}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")
        else:
            print(f"{label} OK       {n} cells reproduced, {added} anatomy cells added")

    print("\nFAIL — the re-run is not a superset of the committed columns" if failed
          else "\nOK — every committed cell reproduced; anatomy targets are additive")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
