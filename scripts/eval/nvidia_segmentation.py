"""NVIDIA per-pixel segmentation probe and mask figure (WI-19).

The results matrix already probes the segmentation map — but only after
reducing it to a scalar. `contains_cyst` asks whether a crop contains a cyst and
`cyst_fraction_crop` asks how much; neither asks **where**, which is the
question that separates a representation that localises anatomy from one that
reports darkness. This script asks it, two ways:

1. a **Dice column** over frozen pre-pool features, frame-grouped and
   frame-bootstrapped (`eval/dense_probe.py`), and
2. a **mask figure** — B-mode, ground truth, then one predicted mask per
   encoder, on whole frames stitched from a sliding window.

    python scripts/eval/nvidia_segmentation.py --k 4 --out outputs/nvidia/k4_segmentation

⚠ **Read the trained arms against `trivial_dense`.** Cysts are anechoic, so a
per-cell brightness threshold segments them competently, and both the number
and the picture will look respectable for every encoder in the table. The
finding this script exists to produce is the *gap* between a trained arm and
that threshold — the figure's columns are ordered so the baselines are read
first, and a run whose trained arms do not clear them has reported that darkness
explains the masks, which is a result and should be presented as one.

**Figure frames are chosen by rule, not by eye**: the median-Dice and
worst-Dice frame of `--reference-encoder`, among frames with a non-empty mask.
One hand-picked frame is an anecdote; the worst frame is what keeps the slide
honest. The head that renders them is fit on the *other* frames of the split
(`--figure-frames` are held out), so the picture is out-of-sample even though
the encoders never saw any of these frames to begin with.

Crops from the 75 empty-`segmentation_map` frames are excluded, exactly as the
`contains_cyst` cells exclude them and for PLAN.md 2.6's reason: an empty mask
is an unknown mix of true negatives and unlabelled edge-touching cysts, and as
a dense target it would train the head to erase real cysts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.data.label_maps import DEFAULT_ROOT as LABEL_MAPS_DEFAULT_ROOT
from ssl_us.data.label_maps import open_label_maps
from ssl_us.data.splits import outer_split
from ssl_us.eval import viz
from ssl_us.eval.dense_features import (
    DenseSpec,
    build_dense_encoders,
    extract_dense_features,
    frame_bmode_db,
    reduced_frame,
    whole_frame_probability,
)
from ssl_us.eval.dense_probe import (
    TARGET,
    cell_targets,
    dense_probe,
    fit_dense_head,
)
from ssl_us.eval.store_features import StoreCropSpec

DEFAULT_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
CKPT_ROOT = "/mnt/home/benbarkow/checkpoints"

# Baselines first. The order is the argument: a reader who meets `trivial_dense`
# before `vicreg_A` cannot mistake "the cysts are dark" for "the representation
# found the cysts". Encoders absent from a run are dropped from the order;
# anything present but unlisted is appended, so a new arm cannot vanish.
FIGURE_ORDER = (
    "trivial_dense",
    "pca_raw_dense",
    "random_resnet18",
    "random_vit",
    "vicreg_A",
    "vicreg_B",
    "vicreg_vit_A",
    "mae_norm",
    "mae_raw",
    "ijepa",
)


def ordered(names) -> list[str]:
    present = list(names)
    head = [n for n in FIGURE_ORDER if n in present]
    return head + [n for n in present if n not in head]


def pick_figure_frames(per_frame_dice: dict[str, float], n: int = 2) -> list[str]:
    """The median-Dice and worst-Dice frame, by a stated rule.

    Returns them worst-last so the figure ends on the failure case rather than
    opening with it. With `n > 2` the extra frames are drawn evenly across the
    ranking so the row set still spans the distribution instead of clustering.
    """
    ranked = sorted(per_frame_dice, key=lambda f: per_frame_dice[f])
    if not ranked:
        return []
    if n >= len(ranked):
        return list(reversed(ranked))
    picks = {ranked[len(ranked) // 2], ranked[0]}
    for q in np.linspace(0, len(ranked) - 1, num=n):
        if len(picks) >= n:
            break
        picks.add(ranked[int(round(q))])
    return sorted(picks, key=lambda f: -per_frame_dice[f])[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--checkpoint-root", default=CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default="outputs/nvidia/k4_segmentation")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--z-min-mm", type=float, default=None, dest="z_min_mm")
    p.add_argument("--crops-per-frame", type=int, default=8, dest="crops_per_frame")
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument(
        "--frame-chunk", type=int, default=0, dest="frame_chunk",
        help="materialize crops this many frames at a time (0 = all at once)",
    )
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--dense-grid", type=int, default=DenseSpec.grid, dest="dense_grid")
    p.add_argument("--dense-batch", type=int, default=64, dest="dense_batch")
    p.add_argument(
        "--max-fit-cells", type=int, default=200_000, dest="max_fit_cells",
        help="cap on rows per linear-head fit; scoring still uses every held-out cell",
    )
    p.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    p.add_argument("--reference-encoder", default="vicreg_A", dest="reference_encoder")
    p.add_argument("--figure-frames", type=int, default=2, dest="figure_frames")
    p.add_argument(
        "--window-stride", type=int, default=32, dest="window_stride",
        help="sliding-window stride in pixels for whole-frame inference",
    )
    p.add_argument("--no-figure", action="store_true", dest="no_figure")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-maps", default=str(LABEL_MAPS_DEFAULT_ROOT), dest="label_maps")
    args = p.parse_args()

    # Same derivation as `nvidia_matrix.py`, from the same call the *training*
    # loader makes, so the encoder is never read on crops it was not trained on.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size
    if args.z_min_mm is None:
        args.z_min_mm = arm.z_min_mm

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Not optional here, unlike in `nvidia_matrix.py`: the per-pixel map *is*
    # the target. Failing now with the fetch command beats failing after an
    # hour of feature extraction.
    label_maps = open_label_maps(args.label_maps)
    print(f"[wi19] label maps: {label_maps.root} ({len(label_maps)} frames)", flush=True)

    store = open_store(args.store)
    frame_ids = outer_split(np.array(store.frame_ids))[args.split]
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]

    spec = StoreCropSpec(
        k=args.k,
        crop_size=args.crop_size,
        z_min_mm=args.z_min_mm,
        crops_per_frame=args.crops_per_frame,
        seed=args.seed,
    )
    dense_spec = DenseSpec(grid=args.dense_grid, batch_size=args.dense_batch)
    cell = dense_spec.cell_size(spec.crop_size)
    print(
        f"[wi19] K={spec.k} crop={spec.crop_size} in_ch={spec.in_channels} "
        f"split={args.split} frames={len(frame_ids)} crops/frame={spec.crops_per_frame} "
        f"grid={dense_spec.grid} ({cell} px/cell)",
        flush=True,
    )

    encoders = build_dense_encoders(args, args.k, dense_spec)
    chunked = bool(args.frame_chunk) and args.frame_chunk < len(frame_ids)
    if chunked and "pca_raw_dense" in encoders:
        print(
            f"[wi19] ⚠ pca_raw_dense is fitted per chunk ({args.frame_chunk} frames of "
            f"{len(frame_ids)}), not over the corpus — its Dice is not comparable to a "
            "--frame-chunk 0 column. Recorded in results.json.",
            flush=True,
        )

    t0 = time.time()
    batch = extract_dense_features(
        args.store, frame_ids, spec, encoders, label_maps,
        dense_spec=dense_spec, frame_chunk=args.frame_chunk, tag="wi19",
    )
    print(f"[wi19] extracted {len(batch)} crops in {time.time() - t0:.0f} s", flush=True)

    # PLAN.md 2.6 — the same exclusion the `contains_cyst` cells apply.
    keep = ~batch.segmentation_empty
    dropped = int((~keep).sum())
    seg = batch.seg[keep]
    fids = batch.frame_id[keep]
    print(
        f"[wi19] cyst pixels {seg.mean() * 100:.2f}% of kept pixels | "
        f"dropped {dropped}/{len(batch)} crops from empty-mask frames | "
        f"{len(np.unique(fids))} frames remain",
        flush=True,
    )
    if len(np.unique(fids)) < 2:
        print("[wi19] fewer than 2 frames with a non-empty mask — nothing to probe")
        return 1

    results, oof = {}, {}
    for name in ordered(encoders):
        maps = batch.maps[name][keep]
        t1 = time.time()
        res, prob = dense_probe(
            maps, seg, fids,
            encoder=name, stage=args.stage,
            max_fit_cells=args.max_fit_cells, n_boot=args.n_boot, seed=args.seed,
        )
        results[name], oof[name] = res, prob
        print(
            f"[wi19] {name:16s} dice {res.score:.4f} {res.row()['ci95']} "
            f"iou {res.secondary['iou']:.4f} cell-dice {res.secondary['dice_cell']:.4f} "
            f"({time.time() - t1:.0f} s)",
            flush=True,
        )

    table_rows = [results[n].row() for n in ordered(results)]
    record = {
        "wi": 19,
        "k": spec.k,
        "crop_size": spec.crop_size,
        "z_min_mm": spec.z_min_mm,
        "crops_per_frame": spec.crops_per_frame,
        "split": args.split,
        "stage": args.stage,
        "target": TARGET,
        "dense_grid": dense_spec.grid,
        "cell_px": cell,
        "hypercolumn_layers": list(dense_spec.layers),
        "seed": args.seed,
        "frames": len(frame_ids),
        "frames_probed": int(len(np.unique(fids))),
        "crops_kept": int(keep.sum()),
        "crops_dropped_empty_mask": dropped,
        "pca_raw_dense_fitted_per_chunk": bool(chunked),
        "frame_chunk": args.frame_chunk,
        "rows": table_rows,
        "per_frame_dice": {n: results[n].per_frame_dice for n in results},
        "not_run": {},
    }

    figure_frames: list[str] = []
    if not args.no_figure:
        ref = args.reference_encoder if args.reference_encoder in results else next(iter(results))
        if ref != args.reference_encoder:
            print(f"[wi19] reference {args.reference_encoder!r} absent — using {ref!r}")
        figure_frames = pick_figure_frames(results[ref].per_frame_dice, args.figure_frames)
        print(
            f"[wi19] figure frames (by {ref} Dice): "
            + ", ".join(f"{f} {results[ref].per_frame_dice[f]:.3f}" for f in figure_frames),
            flush=True,
        )

        # Held out from the head fit, so the rendered mask is out-of-sample.
        fit_mask = ~np.isin(fids, figure_frames)
        y_cell = cell_targets(seg, dense_spec.grid)

        bmode = {f: frame_bmode_db(reduced_frame(store, f, args.k), args.k) for f in figure_frames}
        truth = {
            f: np.asarray(label_maps.seg[label_maps.index_of([f])[0]]) for f in figure_frames
        }
        frames_px = {f: reduced_frame(store, f, args.k) for f in figure_frames}

        predictions: dict[str, dict[str, np.ndarray]] = {}
        frame_dice: dict[str, dict[str, float]] = {}
        for name in ordered(results):
            head = fit_dense_head(
                batch.maps[name][keep][fit_mask], y_cell[fit_mask],
                max_fit_cells=args.max_fit_cells, seed=args.seed,
            )
            predictions[name], frame_dice[name] = {}, {}
            for f in figure_frames:
                prob = whole_frame_probability(
                    frames_px[f], encoders[name], head,
                    size=spec.crop_size, stride=args.window_stride,
                )
                predictions[name][f] = prob
                pred, gt = prob >= 0.5, truth[f].astype(bool)
                denom = pred.sum() + gt.sum()
                frame_dice[name][f] = (
                    1.0 if denom == 0 else float(2.0 * np.logical_and(pred, gt).sum() / denom)
                )
            print(
                f"[wi19] whole-frame {name:16s} "
                + " ".join(f"{f}:{frame_dice[name][f]:.3f}" for f in figure_frames),
                flush=True,
            )

        path = viz.plot_mask_grid(
            figure_frames, bmode, truth, predictions, frame_dice,
            out_dir / "masks.png",
            title=(
                f"WI-19 cyst segmentation from frozen features — K={spec.k}, "
                f"{args.split}, {dense_spec.grid}x{dense_spec.grid} readout "
                f"(baselines first; Dice is whole-frame, stride {args.window_stride})"
            ),
        )
        print(f"[wi19] wrote {path}", flush=True)
        record["figure_frames"] = figure_frames
        record["whole_frame_dice"] = frame_dice
        record["window_stride"] = args.window_stride

    (out_dir / "results.json").write_text(json.dumps(record, indent=2))
    header = ["encoder", "dice", "ci95", "iou", "dice_cell", "n_frames", "n_feat", "grid"]
    lines = [
        f"# WI-19 segmentation probe — K={spec.k}, {args.split}, stage={args.stage}",
        "",
        "Read every trained arm against `trivial_dense`: cysts are anechoic, so a",
        "per-cell brightness threshold is already a competent segmenter and the gap",
        "to it — not the absolute Dice — is what this table measures.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in header) + " |")
    (out_dir / "dice_table.md").write_text("\n".join(lines) + "\n")
    print(f"[wi19] wrote {out_dir / 'results.json'} and {out_dir / 'dice_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
