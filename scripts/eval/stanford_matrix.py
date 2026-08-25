"""Stanford transfer table: frozen NVIDIA-pretrained encoders, real tissue.

The transfer half of WI-13's results matrix. Same protocol, same probes, same
baselines — the only things that change are *which store* the crops come from
and *which axes exist to probe*:

    python scripts/eval/stanford_matrix.py --k 4  --variant mm_matched
    python scripts/eval/stanford_matrix.py --k 4  --variant lambda_matched
    python scripts/eval/stanford_matrix.py --k 16 --variant mm_matched

**Frozen features only — nothing is retrained here** (PLAN.md WI-16). The
encoders are the WI-12/WI-13 VICReg arms and WI-14's MAE, all pre-trained on
NVIDIA *simulation*; Stanford is real in-vivo rat tissue they have never seen.
Checkpoint discovery and the baselines (`random_resnet18`, `pca_raw`,
`trivial`) come from `ssl_us.eval.encoders` — the same module `wi13_matrix`
uses — so the two corpora are read through one definition and a difference
between the tables can only be the data.

**No train/validation split is taken, and none is needed.** WI-13 probes
NVIDIA's 93 held-out frames because the other 830 shaped the weights. Every one
of Stanford's 117 frames is held out by construction — no Stanford frame
entered any training run — so splitting would only shrink `n_eff`.

**Two grid variants, run separately** (PLAN.md §3.6's ablation): `mm_matched`
keeps anatomy scale comparable to NVIDIA and `lambda_matched` keeps speckle
grain comparable. Which transfers better is the open question §5.2's question
2c poses; running one variant and reporting it as "the" transfer number would
answer it by assumption.

**Amplitude scale is matched to the pre-training corpus by default.** Nothing
in the pipeline standardizes intensity — within one corpus it never needed to —
so Stanford crops arrive ~600x larger than the NVIDIA crops these encoders were
fitted on, and a frozen backbone in `.eval()` mode normalizes by *its training
run's* BatchNorm statistics rather than the batch's. Left alone, the transfer
score measures a units mismatch. One corpus-level scalar fixes it without
touching any relative structure; `--intensity-scale raw` runs the unmatched
control. See `ssl_us.eval.store_features.corpus_intensity_scale`.

**Depth bands.** The headline row is the full 0-19.9 mm range, and the
8-19.9 mm restricted row is reported alongside it — §4.5c's falsification
instrument: if the transfer signal is really depth-structure rather than
tissue, restricting the range collapses it. At K >= 8 the two coincide, because
our own 8 mm sector-occupancy floor (PLAN.md §3.2c) already applies there — to
*both* corpora identically, so it is a matched constraint rather than a domain
gap. That is recorded in `not_run` rather than left as a silently missing row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ssl_us.beamform.store import open_store
from ssl_us.data.dataset import LoaderConfig
from ssl_us.eval import viz
from ssl_us.eval.cluster import cluster_report
from ssl_us.eval.encoders import build_encoders, find_checkpoint
from ssl_us.eval.external import FrozenViTEncoder, assert_same_crops
from ssl_us.eval.guards import LabelLeakageError, filter_targets
from ssl_us.eval.probes import probe
from ssl_us.eval.stanford_targets import (
    STANFORD_CROP_TARGETS,
    STANFORD_FRAME_TARGETS,
    UNRUNNABLE,
    stanford_frame_labels,
)
from ssl_us.eval.store_features import (
    StoreCropSpec,
    corpus_intensity_scale,
    sample_store_crops,
)

STORE_TEMPLATE = "/mnt/home/benbarkow/stores/wi16_{variant}_k16.zarr"
# The corpus the encoders were pre-trained on — the reference amplitude
# scale for `--intensity-scale matched`.
NVIDIA_STORE = "/mnt/home/benbarkow/stores/wi8_fsa_k16.zarr"
CKPT_ROOT = "/mnt/home/benbarkow/checkpoints"
VARIANTS = ("mm_matched", "lambda_matched")

# PLAN.md §3.6 rev-6: the corpus's real signal extent, and the range the
# headline transfer row uses.
DEPTH_MAX_MM = 19.9
# PLAN.md §3.2c's occupancy floor, and §4.5c's restricted control row.
DEPTH_FLOOR_MM = 8.0

TARGETS = STANFORD_CROP_TARGETS + STANFORD_FRAME_TARGETS

# PLAN.md §4.4's reporting rule keeps the external frozen models in a visually
# separate block from the internal arms, so they are never read as one ranking.
# One name, so every consumer of `results.json` can split on it.
EXTERNAL_NAME = "dino_bmode"

_EXTERNAL_CACHE: dict[str, FrozenViTEncoder] = {}


def external_encoder(args) -> FrozenViTEncoder:
    """The §4.4 baseline-5 encoder, loaded once per process.

    Cached because both depth bands use it and downloading/initialising the ViT
    twice would be pure waste — and because the B-mode `reference` level it
    learns on its first call is a corpus constant that must not be recomputed
    per band (that would put the two bands on different dB scales).
    """
    if "encoder" not in _EXTERNAL_CACHE:
        _EXTERNAL_CACHE["encoder"] = FrozenViTEncoder(
            model_id=args.external_model or None, device=args.device
        )
    return _EXTERNAL_CACHE["encoder"]


def depth_bands(k: int, arm_z_min_mm: float) -> dict[str, float]:
    """Band name -> `z_min_mm`, for one K.

    At K <= 4 the arm spans the full range, so `full` and `restricted` are two
    genuinely different rows. At K >= 8 the arm already starts at the 8 mm
    occupancy floor and the full-range row does not exist to be run — probing
    it would additionally mean handing the encoder crops from a depth range it
    never trained on, which is the failure `wi13_matrix` documents.
    """
    if arm_z_min_mm >= DEPTH_FLOOR_MM:
        return {"restricted": arm_z_min_mm}
    return {"full": arm_z_min_mm, "restricted": DEPTH_FLOOR_MM}


def band_spec(args, z_min_mm: float, intensity_scale: float) -> StoreCropSpec:
    """The crop spec for one depth band.

    Split out so `scripts/eval/stanford_qualitative.py` draws its figures from *exactly*
    the crops this table was computed on. A second copy of these six fields
    would drift silently, and the figure would then illustrate a different
    experiment than the R^2 printed on it.
    """
    return StoreCropSpec(
        k=args.k,
        crop_size=args.crop_size,
        z_min_mm=z_min_mm,
        crops_per_frame=args.crops_per_frame,
        seed=args.seed,
        # The corpus is unlabelled tissue, so brightness is the one anatomy-ish
        # axis available and it has to be derived from the crops themselves.
        derive_echogenicity=True,
        intensity_scale=intensity_scale,
    )


def resolve_intensity_scale(args, quiet: bool = False) -> float:
    """The corpus amplitude factor for `--intensity-scale`, or 1.0 for `raw`.

    See `corpus_intensity_scale` for why a frozen `.eval()`-mode encoder cannot
    cross a ~600x scale gap unaided, and why the factor is one scalar for the
    corpus rather than one per crop.
    """
    if args.intensity_scale != "matched":
        if not quiet:
            print("[wi16] intensity scale: raw (control — encoder sees Stanford's own units)")
        return 1.0
    probe_spec = StoreCropSpec(k=args.k, crop_size=args.crop_size, crops_per_frame=2)
    scale = corpus_intensity_scale(
        args.store, NVIDIA_STORE, probe_spec, frame_label_fn=stanford_frame_labels
    )
    if not quiet:
        print(f"[wi16] intensity scale: x{scale:.4g} (matched to {NVIDIA_STORE})", flush=True)
    return scale


def _run_band(
    args, band: str, z_min_mm: float, frame_ids, encoders_for, intensity_scale: float
) -> tuple[list, dict, dict]:
    """Every encoder x every legal target, for one depth band."""
    spec = band_spec(args, z_min_mm, intensity_scale)
    batch = sample_store_crops(
        args.store, frame_ids, spec, frame_label_fn=stanford_frame_labels
    )
    encoders = encoders_for(spec.in_channels)
    feats = {name: fn(batch.images) for name, fn in encoders.items()}
    batch.images = np.empty((0,), dtype=np.float32)  # 1 GB at K=4; nothing below needs it

    # PLAN.md §4.4 baseline 5. A natural-image ViT cannot read a sector stack,
    # so it gets the K=1 log-envelope stage §4.4 prescribes — which means a
    # second sampling pass at k=1. Origins do not depend on K, so these are the
    # *same crops*; `assert_same_crops` refuses to proceed if that ever stops
    # being true, because the labels come from `batch`, not from this pass.
    if args.external:
        ext_spec = replace(spec, k=1)
        ext_batch = sample_store_crops(
            args.store, frame_ids, ext_spec, frame_label_fn=stanford_frame_labels
        )
        assert_same_crops(batch, ext_batch)
        encoder = external_encoder(args)
        feats[EXTERNAL_NAME] = encoder(ext_batch.images)
        ext_batch.images = np.empty((0,), dtype=np.float32)
        print(f"[wi16] external baseline: {encoder.model_id} "
              f"({feats[EXTERNAL_NAME].shape[1]}-d)", flush=True)

    zc, ec = batch.labels["z_centre"], batch.labels["echogenicity"]
    print(
        f"\n[wi16/{band}] crops {len(batch)} | crop {spec.crop_size} px | "
        f"depth {zc.min():.1f}-{zc.max():.1f} mm | "
        f"echogenicity {ec.min():.1f} to {ec.max():.1f} dB",
        flush=True,
    )

    legal = filter_targets(args.stage, [t for t, _, _ in TARGETS])
    rows = []
    for name, f in feats.items():
        print(f"[{name}] features {f.shape}", flush=True)
        for target, level, kind in TARGETS:
            if target not in legal:
                continue
            try:
                r = probe(
                    f,
                    batch.labels[target],
                    batch.frame_id,
                    target,
                    stage=args.stage,
                    level=level,
                    kind=kind,
                    seed=args.seed,
                )
            except (ValueError, LabelLeakageError) as exc:
                print(f"  {target}: SKIPPED ({exc})")
                continue
            rows.append({
                "encoder": name,
                "band": band,
                "z_min_mm": z_min_mm,
                "crop_size": spec.crop_size,
                **r.row(),
                "score_lo": r.score_lo,
                "score_hi": r.score_hi,
                "n_crops": len(batch),
            })
            print(
                f"  {target:<14} {r.metric:<18} {r.score:+.3f}  "
                f"ci95 [{r.score_lo:+.3f},{r.score_hi:+.3f}]  n_eff={r.n_eff}"
            )
    return rows, feats, dict(batch.labels)


def _contrast(rows: list[dict], band: str) -> list[dict]:
    """Policy A vs Policy B on one band, against the same baselines WI-13 uses."""
    by_key = {(r["encoder"], r["target"]): r for r in rows if r["band"] == band}
    out = []
    for target, level, _kind in TARGETS:
        a, b = by_key.get(("vicreg_A", target)), by_key.get(("vicreg_B", target))
        if not (a and b):
            continue
        rand = by_key.get(("random_resnet18", target))
        mae = by_key.get(("mae_raw", target)) or by_key.get(("mae_norm", target))
        triv = by_key.get(("trivial", target))
        out.append({
            "band": band,
            "target": target,
            "level": level,
            "metric": a["metric"],
            "policy_a": a["score"],
            "policy_b": b["score"],
            "delta_b_minus_a": b["score"] - a["score"],
            "separated": bool(
                a["score_hi"] < b["score_lo"] or b["score_hi"] < a["score_lo"]
            ),
            "random_baseline": rand["score"] if rand else None,
            "trivial_baseline": triv["score"] if triv else None,
            "mae": mae["score"] if mae else None,
            # The transfer claim in one boolean: a simulation-pretrained encoder
            # only carries something to real tissue if it clears the *untrained*
            # network of the same architecture on the same crops.
            "beats_random": (
                None if rand is None else bool(max(a["score"], b["score"]) > rand["score"])
            ),
            "best_vicreg_beats_mae": (
                None if mae is None else bool(max(a["score"], b["score"]) > mae["score"])
            ),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="mm_matched")
    p.add_argument("--store", default=None, help="defaults to the store for --variant")
    p.add_argument("--checkpoint-root", default=CKPT_ROOT, dest="checkpoint_root")
    p.add_argument("--out", default=None, help="defaults to outputs/stanford/{variant}_k{k}")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=None, dest="crop_size")
    p.add_argument("--crops-per-frame", type=int, default=16, dest="crops_per_frame")
    p.add_argument("--limit-frames", type=int, default=0, help="smoke testing only")
    p.add_argument("--stage", default="c0_const", choices=["c0_const", "gt_sos"])
    p.add_argument("--reference-encoder", default="vicreg_A", dest="reference_encoder")
    p.add_argument("--projections", nargs="+", default=["pca"])
    p.add_argument("--max-points", type=int, default=4000, dest="max_points")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--external",
        action="store_true",
        help="add PLAN.md §4.4 baseline 5, a frozen natural-image ViT on the "
        "K=1 log-envelope stage; reported as a ceiling, in its own block",
    )
    p.add_argument(
        "--external-model",
        default=None,
        dest="external_model",
        help="override the backbone; default prefers ungated DINOv3, then DINOv2",
    )
    p.add_argument(
        "--intensity-scale",
        choices=["matched", "raw"],
        default="matched",
        dest="intensity_scale",
        help="match Stanford's amplitude scale to the pre-training corpus (default), "
        "or leave it raw as the control",
    )
    args = p.parse_args()

    if args.store is None:
        args.store = STORE_TEMPLATE.format(variant=args.variant)
    suffix = "" if args.intensity_scale == "matched" else "_rawscale"
    out_dir = Path(args.out or f"outputs/stanford/{args.variant}_k{args.k}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crop geometry from the same call the training loader made, so the encoder
    # is never handed crops of a size it was not trained on.
    arm = LoaderConfig.for_arm(args.store, k=args.k)
    if args.crop_size is None:
        args.crop_size = arm.crop_size

    store = open_store(args.store)
    frame_ids = list(store.frame_ids)
    if args.limit_frames:
        frame_ids = frame_ids[: args.limit_frames]
    print(
        f"[wi16] {args.variant} store {store.spec.nz}x{store.spec.nx} px | "
        f"K={args.k} crop={args.crop_size} in_ch={2 * args.k} | "
        f"frames={len(frame_ids)} (all held out — none was trained on)",
        flush=True,
    )

    # Built once per band because `in_channels` is fixed by K, not by the band;
    # the closure keeps the checkpoint-loading in `wi13_matrix`'s single copy.
    built: dict[int, dict] = {}

    def encoders_for(in_channels: int) -> dict:
        if in_channels not in built:
            built[in_channels] = build_encoders(args, in_channels, tag="stanford")
        return built[in_channels]

    bands = depth_bands(args.k, arm.z_min_mm)

    # `--intensity-scale raw` is the control: it is what "just run the encoder
    # on the other corpus" gives, and reporting it next to the matched row is
    # what separates a transfer result from a units mismatch.
    scale = resolve_intensity_scale(args)

    results: list[dict] = []
    contrast: list[dict] = []
    ref_feats, ref_labels = None, None
    for band, z_min_mm in bands.items():
        rows, feats, labels = _run_band(
            args, band, z_min_mm, frame_ids, encoders_for, scale
        )
        results += rows
        contrast += _contrast(rows, band)
        if band == next(iter(bands)):
            ref_feats, ref_labels = feats, labels

    ref = args.reference_encoder if args.reference_encoder in ref_feats else next(iter(ref_feats))
    clusters = cluster_report(ref_feats[ref], ref_labels)

    figs, geometry_r2 = [], None
    for method in args.projections:
        n = len(ref_feats[ref])
        sub = np.arange(n)
        if n > args.max_points:
            sub = np.random.default_rng(0).choice(n, args.max_points, replace=False)
        # Only the numeric axes are plottable as a colour; `session`/`source_file`
        # are object arrays of strings.
        labs = {
            k: v[sub] for k, v in ref_labels.items()
            if np.asarray(v).dtype.kind in "ifb"
        }
        figs.append(viz.plot_embedding(
            viz.project(ref_feats[ref][sub], method), labs,
            f"{ref} — Stanford {args.variant} — {method.upper()}",
            out_dir / f"embed_{ref}_{method}.png"))
        resid, geometry_r2 = viz.regress_out(
            ref_feats[ref], ref_labels, ["z_centre", "x_centre"])
        print(f"\n[wi16] geometry (z,x) explains {geometry_r2 * 100:.1f}% of {ref} variance")
        figs.append(viz.plot_embedding(
            viz.project(resid[sub], method), labs,
            f"{ref} — Stanford {args.variant} — {method.upper()} (geometry regressed out)",
            out_dir / f"embed_{ref}_{method}_residual.png"))

    not_run = dict(UNRUNNABLE)
    if "full" not in bands:
        not_run["full-depth band"] = (
            f"K={args.k} >= 8 sits below PLAN.md §3.2c's 8 mm sector-occupancy "
            "floor, which applies identically to both corpora — the restricted "
            "band IS the arm here, not a subset of a wider one"
        )
    other = "lambda_matched" if args.variant == "mm_matched" else "mm_matched"
    if not Path(STORE_TEMPLATE.format(variant=other)).exists():
        not_run[f"{other} variant"] = (
            "PLAN.md §3.6's grid ablation needs both; run "
            f"scripts/beamform/build_stanford_store.py --variant {other}"
        )
    for k in (16, 8, 4, 1):
        if k == args.k:
            continue
        if all(find_checkpoint(args.checkpoint_root, k, pol) for pol in ("A", "B")):
            continue
        not_run[f"K={k} column"] = (
            "K sets the stem's input channel count, so this cell needs its own "
            "VICReg run; no trained pair found for this K"
        )

    payload = {
        "wi": 16,
        "corpus": "stanford_fsa",
        "variant": args.variant,
        "store": str(args.store),
        "grid": {"nz": store.spec.nz, "nx": store.spec.nx,
                 "z_max_mm": DEPTH_MAX_MM,
                 "spacing_mm": store.beamform_params.get("grid_spacing_mm")},
        "k": args.k,
        "crop_size": args.crop_size,
        "stage": args.stage,
        "split": "none — the whole Stanford corpus is held out by construction",
        "n_frames": len(frame_ids),
        "crops_per_frame": args.crops_per_frame,
        "bands": bands,
        "intensity_scale_mode": args.intensity_scale,
        "intensity_scale": scale,
        # PLAN.md §4.4 requires reporting *which* external model was used, since
        # the DINOv3 gate means it is not always the one the plan names.
        "external_baseline": (
            None if not args.external else {
                "encoder": EXTERNAL_NAME,
                "model_id": external_encoder(args).model_id,
                "skipped": external_encoder(args).skipped,
                "stage": "K=1 log-envelope B-mode, grey replicated to 3 channels",
                "read_as": "ceiling — §4.4's reporting rule, not part of the "
                           "internal ranking",
            }
        ),
        "encoders": sorted({r["encoder"] for r in results}),
        "reference_encoder": ref,
        "geometry_variance_explained": geometry_r2,
        "clusters": clusters,
        "results": results,
        "policy_contrast": contrast,
        "not_run": not_run,
        "figures": [str(f) for f in figs],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    _write_tables(payload, out_dir / "results.md")
    print(f"\n[stanford-matrix] wrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")
    return 0


def _write_tables(payload: dict, path: Path) -> None:
    lines = [
        f"# WI-16 Stanford transfer — {payload['variant']}, K={payload['k']}",
        "",
        f"Frozen NVIDIA-pretrained encoders on {payload['n_frames']} real in-vivo "
        f"rat frames ({payload['crop_size']} px crops). No retraining.",
        "",
        "## Policy A vs Policy B",
        "",
    ]
    cols = ["band", "target", "level", "metric", "policy_a", "policy_b",
            "delta_b_minus_a", "separated", "random_baseline", "beats_random",
            "trivial_baseline", "mae", "best_vicreg_beats_mae"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in payload["policy_contrast"]:
        lines.append("| " + " | ".join(
            f"{c[k]:+.3f}" if isinstance(c.get(k), float) else str(c.get(k, ""))
            for k in cols) + " |")

    cols2 = ["encoder", "band", "crop_size", "target", "level", "metric",
             "score", "ci95", "n_eff", "n_feat"]

    def _block(rows):
        out = ["| " + " | ".join(cols2) + " |", "|" + "---|" * len(cols2)]
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols2) + " |")
        return out

    internal = [r for r in payload["results"] if r["encoder"] != EXTERNAL_NAME]
    external = [r for r in payload["results"] if r["encoder"] == EXTERNAL_NAME]
    lines += ["", "## All cells — internal arms and baselines", ""]
    lines += _block(internal)

    # §4.4: "kept in visually separate blocks of the results table so they are
    # not read as a single ranking." A frozen natural-image model on B-mode is a
    # ceiling; it is not competing with the arms on the same input.
    if external:
        ext = payload.get("external_baseline") or {}
        lines += [
            "", "## External frozen baseline — a CEILING, read separately", "",
            f"`{ext.get('model_id', '?')}` on {ext.get('stage', 'B-mode')}. "
            "It sees a different input stage from every row above (log-envelope "
            "B-mode, not the K-sector IQ stack), so this block answers *\"does an "
            "off-the-shelf encoder already do this?\"* and **not** *\"are "
            "sub-apertures better than B-mode?\"* — PLAN.md §4.4 names the "
            "internal signal-stage ablation as the number to cite for that.", "",
        ]
        lines += _block(external)

    lines += ["", "## Cells not run", "", "| cell | reason |", "|---|---|"]
    for cell, reason in payload["not_run"].items():
        lines.append(f"| {cell} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
