"""
Audit the train/val/test splits produced by the dataset loader for leakage.

Given a split strategy (``subject`` or ``file``), this script rebuilds the exact
splits the training code would see and then answers two questions:

  1. **Is the partition clean?** — every frame lands in exactly one split, no
     frame index is shared, no ``.mat`` cine is torn across splits (both of its
     temporal frames must stay together), and — for the ``subject`` strategy —
     no rat appears in more than one split. These are hard assertions; the
     script exits non-zero if any fails.

  2. **Are the splits *too similar* to be a fair test?** Even a formally-disjoint
     split leaks information if a held-out frame is nearly identical to a
     training frame (adjacent temporal frames, same rat/probe pose, …). We
     fingerprint every frame from its log-compressed IQ envelope and compare:
       * within-split vs. cross-split pairwise-similarity distributions — if the
         two overlap, held-out frames are no more novel than training frames;
       * each held-out frame's nearest *training* neighbour — the concrete
         "how close is the closest thing the model already saw" leakage metric;
       * a 2D PCA embedding coloured by split / marked by subject, to eyeball
         whether the splits occupy distinct or overlapping regions.

The figure bundles these together with the subject- and file-composition of each
split. A clean subject split should show clearly separated similarity
distributions and per-rat-disjoint splits; a frame/file split deliberately mixes
rats and will show heavy distribution overlap — which is exactly the leakage this
plot is meant to make visible.

Usage:
    poetry run python scripts/dataset_scripts/analyze_splits.py --split-type subject
    poetry run python scripts/dataset_scripts/analyze_splits.py \
        --split-type subject --train-subjects rat1 --val-subjects rat2
    poetry run python scripts/dataset_scripts/analyze_splits.py \
        --split-type file --focal-zone 0 --out figures/split_audit_file.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Make the installed package importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ssl_ultrasound_representation.dataset.dataset import (  # noqa: E402
    load_dataset,
    load_local_dataset,
    split_by_file,
    split_by_subject,
)

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
# Cross-split similarity at or above this (cosine of frame fingerprints) is
# flagged: a held-out frame that resembles a training frame this closely is a
# near-duplicate the model effectively already saw.
LEAKAGE_SIM_THRESHOLD = 0.98


# ── data + split construction ─────────────────────────────────────────────────
def build_splits(args) -> tuple["object", dict, dict[str, list[int]]]:
    """Load the array+metadata and return the requested split index lists."""
    if args.repo_id is not None:
        arr, meta = load_dataset(
            args.repo_id, max_frames=args.max_frames, cache_dir=args.cache_dir
        )
    else:
        arr, meta = load_local_dataset(args.local_zip, args.local_metadata)

    if args.split_type == "subject":
        subj_kwargs = {}
        if args.train_subjects or args.val_subjects or args.test_subjects:
            subj_kwargs = dict(
                train_subjects=args.train_subjects or [],
                val_subjects=args.val_subjects or [],
                test_subjects=args.test_subjects or [],
            )
        splits = split_by_subject(
            meta, ratios=tuple(args.ratios), seed=args.seed, **subj_kwargs
        )
    else:
        splits = split_by_file(
            meta, ratios=tuple(args.ratios), seed=args.seed
        )
    return arr, meta, splits


# ── leakage audit (hard checks) ───────────────────────────────────────────────
def audit_leakage(meta: dict, splits: dict[str, list[int]]) -> list[str]:
    """Return a list of human-readable failures (empty == clean partition)."""
    frames = meta["frames"]
    by_index = {fm["global_frame_index"]: fm for fm in frames}
    failures: list[str] = []

    # 1. disjoint frame indices + full coverage
    seen: dict[int, str] = {}
    for name in SPLIT_ORDER:
        for idx in splits.get(name, []):
            if idx in seen:
                failures.append(
                    f"frame {idx} appears in both {seen[idx]!r} and {name!r}"
                )
            seen[idx] = name
    all_meta_idx = set(by_index)
    assigned = set(seen)
    if assigned - all_meta_idx:
        failures.append(f"split references unknown frame indices {assigned - all_meta_idx}")
    missing = all_meta_idx - assigned
    if missing:
        failures.append(f"{len(missing)} metadata frames unassigned to any split: {sorted(missing)}")

    # 2. no .mat cine torn across splits (its temporal frames must stay together)
    file_splits: dict[str, set[str]] = defaultdict(set)
    for idx, name in seen.items():
        file_splits[by_index[idx]["relative_path"]].add(name)
    for rp, names in file_splits.items():
        if len(names) > 1:
            failures.append(f"file {rp!r} split across {sorted(names)} (temporal-frame leakage)")

    # 3. subject disjointness (only a *requirement* for the subject strategy)
    subj_splits: dict[str, set[str]] = defaultdict(set)
    for idx, name in seen.items():
        subj_splits[by_index[idx]["subject"]].add(name)
    shared = {s: sorted(n) for s, n in subj_splits.items() if len(n) > 1}
    return failures, shared, seen, by_index


# ── frame fingerprints + similarity ───────────────────────────────────────────
def frame_fingerprint(arr, idx: int, focal_zone: int, downsample: int) -> np.ndarray:
    """
    A compact, pose-sensitive descriptor of one frame.

    Read a single focal zone ``(transducer, scanline, iq, sample)``, take the IQ
    envelope (|I+jQ|), average over the transducer aperture to a ``(scanline,
    sample)`` pseudo-image, log-compress (dB), and coarse-bin to a small fixed
    grid. Near-duplicate acquisitions map to near-identical vectors, so cosine
    similarity between fingerprints exposes "too similar" held-out frames.
    """
    block = np.asarray(arr[idx, :, :, focal_zone])  # (T, S, 2, samp)
    env = np.sqrt(block[..., 0, :] ** 2 + block[..., 1, :] ** 2)  # (T, S, samp)
    img = env.mean(axis=0)  # (S, samp) — average over transducer aperture
    img = 20.0 * np.log10(img + 1e-6)
    # coarse-bin to (downsample, downsample) via block-mean (trim to divisible)
    s, t = img.shape
    ds = downsample
    img = img[: s - s % ds, : t - t % ds]
    img = img.reshape(ds, img.shape[0] // ds, ds, img.shape[1] // ds).mean(axis=(1, 3))
    vec = img.ravel()
    vec = vec - vec.mean()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_fingerprints(arr, indices, focal_zone, downsample):
    return {i: frame_fingerprint(arr, i, focal_zone, downsample) for i in indices}


def by_index_subject(by_index: dict, relative_path: str) -> str:
    """The subject (rat) owning a given file's frames, for compact row labels."""
    for fm in by_index.values():
        if fm["relative_path"] == relative_path:
            return fm["subject"]
    return "?"


def cosine_matrix(order, fps) -> np.ndarray:
    mat = np.stack([fps[i] for i in order])
    return mat @ mat.T


# ── plotting ──────────────────────────────────────────────────────────────────
def make_figure(meta, splits, seen, by_index, fps, args, shared, failures):
    order = [i for name in SPLIT_ORDER for i in splits.get(name, [])]
    split_of = [seen[i] for i in order]
    subj_of = [by_index[i]["subject"] for i in order]
    sim = cosine_matrix(order, fps)
    n = len(order)

    # within- vs cross-split similarity (upper triangle, excl. diagonal)
    iu, ju = np.triu_indices(n, k=1)
    within, cross = [], []
    for a, b in zip(iu, ju):
        (within if split_of[a] == split_of[b] else cross).append(sim[a, b])
    within, cross = np.array(within), np.array(cross)

    # nearest *training* neighbour for each held-out frame
    train_pos = [k for k, nm in enumerate(split_of) if nm == "train"]
    nn = []  # (label, frame_idx, best_sim, best_train_idx)
    for k, nm in enumerate(split_of):
        if nm == "train" or not train_pos:
            continue
        sims = [(sim[k, tp], order[tp]) for tp in train_pos]
        best_s, best_i = max(sims)
        nn.append((nm, order[k], best_s, best_i))

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (a) subject composition per split — the "subject overlap" view
    ax = fig.add_subplot(gs[0, 0])
    subjects = sorted({by_index[i]["subject"] for i in order})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(subjects), 1)))
    subj_color = dict(zip(subjects, cmap))
    counts = {nm: defaultdict(int) for nm in SPLIT_ORDER}
    for i in order:
        counts[seen[i]][by_index[i]["subject"]] += 1
    x = np.arange(len(SPLIT_ORDER))
    bottom = np.zeros(len(SPLIT_ORDER))
    for s in subjects:
        h = np.array([counts[nm][s] for nm in SPLIT_ORDER])
        ax.bar(x, h, bottom=bottom, color=subj_color[s], label=s, edgecolor="white")
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylabel("# frames")
    title = "Subject composition per split"
    if shared:
        title += f"\n⚠ shared subjects: {shared}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, title="rat", loc="upper right", framealpha=0.9)

    # (b) file → split assignment matrix (confirms cines stay intact)
    ax = fig.add_subplot(gs[0, 1])
    files = sorted({by_index[i]["relative_path"] for i in order})
    fmat = np.full((len(files), len(SPLIT_ORDER)), np.nan)
    for i in order:
        r = files.index(by_index[i]["relative_path"])
        c = SPLIT_ORDER.index(seen[i])
        fmat[r, c] = (fmat[r, c] if not np.isnan(fmat[r, c]) else 0) + 1
    ax.imshow(np.nan_to_num(fmat), aspect="auto", cmap="Blues", vmin=0)
    for r in range(len(files)):
        for c in range(len(SPLIT_ORDER)):
            if not np.isnan(fmat[r, c]):
                ax.text(c, r, int(fmat[r, c]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(SPLIT_ORDER)))
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_yticks(range(len(files)))
    # label by rat + the distinguishing time tail of the .mat name (…_HHMMSS.mat)
    ax.set_yticklabels(
        [f"{by_index_subject(by_index, f)}/…{Path(f).stem.split('_')[-1]}" for f in files],
        fontsize=7,
    )
    ax.set_title("Frames per file per split\n(a row in >1 column = leakage)", fontsize=10)

    # (c) frame similarity heatmap, blocked by split
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(sim, cmap="magma", vmin=-1, vmax=1)
    boundaries, acc = [], 0
    for nm in SPLIT_ORDER:
        c = len(splits.get(nm, []))
        if c:
            acc += c
            boundaries.append(acc)
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="cyan", lw=1)
        ax.axvline(b - 0.5, color="cyan", lw=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine sim")
    ax.set_title("Frame fingerprint similarity\n(blocks = splits)", fontsize=10)
    ax.set_xlabel("frame (ordered by split)")

    # (d) within- vs cross-split similarity distributions — the leakage histogram
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-1, 1, 31)
    if within.size:
        ax.hist(within, bins=bins, alpha=0.6, density=True, label=f"within (n={within.size})", color="#4c72b0")
    if cross.size:
        ax.hist(cross, bins=bins, alpha=0.6, density=True, label=f"cross (n={cross.size})", color="#c44e52")
    ax.axvline(LEAKAGE_SIM_THRESHOLD, color="k", ls="--", lw=1, label=f"leak≥{LEAKAGE_SIM_THRESHOLD}")
    ax.set_xlabel("pairwise cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("Within- vs cross-split similarity\n(overlap ⇒ held-out not novel)", fontsize=10)
    ax.legend(fontsize=8)

    # (e) nearest-training-neighbour similarity for each held-out frame
    ax = fig.add_subplot(gs[1, 1])
    if nn:
        labels = [f"{nm[:3]} f{fi}→f{ti}" for nm, fi, _, ti in nn]
        vals = [s for _, _, s, _ in nn]
        colors = [SPLIT_COLORS[nm] for nm, *_ in nn]
        ax.barh(range(len(nn)), vals, color=colors)
        ax.axvline(LEAKAGE_SIM_THRESHOLD, color="k", ls="--", lw=1)
        ax.set_yticks(range(len(nn)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("max cosine sim to a TRAIN frame")
        ax.set_xlim(min(0.0, min(vals) - 0.05), 1.0)
    else:
        ax.text(0.5, 0.5, "no held-out frames", ha="center", va="center")
    ax.set_title("Held-out → nearest train neighbour\n(bars past dashed line = leakage)", fontsize=10)

    # (f) 2D PCA embedding of fingerprints
    ax = fig.add_subplot(gs[1, 2])
    mat = np.stack([fps[i] for i in order])
    mc = mat - mat.mean(0)
    try:
        u, s_, vt = np.linalg.svd(mc, full_matrices=False)
        emb = u[:, :2] * s_[:2]
    except np.linalg.LinAlgError:
        emb = np.zeros((n, 2))
    subj_markers = {s: m for s, m in zip(subjects, ["o", "s", "^", "D", "v", "P", "*"])}
    for k in range(n):
        ax.scatter(
            emb[k, 0], emb[k, 1], c=SPLIT_COLORS[split_of[k]],
            marker=subj_markers.get(subj_of[k], "o"), s=70, edgecolor="k", linewidth=0.4,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Fingerprint PCA\n(color=split, marker=rat)", fontsize=10)
    legend_el = [Patch(color=SPLIT_COLORS[nm], label=nm) for nm in SPLIT_ORDER if splits.get(nm)]
    ax.legend(handles=legend_el, fontsize=8)

    status = "CLEAN" if not failures else f"{len(failures)} LEAKAGE FAILURE(S)"
    fig.suptitle(
        f"Split audit — strategy={args.split_type!r}  focal_zone={args.focal_zone}  "
        f"n_frames={n}  [{status}]",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nFigure written to {args.out}")
    return within, cross, nn


# ── report ────────────────────────────────────────────────────────────────────
def print_report(meta, splits, failures, shared, within, cross, nn, args):
    bar = "─" * 64
    print(f"\n{bar}\nSPLIT AUDIT — strategy={args.split_type!r}\n{bar}")
    for nm in SPLIT_ORDER:
        idx = splits.get(nm, [])
        subs = sorted({fm["subject"] for fm in meta["frames"] if fm["global_frame_index"] in idx})
        print(f"  {nm:5s}: {len(idx):3d} frames  subjects={subs}  indices={idx}")

    print(f"\n  Hard checks:")
    if not failures:
        print("    ✓ disjoint indices, full coverage, no torn cines")
    else:
        for f in failures:
            print(f"    ✗ {f}")
    if shared:
        msg = "    ⚠ subjects shared across splits: " + str(shared)
        print(msg + ("  (expected for 'file' strategy)" if args.split_type == "file"
                     else "  (UNEXPECTED for 'subject' strategy)"))
    else:
        print("    ✓ no subject appears in more than one split")

    print(f"\n  Distribution similarity (frame fingerprints):")
    if within.size:
        print(f"    within-split : mean={within.mean():.3f}  max={within.max():.3f}")
    if cross.size:
        print(f"    cross-split  : mean={cross.mean():.3f}  max={cross.max():.3f}")
        n_leak = int((cross >= LEAKAGE_SIM_THRESHOLD).sum())
        flag = "✗" if n_leak else "✓"
        print(f"    {flag} {n_leak} cross-split pair(s) ≥ {LEAKAGE_SIM_THRESHOLD} (near-duplicate)")
    if nn:
        worst = max(nn, key=lambda r: r[2])
        print(f"    worst held-out→train: {worst[0]} frame {worst[1]} ~ train frame "
              f"{worst[3]} (sim={worst[2]:.3f})")
    print(bar)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-type", choices=["subject", "file"], default="subject",
                   help="how the loader partitions frames (default: subject)")
    p.add_argument("--local-zip", default="data/processed/ultrasonic_dataset.zarr.zip")
    p.add_argument("--local-metadata", default="data/processed/ultrasonic_dataset.metadata.json")
    p.add_argument("--repo-id", default=None, help="load from HF instead of local zip")
    p.add_argument("--max-frames", type=int, default=None, help="HF path only")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-subjects", nargs="*", default=None)
    p.add_argument("--val-subjects", nargs="*", default=None)
    p.add_argument("--test-subjects", nargs="*", default=None)
    p.add_argument("--focal-zone", type=int, default=0)
    p.add_argument("--downsample", type=int, default=24,
                   help="fingerprint grid resolution (NxN)")
    p.add_argument("--out", default="figures/split_audit.png")
    args = p.parse_args()
    args.out = str(args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    arr, meta, splits = build_splits(args)
    failures, shared, seen, by_index = audit_leakage(meta, splits)

    indices = [i for nm in SPLIT_ORDER for i in splits.get(nm, [])]
    if not indices:
        print("No frames in any split — nothing to analyse.")
        sys.exit(1)
    print(f"Fingerprinting {len(indices)} frames (focal_zone={args.focal_zone}, "
          f"grid={args.downsample}x{args.downsample}) ...")
    fps = compute_fingerprints(arr, indices, args.focal_zone, args.downsample)

    within, cross, nn = make_figure(meta, splits, seen, by_index, fps, args, shared, failures)
    print_report(meta, splits, failures, shared, within, cross, nn, args)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
