"""Presentation plots and number tables, built from the committed `results/`.

No GPU, no store, no checkpoints — everything here reads `results/*.json`, so
it re-renders in a couple of seconds and cannot silently drift from the
published tables. The plots that *do* need the corpus and the trained weights
live in `scripts/figures/latent_3d.py`.

    python scripts/figures/presentation.py                  # every plot + table
    python scripts/figures/presentation.py --only rankme
    python scripts/figures/presentation.py --no-tables

**Figures are bare on purpose**: no title, subtitle or caption is drawn, because
these go into a deck that supplies its own. What stays inside the frame is only
what is needed to *read the data* — axes, legends, the chance level of a
classification column, and any annotation that states a number. See
`outputs/presentation/README.md` for what each one shows.

Every plot also gets a matching table under `outputs/presentation/tables/`, as
both Markdown (paste into a slide) and CSV (re-plot it yourself).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssl_us.eval import slides

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "outputs" / "presentation"
TABLES = OUT / "tables"

# Display names. The keys are the encoder ids used throughout `results/`.
NAMES = {
    "vicreg_A": "VICReg A\nResNet-18",
    "vicreg_B": "VICReg B\nResNet-18",
    "vicreg_vit_A": "VICReg A\nViT-S",
    "ijepa": "I-JEPA\nViT-S",
    "mae_norm": "MAE\nnorm-pixel",
    "mae_raw": "MAE\nraw-pixel",
    "random_resnet18": "random\nResNet-18",
    "random_vit": "random\nViT-S",
    "pca_raw": "PCA\non pixels",
    "trivial": "trivial\n(2 feats)",
}
# Trained arms first, then the untrained floors. A reader compares down the
# column; the gap between the two groups is where "did training buy anything"
# is answered.
TRAINED = ["vicreg_A", "vicreg_B", "vicreg_vit_A", "ijepa", "mae_norm", "mae_raw"]
BASELINE = ["random_resnet18", "random_vit", "pca_raw", "trivial"]

TARGET_NAMES = {
    "sos_mean_frame": "sound speed\n(frame)",
    "sos_std_frame": "SoS spread\n(frame)",
    "sos_mean_crop": "sound speed\n(crop)",
    "phase_error": "phase error\n(frame)",
    "cyst_fraction": "cyst fraction\n(frame)",
    "cyst_fraction_crop": "cyst fraction\n(crop)",
    "z_centre": "depth\n(crop)",
    "x_centre": "lateral pos\n(crop)",
    "contains_cyst": "contains cyst\n(crop)",
    "t0_group": "t0 group\n(frame)",
    "echogenicity": "echogenicity\n(crop)",
    "session_id": "session\n(frame)",
    "date_id": "acq. day\n(frame)",
    "kind_fast": "preset\n(frame)",
}

# Chance level for every balanced-accuracy target, = 1 / n_classes. Balanced
# accuracy does NOT have a fixed 0.5 floor: `session_id` is a 30-way problem
# and `date_id` a 10-way one, so 0.28 there is ~8x chance while 0.28 on a
# binary target is below it. Colouring both against a 0.5 floor made the
# multi-class columns read as "nothing". Counts verified against the corpora:
# Stanford `frame_metadata.json` has 30 distinct `session` and 10 distinct
# `date` values over its 117 frames and two `kind` presets; NVIDIA's has
# exactly 3 distinct `t0` values over its 923.
CHANCE = {
    "t0_group": 1 / 3,
    "contains_cyst": 0.5,
    "session_id": 1 / 30,
    "date_id": 1 / 10,
    "kind_fast": 0.5,
}

COLUMNS = {
    "nvidia": "nvidia/k4_validation.json",
    "stanford": "stanford/mm_matched_k4.json",
}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _rows(doc: dict, metric: str, band: str | None = None) -> tuple[list, list, np.ndarray]:
    """(encoders, targets, score matrix) for one metric, in stable order."""
    res = [r for r in doc["results"] if r["metric"] == metric]
    if band is not None:
        res = [r for r in res if r.get("band", band) == band]
    if not res:
        return [], [], np.zeros((0, 0))
    targets, seen = [], set()
    for r in res:
        if r["target"] not in seen:
            targets.append(r["target"])
            seen.add(r["target"])
    order = [e for e in TRAINED + BASELINE if e in doc["encoders"]]
    lookup = {(r["encoder"], r["target"]): r["score"] for r in res}
    mat = np.full((len(order), len(targets)), np.nan)
    for i, e in enumerate(order):
        for j, t in enumerate(targets):
            if (e, t) in lookup:
                mat[i, j] = lookup[(e, t)]
    return order, targets, mat


def _flat(name: str) -> str:
    return NAMES.get(name, name).replace("\n", " ")


# ── tables ──────────────────────────────────────────────────────────────────

def _write_table(name: str, header: list[str], rows: list[list], note: str) -> None:
    """Write `<name>.md` and `<name>.csv`. Markdown to paste, CSV to re-plot."""
    TABLES.mkdir(parents=True, exist_ok=True)
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    md += ["", f"_{note}_", ""]
    (TABLES / f"{name}.md").write_text("\n".join(md))
    with (TABLES / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def table_probe_matrix(corpus: str) -> None:
    doc = _load(COLUMNS[corpus])
    band = "full" if corpus == "stanford" else None
    for metric, tag, fmt in (("r2", "r2", "{:+.3f}"),
                             ("balanced_accuracy", "accuracy", "{:.3f}")):
        enc, tgt, mat = _rows(doc, metric, band)
        if not enc:
            continue
        head = ["encoder"] + [TARGET_NAMES.get(x, x).replace("\n", " ") for x in tgt]
        rows = [[_flat(e)] + [("" if np.isnan(v) else fmt.format(v)) for v in mat[i]]
                for i, e in enumerate(enc)]
        if metric == "balanced_accuracy":
            rows.append(["**chance**"] + [f"{CHANCE[x]:.3f}" for x in tgt])
        note = (f"{corpus} · results/{COLUMNS[corpus]} · K={doc['k']} · "
                f"{doc['n_frames']} frames · "
                + (f"{doc['variant']} grid, full band, amplitude "
                   f"{doc['intensity_scale_mode']}" if corpus == "stanford"
                   else f"{doc['split']} split")
                + f" · metric: {metric}")
        _write_table(f"{corpus}_probe_{tag}", head, rows, note)


def table_rankme() -> None:
    doc = _load("stanford/mm_matched_k4_rankme.json")
    rows = []
    for e in [x for x in TRAINED + BASELINE if x in doc["encoders"]]:
        v = doc["encoders"][e]
        rows.append([_flat(e), v["nvidia"]["dim"],
                     f"{v['nvidia']['rankme']:.1f}", f"{v['stanford']['rankme']:.1f}",
                     f"{v['stanford']['pc1_variance_frac']:.3f}",
                     v["stanford"]["dims_for_90pct"]])
    _write_table("rankme",
                 ["encoder", "feature dim", "RankMe NVIDIA", "RankMe Stanford",
                  "Stanford PC1 var frac", "Stanford dims for 90% var"], rows,
                 "results/stanford/mm_matched_k4_rankme.json · K=4 · mm-matched · "
                 "117 frames per corpus so N is not a confound · "
                 "vicreg_vit_A predates this column")


def table_mask_sweep() -> None:
    doc = _load("mae/mask_sweep.json")
    rows = []
    for batch, runs in (("256", doc["runs_batch256"]), ("64", doc["runs_batch64"])):
        for r in runs:
            if r["status"] != "ok":
                continue
            rows.append([r["patch"], f"{r['mask']:.2f}", batch, r["tokens"],
                         f"{r['loss_min']:.4f}", f"{r['variance_explained'] * 100:+.2f}"])
    rows.sort(key=lambda r: (-int(r[0]), -float(r[1])))
    acf = doc["speckle_acf"]
    _write_table("mae_mask_sweep",
                 ["patch px", "mask ratio", "batch", "tokens", "min loss",
                  "variance explained %"], rows,
                 "results/mae/mask_sweep.json · 12 epochs on 120 frames per cell · "
                 "norm_pix_loss makes the target unit-variance, so variance explained "
                 f"= 1 - loss · speckle autocorrelation halves within "
                 f"{acf['axial_half_width_px']} px axially and "
                 f"{acf['lateral_half_width_px']} px laterally")


def table_policy_contrast() -> None:
    doc = _load("nvidia/k4_validation.json")
    rows = [[TARGET_NAMES.get(r["target"], r["target"]).replace("\n", " "), r["metric"],
             f"{r['policy_a']:+.3f}", f"{r['policy_b']:+.3f}",
             f"{r['delta_b_minus_a']:+.3f}", "yes" if r["separated"] else "no",
             f"{r['random_baseline']:+.3f}"] for r in doc["policy_contrast"]]
    _write_table("policy_contrast",
                 ["target", "metric", "Policy A", "Policy B", "B - A", "CIs separated",
                  "random baseline"], rows,
                 "results/nvidia/k4_validation.json · policy_contrast block · K=4 · "
                 "validation split · 'CIs separated' = the 95% intervals do not overlap")


# ── plots ───────────────────────────────────────────────────────────────────

def _heatmap(ax, mat, encoders, targets, theme, *, vmin, vmax, fmt="{:+.2f}",
             col_floor=None, col_labels=None):
    """Sequential-blue cell grid with in-cell values.

    `col_floor` gives each column its own zero point — needed for balanced
    accuracy, where chance depends on the class count. Colour is then "fraction
    of the way from this column's chance level to perfect" in every column, a
    monotone transform of the printed number within the column.

    In-cell text ink is picked per cell by `slides.ink_on`, which measures the
    cell's luminance rather than branching on the theme; both candidates are
    text tokens, so a series colour is never used for text.
    """
    t = slides.tokens(theme)
    cmap = slides.sequential_cmap(theme)
    floors = col_floor if col_floor is not None else [vmin] * mat.shape[1]
    frac = np.full(mat.shape, np.nan)
    for j, f in enumerate(floors):
        frac[:, j] = (np.clip(mat[:, j], f, vmax) - f) / max(vmax - f, 1e-9)
    ax.imshow(frac, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(col_labels or [TARGET_NAMES.get(x, x) for x in targets], fontsize=9.5)
    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels([NAMES.get(e, e) for e in encoders], fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(encoders), 1), minor=True)
    # A 2px surface gap between cells, not a border drawn around each mark.
    ax.grid(which="minor", color=t["surface"], linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color=slides.ink_on(cmap(frac[i, j]), theme))
    return cmap


def fig_probe_matrix(theme: str, corpus: str) -> Path:
    """Encoder x target decodability, as a cell grid."""
    t = slides.use(theme)
    doc = _load(COLUMNS[corpus])
    band = "full" if corpus == "stanford" else None
    enc_r2, tgt_r2, mat_r2 = _rows(doc, "r2", band)
    enc_ba, tgt_ba, mat_ba = _rows(doc, "balanced_accuracy", band)
    has_ba = mat_ba.size > 0

    widths = [len(tgt_r2), max(len(tgt_ba), 1)]
    fig, axes = plt.subplots(
        1, 2 if has_ba else 1, figsize=(15.5, 5.9),
        gridspec_kw={"width_ratios": widths if has_ba else [1], "wspace": 0.12},
        squeeze=False)
    ax0 = axes[0][0]
    _heatmap(ax0, mat_r2, enc_r2, tgt_r2, theme, vmin=0.0, vmax=1.0)
    ax0.set_title("linear probe R²   0 = nothing decodable · 1 = fully decodable",
                  fontsize=11.5, color=t["secondary"], pad=10, loc="left")
    if has_ba:
        ax1 = axes[0][1]
        _heatmap(ax1, mat_ba, enc_ba, tgt_ba, theme, vmin=0.0, vmax=1.0, fmt="{:.2f}",
                 col_floor=[CHANCE[x] for x in tgt_ba],
                 col_labels=[f"{TARGET_NAMES.get(x, x)}\nchance {CHANCE[x]:.2f}"
                             for x in tgt_ba])
        ax1.set_yticklabels([])
        ax1.set_title("balanced accuracy   shaded from each column's own chance level",
                      fontsize=11.5, color=t["secondary"], pad=10, loc="left")

    # No colourbar: every cell already carries its value, so a scale strip
    # would re-encode what is printed and steal width from the grid.
    n_trained = sum(1 for e in enc_r2 if e in TRAINED)
    n_rows = len(enc_r2)
    for ax in [a for row in axes for a in row]:
        ax.axhline(n_trained - 0.5, color=t["secondary"], linewidth=1.6)
    # Name the two row groups: with no caption, the rule is otherwise unexplained.
    last = axes[0][-1]
    for label, lo, hi in (("trained", 0, n_trained), ("untrained", n_trained, n_rows)):
        last.text(1.012, 1 - (lo + hi) / 2 / n_rows, label, transform=last.transAxes,
                  rotation=90, ha="left", va="center", fontsize=10, color=t["muted"])

    slides.tight(fig, top=0.90, bottom=0.155, left=0.078, right=0.965)
    return slides.save(fig, OUT, f"03_probe_matrix_{corpus}", theme)


def fig_rankme(theme: str) -> Path:
    """Does the representation survive the domain shift, or collapse?"""
    t = slides.use(theme)
    doc = _load("stanford/mm_matched_k4_rankme.json")
    enc = [e for e in TRAINED + BASELINE if e in doc["encoders"]]
    nv = [doc["encoders"][e]["nvidia"]["rankme"] for e in enc]
    st = [doc["encoders"][e]["stanford"]["rankme"] for e in enc]
    c_nv, c_st = slides.series(theme, 2)

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    y = np.arange(len(enc))
    h = 0.36
    # 2px surface gap between adjacent bars, not a stroke around them.
    ax.barh(y - h / 2, nv, height=h, color=c_nv, label="NVIDIA (simulation, in-domain)",
            linewidth=2, edgecolor=t["surface"])
    ax.barh(y + h / 2, st, height=h, color=c_st, label="Stanford (real tissue, transfer)",
            linewidth=2, edgecolor=t["surface"])
    ax.set_yticks(y)
    ax.set_yticklabels([_flat(e) for e in enc], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("RankMe  (effective dimensions used; higher = richer representation)")
    slides.strip_chrome(ax, grid_axis="x")

    # Direct-label selectively: the collapse is the point of the chart.
    i = enc.index("vicreg_B")
    ax.annotate(
        f"collapse: {doc['encoders']['vicreg_B']['nvidia']['rankme']:.0f} → "
        f"{doc['encoders']['vicreg_B']['stanford']['rankme']:.0f} dims\n"
        f"{doc['encoders']['vicreg_B']['stanford']['pc1_variance_frac'] * 100:.0f}% "
        "of variance in one direction",
        xy=(st[i], i + h / 2), xytext=(30, i + h / 2), fontsize=10.5,
        color=t["primary"], ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=t["secondary"], linewidth=1.1))
    ax.legend(loc="lower right")
    slides.tight(fig, top=0.975, bottom=0.115, left=0.145, right=0.985)
    return slides.save(fig, OUT, "04_rankme_collapse", theme)


def fig_mask_sweep(theme: str) -> Path:
    """Why pixel-space masked reconstruction has nothing to predict on speckle.

    The axis is zoomed to the data, which is the only way the individual cells
    are distinguishable at all — so the annotation stating the axis span is
    load-bearing, not decoration. Without it the plot reads as a real effect.
    """
    t = slides.use(theme)
    doc = _load("mae/mask_sweep.json")
    runs = [r for r in doc["runs_batch256"] + doc["runs_batch64"] if r["status"] == "ok"]
    # De-duplicate (patch, mask) keeping the best variance explained: the
    # batch-64 rows repeat p16/p8 as the matched control for p4, so the same
    # cell appears twice and the *kinder* reading is the one to report.
    best: dict[tuple[int, float], float] = {}
    for r in runs:
        key = (r["patch"], r["mask"])
        best[key] = max(best.get(key, -9e9), r["variance_explained"])
    keys = sorted(best, key=lambda kv: (-kv[0], -kv[1]))
    vals = [best[k] * 100 for k in keys]
    c = slides.series(theme, 1)[0]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(keys))
    ax.axhline(0, color=t["axis"], linewidth=1.4, zorder=1)
    ax.vlines(x, 0, vals, color=t["grid"], linewidth=1.4, zorder=2)
    ax.scatter(x, vals, s=120, color=c, zorder=3, edgecolors=t["surface"], linewidths=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}px\n{int(m * 100)}%" for p, m in keys], fontsize=10)
    ax.set_ylabel("variance of the masked patch explained  (%)")
    ax.set_xlabel("patch size / mask ratio")
    ax.set_ylim(-0.85, 1.15)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    slides.strip_chrome(ax, grid_axis="y")
    ax.text(0.5, 0.965,
            "note the axis: it spans two percentage points — every cell is at zero",
            transform=ax.transAxes, ha="center", va="top", fontsize=11, color=t["primary"])
    slides.tight(fig, top=0.975, bottom=0.145, left=0.085, right=0.985)
    return slides.save(fig, OUT, "05_mae_mask_sweep", theme)


def fig_policy_contrast(theme: str) -> Path:
    """The augmentation policy decides which axes survive."""
    t = slides.use(theme)
    doc = _load("nvidia/k4_validation.json")
    rows = [r for r in doc["policy_contrast"] if r["metric"] == "r2"]
    rows.sort(key=lambda r: r["policy_a"] - r["policy_b"])
    labels = [TARGET_NAMES.get(r["target"], r["target"]).replace("\n", " ") for r in rows]
    a = [r["policy_a"] for r in rows]
    b = [r["policy_b"] for r in rows]
    c_a, c_b = slides.series(theme, 2)

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    y = np.arange(len(rows))
    for i, (va, vb) in enumerate(zip(a, b, strict=True)):
        ax.plot([vb, va], [i, i], color=t["axis"], linewidth=1.6, zorder=1)
    # 2px surface ring on the overlapping markers.
    ax.scatter(b, y, s=132, color=c_b, zorder=3, label="Policy B",
               edgecolors=t["surface"], linewidths=2)
    ax.scatter(a, y, s=132, color=c_a, zorder=3, label="Policy A",
               edgecolors=t["surface"], linewidths=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("linear probe R²")
    ax.axvline(0, color=t["axis"], linewidth=1.2)
    slides.strip_chrome(ax, grid_axis="x")
    ax.legend(loc="lower right")

    i = next(i for i, r in enumerate(rows) if r["target"] == "x_centre")
    ax.annotate("Policy A keeps lateral position;\nPolicy B destroys it entirely",
                xy=(b[i] + 0.006, i - 0.25), xytext=(0.06, i - 3.0), fontsize=11,
                color=t["primary"], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=t["secondary"], linewidth=1.1))
    slides.tight(fig, top=0.975, bottom=0.115, left=0.155, right=0.985)
    return slides.save(fig, OUT, "06_policy_contrast", theme)


def fig_pipeline(theme: str) -> Path:
    """The method schematic: what the data is and what happens to it.

    Hand-authored rather than data-derived, but it lives here because it needs
    the same palette and type as the rest. The one thing it has to communicate
    is *where training happens* — exactly one stage — since every evaluation in
    the rest of the set is on frozen weights, and an audience that misses that
    reads the transfer table as fine-tuning.
    """
    t = slides.use(theme)
    accent = slides.series(theme, 1)[0]
    stages = [
        ("RF channel data", ["NVIDIA: 923 simulated frames", "Stanford: 117 in-vivo rat",
                             "raw per-element echoes"]),
        ("Beamform", ["AngularSectorSum, K sectors", "raw sums kept, not normalised",
                      "K becomes a knob, not a rebuild"]),
        ("Crops + views", ["128 px, depth-stratified", "GPU augmentation",
                           "Policy A / Policy B"]),
        ("SSL objective", ["VICReg  ·  MAE  ·  I-JEPA", "ResNet-18 or ViT-S",
                           "no labels used"]),
        ("Frozen evaluation", ["linear probes on 10 axes", "RankMe collapse check",
                               "clustering + embeddings"]),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 27)
    ax.axis("off")
    w, gap = 17.0, 3.6
    for i, (name, lines) in enumerate(stages):
        x = i * (w + gap)
        trained = name == "SSL objective"
        ax.add_patch(plt.Rectangle((x, 7), w, 17, facecolor=t["surface"],
                                   edgecolor=accent if trained else t["axis"],
                                   linewidth=2.2 if trained else 1.2, zorder=2))
        ax.text(x + w / 2, 20.6, name, ha="center", va="center", fontsize=13,
                fontweight="bold", color=accent if trained else t["primary"], zorder=3)
        for j, line in enumerate(lines):
            ax.text(x + w / 2, 16.4 - j * 3.0, line, ha="center", va="center",
                    fontsize=9.8, color=t["secondary"], zorder=3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap - 0.5, 15.5), xytext=(x + w + 0.5, 15.5),
                        arrowprops=dict(arrowstyle="-|>", color=t["muted"], linewidth=1.4))
    xs = 3 * (w + gap)
    ax.annotate("the only stage with gradients", xy=(xs + w / 2, 6.4),
                xytext=(xs + w / 2, 2.4), ha="center", va="center", fontsize=11,
                color=accent,
                arrowprops=dict(arrowstyle="-", color=accent, linewidth=1.2))
    ax.text(4 * (w + gap) + w / 2, 2.4, "weights never updated here", ha="center",
            va="center", fontsize=11, color=t["secondary"])
    slides.tight(fig, top=0.99, bottom=0.01, left=0.008, right=0.992)
    return slides.save(fig, OUT, "00_pipeline", theme)


FIGURES = {
    "pipeline": fig_pipeline,
    "probe_nvidia": lambda th: fig_probe_matrix(th, "nvidia"),
    "probe_stanford": lambda th: fig_probe_matrix(th, "stanford"),
    "rankme": fig_rankme,
    "mask_sweep": fig_mask_sweep,
    "policy": fig_policy_contrast,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=sorted(FIGURES), default=None)
    p.add_argument("--theme", choices=slides.THEMES, default="light")
    p.add_argument("--no-tables", action="store_true", dest="no_tables")
    args = p.parse_args()

    for name in ([args.only] if args.only else list(FIGURES)):
        print(f"  {FIGURES[name](args.theme).relative_to(REPO)}")

    if not args.no_tables:
        table_probe_matrix("nvidia")
        table_probe_matrix("stanford")
        table_rankme()
        table_mask_sweep()
        table_policy_contrast()
        for f in sorted(TABLES.glob("*")):
            print(f"  {f.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
