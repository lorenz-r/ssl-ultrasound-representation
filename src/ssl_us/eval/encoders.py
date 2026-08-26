"""Encoder construction for the results matrices — one definition, two corpora.

WI-13 probes the NVIDIA store and WI-16 probes Stanford, and the only honest way
to compare those two tables is for the *encoders* to be built identically: same
checkpoint-discovery rules, same baselines, same frozen-feature path. This
module is that single copy. It began inside `scripts/eval/nvidia_matrix.py` and moved
here when `scripts/eval/stanford_matrix.py` needed it — scripts are not importable from
one another (no package on `sys.path` when run as `python scripts/x.py`), and
duplicating checkpoint discovery is exactly how two tables come to disagree for
a reason that is not the data.

The baselines are not decoration. `random_resnet18` is the same architecture on
the same crops, never trained: a pre-trained arm that cannot clear it has
learned nothing the architecture did not already supply. `pca_raw` and
`trivial` bound what the raw pixels give for free.
"""

from __future__ import annotations

from pathlib import Path

from . import features as feat
from .store_features import (
    IJEPACheckpointEncoder,
    MAECheckpointEncoder,
    VICRegCheckpointEncoder,
)

# Checkpoint directory layouts on the cluster, tried in order. These are real
# on-disk directory names under `--checkpoint-root` and are deliberately NOT
# renamed with the rest of the repo — the trained weights live at these paths.
# The K-ablation cells `train_vicreg_k_ablation.sh` trains carry K in their
# directory name; K=4 was trained first and lives under the `wi12_*` name,
# which predates the convention. Ordered so an explicit per-K directory wins.
CKPT_TEMPLATES = (
    "wi13_vicreg_k{k}_policy{policy}",
    "wi12_vicreg_policy{policy}",
)

# ⚠ The wi12_* fallback carries no `{k}`, so it matches *every* K, and WI-12
# only ever trained K=4. Left ungated it reports a trained pair for K=8 — which
# does not exist — and the availability probe below then drops "K=8 column"
# from `not_run`, i.e. exactly the silent truncation WI-13's done-condition
# forbids. It is a wrong *answer*, not a crash: loading it for a real `--k 8`
# run would fail loudly on the stem's channel count in `load_state_dict`, so
# the only thing it ever corrupted is the not-run bookkeeping.
LEGACY_TEMPLATE_K = 4

# The single-run arms, whose directory names carry no policy. Named here rather
# than inlined at their use sites because WI-19's segmentation probe builds the
# *same* set of encoders through a pre-pool path (`eval/dense_features.py`), and
# two copies of a checkpoint path is how one table comes to include an arm the
# other silently skipped. `docs/work-item-map.md` records why these keep their
# WI names: they are directories on the cluster filesystem.
VIT_TEMPLATE = "wi18_vicreg_vit_k{k}_policyA"
IJEPA_TEMPLATE = "wi17_ijepa_k{k}"
MAE_TEMPLATES = (("wi14_mae_k{k}", "mae_norm"), ("wi14_mae_k{k}_rawtarget", "mae_raw"))


def find_checkpoint(root: str | Path, k: int, policy: str) -> Path | None:
    for template in CKPT_TEMPLATES:
        if "{k}" not in template and k != LEGACY_TEMPLATE_K:
            continue
        candidate = Path(root) / template.format(k=k, policy=policy) / "last.pt"
        if candidate.exists():
            return candidate
    return None


def build_encoders(args, in_channels: int, tag: str = "nvidia") -> dict:
    """Encoder name -> callable. Both arms go through the identical path.

    `args` supplies `checkpoint_root`, `k`, `device` and `seed`; `tag` only
    labels the progress lines, so a caller can tell whose run they belong to.
    """
    encoders = {}
    for policy in ("A", "B"):
        ckpt = find_checkpoint(args.checkpoint_root, args.k, policy)
        if ckpt is None:
            print(f"[{tag}] policy {policy}: no K={args.k} checkpoint found — skipping")
            continue
        print(f"[{tag}] policy {policy}: {ckpt}")
        encoders[f"vicreg_{policy}"] = VICRegCheckpointEncoder(
            ckpt, in_channels=in_channels, device=args.device
        )
    # WI-18: VICReg on a ViT-S — the missing cell of the (objective x backbone)
    # 2x2 that docs/MEMORY.md §16.7 flags. §16 could not attribute I-JEPA's result to
    # the objective or the backbone because both moved at once. This column
    # holds the objective fixed at VICReg Policy A and swaps *only* the
    # backbone, so `vicreg_A` -> `vicreg_vit_A` isolates architecture and
    # `vicreg_vit_A` -> `ijepa` isolates the objective. Registered only when
    # trained, so every existing column is unchanged until it is.
    vit_ckpt = Path(args.checkpoint_root) / VIT_TEMPLATE.format(k=args.k) / "last.pt"
    if vit_ckpt.exists():
        print(f"[{tag}] vicreg_vit_A: {vit_ckpt}")
        encoders["vicreg_vit_A"] = VICRegCheckpointEncoder(
            vit_ckpt, in_channels=in_channels, device=args.device
        )
    else:
        print(f"[{tag}] no vicreg_vit checkpoint for K={args.k} — skipping")
    # WI-14's MAE baselines, when trained for this K. Both target variants are
    # probed because they are not interchangeable on speckle: `norm_pix_loss`
    # divides each patch by its own std, and the per-patch scale is the *only*
    # predictable component here (neighbour correlation 0.87 lateral / 0.64
    # axial, against 0.000 for the normalised within-patch pattern). The
    # normalised arm therefore trains on pure noise, and the raw arm is the
    # fair MAE baseline for this data even though it is not the paper default.
    for template, name in MAE_TEMPLATES:
        mae_ckpt = Path(args.checkpoint_root) / template.format(k=args.k) / "last.pt"
        if mae_ckpt.exists():
            print(f"[{tag}] {name}: {mae_ckpt}")
            encoders[name] = MAECheckpointEncoder(
                mae_ckpt, in_channels=in_channels, device=args.device
            )
        else:
            print(f"[{tag}] no {name} checkpoint for K={args.k} — skipping")

    # WI-17's I-JEPA arm, when trained for this K. One run per K, like the MAE
    # arm and for the same reason: it trains on the un-augmented crop, so there
    # is no Policy A/B split to make. It is also the only trained arm here that
    # never saw an augmentation, which is the comparison it exists to supply.
    # It brings its own floor: `random_vit` is the identical ViT on the
    # identical crops, never trained. `random_resnet18` cannot serve that role
    # for a transformer arm, and it is registered only when the trained arm is
    # present so existing matrix columns are unchanged until WI-17 is run.
    ijepa_ckpt = Path(args.checkpoint_root) / IJEPA_TEMPLATE.format(k=args.k) / "last.pt"
    if ijepa_ckpt.exists():
        print(f"[{tag}] ijepa: {ijepa_ckpt}")
        encoders["ijepa"] = IJEPACheckpointEncoder(
            ijepa_ckpt, in_channels=in_channels, device=args.device
        )
        encoders["random_vit"] = IJEPACheckpointEncoder(
            ijepa_ckpt, in_channels=in_channels, device=args.device,
            random_init=True, seed=args.seed,
        )
    else:
        print(f"[{tag}] no ijepa checkpoint for K={args.k} — skipping")

    # The floor: same architecture, same input, never trained. If a trained arm
    # cannot beat this, it learned nothing that the architecture did not supply.
    encoders["random_resnet18"] = feat.RandomResNet18(
        in_channels=in_channels, seed=args.seed, device=args.device
    )
    encoders["pca_raw"] = lambda c: feat.pca_features(c, seed=args.seed)
    encoders["trivial"] = feat.trivial_features
    return encoders
