"""
Latent-representation analysis for a trained ultrasound MAE.

Loads a checkpoint (local file or wandb model artifact), rebuilds the encoder
from the self-contained ``model_config`` saved alongside the weights, extracts
latents over the dataset, and renders PCA + t-SNE projections coloured by the
labels that are available *for free* in this dataset (no manual annotation):

  * frame level  — one pooled vector per frame; labels: subject, split.
  * token level  — one vector per (scanline, transducer-group, depth-window)
    token; labels: axial depth, lateral position, aperture group, subject, and
    per-token signal energy (echogenicity proxy from the input patch).

There is no anatomical ground truth in this dataset, so these geometry/physics
labels are the yardstick: a useful representation should organise by depth
(attenuation / focal physics), echogenicity (tissue vs. anechoic), and separate
subjects — while *not* trivially collapsing onto a single nuisance axis.

Run:
    poetry run python -m ssl_ultrasound_representation.eval.latent \
        --checkpoint checkpoints/best_mae_checkpoint.pt \
        --data-zip data/processed/ultrasonic_dataset.zarr.zip \
        --data-metadata data/processed/ultrasonic_dataset.metadata.json \
        --out figures/latent

    # or pull the checkpoint straight from wandb:
    poetry run python -m ssl_ultrasound_representation.eval.latent \
        --artifact entity/ssl-ultrasound-mae/mae-checkpoint:best ...
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ssl_ultrasound_representation.dataset import build_split_datasets
from ssl_ultrasound_representation.mae.model.pos_encoding import (
    build_grid_positional_encoding,
)
from ssl_ultrasound_representation.mae.model.us_mae import SignalMAE
from ssl_ultrasound_representation.train import default_device, normalize_frame


# ── checkpoint loading ─────────────────────────────────────────────────────────
def load_checkpoint(
    checkpoint: str | None = None, artifact: str | None = None
) -> dict:
    """Return the checkpoint dict from a local path or a wandb model artifact.

    Exactly one of ``checkpoint`` / ``artifact`` must be given. A wandb artifact
    ref looks like ``entity/project/mae-checkpoint:best``; its single ``*.pt`` is
    downloaded and loaded.
    """
    if (checkpoint is None) == (artifact is None):
        raise ValueError("pass exactly one of --checkpoint / --artifact")

    if artifact is not None:
        import wandb

        art = wandb.Api().artifact(artifact, type="model")
        ddir = art.download()
        pts = sorted(glob.glob(os.path.join(ddir, "*.pt")))
        if not pts:
            raise FileNotFoundError(f"no .pt file inside artifact {artifact!r}")
        path = pts[0]
    else:
        path = checkpoint

    # weights_only=False: our checkpoints store python dicts (config /
    # model_config) alongside the tensors, which the safe loader rejects.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def infer_model_config(ckpt: dict) -> dict:
    """The model spec needed to rebuild the encoder.

    Uses the self-contained ``model_config`` when present; otherwise (older
    checkpoints) reconstructs it from state-dict shapes plus the saved training
    config. Most dims are read straight from the weights; the head counts and
    window sizes — which don't change tensor shapes — come from the training
    config, falling back to train.py's defaults, and are cross-checked against
    the checkpoint's token grid so a wrong guess fails loudly.
    """
    if "model_config" in ckpt:
        return ckpt["model_config"]

    import re

    sd = ckpt["model_state_dict"]
    cfg = ckpt.get("config", {})

    def n_layers(prefix: str) -> int:
        return max(
            int(re.search(rf"{prefix}\.layers\.(\d+)\.", k).group(1))
            for k in sd if k.startswith(f"{prefix}.layers.")
        ) + 1

    # Data geometry is fixed by the pipeline (hardcoded in train.py).
    n_transducers, n_scanlines, n_samples = 192, 192, 215
    transducer_window = cfg.get("transducer_window", 64)
    depth_window = cfg.get("depth_window", 43)
    num_tokens = sd["decoder_pos_embed"].shape[1]
    expected = (
        n_scanlines
        * (n_transducers // transducer_window)
        * (n_samples // depth_window)
    )
    if expected != num_tokens:
        raise ValueError(
            f"inferred token grid ({expected}) != checkpoint num_tokens "
            f"({num_tokens}); this checkpoint used non-default window sizes — "
            f"set transducer_window/depth_window in its training config."
        )

    return {
        "n_transducers": n_transducers,
        "transducer_window": transducer_window,
        "n_scanlines": n_scanlines,
        "n_samples": n_samples,
        "depth_window": depth_window,
        "embed_dim": sd["encoder_norm.weight"].shape[0],
        "encoder_depth": n_layers("encoder"),
        "encoder_heads": cfg.get("encoder_heads", 8),
        "decoder_embed_dim": sd["decoder_norm.weight"].shape[0],
        "decoder_depth": n_layers("decoder"),
        "decoder_heads": cfg.get("decoder_heads", 8),
        "dim_feedforward": sd["encoder.layers.0.linear1.weight"].shape[0],
        "mask_ratio": cfg.get("mask_ratio", 0.0),
        # Doesn't change tensor shapes; older runs all used the default (True).
        "pos_in_encoder": cfg.get("pos_in_encoder", True),
    }


def build_model(model_config: dict, device: str) -> SignalMAE:
    model = SignalMAE(**model_config).to(device)
    model.eval()
    return model


# ── latent extraction ──────────────────────────────────────────────────────────
def _token_grid_labels(model: SignalMAE) -> dict[str, np.ndarray]:
    """Per-token geometry labels in patchify order (scanline, group, depth)."""
    W = model.n_depth_windows
    G = model.n_transducer_groups
    Sc = model.n_scanlines
    t = np.arange(model.num_tokens)
    return {
        "depth_window": t % W,
        "aperture_group": (t // W) % G,
        "scanline": t // (W * G),
    }


def extract_latents(
    model: SignalMAE,
    datasets: dict,
    meta: dict,
    pos_embed: torch.Tensor,
    device: str,
    batch_size: int = 2,
) -> dict:
    """Run the encoder over the given splits and collect latents + labels.

    Returns a dict with frame-level and token-level arrays:
        frame_latents (Nf, D), frame_subject, frame_split
        token_latents (Nt, D), token_depth/aperture/scanline/subject/split/energy
    """
    idx_to_subject = {
        fm["global_frame_index"]: fm["subject"] for fm in meta["frames"]
    }
    grid = _token_grid_labels(model)

    frame_lat, frame_subj, frame_split = [], [], []
    tok_lat = []
    tok_depth, tok_ap, tok_scan, tok_subj, tok_split, tok_energy = (
        [], [], [], [], [], [],
    )

    for split_name, ds in datasets.items():
        if len(ds) == 0:
            continue
        subjects = [idx_to_subject.get(gi, "unknown") for gi in ds.indices]
        for start in range(0, len(ds), batch_size):
            stop = min(start + batch_size, len(ds))
            x = torch.stack([ds[i] for i in range(start, stop)]).to(device)

            frame_z = model.encode(x, pos_embed, pool="mean").cpu().numpy()
            token_z = model.encode(x, pos_embed, pool="none").cpu().numpy()
            # Per-token energy (echogenicity) from the normalized input patch.
            patches = model.patchify(x)  # (B, L, 2, tw, w)
            energy = patches.pow(2).mean(dim=(2, 3, 4)).cpu().numpy()  # (B, L)

            B, L, _ = token_z.shape
            frame_lat.append(frame_z)
            tok_lat.append(token_z.reshape(B * L, -1))
            for b in range(B):
                subj = subjects[start + b]
                frame_subj.append(subj)
                frame_split.append(split_name)
                tok_depth.append(grid["depth_window"])
                tok_ap.append(grid["aperture_group"])
                tok_scan.append(grid["scanline"])
                tok_subj.append(np.full(L, subj, dtype=object))
                tok_split.append(np.full(L, split_name, dtype=object))
            tok_energy.append(energy.reshape(B * L))

    return {
        "frame_latents": np.concatenate(frame_lat, axis=0),
        "frame_subject": np.array(frame_subj, dtype=object),
        "frame_split": np.array(frame_split, dtype=object),
        "token_latents": np.concatenate(tok_lat, axis=0),
        "token_depth_window": np.concatenate(tok_depth),
        "token_aperture_group": np.concatenate(tok_ap),
        "token_scanline": np.concatenate(tok_scan),
        "token_subject": np.concatenate(tok_subj),
        "token_split": np.concatenate(tok_split),
        "token_energy": np.concatenate(tok_energy),
    }


# ── projections ────────────────────────────────────────────────────────────────
def project(feats: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    """2-D PCA or t-SNE of standardized features."""
    x = StandardScaler().fit_transform(feats)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "tsne":
        n = x.shape[0]
        perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
        return TSNE(
            n_components=2,
            init="pca",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(x)
    raise ValueError(f"unknown method {method!r}")


def regress_out(
    feats: np.ndarray, label_dict: dict, factors: list[str]
) -> tuple[np.ndarray, float]:
    """Linearly remove the given (categorical) position factors from features.

    Fits ``feats ~ [intercept, one-hot(factor) ...]`` by least squares and
    returns the residual plus the fraction of variance the position factors
    explained. One-hot (not raw index) because the latent's dependence on a
    coordinate is not linear in the index — but it *is* additive per axis (that
    is how the positional encoding was injected), so per-level dummies remove it
    cleanly. What remains is the content the position shell was hiding.
    """
    cols = [np.ones((feats.shape[0], 1))]
    for f in factors:
        v = label_dict[f]
        cats = np.unique(v)
        cols.append((v[:, None] == cats[None, :]).astype(float))
    X = np.concatenate(cols, axis=1)
    beta, *_ = np.linalg.lstsq(X, feats, rcond=None)
    pred = X @ beta
    resid = feats - pred
    ss_res = float(((feats - pred) ** 2).sum())
    ss_tot = float(((feats - feats.mean(0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return resid, r2


# ── plotting ───────────────────────────────────────────────────────────────────
def _scatter(ax, xy, values, name, point_size):
    is_categorical = values.dtype == object or values.dtype.kind in "US"
    if is_categorical:
        cats = sorted(set(values.tolist()))
        cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
        for k, c in enumerate(cats):
            m = values == c
            ax.scatter(
                xy[m, 0], xy[m, 1], s=point_size, alpha=0.6,
                color=cmap(k % cmap.N), label=str(c), linewidths=0,
            )
        ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.7)
    else:
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], s=point_size, alpha=0.6,
            c=values, cmap="viridis", linewidths=0,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_embedding(xy, label_dict, title, out_path, point_size):
    keys = list(label_dict)
    ncol = min(3, len(keys))
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, key in zip(axes.flat, keys):
        ax.axis("on")
        _scatter(ax, xy, label_dict[key], key, point_size)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── orchestration ──────────────────────────────────────────────────────────────
def run(config: dict) -> list[str]:
    device = config.get("device", default_device())
    ckpt = load_checkpoint(config.get("checkpoint"), config.get("artifact"))
    model_config = infer_model_config(ckpt)
    model = build_model(model_config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.4f})")

    pos_embed = build_grid_positional_encoding(
        (model.n_scanlines, model.n_transducer_groups, model.n_depth_windows),
        model.embed_dim,
    ).to(device)

    data_source = (
        {"repo_id": config["hf_repo"], "max_frames": config.get("max_frames"),
         "token": config.get("hf_token")}
        if config.get("hf_repo")
        else {"local_zip": config["data_zip"],
              "local_metadata": config["data_metadata"]}
    )
    datasets, meta = build_split_datasets(
        **data_source,
        split_strategy="file",
        ratios=config.get("split_ratios", (0.85, 0.15, 0.0)),
        transform=normalize_frame,
        focal_zone=config.get("focal_zone", 0),
        preload=True,
    )
    # Default: analyse every split together (only ~14 frames total).
    want = config.get("split", "all")
    if want != "all":
        datasets = {want: datasets[want]}

    print("Extracting latents ...")
    data = extract_latents(model, datasets, meta, pos_embed, device,
                           batch_size=config.get("batch_size", 2))
    print(f"  frame latents: {data['frame_latents'].shape}  |  "
          f"token latents: {data['token_latents'].shape}")

    out_dir = Path(config.get("out", "figures/latent"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "latents.npz", **data)

    frame_labels = {"subject": data["frame_subject"], "split": data["frame_split"]}
    token_labels = {
        "depth_window": data["token_depth_window"],
        "energy": data["token_energy"],
        "aperture_group": data["token_aperture_group"],
        "scanline": data["token_scanline"],
        "subject": data["token_subject"],
    }

    # t-SNE is O(n^2)-ish; subsample tokens for it (PCA can use all points).
    max_tsne = config.get("max_tsne_points", 8000)
    n_tok = data["token_latents"].shape[0]
    rng = np.random.default_rng(0)
    sub = (rng.choice(n_tok, size=max_tsne, replace=False)
           if n_tok > max_tsne else np.arange(n_tok))

    methods = config.get("methods", ("pca", "tsne"))
    outputs = []
    for method in methods:
        # frame level
        if data["frame_latents"].shape[0] >= 3:
            xy = project(data["frame_latents"], method)
            outputs.append(plot_embedding(
                xy, frame_labels, f"frame latents — {method.upper()}",
                out_dir / f"frame_{method}.png", point_size=60))
        else:
            print(f"  (skipping frame {method}: <3 frames)")
        # token level
        feats = data["token_latents"] if method == "pca" else data["token_latents"][sub]
        tlabels = (token_labels if method == "pca"
                   else {k: v[sub] for k, v in token_labels.items()})
        xy = project(feats, method)
        outputs.append(plot_embedding(
            xy, tlabels, f"token latents — {method.upper()}"
            + ("" if method == "pca" else f" (n={len(sub)})"),
            out_dir / f"token_{method}.png", point_size=4))

    # Token latents with grid position (depth+aperture+scanline) linearly
    # regressed out — exposes the content structure the positional shell hides.
    if config.get("regress_position", True):
        factors = ["depth_window", "aperture_group", "scanline"]
        resid, r2 = regress_out(data["token_latents"], token_labels, factors)
        print(f"Position {factors} explains {r2 * 100:.1f}% of token-latent "
              f"variance; plotting the {(1 - r2) * 100:.1f}% residual.")
        for method in methods:
            feats = resid if method == "pca" else resid[sub]
            tlabels = (token_labels if method == "pca"
                       else {k: v[sub] for k, v in token_labels.items()})
            xy = project(feats, method)
            outputs.append(plot_embedding(
                xy, tlabels,
                f"token latents, position regressed out — {method.upper()}"
                + ("" if method == "pca" else f" (n={len(sub)})"),
                out_dir / f"token_{method}_residual.png", point_size=4))

    print("Wrote:")
    for p in outputs:
        print(f"  {p}")

    if config.get("log_wandb"):
        import wandb

        run_obj = wandb.init(
            project=config.get("wandb_project", "ssl-ultrasound-mae"),
            job_type="latent-eval",
            name=config.get("run_name"),
        )
        run_obj.log({Path(p).stem: wandb.Image(str(p)) for p in outputs})
        run_obj.finish()

    return [str(p) for p in outputs]


def _cli() -> dict:
    p = argparse.ArgumentParser(description="Latent-space analysis of an MAE checkpoint.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="local checkpoint .pt path")
    src.add_argument("--artifact", help="wandb model artifact ref (entity/proj/name:alias)")
    p.add_argument("--data-zip", dest="data_zip",
                   default="data/processed/ultrasonic_dataset.zarr.zip")
    p.add_argument("--data-metadata", dest="data_metadata",
                   default="data/processed/ultrasonic_dataset.metadata.json")
    p.add_argument("--hf-repo", dest="hf_repo")
    p.add_argument("--max-frames", type=int, dest="max_frames")
    p.add_argument("--hf-token", dest="hf_token")
    p.add_argument("--split", default="all",
                   help="'all' (default) or train/val/test")
    p.add_argument("--focal-zone", type=int, dest="focal_zone")
    p.add_argument("--out", default="figures/latent")
    p.add_argument("--max-tsne-points", type=int, dest="max_tsne_points")
    p.add_argument("--no-regress-position", action="store_true",
                   help="skip the position-regressed-out token plots")
    p.add_argument("--device")
    p.add_argument("--log-wandb", action="store_true", dest="log_wandb")
    p.add_argument("--run-name", dest="run_name")
    cfg = {k: v for k, v in vars(p.parse_args()).items() if v not in (None, False)}
    if cfg.pop("no_regress_position", None):
        cfg["regress_position"] = False
    return cfg


if __name__ == "__main__":
    run(_cli())
