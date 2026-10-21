"""Augmentation-policy figure: 8 augmented views of one crop under each of
Policy A and Policy B (PLAN.md WI-11). Synthetic scene — no dataset download.

    poetry run python scripts/figures/augmentations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssl_us.data import augment as aug
from ssl_us.data import transforms as tf

NZ, NX, K = 64, 64, 16


def synthetic_crop(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic (2, NZ, NX, K) raw sector-sum crop with a bright blob and
    depth-decaying speckle, plus a matching occupancy map (sparse at shallow
    depth, full at depth — the real occupancy-vs-depth shape from WI-3)."""
    rng = np.random.default_rng(seed)
    z, x = np.meshgrid(np.arange(NZ), np.arange(NX), indexing="ij")
    blob = np.exp(-(((z - NZ * 0.6) / 6) ** 2 + ((x - NX * 0.5) / 6) ** 2))
    amplitude = (10.0 * np.exp(-0.02 * z) * (1.0 + 2.0 * blob))[:, :, None]

    occupancy = np.clip(np.round(np.linspace(0.5, 11, NZ)), 1, None).astype(np.uint16)
    occupancy = np.broadcast_to(occupancy[:, None, None], (NZ, NX, K)).copy()

    phase = rng.uniform(0, 2 * np.pi, size=(NZ, NX, K))
    per_sector = amplitude * np.sqrt(occupancy) / np.sqrt(occupancy.mean())
    iq = np.stack([per_sector * np.cos(phase), per_sector * np.sin(phase)], axis=0)
    return iq.astype(np.float32), occupancy


def to_bmode_db(view_k1_or_more: np.ndarray) -> np.ndarray:
    """Log-envelope of a (2, Z, X, [K]) view, summed over any remaining K."""
    if view_k1_or_more.ndim == 4:
        view_k1_or_more = view_k1_or_more.sum(axis=-1)
    env = np.sqrt(view_k1_or_more[0] ** 2 + view_k1_or_more[1] ** 2)
    db = 20 * np.log10(np.maximum(env, 1e-6))
    return db - db.max()


def main(out_path: str = "outputs/diagnostics/augmentations.png") -> Path:
    import torch

    sums_np, occupancy_np = synthetic_crop()
    # The policies are batched torch ops (they run on the GPU during training);
    # here a batch of 8 produces the 8 views in one call, on CPU.
    n_views = 8
    sums = torch.from_numpy(sums_np).unsqueeze(0).repeat(n_views, 1, 1, 1, 1)
    occupancy = torch.from_numpy(occupancy_np.astype("float32")).unsqueeze(0).repeat(
        n_views, 1, 1, 1
    )
    gen_b, gen_a = torch.Generator(), torch.Generator()
    gen_b.manual_seed(1)
    gen_a.manual_seed(2)

    views_b = aug.policy_b(tf.reduce_and_normalize_torch(sums, occupancy, target_k=1), gen_b)
    # expand_to=16 keeps the whole ladder available, so the drawn K actually
    # varies across views — the point of the Policy A panel. to_bmode_db
    # sums over the sector axis for display.
    views_a, ks = aug.policy_a(sums, occupancy, gen_a, expand_to=16)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    axes[0, 0].set_ylabel("Policy B\n(geometry-only)", fontsize=10)
    axes[1, 0].set_ylabel("Policy A\n(anatomy-focused)", fontsize=10)

    for i in range(n_views):
        axes[0, i].imshow(to_bmode_db(views_b[i].numpy()), cmap="gray", vmin=-40, vmax=0)
        axes[0, i].set_title(f"view {i}", fontsize=8)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        axes[1, i].imshow(to_bmode_db(views_a[i].numpy()), cmap="gray", vmin=-40, vmax=0)
        axes[1, i].set_title(f"K={int(ks[i])}", fontsize=8)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    fig.suptitle("PLAN.md WI-11 — 8 augmented views per policy, one synthetic crop")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
