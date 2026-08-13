"""WI-11 done-condition figure: 8 augmented views of one crop under each of
Policy A and Policy B (PLAN.md WI-11). Synthetic scene — no dataset download.

    poetry run python scripts/wi11_augmentation_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssl_us.data import augment as aug

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


def main(out_path: str = "figures/wi11_augmentations.png") -> Path:
    sums, occupancy = synthetic_crop()
    rng_b = np.random.default_rng(1)
    rng_a = np.random.default_rng(2)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    axes[0, 0].set_ylabel("Policy B\n(geometry-only)", fontsize=10)
    axes[1, 0].set_ylabel("Policy A\n(anatomy-focused)", fontsize=10)

    for i in range(8):
        view_b = aug.policy_b(sums, rng_b, z_axis=1, x_axis=2)
        axes[0, i].imshow(to_bmode_db(view_b), cmap="gray", vmin=-40, vmax=0)
        axes[0, i].set_title(f"view {i}", fontsize=8)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        view_a, k = aug.policy_a(sums, occupancy, rng_a, z_axis=1)
        axes[1, i].imshow(to_bmode_db(view_a), cmap="gray", vmin=-40, vmax=0)
        axes[1, i].set_title(f"K={k}", fontsize=8)
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
