"""Frozen feature extractors, including every baseline in PLAN.md 4.4.

All extractors share one signature: ``(crops, ...) -> (N, D) float32``. Nothing here
trains; the encoders are frozen or randomly initialised by construction, which is
what makes the harness runnable before any model exists (PLAN.md 4.6).

Baselines implemented here:
  1. random-init ResNet-18 with the adapted stem  (`RandomResNet18`)
  2. PCA on raw crops                             (`pca_features`)
  3. handcrafted speckle statistics               (`speckle_features`)
  4. trivial mean/std                             (`trivial_features`)

Baselines 5-8 (DINOv3, HVAE, Speckle2Self, TAESD) are external pretrained models;
they are extracted separately — the Keras ones in the zea virtualenv — and joined
into the matrix as precomputed arrays. See PLAN.md 4.4 for why they are reported as
a *ceiling* and not as the answer to "B-mode vs sub-apertures".
"""

from __future__ import annotations

import numpy as np


class RandomResNet18:
    """Randomly-initialised ResNet-18 with a stem adapted to ``in_channels``.

    Deliberately never trained: this is the floor plotted on every chart and the
    encoder the harness is validated against before any real model exists. A random
    CNN still encodes crop position through boundary effects and amplitude, so a
    healthy harness shows ``z_centre`` well above chance and ``cyst_fraction`` near
    chance. The reverse pattern means the harness is leaking.
    """

    def __init__(self, in_channels: int = 1, seed: int = 0, device: str | None = None):
        import torch

        from ..models.resnet import ResNetBackbone

        self._torch = torch
        backbone = ResNetBackbone(in_channels, seed=seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = backbone.to(self.device).eval()
        self.in_channels = in_channels
        self.dim = backbone.dim

    def __call__(self, crops: np.ndarray, batch_size: int = 128) -> np.ndarray:
        torch = self._torch
        x = crops
        if x.ndim == 3:
            x = x[:, None]  # (N, 1, H, W)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                b = torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size])).float()
                out.append(self.net(b.to(self.device)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def pca_features(crops: np.ndarray, dims: int = 64, seed: int = 0) -> np.ndarray:
    """PCA on flattened raw crops — the linear baseline on the input itself."""
    from sklearn.decomposition import PCA

    flat = crops.reshape(len(crops), -1).astype(np.float32)
    n = min(dims, flat.shape[0], flat.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(flat).astype(np.float32)


def trivial_features(crops: np.ndarray) -> np.ndarray:
    """Mean and std of the crop envelope. Two numbers.

    Doubles as the PLAN.md 4.1 frame-level-cue control for the sound-speed probes:
    because sound speed is near-constant within a frame, any global brightness cue
    suffices, so an encoder that fails to clear these two features has learned
    nothing about sound speed.
    """
    env = np.abs(crops).reshape(len(crops), -1)
    return np.stack([env.mean(1), env.std(1)], axis=1).astype(np.float32)


def _nakagami(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method-of-moments Nakagami (m, omega) per crop on the envelope."""
    e2 = (env**2).mean(1)
    e4 = (env**4).mean(1)
    var = np.maximum(e4 - e2**2, 1e-12)
    return (e2**2 / var).astype(np.float32), e2.astype(np.float32)


def _glcm_stats(crop: np.ndarray, levels: int = 16) -> np.ndarray:
    """Contrast / homogeneity / energy / correlation at 4 offsets, on a quantised crop."""
    from skimage.feature import graycomatrix, graycoprops

    lo, hi = crop.min(), crop.max()
    q = np.zeros_like(crop, dtype=np.uint8) if hi <= lo else (
        ((crop - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)
    )
    g = graycomatrix(
        q, distances=[1, 4], angles=[0, np.pi / 2], levels=levels, symmetric=True, normed=True
    )
    return np.concatenate(
        [graycoprops(g, p).ravel() for p in ("contrast", "homogeneity", "energy", "correlation")]
    ).astype(np.float32)


def speckle_features(crops: np.ndarray, with_glcm: bool = True) -> np.ndarray:
    """Handcrafted speckle statistics: Nakagami, SNR, skew, kurtosis, GLCM texture.

    This is the baseline that most directly asks whether a learned representation is
    adding anything over classical speckle analysis.
    """
    env = np.abs(crops).reshape(len(crops), -1).astype(np.float64)
    mu = env.mean(1)
    sd = env.std(1) + 1e-12
    m, omega = _nakagami(env)
    z = (env - mu[:, None]) / sd[:, None]
    feats = [
        mu.astype(np.float32),
        sd.astype(np.float32),
        (mu / sd).astype(np.float32),  # envelope SNR
        (z**3).mean(1).astype(np.float32),  # skewness
        (z**4).mean(1).astype(np.float32),  # kurtosis
        m,
        omega,
    ]
    base = np.stack(feats, axis=1)
    if not with_glcm:
        return base
    try:
        glcm = np.stack([_glcm_stats(c) for c in crops])
    except ImportError:
        return base
    return np.concatenate([base, glcm], axis=1).astype(np.float32)


EXTRACTORS = {
    "random_resnet18": lambda c: RandomResNet18()(c),
    "pca_raw": pca_features,
    "speckle_stats": speckle_features,
    "trivial": trivial_features,
}
