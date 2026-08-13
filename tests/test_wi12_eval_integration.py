"""WI-12 done-condition: "the frozen-feature results matrix from WI-2 runs
unmodified against the trained encoder." Trains a tiny VICReg model on
synthetic crops, extracts backbone features, and feeds them straight into
`ssl_us.eval.probes.probe` (WI-2, untouched) to prove the harness works
against a real (if toy) trained encoder, not only the random-init baseline.
"""

from __future__ import annotations

import numpy as np
import torch

from ssl_us.eval.probes import probe
from ssl_us.models.resnet import BACKBONE_DIM, ResNetBackbone
from ssl_us.models.vicreg import VICReg
from ssl_us.train.vicreg import VICRegTrainConfig, train_vicreg


def test_trained_vicreg_features_run_through_the_wi2_probe_harness_unmodified():
    torch.manual_seed(0)
    model = VICReg(ResNetBackbone(2, seed=0), backbone_dim=BACKBONE_DIM, projector_dim=32)

    def synthetic_loader(n_batches=4, batch_size=6):
        base = torch.randn(n_batches, batch_size, 2, 32, 32)
        for i in range(n_batches):
            yield base[i], base[i] + 0.05 * torch.randn_like(base[i])

    config = VICRegTrainConfig(epochs=2, device="cpu", amp=False)
    train_vicreg(model, list(synthetic_loader()), config)

    # extract frozen backbone features for a fresh batch of crops, exactly the
    # (crops) -> (N, D) contract every extractor in eval/features.py follows.
    model.eval()
    n_frames, per_frame = 25, 4
    n_crops = n_frames * per_frame  # >= 64 so PCA-to-64 capacity control (WI-2) has enough samples
    crops = torch.randn(n_crops, 2, 32, 32)
    with torch.no_grad():
        feats = model.backbone(crops).numpy().astype(np.float32)

    rng = np.random.default_rng(0)
    y_frame = rng.normal(size=n_frames)
    frame_id = np.repeat([f"f{i:02d}" for i in range(n_frames)], per_frame)
    y = np.repeat(y_frame, per_frame)

    result = probe(feats, y, frame_id, "sos_mean_frame", stage="c0_const", level="frame")

    assert result.n_eff == n_frames  # frame-level, not crop-level (WI-2's core invariant)
    assert np.isfinite(result.score)
    assert result.n_features == min(feats.shape[1], 64)  # WI-2's capacity control applied
