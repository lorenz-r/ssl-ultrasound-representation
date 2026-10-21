"""CPU smoke test for the MAE training loop (PLAN.md WI-14).

Generic over the data source, like `test_train_vicreg.py`, so a synthetic
loader here exercises the same code path the real store-backed loader will.
"""

from __future__ import annotations

import json

import torch

from ssl_us.models.mae2d import MAE2D
from ssl_us.train.mae import MAETrainConfig, train_mae


def _small_model(in_channels: int = 2) -> MAE2D:
    torch.manual_seed(0)
    return MAE2D(
        in_channels=in_channels, img_size=32, patch_size=8, embed_dim=16,
        depth=1, heads=2, decoder_embed_dim=8, decoder_depth=1, decoder_heads=2,
    )


def _loader(n_batches: int = 3, batch_size: int = 4, in_channels: int = 2):
    torch.manual_seed(1)
    return [torch.randn(batch_size, in_channels, 32, 32) for _ in range(n_batches)]


def test_train_mae_runs_and_records_history():
    history = train_mae(
        _small_model(), _loader(),
        MAETrainConfig(epochs=2, lr=1e-3, device="cpu", amp=False, warmup_epochs=0),
    )
    assert len(history["loss"]) == 2
    assert len(history["rankme"]) == 2
    assert all(v > 0 for v in history["crops_per_second"])
    assert all(v == v for v in history["loss"])  # no NaN


def test_train_mae_accepts_a_view_pair_loader():
    """A (view1, view2) loader is accepted and its first element used.

    The WI-12 loader yields pairs; MAE needs one view. Silently taking the
    first is deliberate so the VICReg loader can be reused unchanged.
    """
    pairs = [(x, x.clone()) for x in _loader(n_batches=2)]
    history = train_mae(
        _small_model(), pairs,
        MAETrainConfig(epochs=1, lr=1e-3, device="cpu", amp=False, warmup_epochs=0),
    )
    assert len(history["loss"]) == 1


def test_train_mae_checkpoints_and_resumes(tmp_path):
    """A resumed run continues from the stored epoch rather than restarting."""
    cfg = dict(lr=1e-3, device="cpu", amp=False, warmup_epochs=0,
               checkpoint_dir=tmp_path, checkpoint_every=0)
    first = train_mae(_small_model(), _loader(),
                      MAETrainConfig(epochs=2, **cfg))
    assert (tmp_path / "last.pt").exists()
    assert len(json.loads((tmp_path / "history.json").read_text())["loss"]) == 2

    # Same directory, more epochs: history carries over, not restarts.
    resumed = train_mae(_small_model(), _loader(),
                        MAETrainConfig(epochs=4, **cfg))
    assert len(resumed["loss"]) == 4
    assert resumed["loss"][:2] == first["loss"]


def test_rankme_is_measured_on_the_unmasked_path(monkeypatch):
    """RankMe must come from `encode`, never from the 75 %-masked forward.

    Measuring the pretext path would report the reconstruction problem's rank
    rather than the representation's, and the two are not the same thing.
    """
    model = _small_model()
    calls = {"encode": 0}
    real_encode = model.encode

    def counting_encode(x):
        calls["encode"] += 1
        return real_encode(x)

    monkeypatch.setattr(model, "encode", counting_encode)
    train_mae(model, _loader(n_batches=2),
              MAETrainConfig(epochs=1, lr=1e-3, device="cpu", amp=False, warmup_epochs=0))
    assert calls["encode"] > 0
