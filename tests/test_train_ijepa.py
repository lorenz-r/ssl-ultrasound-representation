"""CPU smoke test for the I-JEPA training loop (WI-17).

Generic over the data source, like `test_train_mae.py`, so a synthetic loader
here exercises the same code path the store-backed loader will. The two tests
that matter beyond "it runs" are the EMA (the target must actually track, and
must not be an optimizer parameter) and the collapse halt, which this arm has
and the MAE arm deliberately does not.
"""

from __future__ import annotations

import json

import pytest
import torch

from ssl_us.models.ijepa import IJEPA
from ssl_us.train.ijepa import IJEPATrainConfig, train_ijepa
from ssl_us.train.vicreg import DimensionalCollapse


def _small_model(in_channels: int = 2) -> IJEPA:
    torch.manual_seed(0)
    return IJEPA(
        in_channels=in_channels, img_size=32, patch_size=4, embed_dim=16, depth=1, heads=2,
        predictor_dim=8, predictor_depth=1, predictor_heads=2,
    )


def _loader(n_batches: int = 3, batch_size: int = 4, in_channels: int = 2):
    torch.manual_seed(1)
    return [torch.randn(batch_size, in_channels, 32, 32) for _ in range(n_batches)]


def _config(**kw) -> IJEPATrainConfig:
    base = dict(lr=1e-3, device="cpu", amp=False, warmup_epochs=0, checkpoint_every=0)
    return IJEPATrainConfig(**{**base, **kw})


def test_train_ijepa_runs_and_records_history():
    history = train_ijepa(_small_model(), _loader(), _config(epochs=2))
    assert len(history["loss"]) == 2
    assert len(history["rankme"]) == 2
    assert len(history["target_std"]) == 2
    assert all(v > 0 for v in history["crops_per_second"])
    assert all(v == v for v in history["loss"])  # no NaN


def test_train_ijepa_accepts_a_view_pair_loader():
    """The WI-12 loader yields pairs; this arm needs one crop. Taking the first
    is deliberate, matching `train_mae` — but WI-17 trains on un-augmented
    crops by design, which is what `scripts/train/ijepa.py` supplies."""
    pairs = [(x, x.clone()) for x in _loader(n_batches=2)]
    history = train_ijepa(_small_model(), pairs, _config(epochs=1))
    assert len(history["loss"]) == 1


def test_ema_momentum_is_recorded_and_rises_toward_one():
    history = train_ijepa(
        _small_model(), _loader(n_batches=4), _config(epochs=3, ema_base=0.9, ema_final=1.0)
    )
    assert history["ema_momentum"][0] < history["ema_momentum"][-1]
    assert history["ema_momentum"][-1] == pytest.approx(1.0)


def test_target_encoder_tracks_but_is_never_optimized():
    """It must move (the EMA ran) and it must not have moved the way an
    optimizer would (it is not in the parameter group)."""
    model = _small_model()
    before = [p.clone() for p in model.target_encoder.parameters()]
    train_ijepa(model, _loader(), _config(epochs=1, ema_base=0.5, ema_final=0.5))
    moved = [
        not torch.allclose(a, b)
        for a, b in zip(before, model.target_encoder.parameters(), strict=True)
    ]
    assert any(moved)
    assert all(p.grad is None for p in model.target_encoder.parameters())


def test_min_target_std_halts_a_collapsed_run():
    """The fast collapse instrument: a target that is constant across the batch
    is I-JEPA's degenerate solution, and it shows up in one epoch."""
    with pytest.raises(DimensionalCollapse, match="target representation std"):
        train_ijepa(_small_model(), _loader(), _config(epochs=1, min_target_std=1e6))


def test_collapse_halt_carries_the_history_for_diagnosis():
    try:
        train_ijepa(_small_model(), _loader(), _config(epochs=1, min_target_std=1e6))
    except DimensionalCollapse as exc:
        assert exc.history["loss"]
        assert exc.history["target_std"]
    else:  # pragma: no cover - the halt is asserted above
        pytest.fail("expected DimensionalCollapse")


def test_min_target_std_off_by_default_does_not_halt():
    history = train_ijepa(_small_model(), _loader(), _config(epochs=1))
    assert len(history["loss"]) == 1


def test_train_ijepa_checkpoints_and_resumes(tmp_path):
    first = train_ijepa(_small_model(), _loader(), _config(epochs=2, checkpoint_dir=tmp_path))
    assert (tmp_path / "last.pt").exists()
    assert json.loads((tmp_path / "history.json").read_text())["loss"]

    # A resumed run continues rather than restarting: two more epochs of a
    # 4-epoch budget append to the same history.
    resumed = train_ijepa(_small_model(), _loader(), _config(epochs=4, checkpoint_dir=tmp_path))
    assert len(resumed["loss"]) == 4
    assert resumed["loss"][:2] == first["loss"]


def test_empty_loader_says_what_is_wrong():
    with pytest.raises(ValueError, match="loader yielded nothing"):
        train_ijepa(_small_model(), [], _config(epochs=1))
