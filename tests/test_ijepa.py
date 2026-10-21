"""I-JEPA arm (WI-17) — masking, the three components, and the EMA target.

The tests that carry weight here are the ones guarding the two properties the
objective silently depends on: the predictor must never be handed a token it is
being asked to predict (or the task is a copy), and the target encoder must
never receive gradient (or the "target" is just the model agreeing with itself).
"""

from __future__ import annotations

import pytest
import torch

from ssl_us.models.ijepa import (
    IJEPA,
    MaskConfig,
    Predictor,
    ViTEncoder,
    block_token_ids,
    ijepa_loss,
    momentum_schedule,
    multiblock_masks,
)

# ── masking ────────────────────────────────────────────────────────────────


def test_block_token_ids_is_row_major_over_the_grid():
    ids = block_token_ids(grid_w=8, top=1, left=2, h=2, w=3)
    assert ids.tolist() == [10, 11, 12, 18, 19, 20]


def test_multiblock_masks_returns_the_configured_number_of_targets():
    gen = torch.Generator().manual_seed(0)
    context, targets = multiblock_masks(8, 8, MaskConfig(n_targets=4), generator=gen)
    assert len(targets) == 4
    assert context.dtype == torch.int64
    assert context.numel() >= MaskConfig().min_keep


def test_target_blocks_all_share_one_size_so_predictions_stack():
    """The reference collator samples one (h, w) and reuses it for every target
    block; equal token counts are what lets the per-block predictions stack."""
    gen = torch.Generator().manual_seed(1)
    for _ in range(20):
        _, targets = multiblock_masks(16, 16, MaskConfig(n_targets=4), generator=gen)
        assert len({t.numel() for t in targets}) == 1


def test_context_never_contains_a_target_token():
    """The whole design. Leave the overlap in and the predictor solves the task
    by copying one of its own input tokens."""
    gen = torch.Generator().manual_seed(2)
    for _ in range(50):
        context, targets = multiblock_masks(16, 16, generator=gen)
        ctx = set(context.tolist())
        for t in targets:
            assert ctx.isdisjoint(set(t.tolist()))


def test_masks_are_reproducible_from_a_seeded_generator():
    a = multiblock_masks(16, 16, generator=torch.Generator().manual_seed(7))
    b = multiblock_masks(16, 16, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0])
    for ta, tb in zip(a[1], b[1], strict=True):
        assert torch.equal(ta, tb)


def test_masks_differ_between_draws():
    gen = torch.Generator().manual_seed(3)
    first = multiblock_masks(16, 16, generator=gen)[0]
    for _ in range(10):
        if not torch.equal(first, multiblock_masks(16, 16, generator=gen)[0]):
            return
    pytest.fail("10 consecutive draws produced the same context mask")


def test_impossible_geometry_raises_rather_than_returning_a_degenerate_context():
    """A grid too small for the requested blocks is a config error, and the
    failure mode it would otherwise cause — a one-token context — is invisible
    in the loss."""
    with pytest.raises(ValueError, match="could not sample a context"):
        multiblock_masks(2, 2, MaskConfig(n_targets=4, target_scale=(0.9, 1.0)))


# ── components ─────────────────────────────────────────────────────────────


def _tiny_encoder(in_channels: int = 2) -> ViTEncoder:
    torch.manual_seed(0)
    return ViTEncoder(in_channels, img_size=32, patch_size=4, embed_dim=16, depth=2, heads=2)


def test_encoder_emits_one_token_per_patch():
    out = _tiny_encoder()(torch.randn(3, 2, 32, 32))
    assert out.shape == (3, 64, 16)  # 32/4 = 8 -> 8x8 tokens


def test_encoder_keep_ids_subsets_tokens():
    enc = _tiny_encoder()
    keep = torch.tensor([0, 5, 17])
    assert enc(torch.randn(2, 2, 32, 32), keep_ids=keep).shape == (2, 3, 16)


def test_encoder_rejects_indivisible_crop():
    with pytest.raises(ValueError, match="divisible"):
        ViTEncoder(2, img_size=30, patch_size=4, embed_dim=16, depth=1, heads=2)


def test_predictor_emits_one_vector_per_requested_target_token():
    torch.manual_seed(0)
    pred = Predictor(embed_dim=16, grid=8, predictor_dim=8, depth=1, heads=2)
    context = torch.randn(3, 10, 16)
    out = pred(context, torch.arange(10), torch.arange(20, 26))
    assert out.shape == (3, 6, 16)


def test_predictor_output_depends_on_which_positions_are_asked_for():
    """Position is the only thing distinguishing one query from another, so a
    predictor that ignores it cannot be doing the task."""
    torch.manual_seed(0)
    pred = Predictor(embed_dim=16, grid=8, predictor_dim=8, depth=1, heads=2).eval()
    context = torch.randn(1, 10, 16)
    with torch.no_grad():
        a = pred(context, torch.arange(10), torch.tensor([20, 21]))
        b = pred(context, torch.arange(10), torch.tensor([40, 41]))
    assert not torch.allclose(a, b)


# ── the model ──────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_ijepa() -> IJEPA:
    torch.manual_seed(0)
    return IJEPA(
        in_channels=2, img_size=32, patch_size=4, embed_dim=16, depth=2, heads=2,
        predictor_dim=8, predictor_depth=1, predictor_heads=2,
    )


def test_forward_shapes(tiny_ijepa):
    gen = torch.Generator().manual_seed(0)
    predictions, targets = tiny_ijepa(torch.randn(4, 2, 32, 32), generator=gen)
    assert predictions.shape == targets.shape
    assert predictions.shape[0] == 4
    assert predictions.shape[1] == 4  # n_targets
    assert predictions.shape[-1] == tiny_ijepa.dim


def test_targets_carry_no_gradient(tiny_ijepa):
    _, targets = tiny_ijepa(torch.randn(2, 2, 32, 32), generator=torch.Generator().manual_seed(0))
    assert not targets.requires_grad


def test_backward_leaves_the_target_encoder_without_gradients(tiny_ijepa):
    """If the target encoder is trained, the "target" is the model agreeing
    with itself and the objective is minimised by a constant."""
    predictions, targets = tiny_ijepa(
        torch.randn(4, 2, 32, 32), generator=torch.Generator().manual_seed(0)
    )
    ijepa_loss(predictions, targets).backward()
    assert all(p.grad is None for p in tiny_ijepa.target_encoder.parameters())
    assert any(p.grad is not None for p in tiny_ijepa.context_encoder.parameters())


def test_trainable_excludes_the_target_encoder(tiny_ijepa):
    """`train_ijepa` builds the optimizer over `.trainable`, so weight decay
    can never shrink the target weights toward zero."""
    trainable = {id(p) for p in tiny_ijepa.trainable.parameters()}
    assert not any(id(p) in trainable for p in tiny_ijepa.target_encoder.parameters())
    assert all(id(p) in trainable for p in tiny_ijepa.context_encoder.parameters())
    assert all(id(p) in trainable for p in tiny_ijepa.predictor.parameters())


def test_target_encoder_starts_as_a_copy_not_a_second_random_init(tiny_ijepa):
    for target, online in zip(
        tiny_ijepa.target_encoder.parameters(),
        tiny_ijepa.context_encoder.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target, online)


def test_update_target_moves_the_target_toward_the_context_encoder(tiny_ijepa):
    with torch.no_grad():
        for p in tiny_ijepa.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in tiny_ijepa.target_encoder.parameters()]
    tiny_ijepa.update_target(0.9)
    for old, new, online in zip(
        before, tiny_ijepa.target_encoder.parameters(),
        tiny_ijepa.context_encoder.parameters(), strict=True,
    ):
        torch.testing.assert_close(new, 0.9 * old + 0.1 * online)


def test_update_target_at_momentum_one_is_a_no_op(tiny_ijepa):
    with torch.no_grad():
        for p in tiny_ijepa.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in tiny_ijepa.target_encoder.parameters()]
    tiny_ijepa.update_target(1.0)
    for old, new in zip(before, tiny_ijepa.target_encoder.parameters(), strict=True):
        torch.testing.assert_close(old, new)


def test_train_mode_keeps_the_target_encoder_in_eval(tiny_ijepa):
    """A train-mode-only layer in the target encoder would inject noise into
    the regression target rather than regularise anything."""
    tiny_ijepa.train()
    assert tiny_ijepa.context_encoder.training
    assert not tiny_ijepa.target_encoder.training


def test_encode_is_unmasked_deterministic_and_the_right_width(tiny_ijepa):
    x = torch.randn(3, 2, 32, 32)
    h1 = tiny_ijepa.encode(x)
    h2 = tiny_ijepa.encode(x)
    assert h1.shape == (3, tiny_ijepa.dim) == (3, 16)
    torch.testing.assert_close(h1, h2)


def test_encode_defaults_to_the_target_encoder(tiny_ijepa):
    """The paper evaluates (and releases) the EMA encoder."""
    with torch.no_grad():
        for p in tiny_ijepa.context_encoder.parameters():
            p.add_(0.5)
    x = torch.randn(2, 2, 32, 32)
    torch.testing.assert_close(tiny_ijepa.encode(x), tiny_ijepa.encode(x, encoder="target"))
    assert not torch.allclose(tiny_ijepa.encode(x), tiny_ijepa.encode(x, encoder="context"))


def test_training_reduces_the_loss(tiny_ijepa):
    torch.manual_seed(0)
    opt = torch.optim.Adam(tiny_ijepa.trainable.parameters(), lr=1e-3)
    x = torch.randn(4, 2, 32, 32)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        predictions, targets = tiny_ijepa(x, generator=gen)
        loss = ijepa_loss(predictions, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ── loss and schedule ──────────────────────────────────────────────────────


def test_loss_kinds_agree_on_a_perfect_prediction():
    x = torch.randn(2, 3, 4, 5)
    assert ijepa_loss(x, x, "smooth_l1").item() == pytest.approx(0.0)
    assert ijepa_loss(x, x, "l2").item() == pytest.approx(0.0)


def test_unknown_loss_kind_raises():
    with pytest.raises(ValueError, match="unknown loss kind"):
        ijepa_loss(torch.zeros(1), torch.zeros(1), "cosine")


def test_momentum_schedule_runs_base_to_final():
    assert momentum_schedule(0, 100) == pytest.approx(0.996)
    assert momentum_schedule(50, 100) == pytest.approx(0.998)
    assert momentum_schedule(100, 100) == pytest.approx(1.0)
    # Past the end (a resumed run overshooting its recorded total) clamps.
    assert momentum_schedule(500, 100) == pytest.approx(1.0)


# ── the eval wiring (WI-13/WI-16 read the arm through this) ────────────────


def _write_run(tmp_path, model, **overrides):
    """A checkpoint plus the `run_config.json` the encoder rebuilds from.

    The weights are perturbed off their initialisation first, standing in for
    training: without that a rebuild that silently re-initialised instead of
    loading would still match, and these tests would prove nothing.
    """
    import json

    torch.manual_seed(11)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    torch.save({"epoch": 7, "model": model.state_dict()}, tmp_path / "last.pt")
    config = {
        "wi": 17, "arch": "ijepa", "k": 1, "crop_size": 32, "in_channels": 2,
        "patch_size": 4, "embed_dim": 16, "depth": 2, "heads": 2,
        "predictor_dim": 8, "predictor_depth": 1, "predictor_heads": 2,
    }
    (tmp_path / "run_config.json").write_text(json.dumps({**config, **overrides}))
    return tmp_path / "last.pt"


def test_checkpoint_encoder_reproduces_encode(tmp_path, tiny_ijepa):
    from ssl_us.eval.store_features import IJEPACheckpointEncoder

    ckpt = _write_run(tmp_path, tiny_ijepa)
    encoder = IJEPACheckpointEncoder(ckpt, in_channels=2, device="cpu")
    crops = torch.randn(5, 2, 32, 32)
    torch.testing.assert_close(
        torch.from_numpy(encoder(crops.numpy())), tiny_ijepa.encode(crops)
    )
    assert encoder.epoch == 7


def test_checkpoint_encoder_rejects_the_wrong_k(tmp_path, tiny_ijepa):
    """K sets the stem's channel count, so a mismatch is the wrong checkpoint
    for these crops — caught here rather than as a shape error later."""
    from ssl_us.eval.store_features import IJEPACheckpointEncoder

    ckpt = _write_run(tmp_path, tiny_ijepa)
    with pytest.raises(ValueError, match="wrong K for this checkpoint"):
        IJEPACheckpointEncoder(ckpt, in_channels=8, device="cpu")


def test_checkpoint_encoder_requires_the_run_config(tmp_path, tiny_ijepa):
    from ssl_us.eval.store_features import IJEPACheckpointEncoder

    ckpt = _write_run(tmp_path, tiny_ijepa)
    (tmp_path / "run_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="run_config.json"):
        IJEPACheckpointEncoder(ckpt, in_channels=2, device="cpu")


def test_random_init_floor_is_the_same_architecture_untrained(tmp_path, tiny_ijepa):
    """`random_vit` must match the trained arm's shape exactly — it is the
    architecture floor for the I-JEPA column, and `random_resnet18` cannot be."""
    from ssl_us.eval.store_features import IJEPACheckpointEncoder

    ckpt = _write_run(tmp_path, tiny_ijepa)
    trained = IJEPACheckpointEncoder(ckpt, in_channels=2, device="cpu")
    floor = IJEPACheckpointEncoder(ckpt, in_channels=2, device="cpu", random_init=True, seed=0)
    crops = torch.randn(4, 2, 32, 32).numpy()
    assert floor(crops).shape == trained(crops).shape
    assert not torch.allclose(torch.from_numpy(floor(crops)), torch.from_numpy(trained(crops)))
    # Seeded, so the floor is the same encoder on every re-run of the matrix.
    other = IJEPACheckpointEncoder(ckpt, in_channels=2, device="cpu", random_init=True, seed=0)
    torch.testing.assert_close(
        torch.from_numpy(floor(crops)), torch.from_numpy(other(crops))
    )
