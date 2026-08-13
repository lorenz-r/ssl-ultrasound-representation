"""Spatial MAE baseline arm (PLAN.md WI-14, documented baseline only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssl_us.models.mae2d import MAE2D, PatchEmbed, mae_loss, random_masking, sincos_2d_pos_embed


def test_sincos_pos_embed_shape_and_uniqueness():
    pe = sincos_2d_pos_embed(4, 4, 16)
    assert pe.shape == (16, 16)
    # every grid position must get a distinct embedding
    assert len({tuple(row.tolist()) for row in pe}) == 16


def test_sincos_pos_embed_rejects_bad_dim():
    with pytest.raises(ValueError, match="divisible by 4"):
        sincos_2d_pos_embed(4, 4, 15)


def test_patch_embed_produces_64_tokens_for_128_over_16():
    embed = PatchEmbed(in_channels=2, patch_size=16, embed_dim=32)
    x = torch.randn(3, 2, 128, 128)
    out = embed(x)
    assert out.shape == (3, 64, 32)


def test_random_masking_keeps_the_configured_fraction():
    x = torch.randn(5, 64, 8)
    visible, mask, ids_restore = random_masking(x, mask_ratio=0.75)
    assert visible.shape == (5, 16, 8)  # 25% of 64
    assert mask.shape == (5, 64)
    assert torch.all((mask.sum(dim=1) - 48).abs() < 1e-6)  # 75% masked
    assert ids_restore.shape == (5, 64)


def test_random_masking_is_a_valid_permutation():
    """ids_restore must be usable to unshuffle: gathering [visible; masked]
    by ids_restore must place every original token in its own slot."""
    torch.manual_seed(0)
    n = 20
    x = torch.arange(n, dtype=torch.float32).reshape(1, n, 1)
    visible, mask, ids_restore = random_masking(x, mask_ratio=0.5)
    n_keep = visible.shape[1]
    # reconstruct: known values where visible, sentinel where masked
    sentinel = -1.0
    masked_placeholder = torch.full((1, n - n_keep, 1), sentinel)
    full = torch.cat([visible, masked_placeholder], dim=1)
    restored = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, 1))
    # every position marked visible (mask==0) must show its *original* value
    visible_positions = mask[0] == 0
    np.testing.assert_allclose(
        restored[0, visible_positions, 0].numpy(), x[0, visible_positions, 0].numpy()
    )
    # every masked position must show the sentinel (never leaked real data)
    assert torch.all(restored[0, mask[0] == 1, 0] == sentinel)


def test_random_masking_different_each_call_without_shared_generator():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 4)
    _, mask1, _ = random_masking(x, mask_ratio=0.75)
    _, mask2, _ = random_masking(x, mask_ratio=0.75)
    assert not torch.equal(mask1, mask2)


@pytest.fixture
def tiny_mae():
    return MAE2D(
        in_channels=2,
        img_size=32,
        patch_size=8,
        embed_dim=16,
        depth=2,
        heads=2,
        decoder_embed_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        mask_ratio=0.75,
    )


def test_mae_forward_shapes(tiny_mae):
    x = torch.randn(3, 2, 32, 32)
    pred, target, mask = tiny_mae(x)
    assert pred.shape == target.shape == (3, 16, 2 * 8 * 8)  # 32/8=4 -> 16 tokens
    assert mask.shape == (3, 16)


def test_mae_patchify_matches_patch_embed_tokenization(tiny_mae):
    """The loss's ground truth must tokenize identically to the encoder input,
    or the model would be scored against the wrong pixels."""
    x = torch.randn(2, 2, 32, 32)
    target = tiny_mae.patchify(x)
    # manually extract patch (0,0) and compare against target's first token
    patch00 = x[:, :, :8, :8].reshape(2, -1)
    np.testing.assert_allclose(target[:, 0].detach().numpy(), patch00.numpy(), atol=1e-6)


def test_mae_loss_only_scores_masked_patches():
    pred = torch.zeros(2, 4, 3)
    target = torch.ones(2, 4, 3)  # every patch is "wrong" by exactly 1.0 -> MSE 1.0
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    loss = mae_loss(pred, target, mask)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_mae_loss_ignores_visible_patches_even_if_wrong():
    pred = torch.zeros(1, 2, 2)
    target = torch.full((1, 2, 2), 100.0)  # huge error, but...
    mask = torch.tensor([[0.0, 0.0]])  # ...nothing is masked
    loss = mae_loss(pred, target, mask)
    assert loss.item() == pytest.approx(0.0)  # clamp(min=1.0) denom avoids 0/0, numerator is 0


def test_mae_training_reduces_loss(tiny_mae):
    torch.manual_seed(0)
    opt = torch.optim.Adam(tiny_mae.parameters(), lr=1e-3)
    x = torch.randn(4, 2, 32, 32)
    gen = torch.Generator().manual_seed(0)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        pred, target, mask = tiny_mae(x, generator=gen)
        loss = mae_loss(pred, target, mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


def test_mae_rejects_img_size_not_divisible_by_patch_size():
    with pytest.raises(ValueError, match="divisible"):
        MAE2D(in_channels=2, img_size=30, patch_size=8)
