"""WI-18 — VICReg on a ViT-S backbone, the missing (objective x backbone) cell.

docs/MEMORY.md §16 measured I-JEPA against the VICReg arms and could not say whether
the result came from the objective or from the backbone, because ViT-S and the
latent-target objective moved together. This arm holds the objective fixed and
swaps only the backbone.

What is worth testing here is not that a ViT runs — `test_ijepa.py` covers that
— but the three places the swap could go wrong silently:

1. the pooled vector must be the same *kind* of vector the other ViT arms are
   probed on, or the comparison gains a second difference;
2. `VICReg` must read its width from the backbone rather than the ResNet
   constant, or the projector is built at the wrong input size;
3. **the pre-WI-18 checkpoints must keep loading as ResNet-18**, since they
   carry no `backbone` key and every published column depends on them.
"""

from __future__ import annotations

import json

import pytest
import torch

from ssl_us.models.ijepa import ViTEncoder
from ssl_us.models.vicreg import VICReg, ViTBackbone, build_backbone

CROP, K = 128, 4
IN_CH = 2 * K


# ── the backbone itself ────────────────────────────────────────────────────


def test_vit_backbone_emits_one_vector_per_crop():
    b = ViTBackbone(IN_CH, CROP, patch_size=8)
    out = b(torch.randn(3, IN_CH, CROP, CROP))
    assert out.shape == (3, 384)
    assert b.dim == 384


def test_pooling_matches_a_plain_mean_over_tokens():
    """The probe-facing vector must be the same construction `IJEPA.encode` and
    `MAE2D.encode` use — a mean over patch tokens, no CLS. If this drifts, the
    three ViT arms stop being comparable on the representation they expose."""
    torch.manual_seed(0)
    b = ViTBackbone(IN_CH, CROP, patch_size=8)
    x = torch.randn(2, IN_CH, CROP, CROP)
    b.eval()
    with torch.no_grad():
        assert torch.allclose(b(x), b.net(x).mean(1), atol=0)


def test_token_count_is_256_at_the_derived_patch_size():
    """`build_backbone`'s patch rule holds the sequence length fixed across K,
    so token count never becomes a third difference in the comparison."""
    for crop, k in ((128, 4), (64, 16)):
        b = build_backbone("vit_s", 2 * k, crop)
        assert isinstance(b, ViTBackbone)
        assert b.net.n_tokens == 256, (crop, k)


def test_build_backbone_still_returns_a_resnet_by_default():
    b = build_backbone("resnet18", IN_CH, CROP)
    assert b.dim == 512
    assert not isinstance(b, ViTBackbone)


def test_build_backbone_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone("convnext", IN_CH, CROP)


def test_the_seed_makes_initialisation_reproducible():
    a = build_backbone("vit_s", IN_CH, CROP, seed=7)
    b = build_backbone("vit_s", IN_CH, CROP, seed=7)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


# ── composed into VICReg ───────────────────────────────────────────────────


def test_vicreg_takes_its_width_from_the_backbone():
    """The projector is built at `backbone.dim`; hardcoding 512 would build a
    512-input projector over a 384-d ViT and fail only at the first forward."""
    b = build_backbone("vit_s", IN_CH, CROP)
    m = VICReg(b, backbone_dim=b.dim, projector_dim=256)
    h, z = m(torch.randn(2, IN_CH, CROP, CROP))
    assert h.shape == (2, 384)
    assert z.shape == (2, 256)


def test_the_backbone_feature_is_what_gets_probed_not_the_projection():
    """PLAN.md §4.7 probes the backbone. `forward` returns it first, and the two
    must not be the same tensor or the projector is silently the encoder."""
    b = build_backbone("vit_s", IN_CH, CROP)
    m = VICReg(b, backbone_dim=b.dim, projector_dim=384)
    h, z = m(torch.randn(2, IN_CH, CROP, CROP))
    assert not torch.allclose(h, z)


def test_gradients_reach_the_vit_backbone():
    b = build_backbone("vit_s", IN_CH, CROP)
    m = VICReg(b, backbone_dim=b.dim, projector_dim=128)
    _, z = m(torch.randn(2, IN_CH, CROP, CROP))
    z.square().mean().backward()
    grads = [p.grad for p in b.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ── the compatibility guarantee ────────────────────────────────────────────


def test_a_checkpoint_without_a_backbone_key_loads_as_resnet18(tmp_path):
    """Every pre-WI-18 checkpoint has no `backbone` field. If the default ever
    flips, all six published VICReg columns silently change architecture."""
    from ssl_us.eval.store_features import VICRegCheckpointEncoder
    from ssl_us.models.resnet import ResNetBackbone

    m = VICReg(ResNetBackbone(IN_CH), backbone_dim=512, projector_dim=64)
    torch.save({"model": m.state_dict(), "epoch": 3}, tmp_path / "last.pt")
    (tmp_path / "run_config.json").write_text(
        json.dumps({"wi": 12, "crop_size": CROP, "projector_dim": 64})
    )

    enc = VICRegCheckpointEncoder(tmp_path / "last.pt", in_channels=IN_CH, device="cpu")
    assert enc.backbone_name == "resnet18"
    assert enc(torch.randn(2, IN_CH, CROP, CROP).numpy()).shape == (2, 512)


def test_a_vit_checkpoint_round_trips_through_the_eval_encoder(tmp_path):
    """The architecture is read from `run_config.json`; a wrong guess would be a
    shape error here rather than a silently different model in the matrix."""
    from ssl_us.eval.store_features import VICRegCheckpointEncoder

    b = build_backbone("vit_s", IN_CH, CROP)
    m = VICReg(b, backbone_dim=b.dim, projector_dim=64)
    torch.save({"model": m.state_dict(), "epoch": 5}, tmp_path / "last.pt")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {"wi": 18, "backbone": "vit_s", "crop_size": CROP,
             "patch_size": 8, "projector_dim": 64}
        )
    )

    enc = VICRegCheckpointEncoder(tmp_path / "last.pt", in_channels=IN_CH, device="cpu")
    assert enc.backbone_name == "vit_s"
    out = enc(torch.randn(3, IN_CH, CROP, CROP).numpy())
    assert out.shape == (3, 384)


def test_the_eval_encoder_reproduces_the_trained_backbone_exactly(tmp_path):
    """Frozen features must be the trained weights' features, not a re-init."""
    from ssl_us.eval.store_features import VICRegCheckpointEncoder

    torch.manual_seed(3)
    b = build_backbone("vit_s", IN_CH, CROP)
    m = VICReg(b, backbone_dim=b.dim, projector_dim=64).eval()
    torch.save({"model": m.state_dict(), "epoch": 1}, tmp_path / "last.pt")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {"wi": 18, "backbone": "vit_s", "crop_size": CROP,
             "patch_size": 8, "projector_dim": 64}
        )
    )

    x = torch.randn(2, IN_CH, CROP, CROP)
    with torch.no_grad():
        expected = m.backbone(x).numpy()
    enc = VICRegCheckpointEncoder(tmp_path / "last.pt", in_channels=IN_CH, device="cpu")
    assert torch.allclose(torch.from_numpy(enc(x.numpy())), torch.from_numpy(expected), atol=1e-5)


def test_the_vit_backbone_wraps_the_shared_vit_not_a_copy():
    """`ViTEncoder` is imported rather than reimplemented, so the WI-17 and
    WI-18 ViTs cannot drift apart."""
    assert isinstance(ViTBackbone(IN_CH, CROP, patch_size=8).net, ViTEncoder)
