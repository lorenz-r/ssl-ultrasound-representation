"""WI-19 dense segmentation probe: pre-pool shapes, honest CIs, stitching.

The dense path is the one place in the project where a row is a *pixel* rather
than a crop, so the tests that matter most here guard the two ways that changes
the statistics: the CV groups must still be frames, and the bootstrap must
resample frames rather than cells.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssl_us.eval.dense_features import (
    DenseSpec,
    PcaRawDense,
    ResNetHypercolumn,
    TokenMap,
    TrivialDense,
    cell_pool,
    crop_envelope,
    dense_from_pooled,
    whole_frame_probability,
    window_origins,
)
from ssl_us.eval.dense_probe import (
    cell_targets,
    dense_probe,
    dice_from_counts,
    upsample_maps,
)
from ssl_us.eval.guards import LabelLeakageError
from ssl_us.eval.store_features import crop_echogenicity_db

SPEC = DenseSpec(grid=4, batch_size=8)


# ── envelope and cell pooling ───────────────────────────────────────────────


def test_crop_envelope_matches_the_echogenicity_convention():
    """The dense envelope must be the same quantity `echogenicity` reports.

    `crop_echogenicity_db` works on the `(2, size, size, k)` reduced crop and
    `crop_envelope` on the `(2k, size, size)` channel-first one; if the I/Q
    block split were transposed the two would disagree, silently, and only on
    K > 1.
    """
    rng = np.random.default_rng(0)
    k, size = 4, 8
    v = rng.normal(size=(2, size, size, k)).astype(np.float32)
    channel_first = np.moveaxis(v, -1, 1).reshape(2 * k, size, size)[None]
    env = crop_envelope(channel_first, k)[0]
    expected = 20.0 * np.log10(env.mean() + 1e-12)
    assert expected == pytest.approx(crop_echogenicity_db(v), rel=1e-5)


def test_cell_pool_partitions_the_crop():
    x = np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8)
    cells = cell_pool(x, grid=4)
    assert cells.shape == (2, 4, 4, 2, 2)
    # Cell (0, 1) is rows 0-1, columns 2-3 of the original.
    np.testing.assert_allclose(cells[0, 0, 1], x[0, 0:2, 2:4])


def test_cell_size_rejects_an_indivisible_crop():
    with pytest.raises(ValueError, match="not divisible"):
        DenseSpec(grid=16).cell_size(100)


# ── pre-pool extractors ─────────────────────────────────────────────────────


def test_resnet_hypercolumn_concatenates_layers_two_to_four():
    from ssl_us.models.resnet import ResNetBackbone

    ext = ResNetHypercolumn(ResNetBackbone(4).net, SPEC, device="cpu")
    assert ext.dim == 128 + 256 + 512
    out = ext(np.random.default_rng(0).normal(size=(3, 4, 64, 64)).astype(np.float32))
    assert out.shape == (3, ext.dim, SPEC.grid, SPEC.grid)


def test_token_map_inverts_the_row_major_tokenization():
    """A token grid must land on the cell it came from, not its transpose.

    `PatchEmbed` unfolds rows then columns, so token `i` is row `i // n_w`.
    Feeding tokens that carry their own index makes a transposed reshape
    visible; a shape assertion alone would not.
    """
    grid = 4
    n = grid * grid
    tokens = torch.arange(n, dtype=torch.float32).reshape(1, n, 1)
    ext = TokenMap(lambda x: tokens, DenseSpec(grid=grid, batch_size=8), device="cpu")
    out = ext(np.zeros((1, 2, 8, 8), dtype=np.float32))
    np.testing.assert_allclose(out[0, 0], np.arange(n).reshape(grid, grid))


def test_token_map_resamples_a_mismatched_grid():
    tokens = torch.zeros(2, 64, 5)
    ext = TokenMap(lambda x: tokens, SPEC, device="cpu")
    assert ext(np.zeros((2, 2, 16, 16), dtype=np.float32)).shape == (2, 5, 4, 4)


def test_token_map_rejects_a_non_square_token_count():
    ext = TokenMap(lambda x: torch.zeros(1, 12, 3), SPEC, device="cpu")
    with pytest.raises(ValueError, match="square grid"):
        ext(np.zeros((1, 2, 8, 8), dtype=np.float32))


def test_trivial_dense_is_two_channels_per_cell():
    crops = np.random.default_rng(0).normal(size=(5, 4, 8, 8)).astype(np.float32)
    out = TrivialDense(k=2, spec=SPEC)(crops)
    assert out.shape == (5, 2, 4, 4)


def test_pca_raw_dense_caps_at_the_patch_dimension():
    crops = np.random.default_rng(0).normal(size=(6, 2, 8, 8)).astype(np.float32)
    out = PcaRawDense(k=1, spec=SPEC, dims=64)(crops)
    # A 2x2 cell has only 4 pixels, so 64 components cannot be produced.
    assert out.shape == (6, 4, 4, 4)


def test_dense_from_pooled_finds_the_resnet_inside_the_random_baseline():
    from ssl_us.eval.features import RandomResNet18

    ext = dense_from_pooled(RandomResNet18(in_channels=2, device="cpu"), SPEC, "cpu")
    assert isinstance(ext, ResNetHypercolumn)


def test_dense_from_pooled_rejects_an_unknown_encoder():
    with pytest.raises(TypeError, match="no pre-pool path"):
        dense_from_pooled(object(), SPEC, "cpu")


# ── the pooled/dense drift guards on the model classes ──────────────────────


def test_mae_encode_is_the_mean_of_encode_tokens():
    from ssl_us.models.mae2d import MAE2D

    torch.manual_seed(0)
    model = MAE2D(in_channels=2, img_size=16, patch_size=4).eval()
    x = torch.randn(2, 2, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(model.encode(x), model.encode_tokens(x).mean(dim=1))


def test_ijepa_encode_is_the_mean_of_encode_tokens_for_both_encoders():
    from ssl_us.models.ijepa import IJEPA

    torch.manual_seed(0)
    model = IJEPA(
        in_channels=2, img_size=16, patch_size=4, embed_dim=16, depth=1, heads=2,
        predictor_dim=8, predictor_depth=1, predictor_heads=2,
    ).eval()
    x = torch.randn(2, 2, 16, 16)
    for which in ("target", "context"):
        torch.testing.assert_close(
            model.encode(x, encoder=which),
            model.encode_tokens(x, encoder=which).mean(dim=1),
        )


# ── targets and upsampling ──────────────────────────────────────────────────


def test_cell_targets_are_the_cyst_fraction_of_each_cell():
    seg = np.zeros((1, 8, 8), dtype=np.uint8)
    seg[0, 0:2, 0] = 1  # half of cell (0, 0) at grid 4
    y = cell_targets(seg, grid=4)
    assert y.shape == (1, 4, 4)
    assert y[0, 0, 0] == pytest.approx(0.5)
    assert y[0, 1, 1] == pytest.approx(0.0)


def test_upsample_maps_returns_pixel_resolution():
    prob = np.random.default_rng(0).random((3, 4, 4)).astype(np.float32)
    assert upsample_maps(prob, 32).shape == (3, 32, 32)


# ── sliding window ──────────────────────────────────────────────────────────


def test_window_origins_reach_the_far_edge():
    """507 and 456 are not multiples of 128, so the flush-to-edge window is the
    normal case; without it the accumulator has an uncovered strip."""
    origins = window_origins(507, 456, size=128, stride=32)
    assert origins[:, 0].max() == 507 - 128
    assert origins[:, 1].max() == 456 - 128


def test_window_origins_refuse_a_gapping_stride():
    """A stride wider than the crop leaves uncovered pixels, which the
    accumulator reports as confident background — a checkerboard figure and a
    depressed Dice for a reason that is not the model."""
    with pytest.raises(ValueError, match="would not cover|exceeds the crop size"):
        window_origins(256, 256, size=32, stride=64)


class _ConstantHead:
    """A head that returns the same probability everywhere."""

    def __init__(self, value: float):
        self.value = value

    def predict_proba_maps(self, maps):
        return np.full((len(maps), 4, 4), self.value, dtype=np.float32)


def test_whole_frame_probability_covers_every_pixel():
    frame = np.zeros((2, 96, 80), dtype=np.float32)
    prob = whole_frame_probability(
        frame, lambda c: np.zeros((len(c), 1, 4, 4), np.float16),
        _ConstantHead(0.75), size=32, stride=16,
    )
    assert prob.shape == (96, 80)
    # Constant everywhere means no window seams and no uncovered pixels.
    np.testing.assert_allclose(prob, 0.75, atol=1e-6)


# ── the probe protocol ──────────────────────────────────────────────────────


@pytest.fixture
def separable_dense():
    """Feature maps whose first channel linearly encodes the mask."""
    rng = np.random.default_rng(0)
    n_frames, per_frame, grid, size = 8, 4, 4, 8
    n = n_frames * per_frame
    fid = np.repeat([f"train/{i}" for i in range(n_frames)], per_frame)
    y = (rng.random((n, grid, grid)) < 0.35).astype(np.uint8)
    seg = np.repeat(np.repeat(y, size // grid, axis=1), size // grid, axis=2).astype(np.uint8)
    maps = np.concatenate(
        [
            (y[:, None] * 3.0 + rng.normal(scale=0.4, size=(n, 1, grid, grid))),
            rng.normal(size=(n, 5, grid, grid)),
        ],
        axis=1,
    ).astype(np.float16)
    return maps, seg, fid


def test_dense_probe_recovers_a_linearly_encoded_mask(separable_dense):
    maps, seg, fid = separable_dense
    res, prob = dense_probe(maps, seg, fid, encoder="synthetic", stage="c0_const", n_boot=200)
    assert res.score > 0.8
    assert prob.shape == (len(fid), 4, 4)
    assert res.score_lo <= res.score <= res.score_hi


def test_dense_probe_counts_frames_not_cells(separable_dense):
    """`n_frames` is the effective sample size. If it ever reports cells, the
    CI beside it is meaningless — that is the whole reason this column exists
    as its own protocol rather than a call into `probes.probe`."""
    maps, seg, fid = separable_dense
    res, _ = dense_probe(maps, seg, fid, encoder="synthetic", stage="c0_const", n_boot=200)
    assert res.n_frames == len(np.unique(fid))
    assert res.n_crops == len(fid)
    assert res.n_cells == len(fid) * 16
    assert set(res.per_frame_dice) == set(np.unique(fid))


def test_dense_probe_ci_is_frame_wide_not_cell_narrow(separable_dense):
    """A cell-resampling bootstrap over thousands of rows returns a CI orders
    of magnitude too tight. With 8 frames the frame-level CI must stay visibly
    wide, which is what this asserts."""
    maps, seg, fid = separable_dense
    res, _ = dense_probe(maps, seg, fid, encoder="synthetic", stage="c0_const", n_boot=500)
    assert res.score_hi - res.score_lo > 0.01


def test_dense_probe_finds_nothing_in_noise():
    rng = np.random.default_rng(1)
    n_frames, per_frame, grid, size = 6, 4, 4, 8
    n = n_frames * per_frame
    fid = np.repeat([f"train/{i}" for i in range(n_frames)], per_frame)
    seg = (rng.random((n, size, size)) < 0.3).astype(np.uint8)
    maps = rng.normal(size=(n, 6, grid, grid)).astype(np.float16)
    res, _ = dense_probe(maps, seg, fid, encoder="noise", stage="c0_const", n_boot=200)
    # Chance Dice for a balanced-weight head on a p-positive target is ~2p/(1+p);
    # anything far above that would mean the protocol is leaking.
    assert res.score < 0.6


def test_dense_probe_routes_through_the_leak_guard(separable_dense):
    maps, seg, fid = separable_dense
    with pytest.raises((LabelLeakageError, ValueError)):
        dense_probe(maps, seg, fid, encoder="synthetic", stage="not_a_stage", n_boot=10)


def test_dense_probe_needs_at_least_two_frames():
    maps = np.zeros((4, 3, 4, 4), dtype=np.float16)
    seg = np.zeros((4, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="need >=2 frames"):
        dense_probe(maps, seg, np.array(["f"] * 4), encoder="x", stage="c0_const", n_boot=10)


def test_dice_is_one_when_both_fields_are_empty():
    assert dice_from_counts(0.0, 0.0, 0.0) == 1.0
