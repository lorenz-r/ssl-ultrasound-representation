import torch

from ssl_ultrasound_representation.mae.model.complex_unet import (
    ComplexConv2d,
    ComplexConv3d,
    ComplexLinear,
    ComplexUNet2d,
    ComplexUNet3d,
    ProjectedComplexUNet2d,
)


def test_complex_conv_matches_definition():
    layer = ComplexConv2d(2, 3, 3, padding=1, bias=True)
    real = torch.randn(2, 2, 9, 7)
    imag = torch.randn(2, 2, 9, 7)

    out_real, out_imag = layer(real, imag)

    expected_real = layer.real(real) - layer.imag(imag)
    expected_imag = layer.real(imag) + layer.imag(real)
    expected_real += layer.bias_real[None, :, None, None]
    expected_imag += layer.bias_imag[None, :, None, None]
    torch.testing.assert_close(out_real, expected_real)
    torch.testing.assert_close(out_imag, expected_imag)


def test_complex_unet_round_trips_nv_grid_shape():
    model = ComplexUNet2d(base_channels=2, levels=3)
    real = torch.randn(1, 2, 180, 180)
    imag = torch.randn(1, 2, 180, 180)

    out_real, out_imag = model(real, imag)

    assert out_real.shape == (1, 1, 180, 180)
    assert out_imag.shape == out_real.shape
    assert torch.isfinite(out_real).all()
    assert torch.isfinite(out_imag).all()


def test_complex_conv3d_matches_definition():
    layer = ComplexConv3d(2, 3, 3, padding=1, bias=True)
    real = torch.randn(2, 2, 5, 9, 7)
    imag = torch.randn(2, 2, 5, 9, 7)

    out_real, out_imag = layer(real, imag)

    expected_real = layer.real(real) - layer.imag(imag)
    expected_imag = layer.real(imag) + layer.imag(real)
    expected_real += layer.bias_real[None, :, None, None, None]
    expected_imag += layer.bias_imag[None, :, None, None, None]
    torch.testing.assert_close(out_real, expected_real)
    torch.testing.assert_close(out_imag, expected_imag)


def test_complex_unet3d_round_trips_volume_shape_and_backward():
    model = ComplexUNet3d(base_channels=2, levels=3)
    real = torch.randn(1, 2, 9, 13, 11, requires_grad=True)
    imag = torch.randn(1, 2, 9, 13, 11, requires_grad=True)

    out_real, out_imag = model(real, imag)
    (out_real.square().mean() + out_imag.square().mean()).backward()

    assert out_real.shape == (1, 1, 9, 13, 11)
    assert out_imag.shape == out_real.shape
    assert torch.isfinite(out_real).all()
    assert model.inc.conv1.real.weight.grad is not None


def test_complex_linear_matches_definition():
    layer = ComplexLinear(7, 5, bias=True)
    real = torch.randn(2, 3, 7)
    imag = torch.randn(2, 3, 7)

    out_real, out_imag = layer(real, imag)

    expected_real = layer.real(real) - layer.imag(imag) + layer.bias_real
    expected_imag = layer.real(imag) + layer.imag(real) + layer.bias_imag
    torch.testing.assert_close(out_real, expected_real)
    torch.testing.assert_close(out_imag, expected_imag)


def test_projected_complex_unet_round_trips_full_depth_shape():
    model = ProjectedComplexUNet2d(
        depth_size=16,
        embedding_dim=4,
        base_channels=2,
        levels=3,
    )
    real = torch.randn(1, 16, 13, 11, requires_grad=True)
    imag = torch.randn(1, 16, 13, 11, requires_grad=True)
    visibility = torch.ones(1, 1, 13, 11)
    visibility[:, :, :, 3] = 0

    out_real, out_imag = model(real * visibility, imag * visibility, visibility)
    loss = out_real.square().mean() + out_imag.square().mean()
    loss.backward()
    embedding = model.sample_embedding(real, imag, visibility)

    assert out_real.shape == real.shape
    assert out_imag.shape == imag.shape
    assert embedding.shape == (1, 24)
    assert torch.isfinite(out_real).all()
    assert model.depth_encoder.real.weight.grad is not None


def test_square_projected_complex_unet_starts_as_identity():
    model = ProjectedComplexUNet2d(
        depth_size=8,
        embedding_dim=8,
        base_channels=2,
        levels=2,
    )
    real = torch.randn(1, 8, 7, 5)
    imag = torch.randn(1, 8, 7, 5)
    visibility = torch.ones(1, 1, 7, 5)

    out_real, out_imag = model(real, imag, visibility)

    torch.testing.assert_close(out_real, real, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_imag, imag, atol=1e-5, rtol=1e-5)
