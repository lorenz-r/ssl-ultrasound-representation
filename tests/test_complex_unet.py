import torch

from ssl_ultrasound_representation.mae.model.complex_unet import (
    ComplexConv2d,
    ComplexUNet2d,
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
