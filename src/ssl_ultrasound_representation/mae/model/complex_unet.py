"""Complex-valued U-Net operating on TX/RX response matrices."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ComplexConv2d(nn.Module):
    """Complex convolution implemented by two coupled real convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kwargs = {
            "stride": stride,
            "padding": padding,
            "bias": False,
        }
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.real(imag) + self.imag(real)
        if self.bias_real is not None:
            out_real = out_real + self.bias_real[None, :, None, None]
            out_imag = out_imag + self.bias_imag[None, :, None, None]
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """Complex transposed convolution with the same coupling as ComplexConv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kwargs = {
            "stride": stride,
            "padding": padding,
            "bias": False,
        }
        self.real = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.imag = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, **kwargs
        )
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.real(imag) + self.imag(real)
        if self.bias_real is not None:
            out_real = out_real + self.bias_real[None, :, None, None]
            out_imag = out_imag + self.bias_imag[None, :, None, None]
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """Whiten the 2x2 real/imag covariance independently per feature map."""

    def __init__(
        self,
        channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.weight_rr = nn.Parameter(torch.ones(channels))
        self.weight_ii = nn.Parameter(torch.ones(channels))
        self.weight_ri = nn.Parameter(torch.zeros(channels))
        self.bias_real = nn.Parameter(torch.zeros(channels))
        self.bias_imag = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean_real", torch.zeros(channels))
        self.register_buffer("running_mean_imag", torch.zeros(channels))
        self.register_buffer("running_covar_rr", torch.ones(channels))
        self.register_buffer("running_covar_ii", torch.ones(channels))
        self.register_buffer("running_covar_ri", torch.zeros(channels))

    def _batch_statistics(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        dims = (0, 2, 3)
        mean_real = real.mean(dims)
        mean_imag = imag.mean(dims)
        centered_real = real - mean_real[None, :, None, None]
        centered_imag = imag - mean_imag[None, :, None, None]
        covar_rr = centered_real.square().mean(dims)
        covar_ii = centered_imag.square().mean(dims)
        covar_ri = (centered_real * centered_imag).mean(dims)
        return mean_real, mean_imag, covar_rr, covar_ii, covar_ri

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            stats = self._batch_statistics(real, imag)
            mean_real, mean_imag, covar_rr, covar_ii, covar_ri = stats
            with torch.no_grad():
                momentum = self.momentum
                self.running_mean_real.lerp_(mean_real.detach(), momentum)
                self.running_mean_imag.lerp_(mean_imag.detach(), momentum)
                self.running_covar_rr.lerp_(covar_rr.detach(), momentum)
                self.running_covar_ii.lerp_(covar_ii.detach(), momentum)
                self.running_covar_ri.lerp_(covar_ri.detach(), momentum)
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            covar_rr = self.running_covar_rr
            covar_ii = self.running_covar_ii
            covar_ri = self.running_covar_ri

        centered_real = real - mean_real[None, :, None, None]
        centered_imag = imag - mean_imag[None, :, None, None]

        covar_rr = covar_rr + self.eps
        covar_ii = covar_ii + self.eps
        determinant_sqrt = (
            covar_rr * covar_ii - covar_ri.square()
        ).clamp_min(self.eps).sqrt()
        trace_sqrt = (
            covar_rr + covar_ii + 2.0 * determinant_sqrt
        ).clamp_min(self.eps).sqrt()
        inverse_scale = (determinant_sqrt * trace_sqrt).reciprocal()
        inverse_rr = (covar_ii + determinant_sqrt) * inverse_scale
        inverse_ii = (covar_rr + determinant_sqrt) * inverse_scale
        inverse_ri = -covar_ri * inverse_scale

        white_real = (
            inverse_rr[None, :, None, None] * centered_real
            + inverse_ri[None, :, None, None] * centered_imag
        )
        white_imag = (
            inverse_ri[None, :, None, None] * centered_real
            + inverse_ii[None, :, None, None] * centered_imag
        )
        out_real = (
            self.weight_rr[None, :, None, None] * white_real
            + self.weight_ri[None, :, None, None] * white_imag
            + self.bias_real[None, :, None, None]
        )
        out_imag = (
            self.weight_ri[None, :, None, None] * white_real
            + self.weight_ii[None, :, None, None] * white_imag
            + self.bias_imag[None, :, None, None]
        )
        return out_real, out_imag


class ComplexSiLU(nn.Module):
    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return F.silu(real), F.silu(imag)


class ComplexDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = ComplexBatchNorm2d(out_channels)
        self.act1 = ComplexSiLU()
        self.conv2 = ComplexConv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ComplexBatchNorm2d(out_channels)
        self.act2 = ComplexSiLU()

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imag = self.conv1(real, imag)
        real, imag = self.norm1(real, imag)
        real, imag = self.act1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.norm2(real, imag)
        return self.act2(real, imag)


class ComplexDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = ComplexConv2d(
            in_channels, out_channels, 4, stride=2, padding=1
        )
        self.block = ComplexDoubleConv(out_channels, out_channels)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imag = self.down(real, imag)
        return self.block(real, imag)


class ComplexUp(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.up = ComplexConvTranspose2d(
            in_channels, out_channels, 2, stride=2
        )
        self.block = ComplexDoubleConv(out_channels + skip_channels, out_channels)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        skip_real: torch.Tensor,
        skip_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real, imag = self.up(real, imag)
        if real.shape[-2:] != skip_real.shape[-2:]:
            size = skip_real.shape[-2:]
            real = F.interpolate(real, size=size, mode="bilinear", align_corners=False)
            imag = F.interpolate(imag, size=size, mode="bilinear", align_corners=False)
        real = torch.cat((skip_real, real), dim=1)
        imag = torch.cat((skip_imag, imag), dim=1)
        return self.block(real, imag)


class ComplexUNet2d(nn.Module):
    """U-Net for one or more complex channels on a two-dimensional grid.

    The NV experiment uses two complex input channels: IQ and a real-valued
    visibility mask whose imaginary component is zero. The output is one
    predicted complex IQ channel. Spatial dimensions are padded internally so
    arbitrary probe sizes, including 180x180, round-trip exactly.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 12,
        levels: int = 4,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        widths = [base_channels * (2**i) for i in range(levels)]
        self.levels = levels
        self.inc = ComplexDoubleConv(in_channels, widths[0])
        self.down = nn.ModuleList(
            ComplexDown(widths[i - 1], widths[i]) for i in range(1, levels)
        )
        self.up = nn.ModuleList(
            ComplexUp(widths[i], widths[i - 1], widths[i - 1])
            for i in range(levels - 1, 0, -1)
        )
        self.out = ComplexConv2d(widths[0], out_channels, 1)

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if real.shape != imag.shape:
            raise ValueError(f"real/imag shape mismatch: {real.shape} vs {imag.shape}")
        multiple = 2 ** (self.levels - 1)
        height, width = real.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        padding = (0, pad_width, 0, pad_height)
        if pad_height or pad_width:
            real = F.pad(real, padding)
            imag = F.pad(imag, padding)

        real, imag = self.inc(real, imag)
        skips = [(real, imag)]
        for layer in self.down:
            real, imag = layer(real, imag)
            skips.append((real, imag))
        for layer, (skip_real, skip_imag) in zip(
            self.up, reversed(skips[:-1]), strict=True
        ):
            real, imag = layer(real, imag, skip_real, skip_imag)
        real, imag = self.out(real, imag)
        return real[..., :height, :width], imag[..., :height, :width]

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
