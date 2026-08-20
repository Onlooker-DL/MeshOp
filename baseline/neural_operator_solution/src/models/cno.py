from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinc_kernel(taps: int, scale: float) -> torch.Tensor:
    """Windowed-sinc 1-D low-pass kernel (Hamming window)."""
    if taps < 2:
        raise ValueError("taps must be at least 2.")
    center = (taps - 1) / 2.0
    x = torch.arange(taps, dtype=torch.float32) - center
    kernel = torch.sinc(x / scale)
    window = 0.54 - 0.46 * torch.cos(
        2.0 * math.pi * x / float(taps - 1)
    )
    kernel = kernel * window
    if float(kernel.abs().sum()) <= 0.0:
        raise ValueError("Sinc kernel vanished.")
    return kernel / kernel.sum()


def _next_spatial_size(value: int, levels: int) -> int:
    """Smallest N >= value such that N/2^k is an even integer for k=1..levels."""
    unit = 2 ** (levels + 1)
    return int(math.ceil(value / unit) * unit)


def _conv(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    bias: bool,
) -> nn.Module:
    if dim == 2:
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
    if dim == 3:
        return nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
    raise ValueError(f"dim must be 2 or 3, got {dim}.")


def _bn(dim: int, channels: int) -> nn.Module:
    if dim == 2:
        return nn.BatchNorm2d(channels)
    if dim == 3:
        return nn.BatchNorm3d(channels)
    raise ValueError(f"dim must be 2 or 3, got {dim}.")


def _dropout(dim: int, p: float) -> nn.Module:
    if p <= 0.0:
        return nn.Identity()
    if dim == 2:
        return nn.Dropout2d(p)
    if dim == 3:
        return nn.Dropout3d(p)
    raise ValueError(f"dim must be 2 or 3, got {dim}.")


def _activation(name: str) -> nn.Module:
    table = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return table[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown activation: {name}") from exc


class SincFilter(nn.Module):
    """Separable fixed windowed-sinc low-pass filter for 2-D or 3-D fields."""

    def __init__(
        self,
        dim: int = 2,
        taps: int = 12,
        scale: float = 2.0,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}.")
        self.dim = dim
        self.taps = int(taps)
        self.padding = int(taps // 2)
        kernel = _sinc_kernel(taps, scale)
        if dim == 2:
            # kernel_w: taps along width; kernel_h: taps along height.
            self.register_buffer(
                "kernel_w",
                kernel.reshape(1, 1, 1, -1),
                persistent=False,
            )
            self.register_buffer(
                "kernel_h",
                kernel.reshape(1, 1, -1, 1),
                persistent=False,
            )
        else:
            self.register_buffer(
                "kernel_w",
                kernel.reshape(1, 1, 1, 1, -1),
                persistent=False,
            )
            self.register_buffer(
                "kernel_h",
                kernel.reshape(1, 1, 1, -1, 1),
                persistent=False,
            )
            self.register_buffer(
                "kernel_d",
                kernel.reshape(1, 1, -1, 1, 1),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if self.dim == 2:
            kw = self.kernel_w.expand(channels, -1, -1, -1)
            kh = self.kernel_h.expand(channels, -1, -1, -1)
            y = F.conv2d(
                x,
                kw,
                padding=(self.padding, 0),
                groups=channels,
            )
            y = F.conv2d(
                y,
                kh,
                padding=(0, self.padding),
                groups=channels,
            )
            return y

        kw = self.kernel_w.expand(channels, -1, -1, -1, -1)
        kh = self.kernel_h.expand(channels, -1, -1, -1, -1)
        kd = self.kernel_d.expand(channels, -1, -1, -1, -1)
        y = F.conv3d(
            x,
            kw,
            padding=(self.padding, 0, 0),
            groups=channels,
        )
        y = F.conv3d(
            y,
            kh,
            padding=(0, self.padding, 0),
            groups=channels,
        )
        y = F.conv3d(
            y,
            kd,
            padding=(0, 0, self.padding),
            groups=channels,
        )
        return y


class SincDown2(nn.Module):
    """Bandlimited 2x downsampling: low-pass filter, then decimate by 2."""

    def __init__(self, dim: int = 2, taps: int = 12) -> None:
        super().__init__()
        self.dim = dim
        self.filter = SincFilter(dim=dim, taps=taps, scale=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.filter(x)
        sl = (slice(None), slice(None)) + (slice(0, None, 2),) * self.dim
        return y[sl]


class SincUp2(nn.Module):
    """Bandlimited 2x upsampling: insert zeros, then low-pass filter."""

    def __init__(self, dim: int = 2, taps: int = 12) -> None:
        super().__init__()
        self.dim = dim
        self.filter = SincFilter(dim=dim, taps=taps, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sizes = [2 * s for s in x.shape[2:]]
        y = x.new_zeros(
            (x.shape[0], x.shape[1], *sizes),
            dtype=x.dtype,
        )
        sl = (slice(None), slice(None)) + tuple(
            slice(0, None, 2) for _ in sizes
        )
        y[sl] = x
        return self.filter(y)


class ConvBlock(nn.Module):
    """Conv + BatchNorm + activation (dimension-agnostic)."""

    def __init__(
        self,
        dim: int = 2,
        in_channels: int = 4,
        out_channels: int = 16,
        kernel_size: int = 3,
        activation: str = "gelu",
        use_bn: bool = True,
        use_act: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            _conv(
                dim,
                in_channels,
                out_channels,
                kernel_size,
                padding,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(_bn(dim, out_channels))
        if use_act:
            layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(_dropout(dim, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """ResNet block R(v) = act(v + K2(act(K1(v)))) with same channels."""

    def __init__(
        self,
        dim: int = 2,
        channels: int = 16,
        kernel_size: int = 3,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(
            dim,
            channels,
            channels,
            kernel_size,
            activation,
            dropout=dropout,
        )
        self.conv2 = _conv(
            dim,
            channels,
            channels,
            kernel_size,
            kernel_size // 2,
            bias=False,
        )
        self.bn2 = _bn(dim, channels)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv2(self.conv1(x))
        out = self.bn2(out)
        return self.act(out + residual)


class DownBlock(nn.Module):
    """Encoder block: conv + activation + bandlimited 2x downsampling."""

    def __init__(
        self,
        dim: int = 2,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            dim,
            in_channels,
            out_channels,
            kernel_size,
            activation,
            dropout=dropout,
        )
        self.down = SincDown2(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.conv(x))


class UpBlock(nn.Module):
    """Decoder block: bandlimited 2x upsampling + conv + activation."""

    def __init__(
        self,
        dim: int = 2,
        in_channels: int = 32,
        out_channels: int = 16,
        kernel_size: int = 3,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = SincUp2(dim)
        self.conv = ConvBlock(
            dim,
            in_channels,
            out_channels,
            kernel_size,
            activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class CNO(nn.Module):
    """Convolutional Neural Operator (operator U-Net), 2-D or 3-D.

    Follows Raonic et al., ICLR 2023: lifting convolution, an encoder of
    bandlimited downsampling blocks, a ResNet bottleneck, a decoder of
    bandlimited upsampling blocks with U-Net patching, and a projection
    convolution. Fixed windowed-sinc filters keep the operator
    representation-equivalent / alias-free.

    Input:  (B, C, H, W)  for dim=2, or (B, C, D, H, W) for dim=3.
    Output: same spatial shape as the input.
    """

    def __init__(
        self,
        dim: int = 2,
        in_channels: int = 4,
        out_channels: int = 1,
        width: int = 16,
        levels: int = 2,
        n_res_bottleneck: int = 2,
        n_res_intermediate: int = 2,
        kernel_size: int = 3,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}.")
        if levels < 1:
            raise ValueError("levels must be at least 1.")

        self.dim = dim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.levels = int(levels)

        channel_at = [self.width * (2 ** level) for level in range(self.levels)]
        bottleneck_channels = channel_at[-1]

        self.lifting = ConvBlock(
            dim,
            in_channels,
            self.width,
            kernel_size,
            activation,
            use_bn=False,
            dropout=dropout,
        )

        encoder: list[nn.Module] = []
        for level in range(self.levels):
            encoder.append(
                DownBlock(
                    dim,
                    channel_at[level],
                    (
                        channel_at[level + 1]
                        if level + 1 < self.levels
                        else bottleneck_channels
                    ),
                    kernel_size,
                    activation,
                    dropout,
                )
            )
        self.encoder = nn.ModuleList(encoder)

        self.bottleneck = nn.Sequential(
            *[
                ResBlock(
                    dim,
                    bottleneck_channels,
                    kernel_size,
                    activation,
                    dropout,
                )
                for _ in range(int(n_res_bottleneck))
            ]
        )

        decoder: list[nn.Module] = []
        for level in reversed(range(self.levels)):
            deep_channels = (
                channel_at[level + 1]
                if level + 1 < self.levels
                else bottleneck_channels
            )
            shallow_channels = channel_at[level]
            decoder.append(
                nn.ModuleList(
                    [
                        UpBlock(
                            dim,
                            deep_channels,
                            shallow_channels,
                            kernel_size,
                            activation,
                            dropout,
                        ),
                        ConvBlock(
                            dim,
                            2 * shallow_channels,
                            shallow_channels,
                            kernel_size,
                            activation,
                            dropout=dropout,
                        ),
                        nn.Sequential(
                            *[
                                ResBlock(
                                    dim,
                                    shallow_channels,
                                    kernel_size,
                                    activation,
                                    dropout,
                                )
                                for _ in range(int(n_res_intermediate))
                            ]
                        ),
                    ]
                )
            )
        self.decoder = nn.ModuleList(decoder)

        self.projection = _conv(
            dim,
            self.width,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[2:]
        targets = [
            _next_spatial_size(s, self.levels)
            for s in spatial
        ]
        pads = [t - s for t, s in zip(targets, spatial)]

        if any(pads):
            if self.dim == 2:
                x = F.pad(x, (0, pads[1], 0, pads[0]))
            else:
                x = F.pad(x, (0, pads[2], 0, pads[1], 0, pads[0]))

        x = self.lifting(x)
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            skips.append(x)
            x = block(x)
        x = self.bottleneck(x)

        for level, (up_block, patch_block, res_blocks) in enumerate(
            self.decoder
        ):
            x = up_block(x)
            skip = skips[self.levels - 1 - level]
            if x.shape[2:] != skip.shape[2:]:
                mode = "bilinear" if self.dim == 2 else "trilinear"
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode=mode,
                    align_corners=False,
                )
            x = torch.cat((x, skip), dim=1)
            x = patch_block(x)
            x = res_blocks(x)

        x = self.projection(x)
        if any(pads):
            if self.dim == 2:
                x = x[:, :, : spatial[0], : spatial[1]]
            else:
                x = x[
                    :,
                    :,
                    : spatial[0],
                    : spatial[1],
                    : spatial[2],
                ]
        return x
