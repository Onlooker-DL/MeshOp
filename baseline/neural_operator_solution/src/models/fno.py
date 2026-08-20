from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_t: int,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_t = int(modes_t)
        scale = 1.0 / math.sqrt(max(in_channels * out_channels, 1))
        shape = (in_channels, out_channels, modes_x, modes_t)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, nt = x.shape
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfft2(x.float(), norm="ortho")
            mx = min(self.modes_x, nx // 2)
            mt = min(self.modes_t, xf.shape[-1])
            out = torch.zeros(
                batch,
                self.out_channels,
                nx,
                xf.shape[-1],
                dtype=torch.cfloat,
                device=x.device,
            )
            if mx > 0 and mt > 0:
                out[:, :, :mx, :mt] = self.multiply(
                    xf[:, :, :mx, :mt],
                    self.weight_positive[:, :, :mx, :mt],
                )
                out[:, :, -mx:, :mt] = self.multiply(
                    xf[:, :, -mx:, :mt],
                    self.weight_negative[:, :, :mx, :mt],
                )
            y = torch.fft.irfft2(out, s=(nx, nt), norm="ortho")
        return y.to(dtype=input_dtype)


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_t: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_t)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.normalization = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x) + self.local(x)
        return self.dropout(F.gelu(self.normalization(x)))


class FNO2dSolutionOperator(nn.Module):
    """2-D FNO mapping (u0(x), f(x)) to u(x,t)."""

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 9,
        modes_x: int = 6,
        modes_t: int = 6,
        layers: int = 4,
        projection_width: int = 96,
        padding_t: int = 9,
        dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.padding_t = max(int(padding_t), 0)
        self.lifting = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(width, modes_x, modes_t, dropout)
                for _ in range(int(layers))
            ]
        )
        self.projection_one = nn.Conv2d(width, projection_width, 1)
        self.projection_two = nn.Conv2d(projection_width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_t:
            x = F.pad(x, (0, self.padding_t, 0, 0))
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection_two(F.gelu(self.projection_one(x)))
        return x[..., :-self.padding_t] if self.padding_t else x
