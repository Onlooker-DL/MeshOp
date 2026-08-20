from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int) -> None:
        super().__init__()
        self.modes_x, self.modes_y, self.modes_z = int(modes_x), int(modes_y), int(modes_z)
        scale = 1.0 / math.sqrt(max(in_channels * out_channels, 1))
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z)
        self.weight_pp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_np = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_pn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_nn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny, nz = x.shape
        dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            mx = min(self.modes_x, nx // 2)
            my = min(self.modes_y, ny // 2)
            mz = min(self.modes_z, xf.shape[-1])
            out = torch.zeros(batch, self.weight_pp.shape[1], nx, ny, xf.shape[-1], dtype=torch.cfloat, device=x.device)
            if mx and my and mz:
                out[:, :, :mx, :my, :mz] = self.multiply(xf[:, :, :mx, :my, :mz], self.weight_pp[:, :, :mx, :my, :mz])
                out[:, :, -mx:, :my, :mz] = self.multiply(xf[:, :, -mx:, :my, :mz], self.weight_np[:, :, :mx, :my, :mz])
                out[:, :, :mx, -my:, :mz] = self.multiply(xf[:, :, :mx, -my:, :mz], self.weight_pn[:, :, :mx, :my, :mz])
                out[:, :, -mx:, -my:, :mz] = self.multiply(xf[:, :, -mx:, -my:, :mz], self.weight_nn[:, :, :mx, :my, :mz])
            y = torch.fft.irfftn(out, s=(nx, ny, nz), dim=(-3, -2, -1), norm="ortho")
        return y.to(dtype=dtype)


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, modes_z: int, dropout: float) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_z)
        self.local = nn.Conv3d(width, width, 1)
        self.normalization = nn.GroupNorm(1, width)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(self.normalization(self.spectral(x) + self.local(x))))


class FNO3dSolutionOperator(nn.Module):
    def __init__(self, in_channels: int = 4, width: int = 11, modes_x: int = 5, modes_y: int = 5, modes_z: int = 5, layers: int = 4, projection_width: int = 64, padding: int = 4, dropout: float = 0.03) -> None:
        super().__init__()
        self.padding = max(int(padding), 0)
        self.lifting = nn.Conv3d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes_x, modes_y, modes_z, dropout) for _ in range(int(layers))])
        self.projection_one = nn.Conv3d(width, projection_width, 1)
        self.projection_two = nn.Conv3d(projection_width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            p = self.padding
            x = F.pad(x, (0, p, 0, p, 0, p))
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection_two(F.gelu(self.projection_one(x)))
        if self.padding:
            p = self.padding
            x = x[..., :-p, :-p, :-p]
        return x
