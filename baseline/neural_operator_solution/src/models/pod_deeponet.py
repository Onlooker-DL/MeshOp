from __future__ import annotations

import torch
import torch.nn as nn

from .common import MLP


class PODCoefficientNet(nn.Module):
    """DeepONet branch network that predicts coefficients of a fixed POD basis."""

    def __init__(
        self,
        branch_dim: int,
        rank: int,
        hidden_dim: int = 256,
        depth: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = MLP(branch_dim, rank, hidden_dim, depth, activation, dropout)

    def forward(self, branch: torch.Tensor) -> torch.Tensor:
        return self.net(branch)
