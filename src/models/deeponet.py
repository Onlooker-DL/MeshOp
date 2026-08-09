from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import MLP


class DeepONet(nn.Module):
    """Unstacked DeepONet with a scalar bias and batched trunk queries."""

    def __init__(
        self,
        branch_dim: int,
        coordinate_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch = MLP(
            branch_dim, latent_dim, hidden_dim, branch_depth, activation, dropout
        )
        self.trunk = MLP(
            coordinate_dim, latent_dim, hidden_dim, trunk_depth, activation, dropout
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / math.sqrt(latent_dim)

    def forward(self, branch: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        branch_features = self.branch(branch)
        trunk_features = self.trunk(coordinates)
        return self.scale * torch.einsum("bp,qp->bq", branch_features, trunk_features) + self.bias


class MultiBranchDeepONet(nn.Module):
    """DeepONet with one branch network per input function.

    The Burgers data loader concatenates the two functional inputs (u0 and
    f) into one sensor vector. This model splits it at ``branch_split`` and
    feeds each half to its own branch network; the two latent features are
    concatenated and contracted with the trunk network. The parameter count
    stays close to the single-branch DeepONet because each branch sees only
    half the sensor dimension.
    """

    def __init__(
        self,
        branch_dim: int,
        coordinate_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 28,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        branch_split: int = 0,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if branch_split > 0:
            if branch_split >= branch_dim:
                raise ValueError(
                    "branch_split must be smaller than branch_dim."
                )
            first_dim = int(branch_split)
        else:
            first_dim = branch_dim // 2
        second_dim = branch_dim - first_dim
        self.branch_split = first_dim

        self.branch1 = MLP(
            first_dim,
            latent_dim,
            hidden_dim,
            branch_depth,
            activation,
            dropout,
        )
        self.branch2 = MLP(
            second_dim,
            latent_dim,
            hidden_dim,
            branch_depth,
            activation,
            dropout,
        )
        self.trunk = MLP(
            coordinate_dim,
            2 * latent_dim,
            hidden_dim,
            trunk_depth,
            activation,
            dropout,
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / math.sqrt(2 * latent_dim)

    def forward(
        self,
        branch: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        b1 = branch[:, : self.branch_split]
        b2 = branch[:, self.branch_split :]
        f1 = self.branch1(b1)
        f2 = self.branch2(b2)
        features = torch.cat((f1, f2), dim=1)
        trunk_features = self.trunk(coordinates)
        return (
            self.scale
            * torch.einsum("bp,qp->bq", features, trunk_features)
            + self.bias
        )
