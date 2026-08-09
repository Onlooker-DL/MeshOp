from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAttention(nn.Module):
    """Slice-token physics attention used by the Transolver backbone."""

    def __init__(self, width: int, heads: int, slices: int, dropout: float) -> None:
        super().__init__()
        self.slice_logits = nn.Linear(width, slices)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.slice_logits(points), dim=1)
        normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        tokens = torch.einsum("bqs,bqw->bsw", weights, points)
        tokens = tokens / normalizer.transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return points + torch.einsum("bqs,bsw->bqw", weights, tokens)


class Transolver(nn.Module):
    """Pointwise score operator with global input conditioning and physics attention."""

    def __init__(
        self,
        branch_dim: int,
        coordinate_dim: int,
        width: int = 128,
        layers: int = 4,
        heads: int = 8,
        slices: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.branch_encoder = nn.Sequential(
            nn.Linear(branch_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.point_encoder = nn.Linear(coordinate_dim + width, width)
        self.blocks = nn.ModuleList(
            PhysicsAttention(width, heads, slices, dropout) for _ in range(layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, branch: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        condition = self.branch_encoder(branch)
        b, q = branch.shape[0], coordinates.shape[0]
        coords = coordinates.unsqueeze(0).expand(b, q, -1)
        cond = condition.unsqueeze(1).expand(b, q, -1)
        points = self.point_encoder(torch.cat((coords, cond), dim=-1))
        for block in self.blocks:
            points = block(points)
        return self.output(points).squeeze(-1)

