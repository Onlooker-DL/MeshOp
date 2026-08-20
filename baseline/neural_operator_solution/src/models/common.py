from __future__ import annotations

import torch
import torch.nn as nn


def activation(name: str) -> nn.Module:
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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        depth: int,
        activation_name: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be positive.")
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    activation(activation_name),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
