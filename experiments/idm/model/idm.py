from __future__ import annotations

import torch
from torch import nn


class FeatureIDM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        proprio_dim: int = 0,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = feature_dim * 3 + proprio_dim
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, action_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, z_current: torch.Tensor, z_future: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        delta = z_future - z_current
        x = torch.cat([z_current, z_future, delta, proprio], dim=-1)
        return self.net(x)
