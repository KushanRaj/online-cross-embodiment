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


class PatchTransformerIDM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_width: int,
        horizon: int,
        proprio_dim: int = 0,
        width: int = 512,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 129,
        max_patches: int = 1024,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_width = int(action_width)
        self.token_proj = nn.Linear(feature_dim, width)
        self.temporal_pos = nn.Parameter(torch.zeros(max_frames, width))
        self.spatial_pos = nn.Parameter(torch.zeros(max_patches, width))
        self.readout_tokens = nn.Parameter(torch.zeros(self.horizon, width))
        self.proprio_proj = nn.Linear(proprio_dim, width) if proprio_dim > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=int(width * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, self.action_width),
        )
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.readout_tokens, std=0.02)

    def forward(self, z_window: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        batch, frames, patches, _ = z_window.shape
        if frames > self.temporal_pos.shape[0]:
            raise ValueError(f"Window has {frames} frames, max_frames={self.temporal_pos.shape[0]}")
        if patches > self.spatial_pos.shape[0]:
            raise ValueError(f"Window has {patches} patches, max_patches={self.spatial_pos.shape[0]}")
        x = self.token_proj(z_window)
        x = x + self.temporal_pos[:frames].view(1, frames, 1, -1)
        x = x + self.spatial_pos[:patches].view(1, 1, patches, -1)
        x = x.reshape(batch, frames * patches, -1)

        readout = self.readout_tokens.unsqueeze(0).expand(batch, -1, -1)
        if self.proprio_proj is not None:
            if proprio is None:
                raise ValueError("proprio is required when proprio_dim > 0")
            readout = readout + self.proprio_proj(proprio).unsqueeze(1)
        x = torch.cat([readout, x], dim=1)
        x = self.blocks(x)
        readout_out = self.norm(x[:, : self.horizon])
        return self.head(readout_out)
