from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CachedPatchWindowDataset:
    def __init__(self, cache_path: str | Path, split: str = "train", cache: dict[str, Any] | None = None) -> None:
        if cache is None:
            cache = torch.load(Path(cache_path).expanduser(), map_location="cpu", weights_only=False)
        if cache.get("cache_type") != "patch_window":
            raise ValueError(f"Expected patch_window cache, got {cache.get('cache_type')!r}")
        split_indices = cache["splits"][split]
        self.cache_path = Path(cache_path).expanduser()
        self.cache = cache
        self.split_indices = split_indices
        self.patch_tokens = cache["patch_tokens"]
        self.window_frame_indices = cache["window_frame_indices"][split_indices].long()
        self.proprio = cache["proprio"][split_indices].float()
        self.actions = cache["actions"][split_indices].float()
        self.horizon_k = cache["horizon_k"][split_indices].long()
        self.sample_ids = [cache["sample_ids"][int(i)] for i in split_indices]
        self.rows_raw = [cache["rows_raw"][int(i)] for i in split_indices]

    @property
    def horizon(self) -> int:
        return int(self.cache["horizon"])

    @property
    def action_width(self) -> int:
        return int(self.cache["action_width"])

    @property
    def patch_count(self) -> int:
        return int(self.patch_tokens.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.patch_tokens.shape[2])

    @property
    def proprio_dim(self) -> int:
        return int(self.proprio.shape[-1])

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        frame_indices = self.window_frame_indices[idx]
        return {
            "z_window": self.patch_tokens[frame_indices].float(),
            "proprio": self.proprio[idx],
            "action": self.actions[idx],
            "horizon_k": self.horizon_k[idx],
            "sample_id": self.sample_ids[idx],
        }
