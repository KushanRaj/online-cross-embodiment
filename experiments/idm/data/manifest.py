from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ManifestRow:
    index: int
    dataset_root: Path
    raw: dict[str, Any]

    @property
    def sample_id(self) -> str:
        return str(self.raw["sample_id"])

    @property
    def source_type(self) -> str:
        return str(self.raw.get("source_type", "unknown"))

    @property
    def horizon_k(self) -> int:
        return int(self.raw["horizon_k"])

    @property
    def action(self) -> np.ndarray:
        return np.asarray(self.raw["action_chunk"], dtype=np.float32)

    @property
    def proprio(self) -> np.ndarray:
        value = self.raw.get("proprio_t")
        if value is None:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)


def read_manifest(path: str | Path, limit: int | None = None) -> list[ManifestRow]:
    manifest_path = Path(path).expanduser()
    dataset_root = manifest_path.parent
    rows: list[ManifestRow] = []
    with manifest_path.open() as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(ManifestRow(index=index, dataset_root=dataset_root, raw=json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _apply_transform(array: np.ndarray, transform: str | None) -> np.ndarray:
    if transform in (None, "", "none"):
        return np.ascontiguousarray(array)
    if transform == "flipud":
        return np.ascontiguousarray(np.flipud(array))
    if transform == "rot180":
        return np.ascontiguousarray(array[::-1, ::-1])
    raise ValueError(f"Unsupported image transform: {transform}")


def _load_hdf5_frame(row: ManifestRow, path_key: str, index_key: str, transform_key: str) -> Image.Image:
    import h5py

    h5_path = Path(row.raw["source_file"]).expanduser()
    with h5py.File(h5_path, "r") as h5:
        frame = h5[row.raw[path_key]][int(row.raw[index_key])]
    frame = _apply_transform(np.asarray(frame), row.raw.get(transform_key))
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def _load_path_frame(row: ManifestRow, key: str) -> Image.Image:
    path = Path(row.raw[key])
    if not path.is_absolute():
        path = row.dataset_root / path
    return Image.open(path).convert("RGB")


def load_current_image(row: ManifestRow) -> Image.Image:
    if "image_t_path" in row.raw:
        return _load_path_frame(row, "image_t_path")
    return _load_hdf5_frame(row, "image_t_hdf5_path", "image_t_index", "image_t_transform")


def load_future_image(row: ManifestRow) -> Image.Image:
    if "image_future_path" in row.raw:
        return _load_path_frame(row, "image_future_path")
    if "model_future_path" in row.raw:
        return _load_path_frame(row, "model_future_path")
    if "real_future_hdf5_path" in row.raw:
        return _load_hdf5_frame(row, "real_future_hdf5_path", "real_future_index", "real_future_transform")
    return _load_hdf5_frame(row, "image_future_hdf5_path", "image_future_index", "image_future_transform")


class CachedFeatureDataset:
    def __init__(self, cache_path: str | Path, split: str = "train") -> None:
        import torch

        cache = torch.load(Path(cache_path).expanduser(), map_location="cpu", weights_only=False)
        split_indices = cache["splits"][split]
        self.cache_path = Path(cache_path).expanduser()
        self.dataset_root = Path(cache.get("dataset_root") or Path(cache["manifest"]).expanduser().parent)
        self.split_indices = split_indices
        self.z_current = cache["z_current"][split_indices].float()
        self.z_future = cache["z_future"][split_indices].float()
        self.proprio = cache["proprio"][split_indices].float()
        self.actions = cache["actions"][split_indices].float()
        self.horizon_k = cache.get("horizon_k")
        if self.horizon_k is not None:
            self.horizon_k = self.horizon_k[split_indices].long()
        self.sample_ids = [cache["sample_ids"][int(i)] for i in split_indices]
        rows_raw = cache.get("rows_raw")
        self.rows_raw = [rows_raw[int(i)] for i in split_indices] if rows_raw is not None else None

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "z_current": self.z_current[idx],
            "z_future": self.z_future[idx],
            "proprio": self.proprio[idx],
            "action": self.actions[idx],
            "horizon_k": self.horizon_k[idx] if self.horizon_k is not None else 0,
            "sample_id": self.sample_ids[idx],
        }

    def manifest_row(self, idx: int) -> ManifestRow | None:
        if self.rows_raw is None:
            return None
        return ManifestRow(index=idx, dataset_root=self.dataset_root, raw=self.rows_raw[idx])
