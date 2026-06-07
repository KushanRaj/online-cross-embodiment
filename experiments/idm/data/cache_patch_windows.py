from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments.idm.data.manifest import ManifestRow, _apply_transform, read_manifest


FrameKey = tuple[str, str, int, str | None]


def _frame_key(row: ManifestRow, offset: int) -> FrameKey:
    raw = row.raw
    if "source_file" not in raw:
        raise ValueError("Patch-window cache currently expects HDF5-backed LIBERO rows")
    start = int(raw["window_start_t"])
    path = raw.get("image_t_hdf5_path") or raw.get("image_future_hdf5_path")
    transform = raw.get("image_t_transform") or raw.get("image_future_transform")
    return (str(Path(raw["source_file"]).expanduser()), str(path), start + offset, transform)


def _load_hdf5_image(key: FrameKey) -> Image.Image:
    import h5py

    source_file, hdf5_path, frame_index, transform = key
    with h5py.File(source_file, "r") as h5:
        frame = h5[hdf5_path][int(frame_index)]
    frame = _apply_transform(np.asarray(frame), transform)
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")


def _encode_patch_batch(model, processor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    batch = processor(images=images, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        if hasattr(model, "vision_model"):
            output = model.vision_model(**batch)
        else:
            output = model(**batch)
    if not hasattr(output, "last_hidden_state") or output.last_hidden_state is None:
        raise RuntimeError("Patch-token cache requires encoder last_hidden_state")
    return output.last_hidden_state.float().cpu()


def _target_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _stack_actions(rows: list[ManifestRow], action_width: int) -> torch.Tensor:
    max_horizon = max(row.horizon_k for row in rows)
    out = torch.zeros((len(rows), max_horizon, action_width), dtype=torch.float32)
    for i, row in enumerate(rows):
        action = torch.from_numpy(row.action.astype(np.float32)).reshape(row.horizon_k, -1)
        if action.shape[-1] != action_width:
            raise ValueError(f"Expected action width {action_width}, got {action.shape[-1]}")
        out[i, : row.horizon_k] = action
    return out


def _stack_proprio(rows: list[ManifestRow]) -> torch.Tensor:
    values = [torch.from_numpy(row.proprio.reshape(-1)) for row in rows]
    max_dim = max((int(value.numel()) for value in values), default=0)
    padded = torch.zeros((len(values), max_dim), dtype=torch.float32)
    for i, value in enumerate(values):
        padded[i, : value.numel()] = value
    return padded


def _make_splits(n: int, val_fraction: float, seed: int) -> dict[str, torch.Tensor]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = max(1, int(round(n * val_fraction))) if n > 1 else 0
    val = sorted(indices[:val_n])
    train = sorted(indices[val_n:])
    return {
        "train": torch.tensor(train, dtype=torch.long),
        "val": torch.tensor(val, dtype=torch.long),
    }


def _filter_rows(args: argparse.Namespace) -> list[ManifestRow]:
    rows = read_manifest(args.manifest)
    if args.horizon is not None:
        rows = [row for row in rows if row.horizon_k == args.horizon]
    if args.source_type is not None:
        rows = [row for row in rows if row.source_type == args.source_type]
    if args.shuffle_before_limit:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No manifest rows found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-before-limit", action="store_true")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--source-type", default="actual_actual")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--action-width", type=int, default=7)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    rows = _filter_rows(args)
    horizon = int(args.horizon)
    if any(row.horizon_k != horizon for row in rows):
        raise SystemExit("Patch-window cache currently expects a single fixed horizon")

    frame_to_index: dict[FrameKey, int] = {}
    frame_keys: list[FrameKey] = []
    window_frame_indices = torch.empty((len(rows), horizon + 1), dtype=torch.long)
    for row_i, row in enumerate(rows):
        for offset in range(horizon + 1):
            key = _frame_key(row, offset)
            if key not in frame_to_index:
                frame_to_index[key] = len(frame_keys)
                frame_keys.append(key)
            window_frame_indices[row_i, offset] = frame_to_index[key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    patch_tokens: torch.Tensor | None = None
    target_dtype = _target_dtype(args.dtype)
    for start in tqdm(range(0, len(frame_keys), args.batch_size), desc="encoding unique frames"):
        chunk = frame_keys[start : start + args.batch_size]
        images = [_load_hdf5_image(key) for key in chunk]
        encoded = _encode_patch_batch(model, processor, images, device).to(dtype=target_dtype)
        if patch_tokens is None:
            patch_tokens = torch.empty(
                (len(frame_keys), encoded.shape[1], encoded.shape[2]),
                dtype=target_dtype,
            )
        patch_tokens[start : start + encoded.shape[0]] = encoded
    if patch_tokens is None:
        raise RuntimeError("No frames were encoded")

    cache: dict[str, Any] = {
        "cache_type": "patch_window",
        "manifest": str(Path(args.manifest).expanduser()),
        "dataset_root": str(rows[0].dataset_root),
        "encoder": args.encoder,
        "horizon": horizon,
        "action_width": int(args.action_width),
        "sample_ids": [row.sample_id for row in rows],
        "rows_raw": [row.raw for row in rows],
        "source_types": [row.source_type for row in rows],
        "horizon_k": torch.tensor([row.horizon_k for row in rows], dtype=torch.long),
        "frame_keys": frame_keys,
        "patch_tokens": patch_tokens,
        "window_frame_indices": window_frame_indices,
        "proprio": _stack_proprio(rows),
        "actions": _stack_actions(rows, args.action_width),
        "splits": _make_splits(len(rows), args.val_fraction, args.seed),
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(
        {
            "out": str(out_path),
            "rows": len(rows),
            "unique_frames": len(frame_keys),
            "horizon": horizon,
            "patch_tokens_shape": tuple(cache["patch_tokens"].shape),
            "actions_shape": tuple(cache["actions"].shape),
            "proprio_dim": int(cache["proprio"].shape[-1]),
        }
    )


if __name__ == "__main__":
    main()
