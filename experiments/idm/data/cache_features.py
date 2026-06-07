from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments.idm.data.manifest import load_current_image, load_future_image, read_manifest


def _encode_batch(model, processor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    batch = processor(images=images, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            output = model.get_image_features(**batch)
        else:
            output = model(**batch)
    if isinstance(output, torch.Tensor):
        features = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        features = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        features = output.last_hidden_state.mean(dim=1)
    else:
        raise RuntimeError("Encoder output has neither pooler_output nor last_hidden_state")
    return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


def _stack_actions(rows) -> torch.Tensor:
    actions = [torch.from_numpy(row.action.reshape(-1)) for row in rows]
    max_dim = max(int(action.numel()) for action in actions)
    padded = torch.zeros((len(actions), max_dim), dtype=torch.float32)
    for i, action in enumerate(actions):
        padded[i, : action.numel()] = action
    return padded


def _stack_proprio(rows) -> torch.Tensor:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-before-limit", action="store_true")
    parser.add_argument("--horizon", type=int, default=None, help="Keep only rows with this horizon_k")
    parser.add_argument("--horizons", default=None, help="Comma-separated horizon_k values to keep")
    parser.add_argument("--source-type", default=None, help="Keep only rows with this source_type")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=195)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.horizon is not None:
        rows = [row for row in rows if row.horizon_k == args.horizon]
    if args.horizons is not None:
        keep = {int(x.strip()) for x in args.horizons.split(",") if x.strip()}
        rows = [row for row in rows if row.horizon_k in keep]
    if args.source_type is not None:
        rows = [row for row in rows if row.source_type == args.source_type]
    if args.shuffle_before_limit:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No manifest rows found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    z_current: list[torch.Tensor] = []
    z_future: list[torch.Tensor] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="encoding"):
        chunk = rows[start : start + args.batch_size]
        current_images = [load_current_image(row) for row in chunk]
        future_images = [load_future_image(row) for row in chunk]
        z_current.append(_encode_batch(model, processor, current_images, device))
        z_future.append(_encode_batch(model, processor, future_images, device))

    cache = {
        "manifest": str(Path(args.manifest).expanduser()),
        "encoder": args.encoder,
        "horizon_filter": args.horizon,
        "source_type_filter": args.source_type,
        "sample_ids": [row.sample_id for row in rows],
        "rows_raw": [row.raw for row in rows],
        "dataset_root": str(rows[0].dataset_root) if rows else str(Path(args.manifest).expanduser().parent),
        "source_types": [row.source_type for row in rows],
        "horizon_k": torch.tensor([row.horizon_k for row in rows], dtype=torch.long),
        "z_current": torch.cat(z_current, dim=0),
        "z_future": torch.cat(z_future, dim=0),
        "proprio": _stack_proprio(rows),
        "actions": _stack_actions(rows),
        "splits": _make_splits(len(rows), args.val_fraction, args.seed),
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(
        {
            "out": str(out_path),
            "rows": len(rows),
            "feature_dim": int(cache["z_current"].shape[-1]),
            "action_dim": int(cache["actions"].shape[-1]),
            "proprio_dim": int(cache["proprio"].shape[-1]),
        }
    )


if __name__ == "__main__":
    main()
