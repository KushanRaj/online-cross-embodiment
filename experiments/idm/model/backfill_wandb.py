from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from experiments.idm.data.manifest import CachedFeatureDataset
from experiments.idm.model.idm import FeatureIDM
from experiments.idm.model.train_idm import _wandb_media


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-media-samples", type=int, default=16)
    args = parser.parse_args()

    import wandb

    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False)
    history = json.loads(Path(args.history).expanduser().read_text())
    val_ds = CachedFeatureDataset(args.cache, split="val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeatureIDM(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        proprio_dim=int(checkpoint["proprio_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        depth=int(checkpoint["args"]["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "online-cross-embodiment-idm"
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    run = wandb.init(
        project=project,
        entity=entity,
        name=args.wandb_name,
        config={
            "backfill": True,
            "cache": str(Path(args.cache).expanduser()),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "history": str(Path(args.history).expanduser()),
            "checkpoint_metrics": checkpoint.get("metrics", {}),
            "train_args": checkpoint.get("args", {}),
            "val_rows": len(val_ds),
        },
    )

    for row in history:
        epoch = int(row["epoch"])
        wandb.log(dict(row), step=epoch)

    if args.wandb_media_samples > 0 and len(val_ds) > 0:
        media_indices = torch.linspace(
            0,
            len(val_ds) - 1,
            steps=min(args.wandb_media_samples, len(val_ds)),
            dtype=torch.long,
        ).tolist()
        wandb.log({"val/action_prediction_examples_best": _wandb_media(model, val_ds, media_indices, device)}, step=len(history))

    run.finish()
    print({"wandb_url": run.url, "epochs_logged": len(history), "media_samples": min(args.wandb_media_samples, len(val_ds))})


if __name__ == "__main__":
    main()
