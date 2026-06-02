from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.idm.data.manifest import CachedFeatureDataset, load_current_image, load_future_image
from experiments.idm.model.idm import FeatureIDM


def _init_wandb(args: argparse.Namespace, metadata: dict) -> object | None:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        if args.wandb_mode == "required":
            raise
        print("wandb is not installed; continuing without W&B logging")
        return None

    mode = "offline" if args.wandb_mode == "offline" else None
    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "online-cross-embodiment-idm"
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=args.wandb_name,
            mode=mode,
            config={**vars(args), **metadata},
        )
    except Exception:
        if args.wandb_mode == "required":
            raise
        print("wandb initialization failed; continuing without W&B logging")
        return None
    return run


def _pearsonr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt((pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)).clamp_min(1e-12))
    return numerator / denominator


def _direction_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-1] < 3:
        return torch.tensor(float("nan"))
    return nn.functional.cosine_similarity(pred[:, :3], target[:, :3], dim=-1).mean()


def _make_transition_panel(current, future):
    from PIL import Image, ImageDraw

    current = current.resize((224, 224))
    future = future.resize((224, 224))
    panel = Image.new("RGB", (448, 248), "white")
    panel.paste(current, (0, 24))
    panel.paste(future, (224, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 6), "current", fill=(0, 0, 0))
    draw.text((232, 6), "future", fill=(0, 0, 0))
    return panel


def _format_action(values: np.ndarray, max_items: int = 8) -> str:
    clipped = values.reshape(-1)[:max_items]
    suffix = " ..." if values.size > max_items else ""
    return "[" + ", ".join(f"{x:.3f}" for x in clipped) + suffix + "]"


def _wandb_media(
    model: FeatureIDM,
    dataset: CachedFeatureDataset,
    indices: list[int],
    device: torch.device,
):
    import wandb

    columns = [
        "sample_id",
        "transition",
        "gt_action",
        "pred_action",
        "l2_deviation",
        "eef_l2_deviation",
        "cosine_xyz",
    ]
    table = wandb.Table(columns=columns)
    model.eval()
    with torch.no_grad():
        for idx in indices:
            row = dataset.manifest_row(idx)
            if row is None:
                continue
            z_current = dataset.z_current[idx : idx + 1].to(device)
            z_future = dataset.z_future[idx : idx + 1].to(device)
            proprio = dataset.proprio[idx : idx + 1].to(device)
            gt = dataset.actions[idx].cpu()
            pred = model(z_current, z_future, proprio).cpu()[0]
            diff = pred - gt
            gt_np = gt.numpy()
            pred_np = pred.numpy()
            diff_np = diff.numpy()
            xyz_cos = float("nan")
            if gt.numel() >= 3:
                xyz_cos = float(nn.functional.cosine_similarity(pred[:3], gt[:3], dim=0))
            panel = _make_transition_panel(load_current_image(row), load_future_image(row))
            table.add_data(
                dataset.sample_ids[idx],
                wandb.Image(panel),
                _format_action(gt_np),
                _format_action(pred_np),
                float(np.linalg.norm(diff_np)),
                float(np.linalg.norm(diff_np[:3])) if diff_np.size >= 3 else float("nan"),
                xyz_cos,
            )
    return table


def _run_eval(model: FeatureIDM, loader: DataLoader, device: torch.device) -> dict[str, float | list[float]]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            z_current = batch["z_current"].to(device)
            z_future = batch["z_future"].to(device)
            proprio = batch["proprio"].to(device)
            action = batch["action"].to(device)
            preds.append(model(z_current, z_future, proprio).cpu())
            targets.append(action.cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    smooth_l1 = nn.functional.smooth_l1_loss(pred, target).item()
    mse = nn.functional.mse_loss(pred, target).item()
    pearson = _pearsonr(pred, target)
    return {
        "smooth_l1": smooth_l1,
        "mse": mse,
        "eef_direction_cosine": float(_direction_cosine(pred, target)),
        "pearson_mean": float(torch.nanmean(pearson)),
        "pearson_per_dim": [float(x) for x in pearson],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "optional", "offline", "required"],
        default="disabled",
        help="W&B logging mode. optional uses W&B if installed/auth works; offline writes local W&B logs.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-media-every", type=int, default=5)
    parser.add_argument("--wandb-media-samples", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CachedFeatureDataset(args.cache, split="train")
    val_ds = CachedFeatureDataset(args.cache, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    feature_dim = int(train_ds.z_current.shape[-1])
    action_dim = int(train_ds.actions.shape[-1])
    proprio_dim = int(train_ds.proprio.shape[-1])
    metadata = {
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
    }
    wandb_run = _init_wandb(args, metadata)

    model = FeatureIDM(
        feature_dim=feature_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    if args.wandb_media_samples > 0 and len(val_ds) > 0:
        media_indices = torch.linspace(
            0,
            len(val_ds) - 1,
            steps=min(args.wandb_media_samples, len(val_ds)),
            dtype=torch.long,
        ).tolist()
    else:
        media_indices = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            z_current = batch["z_current"].to(device)
            z_future = batch["z_future"].to(device)
            proprio = batch["proprio"].to(device)
            action = batch["action"].to(device)
            pred = model(z_current, z_future, proprio)
            loss = loss_fn(pred, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(action.shape[0])
            seen += int(action.shape[0])

        metrics = _run_eval(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_smooth_l1"] = total_loss / max(seen, 1)
        metrics["train/loss"] = metrics["train_smooth_l1"]
        metrics["val/smooth_l1"] = metrics["smooth_l1"]
        metrics["val/mse"] = metrics["mse"]
        metrics["val/eef_direction_cosine"] = metrics["eef_direction_cosine"]
        metrics["val/pearson_mean"] = metrics["pearson_mean"]
        history.append(metrics)
        print(json.dumps(metrics))
        if wandb_run is not None:
            log_payload = dict(metrics)
            if args.wandb_log_media_every > 0 and epoch % args.wandb_log_media_every == 0:
                log_payload["val/action_prediction_examples"] = _wandb_media(model, val_ds, media_indices, device)
            wandb_run.log(log_payload, step=epoch)
        if metrics["smooth_l1"] < best:
            best = float(metrics["smooth_l1"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_dim": feature_dim,
                    "action_dim": action_dim,
                    "proprio_dim": proprio_dim,
                    "args": vars(args),
                    "metrics": metrics,
                },
                out_dir / "best.pt",
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
