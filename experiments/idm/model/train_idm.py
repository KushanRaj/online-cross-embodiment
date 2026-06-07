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
from experiments.idm.data.patch_windows import CachedPatchWindowDataset
from experiments.idm.model.idm import FeatureIDM, PatchTransformerIDM


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
            id=args.wandb_run_id,
            resume=args.wandb_resume,
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
            horizon = int(dataset.horizon_k[idx].item()) if dataset.horizon_k is not None else gt.numel()
            dim = min(horizon * 7, gt.numel(), pred.numel())
            gt = gt[:dim]
            pred = pred[:dim]
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


def _action_prefix(tensor: torch.Tensor, action_prefix: int | None, action_width: int) -> torch.Tensor:
    if action_prefix is None:
        return tensor
    dim = int(action_prefix) * int(action_width)
    if dim <= 0:
        return tensor
    return tensor[:, : min(dim, tensor.shape[-1])]


def _horizon_mask(horizon_k: torch.Tensor, action_dim: int, action_width: int, device: torch.device) -> torch.Tensor:
    if horizon_k.numel() == 0 or int(horizon_k.max().item()) <= 0:
        return torch.ones((horizon_k.shape[0], action_dim), dtype=torch.bool, device=device)
    steps = torch.arange(action_dim, device=device) // int(action_width)
    return steps.unsqueeze(0) < horizon_k.to(device).long().unsqueeze(1)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_k: torch.Tensor,
    action_width: int,
) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if target.ndim == 3:
        target = target.reshape(target.shape[0], -1)
    mask = _horizon_mask(horizon_k, pred.shape[-1], action_width, pred.device)
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return loss[mask].mean()


def _predict(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    proprio = batch["proprio"].to(device)
    if "z_window" in batch:
        return model(batch["z_window"].to(device), proprio)
    return model(batch["z_current"].to(device), batch["z_future"].to(device), proprio)


def _run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_prefix: int | None = None,
    action_width: int = 7,
) -> dict[str, float | list[float]]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    horizons: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            preds.append(_predict(model, batch, device).cpu())
            targets.append(batch["action"].cpu())
            horizons.append(batch["horizon_k"].cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if target.ndim == 3:
        target = target.reshape(target.shape[0], -1)
    pred = _action_prefix(pred, action_prefix, action_width)
    target = _action_prefix(target, action_prefix, action_width)
    horizon = torch.cat(horizons, dim=0)
    if action_prefix is not None:
        horizon = torch.minimum(horizon, torch.full_like(horizon, int(action_prefix)))
    mask = _horizon_mask(horizon, pred.shape[-1], action_width, pred.device)
    smooth_l1 = nn.functional.smooth_l1_loss(pred, target, reduction="none")[mask].mean().item()
    mse = nn.functional.mse_loss(pred, target, reduction="none")[mask].mean().item()
    pred = pred[mask].reshape(-1, action_width)
    target = target[mask].reshape(-1, action_width)
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
    parser.add_argument("--model-type", choices=["feature", "patch"], default="feature")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps in the current run")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--eval-action-prefix", type=int, default=None, help="Evaluate only the first N action steps")
    parser.add_argument("--action-width", type=int, default=7, help="Scalar action width per timestep")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint to load model weights from before training")
    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch offset for resumed logging/checkpoint metadata")
    parser.add_argument("--start-step", type=int, default=0, help="Optimizer-step offset for W&B x-axis")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "optional", "offline", "required"],
        default="disabled",
        help="W&B logging mode. optional uses W&B if installed/auth works; offline writes local W&B logs.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-resume", default=None, choices=[None, "allow", "must", "never", "auto"])
    parser.add_argument("--wandb-log-train-every", type=int, default=50)
    parser.add_argument("--wandb-log-media-every", type=int, default=5)
    parser.add_argument("--wandb-media-samples", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "patch":
        cache = torch.load(Path(args.cache).expanduser(), map_location="cpu", weights_only=False)
        train_ds = CachedPatchWindowDataset(args.cache, split="train", cache=cache)
        val_ds = CachedPatchWindowDataset(args.cache, split="val", cache=cache)
        feature_dim = train_ds.feature_dim
        action_dim = train_ds.horizon * train_ds.action_width
        proprio_dim = train_ds.proprio_dim
        horizon = train_ds.horizon
        patch_count = train_ds.patch_count
        model: nn.Module = PatchTransformerIDM(
            feature_dim=feature_dim,
            action_width=train_ds.action_width,
            horizon=horizon,
            proprio_dim=proprio_dim,
            width=args.hidden_dim,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
        ).to(device)
    else:
        train_ds = CachedFeatureDataset(args.cache, split="train")
        val_ds = CachedFeatureDataset(args.cache, split="val")
        feature_dim = int(train_ds.z_current.shape[-1])
        action_dim = int(train_ds.actions.shape[-1])
        proprio_dim = int(train_ds.proprio.shape[-1])
        horizon = action_dim // args.action_width
        patch_count = None
        model = FeatureIDM(
            feature_dim=feature_dim,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
        ).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    metadata = {
        "device": str(device),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "feature_dim": feature_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "architecture": type(model).__name__,
        "horizon": horizon,
        "patch_count": patch_count,
        "eval_action_prefix": args.eval_action_prefix,
        "action_width": args.action_width,
    }
    wandb_run = _init_wandb(args, metadata)
    resumed_from: str | None = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint).expanduser()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        resumed_from = str(resume_path)
        print({"resumed_from": resumed_from, "checkpoint_metrics": checkpoint.get("metrics", {})})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    if args.start_epoch > 0 and history_path.exists():
        history = json.loads(history_path.read_text())
    else:
        history = []
    best = float("inf")
    if history:
        best = min(float(row["smooth_l1"]) for row in history)
    if args.model_type == "feature" and args.wandb_media_samples > 0 and len(val_ds) > 0:
        media_indices = torch.linspace(
            0,
            len(val_ds) - 1,
            steps=min(args.wandb_media_samples, len(val_ds)),
            dtype=torch.long,
        ).tolist()
    else:
        media_indices = []
    global_step = int(args.start_step)
    stop_training = False
    for local_epoch in range(1, args.epochs + 1):
        epoch = args.start_epoch + local_epoch
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            action = batch["action"].to(device)
            horizon_k = batch["horizon_k"].to(device)
            pred = _predict(model, batch, device)
            loss = _masked_smooth_l1(pred, action, horizon_k, args.action_width)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += float(loss.item()) * int(action.shape[0])
            seen += int(action.shape[0])
            if wandb_run is not None and args.wandb_log_train_every > 0 and global_step % args.wandb_log_train_every == 0:
                wandb_run.log(
                    {
                        "train/batch_loss": float(loss.item()),
                        "train/l1_loss": float(loss.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            if args.max_steps is not None and (global_step - int(args.start_step)) >= args.max_steps:
                stop_training = True
                break

        metrics = _run_eval(
            model,
            val_loader,
            device,
            action_prefix=args.eval_action_prefix,
            action_width=args.action_width,
        )
        metrics["epoch"] = epoch
        metrics["local_epoch"] = local_epoch
        metrics["global_step"] = global_step
        metrics["train_smooth_l1"] = total_loss / max(seen, 1)
        metrics["resumed_from"] = resumed_from
        metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
        metrics["train/loss"] = metrics["train_smooth_l1"]
        metrics["train/l1_loss"] = metrics["train_smooth_l1"]
        metrics["val/smooth_l1"] = metrics["smooth_l1"]
        metrics["val/l1_loss"] = metrics["smooth_l1"]
        metrics["val/mse"] = metrics["mse"]
        metrics["val/eef_direction_cosine"] = metrics["eef_direction_cosine"]
        metrics["val/pearson_mean"] = metrics["pearson_mean"]
        metrics["val/correlation"] = metrics["pearson_mean"]
        history.append(metrics)
        print(json.dumps(metrics))
        if wandb_run is not None:
            log_payload = dict(metrics)
            if args.wandb_log_media_every > 0 and epoch % args.wandb_log_media_every == 0:
                log_payload["val/action_prediction_examples"] = _wandb_media(model, val_ds, media_indices, device)
            wandb_run.log(log_payload, step=global_step)
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
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "proprio_dim": proprio_dim,
                "args": vars(args),
                "metrics": metrics,
            },
            out_dir / "last.pt",
        )
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        if stop_training:
            break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
