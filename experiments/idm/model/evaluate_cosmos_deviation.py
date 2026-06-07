from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments.idm.data.manifest import ManifestRow, _load_hdf5_frame, _load_path_frame, read_manifest
from experiments.idm.model.idm import FeatureIDM


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
    return torch.nn.functional.normalize(features.float(), dim=-1)


def _load_real_future(row: ManifestRow) -> Image.Image:
    return _load_hdf5_frame(row, "real_future_hdf5_path", "real_future_index", "real_future_transform")


def _load_model_future(row: ManifestRow) -> Image.Image:
    return _load_path_frame(row, "model_future_path")


def _load_current(row: ManifestRow) -> Image.Image:
    return _load_hdf5_frame(row, "image_t_hdf5_path", "image_t_index", "image_t_transform")


def _demo_action_chunk(row: ManifestRow) -> np.ndarray:
    demo_key = str(row.raw["image_t_hdf5_path"]).split("/")[1]
    t = int(row.raw["window_start_t"])
    k = int(row.raw["horizon_k"])
    with h5py.File(Path(row.raw["source_file"]).expanduser(), "r") as h5:
        return np.asarray(h5[f"data/{demo_key}/actions"][t : t + k], dtype=np.float32).reshape(-1)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.reshape(-1)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _predict_action(
    idm: FeatureIDM,
    z_current: torch.Tensor,
    z_future: torch.Tensor,
    proprio: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        return idm(z_current, z_future, proprio).cpu().numpy()[0].astype(np.float32)


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(rows))}
    for key in keys:
        values = np.asarray([r[key] for r in rows if np.isfinite(r[key])], dtype=np.float64)
        if values.size == 0:
            continue
        out[f"{key}/mean"] = float(values.mean())
        out[f"{key}/median"] = float(np.median(values))
        out[f"{key}/p90"] = float(np.percentile(values, 90))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", default="google/siglip-base-patch16-224")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--wandb-mode", choices=["disabled", "optional", "required"], default="disabled")
    parser.add_argument("--wandb-project", default="online-cross-embodiment-idm")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="cosmos-idm-deviation")
    args = parser.parse_args()

    rows = read_manifest(args.manifest, limit=args.limit)
    rows = [row for row in rows if row.source_type == "real_model" and row.horizon_k == 16]
    if not rows:
        raise SystemExit("No k=16 real_model rows found")

    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idm = FeatureIDM(
        feature_dim=int(checkpoint["feature_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        proprio_dim=int(checkpoint["proprio_dim"]),
        hidden_dim=int(checkpoint["args"]["hidden_dim"]),
        depth=int(checkpoint["args"]["depth"]),
    ).to(device)
    idm.load_state_dict(checkpoint["model_state"])
    idm.eval()

    processor = AutoProcessor.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config={
                    "manifest": args.manifest,
                    "checkpoint": args.checkpoint,
                    "encoder": args.encoder,
                    "limit": args.limit,
                    "rows": len(rows),
                    "checkpoint_metrics": checkpoint.get("metrics", {}),
                },
            )
        except Exception:
            if args.wandb_mode == "required":
                raise
            wandb_run = None

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc="cosmos-idm"):
        chunk = rows[start : start + args.batch_size]
        current_images = [_load_current(row) for row in chunk]
        real_future_images = [_load_real_future(row) for row in chunk]
        model_future_images = [_load_model_future(row) for row in chunk]

        z_c = _encode_batch(encoder, processor, current_images, device)
        z_r = _encode_batch(encoder, processor, real_future_images, device)
        z_p = _encode_batch(encoder, processor, model_future_images, device)
        proprio = torch.tensor([row.proprio.reshape(-1) for row in chunk], dtype=torch.float32, device=device)

        pred_cr = idm(z_c, z_r, proprio).detach().cpu().numpy().astype(np.float32)
        pred_cp = idm(z_c, z_p, proprio).detach().cpu().numpy().astype(np.float32)
        pred_pr = idm(z_p, z_r, proprio).detach().cpu().numpy().astype(np.float32)

        for i, row in enumerate(chunk):
            cosmos_action = row.action.reshape(-1).astype(np.float32)
            demo_action = _demo_action_chunk(row)
            implied_real = pred_cr[i]
            implied_model = pred_cp[i]
            residual = pred_pr[i]
            feature_diff = (z_r[i] - z_p[i]).detach().cpu().numpy().astype(np.float32)
            metrics_rows.append(
                {
                    "sample_id": row.sample_id,
                    "task_name": row.raw.get("task_name", ""),
                    "trajectory_id": row.raw.get("trajectory_id", ""),
                    "window_start_t": int(row.raw["window_start_t"]),
                    "horizon_k": int(row.raw["horizon_k"]),
                    "model_value": float(row.raw.get("model_value", 0.0)),
                    "real_recovery_l2_vs_demo": _norm(implied_real - demo_action),
                    "real_recovery_cos_vs_demo": _cosine(implied_real, demo_action),
                    "model_consistency_l2_vs_cosmos": _norm(implied_model - cosmos_action),
                    "model_consistency_cos_vs_cosmos": _cosine(implied_model, cosmos_action),
                    "real_model_implied_l2": _norm(implied_real - implied_model),
                    "real_model_implied_cos": _cosine(implied_real, implied_model),
                    "residual_pr_l2": _norm(residual),
                    "residual_pr_cos_vs_demo_minus_cosmos": _cosine(residual, demo_action - cosmos_action),
                    "siglip_real_model_feature_l2": _norm(feature_diff),
                    "siglip_real_model_feature_cos": _cosine(z_r[i].detach().cpu().numpy(), z_p[i].detach().cpu().numpy()),
                    "demo_action_l2": _norm(demo_action),
                    "cosmos_action_l2": _norm(cosmos_action),
                    "demo_cosmos_action_l2": _norm(demo_action - cosmos_action),
                }
            )

    metric_keys = [
        "real_recovery_l2_vs_demo",
        "model_consistency_l2_vs_cosmos",
        "real_model_implied_l2",
        "residual_pr_l2",
        "siglip_real_model_feature_l2",
        "siglip_real_model_feature_cos",
        "demo_cosmos_action_l2",
    ]
    summary = _summarize(metrics_rows, metric_keys)
    summary.update(
        {
            "manifest": str(Path(args.manifest).expanduser()),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "encoder": args.encoder,
            "preprocessing_note": "Cosmos model futures are saved in Cosmos LIBERO flipped convention; HDF5 current/real future frames use manifest flipud before SigLIP.",
        }
    )

    csv_path = out_dir / "cosmos_idm_deviation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if wandb_run is not None:
        import wandb

        wandb.log(summary)
        columns = list(metrics_rows[0].keys())
        wandb.log(
            {
                "cosmos_idm_deviation": wandb.Table(
                    columns=columns,
                    data=[[r[k] for k in columns] for r in metrics_rows[:1000]],
                )
            }
        )
        wandb_run.finish()

    print(json.dumps(summary, indent=2))
    print({"csv": str(csv_path), "summary": str(out_dir / "summary.json")})


if __name__ == "__main__":
    main()
