from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from experiments.idm.model.idm import FeatureIDM, PatchTransformerIDM


def _read_image_stack(h5: h5py.File, name: str) -> np.ndarray | None:
    if name in h5:
        return np.asarray(h5[name][()])
    jpeg_name = f"{name}_jpeg"
    if jpeg_name not in h5:
        return None
    frames = []
    for encoded in h5[jpeg_name]:
        frames.append(np.asarray(Image.open(io.BytesIO(bytes(encoded))).convert("RGB"), dtype=np.uint8))
    if not frames:
        return None
    return np.stack(frames, axis=0)


def _read_array(h5: h5py.File, name: str) -> np.ndarray | None:
    if name not in h5:
        return None
    return np.asarray(h5[name][()])


def _to_images(frames: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames]


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


def _encode_patch_batch(model, processor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    batch = processor(images=images, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        vision_model = getattr(model, "vision_model", model)
        output = vision_model(**batch)
    if not hasattr(output, "last_hidden_state"):
        raise RuntimeError("Patch IDM requires encoder last_hidden_state patch tokens")
    return output.last_hidden_state.float()


def _patch_window(z_current: torch.Tensor, z_future: torch.Tensor, horizon: int) -> torch.Tensor:
    current = z_current.unsqueeze(1).expand(-1, horizon, -1, -1)
    future = z_future.unsqueeze(1)
    return torch.cat([current, future], dim=1)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float32).reshape(-1)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _mean_step_cosine(a: np.ndarray, b: np.ndarray, dims: slice | None = None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 7)
    if b.ndim == 1:
        b = b.reshape(-1, 7)
    if dims is not None:
        a = a[:, dims]
        b = b[:, dims]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 1e-12
    if not np.any(valid):
        return float("nan")
    cos = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return float(np.mean(cos))


def _mean_step_l2(a: np.ndarray, b: np.ndarray, dims: slice | None = None) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 7)
    if b.ndim == 1:
        b = b.reshape(-1, 7)
    if dims is not None:
        a = a[:, dims]
        b = b[:, dims]
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _gripper_mismatch_rate(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 7)[:, -1]
    b = np.asarray(b, dtype=np.float32).reshape(-1, 7)[:, -1]
    return float(np.mean(np.sign(a) != np.sign(b)))


def _gripper_flip_count(a: np.ndarray) -> int:
    g = np.asarray(a, dtype=np.float32).reshape(-1, 7)[:, -1]
    signs = np.sign(g)
    return int(np.sum(signs[1:] != signs[:-1]))


def _load_idm(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model_type = str(args.get("model_type", "feature"))
    if model_type == "patch":
        horizon = int(args.get("eval_action_prefix") or checkpoint.get("metadata", {}).get("horizon") or int(checkpoint["action_dim"]) // 7)
        action_width = int(args.get("action_width", 7))
        model = PatchTransformerIDM(
            feature_dim=int(checkpoint["feature_dim"]),
            action_width=action_width,
            horizon=horizon,
            proprio_dim=int(checkpoint["proprio_dim"]),
            width=int(args.get("hidden_dim", 512)),
            depth=int(args.get("depth", 4)),
            heads=int(args.get("heads", 8)),
            mlp_ratio=float(args.get("mlp_ratio", 4.0)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(device)
    else:
        model = FeatureIDM(
            feature_dim=int(checkpoint["feature_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            proprio_dim=int(checkpoint["proprio_dim"]),
            hidden_dim=int(args.get("hidden_dim", 512)),
            depth=int(args.get("depth", 4)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, model_type


def _episode_rows(
    h5_path: Path,
    idm: torch.nn.Module,
    idm_type: str,
    encoder,
    processor,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        primary = _read_image_stack(h5, "primary_images")
        query_primary = _read_image_stack(h5, "query_primary_images")
        query_future = _read_image_stack(h5, "query_future_primary_images")
        query_t = _read_array(h5, "query_t")
        query_proprio = _read_array(h5, "query_proprio")
        query_actions = _read_array(h5, "query_action_chunks")
        query_values = _read_array(h5, "query_values")
        executed_actions = _read_array(h5, "actions")
        if primary is None or query_primary is None or query_future is None:
            return rows
        if query_t is None or query_proprio is None or query_actions is None:
            return rows

        success = bool(h5.attrs.get("success", False))
        task_description = str(h5.attrs.get("task_description", ""))
        horizon = int(query_actions.shape[1])
        valid_indices = [i for i, t in enumerate(query_t.astype(int)) if t + horizon < len(primary)]

        for start in tqdm(range(0, len(valid_indices), batch_size), desc=h5_path.name):
            ids = valid_indices[start : start + batch_size]
            current_images = _to_images(query_primary[ids])
            model_future_images = _to_images(query_future[ids])
            observed_future_images = _to_images(np.stack([primary[int(query_t[i]) + horizon] for i in ids], axis=0))

            proprio = torch.tensor(query_proprio[ids].reshape(len(ids), -1), dtype=torch.float32, device=device)

            with torch.no_grad():
                if idm_type == "patch":
                    z_current_patch = _encode_patch_batch(encoder, processor, current_images, device)
                    z_model_patch = _encode_patch_batch(encoder, processor, model_future_images, device)
                    z_observed_patch = _encode_patch_batch(encoder, processor, observed_future_images, device)
                    horizon_for_window = int(query_actions.shape[1])
                    implied_model = idm(_patch_window(z_current_patch, z_model_patch, horizon_for_window), proprio)
                    implied_observed = idm(_patch_window(z_current_patch, z_observed_patch, horizon_for_window), proprio)
                    z_model = torch.nn.functional.normalize(z_model_patch.mean(dim=1), dim=-1)
                    z_observed = torch.nn.functional.normalize(z_observed_patch.mean(dim=1), dim=-1)
                else:
                    z_current = _encode_batch(encoder, processor, current_images, device)
                    z_model = _encode_batch(encoder, processor, model_future_images, device)
                    z_observed = _encode_batch(encoder, processor, observed_future_images, device)
                    implied_model = idm(z_current, z_model, proprio)
                    implied_observed = idm(z_current, z_observed, proprio)
                implied_model = implied_model.detach().cpu().numpy().astype(np.float32)
                implied_observed = implied_observed.detach().cpu().numpy().astype(np.float32)

            feature_delta = (z_observed - z_model).detach().cpu().numpy().astype(np.float32)
            feature_cos = torch.nn.functional.cosine_similarity(z_observed, z_model, dim=-1).detach().cpu().numpy()

            for local, i in enumerate(ids):
                t = int(query_t[i])
                selected_chunk = query_actions[i].astype(np.float32)
                executed_chunk = executed_actions[t : t + horizon].astype(np.float32)
                model_chunk = implied_model[local].reshape(horizon, 7)
                observed_chunk = implied_observed[local].reshape(horizon, 7)
                selected_action = selected_chunk.reshape(-1)
                executed_action = executed_chunk.reshape(-1)
                model_action = model_chunk.reshape(-1)
                observed_action = observed_chunk.reshape(-1)
                rows.append(
                    {
                        "episode_file": str(h5_path),
                        "episode": h5_path.stem,
                        "task_description": task_description,
                        "success": success,
                        "query_index": int(i),
                        "query_t": t,
                        "horizon_k": horizon,
                        "model_value": float(query_values[i]) if query_values is not None else float("nan"),
                        "siglip_pred_observed_l2": _norm(feature_delta[local]),
                        "siglip_pred_observed_cos": float(feature_cos[local]),
                        "idm_implied_model_l2": _norm(model_action),
                        "idm_implied_observed_l2": _norm(observed_action),
                        "idm_model_observed_l2": _norm(observed_action - model_action),
                        "idm_model_observed_cos": _cosine(observed_action, model_action),
                        "idm_model_vs_selected_l2": _norm(model_action - selected_action),
                        "idm_observed_vs_selected_l2": _norm(observed_action - selected_action),
                        "idm_model_vs_selected_cos_flattened_legacy": _cosine(model_action, selected_action),
                        "idm_observed_vs_selected_cos_flattened_legacy": _cosine(observed_action, selected_action),
                        "idm_model_vs_selected_step_cos": _mean_step_cosine(model_chunk, selected_chunk),
                        "idm_model_vs_selected_step_cos_dist": 1.0 - _mean_step_cosine(model_chunk, selected_chunk),
                        "idm_observed_vs_selected_step_cos": _mean_step_cosine(observed_chunk, selected_chunk),
                        "idm_observed_vs_selected_step_cos_dist": 1.0 - _mean_step_cosine(observed_chunk, selected_chunk),
                        "idm_model_vs_selected_eef_step_cos": _mean_step_cosine(model_chunk, selected_chunk, slice(0, 6)),
                        "idm_model_vs_selected_eef_step_cos_dist": 1.0 - _mean_step_cosine(model_chunk, selected_chunk, slice(0, 6)),
                        "idm_model_vs_selected_step_l2": _mean_step_l2(model_chunk, selected_chunk),
                        "idm_model_vs_selected_eef_step_l2": _mean_step_l2(model_chunk, selected_chunk, slice(0, 6)),
                        "idm_model_vs_selected_gripper_step_l1": float(np.mean(np.abs(model_chunk[:, -1] - selected_chunk[:, -1]))),
                        "idm_model_vs_selected_gripper_mismatch_rate": _gripper_mismatch_rate(model_chunk, selected_chunk),
                        "selected_gripper_flip_count": _gripper_flip_count(selected_chunk),
                        "idm_model_gripper_flip_count": _gripper_flip_count(model_chunk),
                        "selected_vs_executed_l2": _norm(selected_action - executed_action),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "siglip_pred_observed_l2",
        "siglip_pred_observed_cos",
        "idm_model_observed_l2",
        "idm_model_observed_cos",
        "idm_model_vs_selected_l2",
        "idm_observed_vs_selected_l2",
        "idm_model_vs_selected_step_cos_dist",
        "idm_model_vs_selected_eef_step_cos_dist",
        "idm_model_vs_selected_step_l2",
        "idm_model_vs_selected_eef_step_l2",
        "idm_model_vs_selected_gripper_step_l1",
        "idm_model_vs_selected_gripper_mismatch_rate",
        "selected_vs_executed_l2",
        "model_value",
    ]
    out: dict[str, Any] = {"n": len(rows), "episodes": len({r["episode"] for r in rows})}
    out["success_episodes"] = sum(1 for ep in {r["episode"]: r["success"] for r in rows}.values() if ep)
    for key in keys:
        values = np.asarray([r[key] for r in rows if np.isfinite(float(r[key]))], dtype=np.float64)
        if values.size:
            out[f"{key}/mean"] = float(values.mean())
            out[f"{key}/median"] = float(np.median(values))
            out[f"{key}/p90"] = float(np.percentile(values, 90))
    return out


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode"]), []).append(row)

    metrics = [
        ("siglip_pred_observed_l2", "SigLIP predicted-vs-observed L2"),
        ("idm_model_vs_selected_step_l2", "IDM(C,P) vs selected action mean per-step L2"),
        ("idm_model_vs_selected_step_cos_dist", "IDM(C,P) vs selected action mean per-step 1-cos"),
        ("idm_model_vs_selected_gripper_mismatch_rate", "IDM(C,P) vs selected action gripper sign mismatch rate"),
        ("model_value", "Cosmos value"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 11), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for episode, ep_rows in by_episode.items():
            ep_rows = sorted(ep_rows, key=lambda r: int(r["query_t"]))
            label = episode.split("--task=")[-1][:40] if "--task=" in episode else episode[-40:]
            ax.plot([r["query_t"] for r in ep_rows], [r[metric] for r in ep_rows], marker="o", linewidth=1.5, label=label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("rollout timestep after stabilization")
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "online_cosmos_deviation_timeseries.png", dpi=180)
    plt.close(fig)

    task_names = sorted({str(r["task_description"]) for r in rows})
    x = np.arange(len(task_names))
    fig, ax = plt.subplots(figsize=(max(9, 2.5 * len(task_names)), 5))
    means = []
    for task in task_names:
        values = [r["idm_model_observed_l2"] for r in rows if r["task_description"] == task]
        means.append(float(np.mean(values)))
    ax.bar(x, means)
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean IDM model-observed L2")
    ax.set_title("Cosmos online rollout deviation by LIBERO-10 task")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "online_cosmos_deviation_by_task.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Rollout HDF5 files or directories")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.input:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.hdf5")))
            files.extend(sorted(path.rglob("*.h5")))
        elif path.is_file():
            files.append(path)
    if not files:
        raise SystemExit("No rollout HDF5 files found")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idm, idm_type = _load_idm(Path(args.checkpoint), device)
    processor = AutoProcessor.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()

    rows: list[dict[str, Any]] = []
    for h5_path in files:
        rows.extend(_episode_rows(h5_path, idm, idm_type, encoder, processor, device, args.batch_size))
    if not rows:
        raise SystemExit("No valid query rows found. Check data_collection=True and future predictions in rollout HDF5.")

    _write_csv(out_dir / "online_cosmos_deviation.csv", rows)
    summary = _summary(rows)
    summary.update(
        {
            "input_files": [str(path) for path in files],
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "idm_type": idm_type,
            "encoder": args.encoder,
            "horizon_note": "Observed future is primary_images[query_t + horizon_k]; Cosmos predicted future is query_future_primary_images at the same model-query point.",
            "patch_idm_adapter_note": "For patch IDM checkpoints, online C/P pairs are adapted to the training window as [C repeated k times, P].",
            "preprocessing_note": "Online rollout HDF5 stores the already prepared Cosmos LIBERO frames; no additional flip is applied.",
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(rows, out_dir)
    print(json.dumps(summary, indent=2))
    print({"csv": str(out_dir / "online_cosmos_deviation.csv"), "plots": [str(out_dir / "online_cosmos_deviation_timeseries.png"), str(out_dir / "online_cosmos_deviation_by_task.png")]})


if __name__ == "__main__":
    main()
