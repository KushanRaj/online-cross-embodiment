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

from experiments.idm.model.adapter import ImageIDMAdapter


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


def _episode_rows(
    h5_path: Path,
    idm: ImageIDMAdapter,
    batch_size: int,
    reference_action_dataset: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        primary = _read_image_stack(h5, "primary_images")
        query_primary = _read_image_stack(h5, "query_primary_images")
        query_future = _read_image_stack(h5, "query_future_primary_images")
        query_t = _read_array(h5, "query_t")
        query_proprio = _read_array(h5, "query_proprio")
        query_actions = _read_array(h5, reference_action_dataset)
        if query_actions is None:
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

            proprio = query_proprio[ids].reshape(len(ids), -1)
            implied_model = idm.predict(current_images, model_future_images, proprio)
            implied_observed = idm.predict(current_images, observed_future_images, proprio)
            feature_l2, feature_cos = idm.feature_similarity(model_future_images, observed_future_images)

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
                        "siglip_pred_observed_l2": float(feature_l2[local]),
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
    parser.add_argument("--reference-action-dataset", default="query_action_chunks")
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
    idm = ImageIDMAdapter(Path(args.checkpoint), args.encoder, device)

    rows: list[dict[str, Any]] = []
    for h5_path in files:
        rows.extend(_episode_rows(h5_path, idm, args.batch_size, args.reference_action_dataset))
    if not rows:
        raise SystemExit("No valid query rows found. Check data_collection=True and future predictions in rollout HDF5.")

    _write_csv(out_dir / "online_cosmos_deviation.csv", rows)
    summary = _summary(rows)
    summary.update(
        {
            "input_files": [str(path) for path in files],
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "idm_type": idm.model_type,
            "encoder": args.encoder,
            "reference_action_dataset": args.reference_action_dataset,
            "horizon_note": "Observed future is primary_images[query_t + horizon_k]; Cosmos predicted future is query_future_primary_images at the same model-query point.",
            "patch_idm_adapter_note": idm.adapter_note,
            "preprocessing_note": "Online rollout HDF5 stores the already prepared Cosmos LIBERO frames; no additional flip is applied.",
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(rows, out_dir)
    print(json.dumps(summary, indent=2))
    print({"csv": str(out_dir / "online_cosmos_deviation.csv"), "plots": [str(out_dir / "online_cosmos_deviation_timeseries.png"), str(out_dir / "online_cosmos_deviation_by_task.png")]})


if __name__ == "__main__":
    main()
