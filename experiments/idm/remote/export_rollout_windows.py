#!/usr/bin/env python3
"""Export IDM transition-window rows from Cosmos/GE LIBERO rollout HDF5 files.

The output is intentionally encoder-agnostic:

  manifest.jsonl
  images/<sample_id>_t.png
  images/<sample_id>_future.png

Rows can later be encoded with SigLIP, DINOv3, V-JEPA2, etc. without changing
the data contract.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image


def _read_image_stack(h5: h5py.File, name: str) -> np.ndarray | None:
    if name in h5:
        return h5[name][()]
    jpeg_name = f"{name}_jpeg"
    if jpeg_name not in h5:
        return None

    frames = []
    for encoded in h5[jpeg_name]:
        frame = Image.open(io.BytesIO(bytes(encoded))).convert("RGB")
        frames.append(np.asarray(frame, dtype=np.uint8))
    if not frames:
        return None
    return np.stack(frames, axis=0)


def _read_array(h5: h5py.File, name: str) -> np.ndarray | None:
    if name not in h5:
        return None
    return h5[name][()]


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _as_jsonable(array: np.ndarray) -> list:
    return np.asarray(array).astype(float).tolist()


def _iter_hdf5_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.hdf5")))
            files.extend(sorted(path.rglob("*.h5")))
        elif path.is_file():
            files.append(path)
    return files


def export_file(h5_path: Path, out_dir: Path, manifest, horizons: list[int], model_id: str) -> dict:
    counts = {"actual_actual": 0, "actual_model": 0}
    image_dir = out_dir / "images"

    with h5py.File(h5_path, "r") as h5:
        primary = _read_image_stack(h5, "primary_images")
        wrist = _read_image_stack(h5, "wrist_images")
        proprio = _read_array(h5, "proprio")
        actions = _read_array(h5, "actions")

        task_description = h5.attrs.get("task_description", "")
        success = bool(h5.attrs.get("success", False))
        trajectory_id = h5_path.stem

        if primary is not None and actions is not None:
            for horizon in horizons:
                max_start = min(len(primary) - horizon, len(actions) - horizon + 1)
                for t in range(max_start):
                    sample_id = f"{trajectory_id}--actual_actual--k{horizon}--t{t:05d}"
                    image_t_path = image_dir / f"{sample_id}_t.png"
                    image_future_path = image_dir / f"{sample_id}_future.png"
                    _save_image(image_t_path, primary[t])
                    _save_image(image_future_path, primary[t + horizon])

                    row = {
                        "sample_id": sample_id,
                        "trajectory_id": trajectory_id,
                        "source_file": str(h5_path),
                        "source_type": "actual_actual",
                        "model_id": model_id,
                        "environment": "libero",
                        "task_name": task_description,
                        "instruction": task_description,
                        "camera_name": "primary",
                        "window_start_t": int(t),
                        "horizon_k": int(horizon),
                        "image_t_path": _rel(image_t_path, out_dir),
                        "image_future_path": _rel(image_future_path, out_dir),
                        "proprio_t": _as_jsonable(proprio[t]) if proprio is not None else None,
                        "action_chunk": _as_jsonable(actions[t : t + horizon]),
                        "action_space": "libero_7d_eef_delta_gripper",
                        "success_label": success,
                    }
                    if wrist is not None:
                        wrist_t_path = image_dir / f"{sample_id}_wrist_t.png"
                        wrist_future_path = image_dir / f"{sample_id}_wrist_future.png"
                        _save_image(wrist_t_path, wrist[t])
                        _save_image(wrist_future_path, wrist[t + horizon])
                        row["wrist_image_t_path"] = _rel(wrist_t_path, out_dir)
                        row["wrist_image_future_path"] = _rel(wrist_future_path, out_dir)
                    manifest.write(json.dumps(row) + "\n")
                    counts["actual_actual"] += 1

        query_primary = _read_image_stack(h5, "query_primary_images")
        query_future_primary = _read_image_stack(h5, "query_future_primary_images")
        query_wrist = _read_image_stack(h5, "query_wrist_images")
        query_future_wrist = _read_image_stack(h5, "query_future_wrist_images")
        query_proprio = _read_array(h5, "query_proprio")
        query_action_chunks = _read_array(h5, "query_action_chunks")
        query_t = _read_array(h5, "query_t")
        query_values = _read_array(h5, "query_values")

        if query_primary is not None and query_future_primary is not None and query_action_chunks is not None:
            n = min(len(query_primary), len(query_future_primary), len(query_action_chunks))
            for i in range(n):
                horizon = int(query_action_chunks[i].shape[0])
                step = int(query_t[i]) if query_t is not None else i
                sample_id = f"{trajectory_id}--actual_model--k{horizon}--q{i:05d}"
                image_t_path = image_dir / f"{sample_id}_t.png"
                image_future_path = image_dir / f"{sample_id}_future.png"
                _save_image(image_t_path, query_primary[i])
                _save_image(image_future_path, query_future_primary[i])

                row = {
                    "sample_id": sample_id,
                    "trajectory_id": trajectory_id,
                    "source_file": str(h5_path),
                    "source_type": "actual_model",
                    "model_id": model_id,
                    "environment": "libero",
                    "task_name": task_description,
                    "instruction": task_description,
                    "camera_name": "primary",
                    "window_start_t": step,
                    "horizon_k": horizon,
                    "image_t_path": _rel(image_t_path, out_dir),
                    "image_future_path": _rel(image_future_path, out_dir),
                    "proprio_t": _as_jsonable(query_proprio[i]) if query_proprio is not None else None,
                    "action_chunk": _as_jsonable(query_action_chunks[i]),
                    "action_space": "libero_7d_eef_delta_gripper",
                    "success_label": success,
                    "model_value": float(query_values[i]) if query_values is not None else None,
                }
                if query_wrist is not None and query_future_wrist is not None and i < len(query_wrist):
                    wrist_t_path = image_dir / f"{sample_id}_wrist_t.png"
                    wrist_future_path = image_dir / f"{sample_id}_wrist_future.png"
                    _save_image(wrist_t_path, query_wrist[i])
                    _save_image(wrist_future_path, query_future_wrist[i])
                    row["wrist_image_t_path"] = _rel(wrist_t_path, out_dir)
                    row["wrist_image_future_path"] = _rel(wrist_future_path, out_dir)
                manifest.write(json.dumps(row) + "\n")
                counts["actual_model"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="HDF5 files or directories to export")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--model-id", default="cosmos", help="Model/source id for generated transition rows")
    parser.add_argument("--horizons", default="1,4", help="Comma-separated actual transition horizons")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    files = _iter_hdf5_files(args.input)
    if not files:
        raise SystemExit("No HDF5 files found")

    total = {"actual_actual": 0, "actual_model": 0}
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for h5_path in files:
            counts = export_file(h5_path, out_dir, manifest, horizons, args.model_id)
            for key, value in counts.items():
                total[key] += value

    metadata = {
        "dataset_version": "idm-v0",
        "source_files": [str(p) for p in files],
        "horizons": horizons,
        "model_id": args.model_id,
        "action_space": "libero_7d_eef_delta_gripper",
        "counts": total,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
