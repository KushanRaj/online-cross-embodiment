#!/usr/bin/env python3
"""Export supervised IDM windows from RoboCasa/Cosmos rollout HDF5 files.

The downloaded RoboCasa-Cosmos dataset is episode-shaped rather than LIBERO's
`data/demo_*` layout. Rows reference the HDF5 frame datasets directly so the
feature cache can be rebuilt without materializing thousands of PNGs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ACTION_SPACE = "robocasa_7d_osc_pose_manipulation"
MOBILE_BASE_PAD = [0.0, 0.0, 0.0, 0.0, -1.0]


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


def _dataset_name(h5: h5py.File, base: str) -> str | None:
    if base in h5:
        return base
    jpeg = f"{base}_jpeg"
    if jpeg in h5:
        return jpeg
    return None


def _read_array(h5: h5py.File, name: str) -> np.ndarray | None:
    if name not in h5:
        return None
    return np.asarray(h5[name][()])


def _task_from_path(path: Path) -> str:
    parts = path.parts
    if "all_episodes" in parts:
        idx = parts.index("all_episodes")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    stem = re.sub(r"_demo$", "", path.stem)
    return stem.replace("_", " ")


def export_file(
    h5_path: Path,
    manifest,
    horizons: list[int],
    dataset_id: str,
    frame_transform: str,
    max_rows_per_file: int | None,
) -> Counter:
    counts: Counter = Counter()
    task_name = _task_from_path(h5_path)

    with h5py.File(h5_path, "r") as h5:
        primary_key = _dataset_name(h5, "primary_images")
        if primary_key is None:
            return counts

        actions = _read_array(h5, "actions")
        if actions is None:
            return counts
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 7:
            raise ValueError(f"{h5_path} actions must be [T, >=7], got {actions.shape}")
        actions_7d = actions[:, :7]

        proprio = _read_array(h5, "proprio")
        task_description = str(h5.attrs.get("task_description", task_name))
        success = bool(h5.attrs.get("success", False))
        trajectory_id = h5_path.stem
        image_len = int(len(h5[primary_key]))
        length = min(image_len, len(actions_7d))
        row_count = 0

        for horizon in horizons:
            max_start = min(length - horizon, len(actions_7d) - horizon + 1)
            for t in range(max(0, max_start)):
                if max_rows_per_file is not None and row_count >= max_rows_per_file:
                    break
                sample_id = f"{trajectory_id}--actual_actual--k{horizon}--t{t:05d}"
                row = {
                    "sample_id": sample_id,
                    "trajectory_id": trajectory_id,
                    "source_file": str(h5_path),
                    "source_type": "actual_actual",
                    "dataset_id": dataset_id,
                    "environment": "robocasa",
                    "task_name": task_name,
                    "instruction": task_description,
                    "camera_name": "primary",
                    "window_start_t": int(t),
                    "horizon_k": int(horizon),
                    "pair_direction": "forward",
                    "image_t_hdf5_path": primary_key,
                    "image_t_index": int(t),
                    "image_t_transform": frame_transform,
                    "image_future_hdf5_path": primary_key,
                    "image_future_index": int(t + horizon),
                    "image_future_transform": frame_transform,
                    "proprio_t": _as_jsonable(proprio[t]) if proprio is not None and t < len(proprio) else None,
                    "action_chunk": _as_jsonable(actions_7d[t : t + horizon]),
                    "action_space": ACTION_SPACE,
                    "env_action_pad": MOBILE_BASE_PAD,
                    "success_label": success,
                }
                manifest.write(json.dumps(row) + "\n")
                counts[f"k{horizon}"] += 1
                counts["actual_actual"] += 1
                row_count += 1
            if max_rows_per_file is not None and row_count >= max_rows_per_file:
                break

        counts["trajectories"] += 1
        counts["transitions"] += length

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="RoboCasa HDF5 files or directories")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--horizons", default="16", help="Comma-separated horizons")
    parser.add_argument("--dataset-id", default="robocasa_cosmos_policy_all_episodes")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional per-file smoke cap")
    parser.add_argument(
        "--frame-transform",
        choices=["none", "flipud", "rot180"],
        default="none",
        help="Canonical transform applied to both current and future frames",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    source_files = _iter_hdf5_files(args.input)
    if not source_files:
        raise SystemExit("No HDF5 files found")

    counts: Counter = Counter()
    per_file: dict[str, dict[str, int]] = {}
    with manifest_path.open("w") as manifest:
        for source_file in source_files:
            file_counts = export_file(
                source_file,
                manifest,
                horizons,
                args.dataset_id,
                args.frame_transform,
                args.max_rows_per_file,
            )
            counts.update(file_counts)
            per_file[str(source_file)] = dict(file_counts)

    metadata = {
        "dataset_version": "idm-robocasa-rollout-v0",
        "manifest": str(manifest_path),
        "source_files": [str(path) for path in source_files],
        "horizons": horizons,
        "dataset_id": args.dataset_id,
        "action_space": ACTION_SPACE,
        "counts": dict(counts),
        "per_file": per_file,
        "storage": "hdf5_references",
        "source_type": "actual_actual",
        "pair_direction": "forward",
        "reverse_rows_included": False,
        "frame_convention": f"robocasa_rollout_{args.frame_transform}",
        "hdf5_frame_transform": args.frame_transform,
        "env_action_pad": MOBILE_BASE_PAD,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
