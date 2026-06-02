#!/usr/bin/env python3
"""Export supervised IDM windows from raw LIBERO demonstration HDF5 files.

Rows reference HDF5 frame locations instead of materializing PNGs. This keeps
the base IDM dataset small and avoids duplicating the same frames across
multiple horizons.
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


ACTION_SPACE = "libero_7d_eef_delta_gripper"


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


def _task_from_filename(path: Path) -> str:
    stem = re.sub(r"_demo$", "", path.stem)
    return stem.replace("_", " ")


def _sorted_demo_keys(handle: h5py.File) -> list[str]:
    keys = list(handle["data"].keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]) if k.startswith("demo_") else k)


def export_file(
    h5_path: Path,
    manifest,
    horizons: list[int],
    max_demos_per_file: int | None,
    dataset_id: str,
) -> Counter:
    counts: Counter = Counter()
    task_name = _task_from_filename(h5_path)

    with h5py.File(h5_path, "r") as h5:
        demo_keys = _sorted_demo_keys(h5)
        if max_demos_per_file is not None:
            demo_keys = demo_keys[:max_demos_per_file]

        for demo_key in demo_keys:
            demo = h5["data"][demo_key]
            actions = demo["actions"][()]
            robot_states = demo["robot_states"][()] if "robot_states" in demo else None
            rewards = demo["rewards"][()] if "rewards" in demo else None
            success = bool(rewards is not None and len(rewards) > 0 and rewards[-1] > 0)
            length = int(len(actions))
            trajectory_id = f"{h5_path.stem}--{demo_key}"

            for horizon in horizons:
                max_start = length - horizon
                for t in range(max_start):
                    sample_id = f"{trajectory_id}--actual_actual--k{horizon}--t{t:05d}"
                    row = {
                        "sample_id": sample_id,
                        "trajectory_id": trajectory_id,
                        "source_file": str(h5_path),
                        "source_type": "actual_actual",
                        "dataset_id": dataset_id,
                        "environment": "libero",
                        "task_name": task_name,
                        "instruction": task_name,
                        "camera_name": "agentview_rgb",
                        "window_start_t": int(t),
                        "horizon_k": int(horizon),
                        "image_t_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                        "image_t_index": int(t),
                        "image_t_transform": "flipud",
                        "image_future_hdf5_path": f"data/{demo_key}/obs/agentview_rgb",
                        "image_future_index": int(t + horizon),
                        "image_future_transform": "flipud",
                        "wrist_image_t_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                        "wrist_image_t_index": int(t),
                        "wrist_image_t_transform": "flipud",
                        "wrist_image_future_hdf5_path": f"data/{demo_key}/obs/eye_in_hand_rgb",
                        "wrist_image_future_index": int(t + horizon),
                        "wrist_image_future_transform": "flipud",
                        "proprio_t": _as_jsonable(robot_states[t]) if robot_states is not None else None,
                        "action_chunk": _as_jsonable(actions[t : t + horizon]),
                        "action_space": ACTION_SPACE,
                        "success_label": success,
                    }
                    manifest.write(json.dumps(row) + "\n")
                    counts[f"k{horizon}"] += 1
                    counts["actual_actual"] += 1
            counts["trajectories"] += 1
            counts["transitions"] += length

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Raw LIBERO HDF5 files or directories")
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--horizons", default="1,4", help="Comma-separated horizons")
    parser.add_argument("--dataset-id", default="libero_spatial_demos", help="Dataset id to write into rows")
    parser.add_argument("--max-demos-per-file", type=int, default=None, help="Optional smoke limit")
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
                args.max_demos_per_file,
                args.dataset_id,
            )
            counts.update(file_counts)
            per_file[str(source_file)] = dict(file_counts)

    metadata = {
        "dataset_version": "idm-libero-demo-v0",
        "manifest": str(manifest_path),
        "source_files": [str(path) for path in source_files],
        "horizons": horizons,
        "dataset_id": args.dataset_id,
        "action_space": ACTION_SPACE,
        "counts": dict(counts),
        "per_file": per_file,
        "storage": "hdf5_references",
        "frame_convention": "cosmos_libero_eval_flipud",
        "hdf5_frame_transform": "flipud",
        "generated_frame_transform": "none",
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
