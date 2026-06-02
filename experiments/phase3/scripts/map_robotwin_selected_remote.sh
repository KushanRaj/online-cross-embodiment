#!/usr/bin/env bash
set -euo pipefail

# Remote-only RoboTwin selected-task mapper.
#
# Default behavior:
#   - downloads RoboTwin metadata only,
#   - maps the Phase 3 selected task labels to episode/video/data shards,
#   - writes a JSON report,
#   - does not download selected videos or data shards.
#
# Set DOWNLOAD=1 to download the exact selected paths after reviewing the size.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${DATA_ROOT}/robotwin}"
ROBOTWIN_META_DIR="${ROBOTWIN_META_DIR:-${ROBOTWIN_DATA_ROOT}/meta_only}"
ROBOTWIN_SELECTED_DIR="${ROBOTWIN_SELECTED_DIR:-${ROBOTWIN_DATA_ROOT}/selected_phase3}"
MAPPING_JSON="${MAPPING_JSON:-${ROBOTWIN_DATA_ROOT}/phase3_selected_mapping.json}"
PYTHON="${PYTHON:-python3}"
DOWNLOAD="${DOWNLOAD:-0}"
export ROBOTWIN_META_DIR ROBOTWIN_SELECTED_DIR MAPPING_JSON DOWNLOAD

mkdir -p "${ROBOTWIN_DATA_ROOT}" "${ROBOTWIN_META_DIR}" "${ROBOTWIN_SELECTED_DIR}"

"${PYTHON}" - <<'PY'
import importlib.util
import subprocess
import sys

missing = [m for m in ["huggingface_hub", "pyarrow"] if importlib.util.find_spec(m) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", *missing])
PY

"${PYTHON}" - <<'PY'
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

REPO_ID = "lerobot/robotwin_unified"
META_DIR = Path(os.environ["ROBOTWIN_META_DIR"])
SELECTED_DIR = Path(os.environ["ROBOTWIN_SELECTED_DIR"])
MAPPING_JSON = Path(os.environ["MAPPING_JSON"])
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(META_DIR),
    allow_patterns=["meta/**"],
)

tasks_table = pq.read_table(META_DIR / "meta/tasks.parquet")
episodes_table = pq.read_table(
    META_DIR / "meta/episodes/chunk-000/file-000.parquet",
    columns=[
        "episode_index",
        "data/file_index",
        "tasks",
        "videos/observation.images.cam_high/file_index",
        "videos/observation.images.cam_left_wrist/file_index",
        "videos/observation.images.cam_right_wrist/file_index",
    ],
)

tasks = tasks_table.to_pydict()
episodes = episodes_table.to_pydict()
idx_to_task = dict(zip(tasks["task_index"], tasks["task"]))

def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def words(text):
    return set(norm(text).split())

def anyw(text, candidates):
    text_words = words(text)
    return any(candidate in text_words for candidate in candidates)

patterns = {
    "Grab Roller": lambda i, s: "roller" in words(s),
    "Place Empty Cup": lambda i, s: "cup" in words(s) and "coaster" in words(s),
    "Stack Blocks Two": lambda i, s: 21500 <= i <= 21580 and "block" in words(s),
    "Click Alarmclock": lambda i, s: (
        ("alarm" in words(s) or "alarmclock" in words(s))
        and anyw(s, ["click", "press", "push", "tap", "activate"])
    ),
    "Hanging Mug": lambda i, s: "mug" in words(s) and anyw(s, ["hang", "rack", "attach", "hook"]),
    "Open Microwave": lambda i, s: "microwave" in words(s) and anyw(s, ["open", "pull", "handle"]),
    "Turn Switch": lambda i, s: (
        "switch" in words(s)
        and "microphone" not in words(s)
        and anyw(s, ["turn", "rotate", "toggle", "press", "push", "flip", "activate", "engage", "click"])
    ),
    "Place Can Basket": lambda i, s: "can" in words(s) and "basket" in words(s),
    "Move Stapler Pad": lambda i, s: "stapler" in words(s) and anyw(s, ["mat", "pad", "colored"]),
    "Press Stapler": lambda i, s: 17631 <= i <= 18148 and "stapler" in words(s),
    "Handover Mic": lambda i, s: (
        "microphone" in words(s)
        and anyw(s, ["handover", "hand", "transfer", "pass", "shift", "opposite", "switch", "deliver", "give"])
    ),
    "Lift Pot": lambda i, s: (
        ("pot" in words(s) or "kitchenpot" in words(s))
        and anyw(s, ["lift", "raise", "elevate", "upward"])
        and "can" not in words(s)
    ),
    "Move Can Pot": lambda i, s: (
        "can" in words(s)
        and ("pot" in words(s) or "kitchenpot" in words(s))
        and anyw(s, ["beside", "next", "near", "close", "side"])
    ),
}

api = HfApi()
repo_files = list(api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", recursive=True, expand=True))
size_by_path = {item.path: item.size for item in repo_files if getattr(item, "size", None) is not None}

def video_path(camera, file_index):
    return f"videos/observation.images.{camera}/chunk-000/file-{file_index:03d}.mp4"

def data_path(file_index):
    return f"data/chunk-000/file-{file_index:03d}.parquet"

metadata_paths = [
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
]

union_paths = set(metadata_paths)
summary = []

for label, predicate in patterns.items():
    matched_indices = {idx for idx, text in idx_to_task.items() if predicate(idx, text)}
    matched_strings = {idx_to_task[idx] for idx in matched_indices}
    data_files = set()
    video_files = {"cam_high": set(), "cam_left_wrist": set(), "cam_right_wrist": set()}
    episode_indices = []

    for row, task_list in enumerate(episodes["tasks"]):
        if any(task in matched_strings for task in task_list):
            episode_indices.append(episodes["episode_index"][row])
            data_files.add(episodes["data/file_index"][row])
            video_files["cam_high"].add(episodes["videos/observation.images.cam_high/file_index"][row])
            video_files["cam_left_wrist"].add(episodes["videos/observation.images.cam_left_wrist/file_index"][row])
            video_files["cam_right_wrist"].add(episodes["videos/observation.images.cam_right_wrist/file_index"][row])

    paths = {data_path(idx) for idx in data_files}
    for camera, indices in video_files.items():
        paths |= {video_path(camera, idx) for idx in indices}

    union_paths |= paths

    task_samples = [idx_to_task[idx] for idx in sorted(matched_indices)[:3]]
    bytes_for_label = sum(size_by_path.get(path, 0) for path in paths)
    summary.append(
        {
            "label": label,
            "task_strings": len(matched_indices),
            "episodes": len(episode_indices),
            "data_files": sorted(data_files),
            "video_files": {camera: sorted(indices) for camera, indices in video_files.items()},
            "size_gib": bytes_for_label / (1024**3),
            "sample_tasks": task_samples,
        }
    )

union_paths_sorted = sorted(union_paths)
union_bytes = sum(size_by_path.get(path, 0) for path in union_paths_sorted)

report = {
    "repo_id": REPO_ID,
    "metadata_only_dir": str(META_DIR),
    "selected_download_dir": str(SELECTED_DIR),
    "downloaded_selected_paths": DOWNLOAD,
    "selected_task_count": len(summary),
    "selected_total_gib": union_bytes / (1024**3),
    "selected_file_count": len(union_paths_sorted),
    "summary": summary,
    "selected_paths": union_paths_sorted,
}

MAPPING_JSON.write_text(json.dumps(report, indent=2))

if DOWNLOAD:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(SELECTED_DIR),
        allow_patterns=union_paths_sorted,
    )

print(json.dumps({k: report[k] for k in [
    "repo_id",
    "metadata_only_dir",
    "selected_download_dir",
    "downloaded_selected_paths",
    "selected_task_count",
    "selected_total_gib",
    "selected_file_count",
]}, indent=2))
print(f"Mapping JSON: {MAPPING_JSON}")
PY
