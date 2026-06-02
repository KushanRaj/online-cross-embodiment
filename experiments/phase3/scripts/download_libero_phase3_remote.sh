#!/usr/bin/env bash
set -euo pipefail

# Remote-only data prep for the common LIBERO track.
# It downloads selected demonstrations only; it does not run evaluations.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
LIBERO_REPO="${LIBERO_REPO:-${ROBOTICS_ROOT}/repos/LIBERO}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${ROBOTICS_ROOT}/data/libero}"
PYTHON="${PYTHON:-python3}"
HF_REPO_ID="${HF_REPO_ID:-yifengzhu-hf/LIBERO-datasets}"
LIBERO_DATASETS="${LIBERO_DATASETS:-libero_10}"
export HF_REPO_ID LIBERO_DATA_ROOT LIBERO_DATASETS

mkdir -p "${ROBOTICS_ROOT}/repos" "${LIBERO_DATA_ROOT}"

if [[ ! -d "${LIBERO_REPO}/.git" ]]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REPO}"
fi

"${PYTHON}" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
PY

"${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

datasets = [d.strip() for d in os.environ["LIBERO_DATASETS"].split(",") if d.strip()]
patterns = [f"{dataset}/**" for dataset in datasets]

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["LIBERO_DATA_ROOT"],
    allow_patterns=patterns,
)
print(os.environ["LIBERO_DATA_ROOT"])
print("Downloaded patterns:", ", ".join(patterns))
PY

find "${LIBERO_DATA_ROOT}" -maxdepth 2 -type f \( -name '*.hdf5' -o -name '*.h5' \) | sort
