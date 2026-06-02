#!/usr/bin/env bash
set -euo pipefail

# Remote-only data prep for the Cosmos/RoboCasa track.
# Default download is the selected Phase 3 task-family subset only.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPOS_ROOT="${REPOS_ROOT:-${ROBOTICS_ROOT}/repos}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
ROBOCASA_REPO="${ROBOCASA_REPO:-${REPOS_ROOT}/robocasa}"
ROBOCASA_COSMOS_DIR="${ROBOCASA_COSMOS_DIR:-${DATA_ROOT}/robocasa-cosmos-policy}"
PYTHON="${PYTHON:-python3}"
HF_REPO_ID="${HF_REPO_ID:-nvidia/RoboCasa-Cosmos-Policy}"
ALLOW_PATTERNS="${ALLOW_PATTERNS:-all_episodes/TurnOffStove/**,all_episodes/TurnOnSinkFaucet/**,all_episodes/CoffeeSetupMug/**,all_episodes/TurnOnStove/**,all_episodes/CloseDoubleDoor/**,all_episodes/CloseDrawer/**,all_episodes/CloseSingleDoor/**,all_episodes/CoffeePressButton/**,all_episodes/OpenDrawer/**,all_episodes/TurnSinkSpout/**,*.json,*.md}"
export HF_REPO_ID ALLOW_PATTERNS ROBOCASA_COSMOS_DIR

mkdir -p "${REPOS_ROOT}" "${DATA_ROOT}" "${ROBOCASA_COSMOS_DIR}"

if [[ ! -d "${ROBOCASA_REPO}/.git" ]]; then
  git clone https://github.com/robocasa/robocasa.git "${ROBOCASA_REPO}"
fi

"${PYTHON}" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
PY

"${PYTHON}" - <<PY
import os
from huggingface_hub import snapshot_download

patterns = [p.strip() for p in os.environ["ALLOW_PATTERNS"].split(",") if p.strip()]
snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["ROBOCASA_COSMOS_DIR"],
    allow_patterns=patterns,
)
print(os.environ["ROBOCASA_COSMOS_DIR"])
PY

find "${ROBOCASA_COSMOS_DIR}" -maxdepth 3 -type f | sort | sed -n '1,120p'
