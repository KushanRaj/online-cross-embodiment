#!/usr/bin/env bash
set -euo pipefail

# Readiness check only. No rollouts, training, or evaluation.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPOS_ROOT="${REPOS_ROOT:-${ROBOTICS_ROOT}/repos}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
PHASE3_ROOT="${PHASE3_ROOT:-${DATA_ROOT}/phase3}"

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "OK   ${label}: ${path}"
  else
    echo "MISS ${label}: ${path}"
  fi
}

check_path "phase3 task manifest" "${PHASE3_ROOT}/tasks/phase3_clean_tasks.yaml"
check_path "LIBERO repo" "${REPOS_ROOT}/LIBERO"
check_path "LIBERO data root" "${DATA_ROOT}/libero"
check_path "RoboCasa repo" "${REPOS_ROOT}/robocasa"
check_path "RoboCasa Cosmos data root" "${DATA_ROOT}/robocasa-cosmos-policy"
check_path "RoboTwin repo" "${REPOS_ROOT}/RoboTwin"
check_path "RoboTwin selected tasks" "${DATA_ROOT}/robotwin/phase3_selected_tasks.txt"

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
