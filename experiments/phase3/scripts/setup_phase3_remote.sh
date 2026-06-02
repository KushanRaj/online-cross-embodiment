#!/usr/bin/env bash
set -euo pipefail

# Remote-only setup. This prepares repos and task manifests.
# It does not run policies, rollouts, training, or evaluation.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPOS_ROOT="${REPOS_ROOT:-${ROBOTICS_ROOT}/repos}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
PHASE3_ROOT="${PHASE3_ROOT:-${DATA_ROOT}/phase3}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../../.." && pwd)"

mkdir -p "${REPOS_ROOT}" "${DATA_ROOT}" "${PHASE3_ROOT}/tasks"
mkdir -p "${DATA_ROOT}/robocasa-cosmos-policy" "${DATA_ROOT}/robotwin"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    git -C "${dir}" fetch --all --tags --prune
  else
    git clone "${url}" "${dir}"
  fi
}

clone_or_update "https://github.com/Lifelong-Robot-Learning/LIBERO.git" "${REPOS_ROOT}/LIBERO"
clone_or_update "https://github.com/robocasa/robocasa.git" "${REPOS_ROOT}/robocasa"
clone_or_update "https://github.com/RoboTwin-Platform/RoboTwin.git" "${REPOS_ROOT}/RoboTwin"

rsync -a "${REPO_ROOT}/experiments/phase3/tasks/" "${PHASE3_ROOT}/tasks/"

cat > "${DATA_ROOT}/robotwin/phase3_selected_tasks.txt" <<'EOF'
Grab Roller
Place Empty Cup
Stack Blocks Two
Click Alarmclock
Hanging Mug
Open Microwave
Turn Switch
Place Can Basket
Move Stapler Pad
Press Stapler
Handover Mic
Lift Pot
Move Can Pot
EOF

cat > "${PHASE3_ROOT}/README.txt" <<EOF
Phase 3 clean rollout benchmark assets

Task manifest:
  ${PHASE3_ROOT}/tasks/phase3_clean_tasks.yaml

Environment repos:
  ${REPOS_ROOT}/LIBERO
  ${REPOS_ROOT}/robocasa
  ${REPOS_ROOT}/RoboTwin

Data roots:
  ${DATA_ROOT}/libero
  ${DATA_ROOT}/robocasa-cosmos-policy
  ${DATA_ROOT}/robotwin

This setup step does not run rollouts/evaluations.
EOF

echo "Phase 3 remote setup complete."
echo "Task manifest: ${PHASE3_ROOT}/tasks/phase3_clean_tasks.yaml"
