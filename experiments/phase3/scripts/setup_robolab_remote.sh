#!/usr/bin/env bash
set -euo pipefail

# Remote-only setup for the RoboLab clean-stress task track.
# It clones/updates the RoboLab repo and records a small selected task list.
# It does not run Isaac Lab, policy inference, rollouts, training, or uv sync.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPOS_ROOT="${REPOS_ROOT:-${ROBOTICS_ROOT}/repos}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
ROBOLAB_REPO="${ROBOLAB_REPO:-${REPOS_ROOT}/RoboLab}"
ROBOLAB_DATA_ROOT="${ROBOLAB_DATA_ROOT:-${DATA_ROOT}/robolab}"
ROBOLAB_SELECTED_TASKS="${ROBOLAB_SELECTED_TASKS:-${ROBOLAB_DATA_ROOT}/phase3_selected_tasks.txt}"

mkdir -p "${REPOS_ROOT}" "${ROBOLAB_DATA_ROOT}"

if [[ ! -d "${ROBOLAB_REPO}/.git" ]]; then
  git clone https://github.com/NVLabs/RoboLab.git "${ROBOLAB_REPO}"
else
  git -C "${ROBOLAB_REPO}" pull --ff-only || true
fi

cat > "${ROBOLAB_SELECTED_TASKS}" <<'EOF'
BananaInBowlTask
RubiksCubeAndBananaTask
EOF

printf "RoboLab repo: %s\n" "${ROBOLAB_REPO}"
printf "Selected task manifest: %s\n" "${ROBOLAB_SELECTED_TASKS}"
printf "\nSelected task manifest:\n"
cat "${ROBOLAB_SELECTED_TASKS}"
printf "\n\nRepo size:\n"
du -sh "${ROBOLAB_REPO}" "${ROBOLAB_DATA_ROOT}"
