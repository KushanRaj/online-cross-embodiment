#!/usr/bin/env bash
set -euo pipefail

# Remote-only setup for the Fast-WAM/LingBot RoboTwin track.
# This prepares the repo and task manifest. It does not evaluate policies.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPOS_ROOT="${REPOS_ROOT:-${ROBOTICS_ROOT}/repos}"
DATA_ROOT="${DATA_ROOT:-${ROBOTICS_ROOT}/data}"
ROBOTWIN_REPO="${ROBOTWIN_REPO:-${REPOS_ROOT}/RoboTwin}"
ROBOTWIN_DATA_ROOT="${ROBOTWIN_DATA_ROOT:-${DATA_ROOT}/robotwin}"
PYTHON="${PYTHON:-python3}"
INSTALL="${INSTALL:-0}"

mkdir -p "${REPOS_ROOT}" "${ROBOTWIN_DATA_ROOT}"

if [[ ! -d "${ROBOTWIN_REPO}/.git" ]]; then
  git clone https://github.com/RoboTwin-Platform/RoboTwin.git "${ROBOTWIN_REPO}"
else
  git -C "${ROBOTWIN_REPO}" fetch --all --tags --prune
fi

cat > "${ROBOTWIN_DATA_ROOT}/phase3_selected_tasks.txt" <<'EOF'
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

if [[ "${INSTALL}" == "1" ]]; then
  cd "${ROBOTWIN_REPO}"
  if [[ -f requirements.txt ]]; then
    "${PYTHON}" -m pip install --user -r requirements.txt
  fi
fi

echo "RoboTwin repo: ${ROBOTWIN_REPO}"
echo "Selected tasks: ${ROBOTWIN_DATA_ROOT}/phase3_selected_tasks.txt"
