#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-ubuntu@ec2-13-126-35-29.ap-south-1.compute.amazonaws.com}"
KEY="${KEY:-$HOME/.ssh/kushan-harbor.pem}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/robotics/repos/online-cross-embodiment}"

rsync -az --delete \
  -e "ssh -i ${KEY}" \
  --exclude ".git/" \
  --exclude "run-artifacts/" \
  --exclude "tmp/" \
  --exclude "libero_data/" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "Synced repo to ${REMOTE}:${REMOTE_DIR}"
