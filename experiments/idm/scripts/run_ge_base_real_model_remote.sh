#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/robotics/repos/online-cross-embodiment}"
GE_REPO="${GE_REPO:-/home/ubuntu/robotics/repos/Genie-Envisioner}"
INPUT="${INPUT:-/home/ubuntu/robotics/data/libero/libero_spatial}"
OUT="${OUT:-/home/ubuntu/robotics/runs/idm-datasets/ge-base-real-model-smoke}"
PYTHON="${PYTHON:-/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python}"

CONFIG_FILE="${CONFIG_FILE:-${GE_REPO}/configs/ltx_model/libero/video_model_libero.yaml}"
CKPT_PATH="${CKPT_PATH:-${GE_REPO}/checkpoints/agibot_world/Genie-Envisioner/GE_base_fast_v0.1.safetensors}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/home/ubuntu/robotics/checkpoints/LTX-Video}"
LIMIT="${LIMIT:-2}"
STRIDE="${STRIDE:-36}"
DEVICE="${DEVICE:-cuda:0}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${GE_REPO}:${GE_REPO}/video_gen_examples:${PYTHONPATH:-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing GE-Base config: $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "Missing GE-Base checkpoint: $CKPT_PATH" >&2
  exit 1
fi

if [[ ! -d "$PRETRAINED_PATH" ]]; then
  echo "Missing LTX pretrained path: $PRETRAINED_PATH" >&2
  exit 1
fi

"$PYTHON" experiments/idm/remote/query_ge_base_real_to_model.py \
  --ge-repo "$GE_REPO" \
  --config-file "$CONFIG_FILE" \
  --ckpt-path "$CKPT_PATH" \
  --pretrained-path "$PRETRAINED_PATH" \
  --input "$INPUT" \
  --out "$OUT" \
  --limit "$LIMIT" \
  --stride "$STRIDE" \
  --device "$DEVICE"
