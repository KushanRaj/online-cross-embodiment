#!/usr/bin/env bash
set -euo pipefail

# Remote-only script. Exports RoboCasa/Cosmos real->real windows, caches frozen
# visual features, and trains the same feature-MLP IDM baseline used for LIBERO.

REPO_DIR="${REPO_DIR:-/home/ubuntu/robotics/repos/online-cross-embodiment}"
ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
ROBOCASA_DATA_ROOT="${ROBOCASA_DATA_ROOT:-${ROBOTICS_ROOT}/data/robocasa-cosmos-policy}"
RUN_ROOT="${RUN_ROOT:-${ROBOTICS_ROOT}/runs/idm-phase1}"
PYTHON="${PYTHON:-${ROBOTICS_ROOT}/repos/cosmos-policy/.venv/bin/python}"

SPLIT="${SPLIT:-all_episodes}"
DATASET_ID="${DATASET_ID:-robocasa_cosmos_policy_${SPLIT}}"
HORIZON="${HORIZON:-16}"
LIMIT="${LIMIT:-60000}"
ENCODER="${ENCODER:-google/siglip-base-patch16-224}"
SOURCE_TYPE="${SOURCE_TYPE:-actual_actual}"

ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-400}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-3e-4}"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-online-cross-embodiment-idm}"
WANDB_NAME="${WANDB_NAME:-idm-siglip-robocasa-real-real-k${HORIZON}}"

MANIFEST_DIR="${MANIFEST_DIR:-${ROBOTICS_ROOT}/runs/idm-datasets/robocasa-demos-k${HORIZON}}"
CACHE_PATH="${CACHE_PATH:-${RUN_ROOT}/features_siglip_robocasa_real_real_k${HORIZON}.pt}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/idm_siglip_robocasa_real_real_k${HORIZON}}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

if [[ ! -d "${ROBOCASA_DATA_ROOT}/${SPLIT}" ]]; then
  echo "Missing RoboCasa dataset directory: ${ROBOCASA_DATA_ROOT}/${SPLIT}" >&2
  echo "Run: bash experiments/phase3/scripts/download_robocasa_cosmos_remote.sh" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$MANIFEST_DIR"

"$PYTHON" experiments/idm/remote/export_robocasa_rollout_windows.py \
  --input "${ROBOCASA_DATA_ROOT}/${SPLIT}" \
  --out "$MANIFEST_DIR" \
  --horizons "$HORIZON" \
  --dataset-id "$DATASET_ID" \
  --frame-transform "${FRAME_TRANSFORM:-none}"

CACHE_ARGS=(
  --manifest "$MANIFEST_DIR/manifest.jsonl"
  --out "$CACHE_PATH"
  --encoder "$ENCODER"
  --horizon "$HORIZON"
  --source-type "$SOURCE_TYPE"
  --batch-size "$ENCODE_BATCH_SIZE"
)
if [[ -n "$LIMIT" ]]; then
  CACHE_ARGS+=(--shuffle-before-limit --limit "$LIMIT")
fi

"$PYTHON" -m experiments.idm.data.cache_features "${CACHE_ARGS[@]}"

TRAIN_ARGS=(
  --cache "$CACHE_PATH"
  --out-dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --lr "$LR"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-name "$WANDB_NAME"
  --wandb-log-train-every "${WANDB_LOG_TRAIN_EVERY:-50}"
  --wandb-log-media-every "${WANDB_LOG_MEDIA_EVERY:-5}"
  --wandb-media-samples "${WANDB_MEDIA_SAMPLES:-8}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

"$PYTHON" -m experiments.idm.model.train_idm "${TRAIN_ARGS[@]}"
