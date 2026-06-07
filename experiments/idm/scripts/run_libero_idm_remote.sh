#!/usr/bin/env bash
set -euo pipefail

# Remote-only script. Exports LIBERO demo windows, caches frozen visual
# features, and trains one IDM for a named suite/horizon.
#
# Example:
#   DATASET=libero_10 HORIZON=16 MAX_STEPS=58000 WANDB_MODE=required \
#     bash experiments/idm/scripts/run_libero_idm_remote.sh

REPO_DIR="${REPO_DIR:-/home/ubuntu/robotics/repos/online-cross-embodiment}"
ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${ROBOTICS_ROOT}/data/libero}"
RUN_ROOT="${RUN_ROOT:-${ROBOTICS_ROOT}/runs/idm-phase1}"
PYTHON="${PYTHON:-${ROBOTICS_ROOT}/repos/cosmos-policy/.venv/bin/python}"

DATASET="${DATASET:-libero_10}"
DATASET_ID="${DATASET_ID:-${DATASET}_demos}"
HORIZON="${HORIZON:-16}"
LIMIT="${LIMIT:-}"
ENCODER="${ENCODER:-google/siglip-base-patch16-224}"
SOURCE_TYPE="${SOURCE_TYPE:-actual_actual}"

ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-400}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-3e-4}"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-online-cross-embodiment-idm}"
WANDB_NAME="${WANDB_NAME:-idm-siglip-${DATASET}-real-real-k${HORIZON}}"

MANIFEST_DIR="${MANIFEST_DIR:-${ROBOTICS_ROOT}/runs/idm-datasets/${DATASET}-demos-k${HORIZON}}"
CACHE_PATH="${CACHE_PATH:-${RUN_ROOT}/features_siglip_${DATASET}_real_real_k${HORIZON}.pt}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/idm_siglip_${DATASET}_real_real_k${HORIZON}}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

if [[ ! -d "${LIBERO_DATA_ROOT}/${DATASET}" ]]; then
  echo "Missing LIBERO dataset directory: ${LIBERO_DATA_ROOT}/${DATASET}" >&2
  echo "Run: DATASET=${DATASET} bash experiments/idm/scripts/download_libero_suite_remote.sh" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$MANIFEST_DIR"

"$PYTHON" experiments/idm/remote/export_libero_demo_windows.py \
  --input "${LIBERO_DATA_ROOT}/${DATASET}" \
  --out "$MANIFEST_DIR" \
  --horizons "$HORIZON" \
  --dataset-id "$DATASET_ID" \
  --frame-transform "${FRAME_TRANSFORM:-flipud}"

CACHE_ARGS=(
  --manifest "$MANIFEST_DIR/manifest.jsonl"
  --out "$CACHE_PATH"
  --encoder "$ENCODER"
  --horizon "$HORIZON"
  --source-type "$SOURCE_TYPE"
  --batch-size "$ENCODE_BATCH_SIZE"
)
if [[ -n "$LIMIT" ]]; then
  CACHE_ARGS+=(--limit "$LIMIT")
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
