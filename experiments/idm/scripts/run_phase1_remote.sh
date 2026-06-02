#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/robotics/repos/online-cross-embodiment}"
MANIFEST="${MANIFEST:-/home/ubuntu/robotics/runs/idm-datasets/libero-spatial-demos/manifest.jsonl}"
OUT_ROOT="${OUT_ROOT:-/home/ubuntu/robotics/runs/idm-phase1}"
ENCODER="${ENCODER:-google/siglip-base-patch16-224}"
LIMIT="${LIMIT:-50000}"
HORIZON="${HORIZON:-1}"
SOURCE_TYPE="${SOURCE_TYPE:-actual_actual}"
PYTHON="${PYTHON:-/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

"$PYTHON" -m experiments.idm.data.cache_features \
  --manifest "$MANIFEST" \
  --out "$OUT_ROOT/features_siglip_real_real_k${HORIZON}.pt" \
  --encoder "$ENCODER" \
  --horizon "$HORIZON" \
  --source-type "$SOURCE_TYPE" \
  --limit "$LIMIT" \
  --batch-size "${ENCODE_BATCH_SIZE:-32}"

TRAIN_ARGS=(
  --cache "$OUT_ROOT/features_siglip_real_real_k${HORIZON}.pt"
  --out-dir "$OUT_ROOT/idm_siglip_real_real_k${HORIZON}"
  --epochs "${EPOCHS:-20}"
  --batch-size "${TRAIN_BATCH_SIZE:-256}"
  --wandb-mode "${WANDB_MODE:-disabled}"
  --wandb-project "${WANDB_PROJECT:-online-cross-embodiment-idm}"
  --wandb-log-media-every "${WANDB_LOG_MEDIA_EVERY:-5}"
  --wandb-media-samples "${WANDB_MEDIA_SAMPLES:-8}"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  TRAIN_ARGS+=(--wandb-name "$WANDB_NAME")
fi

"$PYTHON" -m experiments.idm.model.train_idm \
  "${TRAIN_ARGS[@]}"
