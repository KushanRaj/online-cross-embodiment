#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/robotics/repos/online-cross-embodiment}
PY=${PY:-/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/robotics/data/libero}
RUN_ROOT=${RUN_ROOT:-/home/ubuntu/robotics/runs}
HORIZONS=${HORIZONS:-16,32,128}
DATASET_DIR=${DATASET_DIR:-$RUN_ROOT/idm-datasets/libero_downloaded-demos-mixed-k16-32-128}
CACHE_DIR=${CACHE_DIR:-$RUN_ROOT/idm-phase1/mixed-horizon}
TRAIN_ROOT=${TRAIN_ROOT:-$RUN_ROOT/idm-phase1/mixed-horizon}
CACHE_PATH=${CACHE_PATH:-$CACHE_DIR/features_siglip_libero_downloaded_real_real_mixed_k16_32_128.pt}
OUT_DIR=${OUT_DIR:-$TRAIN_ROOT/idm_siglip_libero_downloaded_real_real_mixed_k16_32_128}
MAX_STEPS=${MAX_STEPS:-58000}
BATCH_SIZE=${BATCH_SIZE:-512}
IDM_WANDB_MODE=${IDM_WANDB_MODE:-required}
WANDB_PROJECT=${WANDB_PROJECT:-online-cross-embodiment-idm}
WANDB_NAME=${WANDB_NAME:-idm-siglip-libero-mixed-k16-32-128}

cd "$REPO"
mkdir -p "$CACHE_DIR" "$TRAIN_ROOT"

if [[ ! -f "$DATASET_DIR/manifest.jsonl" ]]; then
  "$PY" -m experiments.idm.remote.export_libero_demo_windows \
    --input "$DATA_ROOT" \
    --out "$DATASET_DIR" \
    --horizons "$HORIZONS" \
    --dataset-id libero_downloaded_demos \
    --frame-transform flipud
fi

if [[ ! -f "$CACHE_PATH" ]]; then
  "$PY" -m experiments.idm.data.cache_features \
    --manifest "$DATASET_DIR/manifest.jsonl" \
    --out "$CACHE_PATH" \
    --horizons "$HORIZONS" \
    --source-type actual_actual \
    --batch-size 64 \
    --shuffle-before-limit
fi

"$PY" -m experiments.idm.model.train_idm \
  --cache "$CACHE_PATH" \
  --out-dir "$OUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --epochs 100000 \
  --max-steps "$MAX_STEPS" \
  --lr 3e-4 \
  --hidden-dim 512 \
  --depth 4 \
  --action-width 7 \
  --wandb-mode "$IDM_WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$WANDB_NAME" \
  --wandb-log-train-every "${WANDB_LOG_TRAIN_EVERY:-50}" \
  --wandb-log-media-every "${WANDB_LOG_MEDIA_EVERY:-5}" \
  --wandb-media-samples "${WANDB_MEDIA_SAMPLES:-8}"
