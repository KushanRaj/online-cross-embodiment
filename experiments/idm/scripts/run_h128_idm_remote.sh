#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/ubuntu/robotics/repos/online-cross-embodiment}
PY=${PY:-/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/robotics/data/libero}
RUN_ROOT=${RUN_ROOT:-/home/ubuntu/robotics/runs}
DATASET_DIR=${DATASET_DIR:-$RUN_ROOT/idm-datasets/libero_downloaded-demos-k128}
CACHE_DIR=${CACHE_DIR:-$RUN_ROOT/idm-phase1/h128}
TRAIN_ROOT=${TRAIN_ROOT:-$RUN_ROOT/idm-phase1/h128}
MAX_STEPS=${MAX_STEPS:-58000}
BATCH_SIZE=${BATCH_SIZE:-512}
WANDB_MODE=${WANDB_MODE:-optional}
WANDB_PROJECT=${WANDB_PROJECT:-online-cross-embodiment-idm}

cd "$REPO"
mkdir -p "$CACHE_DIR" "$TRAIN_ROOT"

if [[ ! -f "$DATASET_DIR/manifest.jsonl" ]]; then
  "$PY" -m experiments.idm.remote.export_libero_demo_windows \
    --input "$DATA_ROOT" \
    --out "$DATASET_DIR" \
    --horizons 128 \
    --dataset-id libero_downloaded_demos \
    --frame-transform flipud
fi

build_cache() {
  local name=$1
  local limit=$2
  local out="$CACHE_DIR/features_siglip_libero_downloaded_real_real_k128_${name}.pt"
  if [[ -f "$out" ]]; then
    echo "cache exists: $out"
    return
  fi
  if [[ "$limit" == "all" ]]; then
    "$PY" -m experiments.idm.data.cache_features \
      --manifest "$DATASET_DIR/manifest.jsonl" \
      --out "$out" \
      --horizon 128 \
      --source-type actual_actual \
      --batch-size 64 \
      --shuffle-before-limit
  else
    "$PY" -m experiments.idm.data.cache_features \
      --manifest "$DATASET_DIR/manifest.jsonl" \
      --out "$out" \
      --horizon 128 \
      --source-type actual_actual \
      --limit "$limit" \
      --batch-size 64 \
      --shuffle-before-limit
  fi
}

train_run() {
  local name=$1
  local eval_prefix=$2
  local cache="$CACHE_DIR/features_siglip_libero_downloaded_real_real_k128_${name}.pt"
  local out="$TRAIN_ROOT/idm_siglip_libero_downloaded_real_real_k128_${name}"
  "$PY" -m experiments.idm.model.train_idm \
    --cache "$cache" \
    --out-dir "$out" \
    --batch-size "$BATCH_SIZE" \
    --epochs 100000 \
    --max-steps "$MAX_STEPS" \
    --lr 3e-4 \
    --hidden-dim 512 \
    --depth 4 \
    --eval-action-prefix "$eval_prefix" \
    --action-width 7 \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "idm-k128-${name}-eval${eval_prefix}"
}

build_cache 16k 16000
build_cache 32k 32000
build_cache all all

train_run 16k 16
train_run 32k 32
train_run all 128
