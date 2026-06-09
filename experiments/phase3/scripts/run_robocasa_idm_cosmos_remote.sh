#!/usr/bin/env bash
set -euo pipefail

# Remote-only orchestration:
# 1. Train RoboCasa real->real k=16 feature MLP IDM.
# 2. Run Cosmos Policy RoboCasa rollouts with Phase 3 query dumps.
# 3. Compute IDM deviation metrics and render canonical videos.
# 4. Optionally shut down the instance immediately after completion.

ROBOTICS_ROOT="${ROBOTICS_ROOT:-/home/ubuntu/robotics}"
REPO_DIR="${REPO_DIR:-${ROBOTICS_ROOT}/repos/online-cross-embodiment}"
PYTHON="${PYTHON:-${ROBOTICS_ROOT}/repos/cosmos-policy/.venv/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"

IDM_ROOT="${IDM_ROOT:-${ROBOTICS_ROOT}/runs/idm-phase1}"
IDM_OUT_DIR="${IDM_OUT_DIR:-${IDM_ROOT}/idm_siglip_robocasa_real_real_k16_${RUN_STAMP}}"
IDM_CACHE_PATH="${IDM_CACHE_PATH:-${IDM_ROOT}/features_siglip_robocasa_real_real_k16_${RUN_STAMP}.pt}"
IDM_MANIFEST_DIR="${IDM_MANIFEST_DIR:-${ROBOTICS_ROOT}/runs/idm-datasets/robocasa-demos-k16-${RUN_STAMP}}"

ROLLOUT_OUT_DIR="${ROLLOUT_OUT_DIR:-${ROBOTICS_ROOT}/runs/phase3/robocasa-cosmos-deviation-${RUN_STAMP}}"
DEVIATION_OUT_DIR="${DEVIATION_OUT_DIR:-${ROLLOUT_OUT_DIR}/deviation}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-${ROLLOUT_OUT_DIR}/prediction_metric_videos}"

TASKS="${TASKS:-CloseDrawer,CoffeeServeMug,CoffeeSetupMug,OpenDrawer,TurnOffMicrowave,TurnOffSinkFaucet,TurnOffStove,TurnOnMicrowave,TurnOnSinkFaucet,TurnOnStove,TurnSinkSpout}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

HORIZON=16 \
LIMIT="${LIMIT:-60000}" \
MANIFEST_DIR="$IDM_MANIFEST_DIR" \
CACHE_PATH="$IDM_CACHE_PATH" \
OUT_DIR="$IDM_OUT_DIR" \
WANDB_MODE="${WANDB_MODE:-optional}" \
WANDB_PROJECT="${WANDB_PROJECT:-online-cross-embodiment-idm}" \
WANDB_NAME="${WANDB_NAME:-idm-siglip-robocasa-real-real-k16-${RUN_STAMP}}" \
MAX_STEPS="${MAX_STEPS:-58000}" \
bash experiments/idm/scripts/run_robocasa_idm_remote.sh

"$PYTHON" experiments/phase3/run_robocasa_cosmos_deviation.py \
  --out-dir "$ROLLOUT_OUT_DIR" \
  --tasks "$TASKS" \
  --obj-instance-split target \
  --num-trials-per-task "$NUM_TRIALS_PER_TASK"

"$PYTHON" experiments/phase3/evaluate_online_cosmos_deviation.py \
  --input "$ROLLOUT_OUT_DIR/rollout_data" \
  --checkpoint "$IDM_OUT_DIR/best.pt" \
  --out-dir "$DEVIATION_OUT_DIR" \
  --batch-size "${EVAL_BATCH_SIZE:-32}"

"$PYTHON" experiments/phase3/make_prediction_metric_videos.py \
  --run-root "$ROLLOUT_OUT_DIR" \
  --csv "$DEVIATION_OUT_DIR/online_cosmos_deviation.csv" \
  --out-dir "$VIDEO_OUT_DIR" \
  --fps "${VIDEO_FPS:-20}" \
  --repeats "${VIDEO_REPEATS:-1}"

cat > "${ROLLOUT_OUT_DIR}/run_manifest.json" <<JSON
{
  "run_stamp": "${RUN_STAMP}",
  "idm_out_dir": "${IDM_OUT_DIR}",
  "idm_cache_path": "${IDM_CACHE_PATH}",
  "idm_manifest_dir": "${IDM_MANIFEST_DIR}",
  "rollout_out_dir": "${ROLLOUT_OUT_DIR}",
  "deviation_out_dir": "${DEVIATION_OUT_DIR}",
  "video_out_dir": "${VIDEO_OUT_DIR}",
  "tasks": "${TASKS}",
  "num_trials_per_task": ${NUM_TRIALS_PER_TASK}
}
JSON

echo "ROBOCASA_PHASE3_DONE ${ROLLOUT_OUT_DIR}"

if [[ "${SHUTDOWN_WHEN_DONE:-0}" == "1" ]]; then
  sudo shutdown -h now
fi
