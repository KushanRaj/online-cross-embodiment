#!/usr/bin/env bash
set -u
set -o pipefail

RUN_ID="${RUN_ID:-libero-language-sweep-20260529-1120}"
ROOT="${ROOT:-$HOME/robotics/runs/$RUN_ID}"
STATUS="${STATUS:-$ROOT/status_ge_fastwam_resume.tsv}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
TRIALS="${TRIALS:-3}"

mkdir -p "$ROOT"
if [ ! -f "$STATUS" ]; then
  printf "timestamp\tmodel\tprompt_id\tstatus\texit_code\tseconds\n" > "$STATUS"
fi

prompt_ids=(P0 P1 P2 P3 C1 C2 C3 C4 O1 O2)
declare -A prompts
prompts[P0]="Pick up the black bowl between the plate and the ramekin and place it on the plate."
prompts[P1]="Take the black bowl that is between the plate and the ramekin, and put it onto the plate."
prompts[P2]="Find the black bowl positioned between the plate and the ramekin. Grasp that black bowl and place it on top of the plate."
prompts[P3]="Move the gripper above the black bowl located between the plate and the ramekin. Lower the gripper, grasp the black bowl, lift it without touching the ramekin, move over the plate, lower the bowl onto the plate, and release it."
prompts[C1]="Pick up the ramekin and place it on the plate."
prompts[C2]="Pick up the black bowl and place it next to the plate, not on it."
prompts[C3]="Pick up the plate and place it near the black bowl."
prompts[C4]="Pick up the black bowl without touching the ramekin, then place the bowl on the plate."
prompts[O1]="Move near the black bowl."
prompts[O2]="Do not pick up anything. Keep the gripper away from the objects."

log_status() {
  local model="$1"
  local prompt_id="$2"
  local status="$3"
  local code="$4"
  local seconds="$5"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "$model" "$prompt_id" "$status" "$code" "$seconds" | tee -a "$STATUS"
}

run_step() {
  local model="$1"
  local prompt_id="$2"
  local log="$3"
  shift 3
  local start
  local end
  local code
  start=$(date +%s)
  mkdir -p "$(dirname "$log")"
  printf "\n===== %s %s START %s =====\n" "$model" "$prompt_id" "$(date --iso-8601=seconds)" | tee -a "$log"
  "$@" >> "$log" 2>&1
  code=$?
  end=$(date +%s)
  if [ "$code" -eq 0 ]; then
    log_status "$model" "$prompt_id" "ok" "$code" "$((end-start))"
  else
    log_status "$model" "$prompt_id" "fail" "$code" "$((end-start))"
  fi
  return 0
}

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

run_ge_act() {
  local REPO="$HOME/robotics/repos/Genie-Envisioner"
  local PY="$HOME/robotics/repos/cosmos-policy/.venv/bin/python"
  cd "$REPO" || return 1
  for pid in P1 P2 P3 C1 C2 C3 C4 O1 O2; do
    local prompt="${prompts[$pid]}"
    run_step "ge_act" "$pid" "$ROOT/ge-act/$pid/eval.resume.log" \
      env PYTHONPATH="$REPO:${PYTHONPATH:-}" MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" \
      "$PY" experiments/eval_libero.py \
        --config_file configs/ltx_model/libero/action_model_libero.local.yaml \
        --output_dir "$ROOT/ge-act/$pid" \
        --ckpt_path checkpoints/agibot_world/Genie-Envisioner/ge_act_libero_spatial.safetensors \
        --exec_step 8 \
        --task_suite_name "$TASK_SUITE" \
        --device 0 \
        --num_trails_per_task "$TRIALS" \
        --task_ids "$TASK_ID" \
        --threshold 30 \
        --prompt_override "$prompt"
  done
}

run_fastwam() {
  local REPO="$HOME/robotics/repos/FastWAM"
  cd "$REPO" || return 1
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "fastwam" "$pid" "$ROOT/fastwam/$pid/eval.log" \
      env MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" FASTWAM_PROMPT_OVERRIDE="$prompt" \
      "$REPO/.venv/bin/python" experiments/libero/eval_libero_single.py \
        ckpt="$REPO/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
        EVALUATION.dataset_stats_path="$REPO/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.task_id="$TASK_ID" \
        EVALUATION.num_trials="$TRIALS" \
        EVALUATION.output_dir="$ROOT/fastwam/$pid/output"
  done
}

run_ge_act
run_fastwam
printf "ge_fastwam_resume_completed_at\t%s\n" "$(date --iso-8601=seconds)" >> "$ROOT/metadata.tsv"
