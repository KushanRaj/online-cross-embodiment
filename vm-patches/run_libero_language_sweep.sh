#!/usr/bin/env bash
set -u
set -o pipefail

RUN_ID="${RUN_ID:-libero-language-sweep-$(date +%Y%m%d-%H%M%S)}"
ROOT="${ROOT:-$HOME/robotics/runs/$RUN_ID}"
STATUS="$ROOT/status.tsv"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-0}"
TRIALS="${TRIALS:-3}"

mkdir -p "$ROOT"
printf "run_id\t%s\nroot\t%s\ntask_suite\t%s\ttask_id\t%s\ttrials\t%s\n" "$RUN_ID" "$ROOT" "$TASK_SUITE" "$TASK_ID" "$TRIALS" > "$ROOT/metadata.tsv"
printf "timestamp\tmodel\tprompt_id\tstatus\texit_code\tseconds\n" > "$STATUS"

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

for pid in "${prompt_ids[@]}"; do
  printf "%s\t%s\n" "$pid" "${prompts[$pid]}" >> "$ROOT/prompts.tsv"
done

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
  return "$code"
}

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"

run_pi() {
  local OPENPI="$HOME/robotics/repos/PolaRiS/third_party/openpi"
  local server_log="$ROOT/pi/server.log"
  local server_pid_file="$ROOT/pi/server.pid"
  mkdir -p "$ROOT/pi"
  pkill -f "scripts/serve_policy.py.*--port 8010" >/dev/null 2>&1 || true
  cd "$OPENPI"
  PYTHONPATH="$OPENPI/third_party/libero:${PYTHONPATH:-}" \
    "$OPENPI/.venv/bin/python" scripts/serve_policy.py --env LIBERO --port 8010 \
    > "$server_log" 2>&1 &
  echo $! > "$server_pid_file"
  for _ in $(seq 1 180); do
    if grep -q "server listening" "$server_log"; then
      break
    fi
    sleep 2
  done
  if ! grep -q "server listening" "$server_log"; then
    echo "OpenPI server did not become ready" >&2
    return 1
  fi
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "pi05_libero" "$pid" "$ROOT/pi/$pid/eval.log" \
      env PYTHONPATH="$OPENPI/third_party/libero:${PYTHONPATH:-}" MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" \
      "$OPENPI/.venv/bin/python" examples/libero/main.py \
        --args.host 127.0.0.1 \
        --args.port 8010 \
        --args.task-suite-name "$TASK_SUITE" \
        --args.task-ids "$TASK_ID" \
        --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$ROOT/pi/$pid/videos" \
        --args.prompt-override "$prompt" || true
  done
  kill "$(cat "$server_pid_file")" >/dev/null 2>&1 || true
}

run_molmo() {
  local OPENPI="$HOME/robotics/repos/PolaRiS/third_party/openpi"
  cd "$HOME/robotics"
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "molmoact2_libero" "$pid" "$ROOT/molmo/$pid/eval.log" \
      env PYTHONPATH="$OPENPI/third_party/libero:${PYTHONPATH:-}" MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" \
      "$OPENPI/.venv/bin/python" runs/molmoact2-libero-runner.py \
        --args.task-suite-name "$TASK_SUITE" \
        --args.task-ids "$TASK_ID" \
        --args.num-trials-per-task "$TRIALS" \
        --args.video-out-path "$ROOT/molmo/$pid/videos" \
        --args.log-path "$ROOT/molmo/$pid/eval.log.inner" \
        --args.prompt-override "$prompt" || true
  done
}

run_cosmos() {
  local REPO="$HOME/robotics/repos/cosmos-policy"
  cd "$REPO"
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "cosmos_policy" "$pid" "$ROOT/cosmos/$pid/eval.log" \
      env MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" \
      "$REPO/.venv/bin/python" -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name "$TASK_SUITE" \
        --task_ids "$TASK_ID" \
        --num_trials_per_task "$TRIALS" \
        --local_log_dir "$ROOT/cosmos/$pid/logs" \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note "$pid" \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --prompt_override "$prompt" || true
  done
}

run_ge_act() {
  local REPO="$HOME/robotics/repos/Genie-Envisioner"
  local PY="$HOME/robotics/repos/cosmos-policy/.venv/bin/python"
  cd "$REPO"
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "ge_act" "$pid" "$ROOT/ge-act/$pid/eval.log" \
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
        --prompt_override "$prompt" || true
  done
}

run_fastwam() {
  local REPO="$HOME/robotics/repos/FastWAM"
  cd "$REPO"
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "fastwam" "$pid" "$ROOT/fastwam/$pid/eval.log" \
      env MUJOCO_GL="$MUJOCO_GL" EGL_DEVICE_ID="$EGL_DEVICE_ID" \
      "$REPO/.venv/bin/python" experiments/libero/eval_libero_single.py \
        ckpt="$REPO/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
        EVALUATION.dataset_stats_path="$REPO/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json" \
        EVALUATION.task_suite_name="$TASK_SUITE" \
        EVALUATION.task_id="$TASK_ID" \
        EVALUATION.num_trials="$TRIALS" \
        EVALUATION.output_dir="$ROOT/fastwam/$pid/output" \
        "EVALUATION.prompt_override=$prompt" || true
  done
}

main() {
  run_pi
  run_molmo
  run_cosmos
  run_ge_act
  run_fastwam
  printf "completed_at\t%s\n" "$(date --iso-8601=seconds)" >> "$ROOT/metadata.tsv"
}

main "$@"
