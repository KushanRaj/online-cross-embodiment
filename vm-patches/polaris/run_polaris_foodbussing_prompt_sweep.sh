#!/usr/bin/env bash
set -u
set -o pipefail

RUN_ID="${RUN_ID:-polaris-foodbussing-prompt-sweep-$(date +%Y%m%d-%H%M%S)}"
ROOT="${ROOT:-$HOME/robotics/runs/$RUN_ID}"
STATUS="$ROOT/status.tsv"
POLARIS="$HOME/robotics/repos/PolaRiS"
OPENPI="$POLARIS/third_party/openpi"
TRIALS="${TRIALS:-3}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-60}"
PORT="${PORT:-8000}"

mkdir -p "$ROOT"
printf "timestamp\tmodel\tprompt_id\tstatus\texit_code\tseconds\n" > "$STATUS"
printf "run_id\t%s\nroot\t%s\ntrials\t%s\tepisode_length_s\t%s\n" "$RUN_ID" "$ROOT" "$TRIALS" "$EPISODE_LENGTH_S" > "$ROOT/metadata.tsv"

prompt_ids=(L1 L2 L3)
declare -A prompts
prompts[L1]="Put all the foods in the bowl."
prompts[L2]="Put each food item into the bowl. Pick up the ice cream cone, the battery-shaped item, and the other loose food items one by one, and place them inside the bowl."
prompts[L3]="Move toward the nearest loose food item first. Align the gripper above it, lower carefully, close the gripper, lift it clear of the table, move over the bowl, lower into the bowl, open the gripper, and repeat this process for the remaining food items until all food items are inside the bowl."

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
  return 0
}

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export POLARIS_EPISODE_LENGTH_S="$EPISODE_LENGTH_S"

start_pi_server() {
  mkdir -p "$ROOT/pi05-polaris"
  pkill -f "scripts/serve_policy.py.*${PORT}" >/dev/null 2>&1 || true
  cd "$OPENPI" || exit 1
  uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config=pi05_droid_jointpos_polaris \
    --policy.dir=gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris \
    > "$ROOT/pi05-polaris/server.log" 2>&1 &
  echo $! > "$ROOT/pi05-polaris/server.pid"
  for _ in $(seq 1 180); do
    if grep -q "server listening" "$ROOT/pi05-polaris/server.log"; then
      return 0
    fi
    sleep 2
  done
  echo "Pi Polaris server did not become ready" >&2
  return 1
}

run_pi_polaris() {
  start_pi_server || return 1
  cd "$POLARIS" || return 1
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "pi05_polaris" "$pid" "$ROOT/pi05-polaris/$pid/eval.log" \
      uv run scripts/eval.py \
        --policy.client DroidJointPos \
        --policy.host 127.0.0.1 \
        --policy.port "$PORT" \
        --policy.open-loop-horizon 8 \
        --environment DROID-FoodBussing \
        --run-folder "$ROOT/pi05-polaris/$pid" \
        --rollouts "$TRIALS" \
        --instruction "$prompt"
  done
  kill "$(cat "$ROOT/pi05-polaris/server.pid")" >/dev/null 2>&1 || true
}

run_molmo() {
  cd "$POLARIS" || return 1
  for pid in "${prompt_ids[@]}"; do
    local prompt="${prompts[$pid]}"
    run_step "molmoact2_droid" "$pid" "$ROOT/molmoact2/$pid/eval.log" \
      env MOLMOACT2_DTYPE=bfloat16 \
      uv run scripts/eval.py \
        --policy.client MolmoAct2Droid \
        --policy.open-loop-horizon 8 \
        --environment DROID-FoodBussing \
        --run-folder "$ROOT/molmoact2/$pid" \
        --rollouts "$TRIALS" \
        --instruction "$prompt"
  done
}

run_pi_polaris
run_molmo
printf "completed_at\t%s\n" "$(date --iso-8601=seconds)" >> "$ROOT/metadata.tsv"
