# 2026-06-09 - RoboCasa Cosmos Policy

Artifact:

```text
run-artifacts/robocasa-cosmos-deviation-20260609-112643
```

## Purpose

Run RoboCasa with the RoboCasa fine-tuned Cosmos Policy checkpoint and compute the same IDM deviation diagnostics used for LIBERO.

## Run Summary

```text
mode = cosmos_policy_robocasa_deviation
policy_ckpt = nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
episodes = 11
successes = 4
success_rate = 0.3636
num_trials_per_task = 1
policy_native_chunk_size = 32
idm_reference_horizon = 16
```

## Tasks

```text
CloseDrawer
CoffeeServeMug
CoffeeSetupMug
OpenDrawer
TurnOffMicrowave
TurnOffSinkFaucet
TurnOffStove
TurnOnMicrowave
TurnOnSinkFaucet
TurnOnStove
TurnSinkSpout
```

## Deviation Summary Highlights

```text
n = 275
success_episodes = 4
siglip_pred_observed_l2/mean = 0.4316
siglip_pred_observed_cos/mean = 0.9047
idm_model_vs_selected_step_l2/mean = 0.8966
idm_model_vs_selected_step_cos_dist/mean = 0.5671
idm_model_vs_selected_eef_step_l2/mean = 0.4771
idm_model_vs_selected_eef_step_cos_dist/mean = 0.6576
idm_model_vs_selected_gripper_mismatch_rate/mean = 0.2807
```

## Caveats

- Only one rollout per task.
- Needs repeated trials for success/failure correlation.
- Use RoboCasa IDM checkpoint for interpretation.
