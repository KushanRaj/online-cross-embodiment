# 2026-06-09 - RoboCasa IDM As Policy

Artifact:

```text
run-artifacts/robocasa-idm-policy-20260609-133354
```

## Purpose

Use Cosmos to predict a future and use the IDM to convert that future into the executed action.

```text
Cosmos: C -> P
IDM: C,P -> action
environment executes IDM action
```

## Run Summary

```text
mode = idm_as_policy_from_robocasa_cosmos_future
episodes = 11
successes = 0
success_rate = 0.0
cosmos_policy_ckpt = nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
idm_checkpoint = idm_siglip_robocasa_real_real_k16_20260609-112643/best.pt
idm_type = feature
cosmos_native_chunk_size = 32
idm_reference_horizon = 16
```

## Correct Reference

Self-reference:

```text
reference_action_dataset = query_action_chunks
meaning = compare IDM(C,P) to the IDM action that was executed
```

Cosmos-action reference:

```text
reference_action_dataset = query_cosmos_action_chunks
meaning = compare IDM(C,P) to Cosmos's own action-head chunk
```

Use the Cosmos-action reference for policy comparison.

## Cosmos-Reference Highlights

```text
n = 353
success_episodes = 0
siglip_pred_observed_l2/mean = 0.4383
siglip_pred_observed_cos/mean = 0.9003
idm_model_vs_selected_step_l2/mean = 0.8422
idm_model_vs_selected_step_cos_dist/mean = 0.4907
idm_model_vs_selected_eef_step_l2/mean = 0.5232
idm_model_vs_selected_eef_step_cos_dist/mean = 0.8525
idm_model_vs_selected_gripper_mismatch_rate/mean = 0.2040
selected_vs_executed_l2/mean = 3.5169
```

## Main Lesson

The IDM-as-policy loop does not solve RoboCasa. It is still useful because it exposes how far "world-model future -> inverse dynamics action" is from a stable closed-loop policy.
