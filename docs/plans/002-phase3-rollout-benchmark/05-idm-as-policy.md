# 05 - IDM As Policy

## Experiment

Use the IDM as the executed policy:

```text
Cosmos predicts future P from current observation.
IDM predicts action chunk IDM(C,P).
Execute IDM action chunk in the environment.
Record observed future R.
```

This asks whether the world-model future plus IDM is enough to act.

## RoboCasa Run

Artifact:

```text
run-artifacts/robocasa-idm-policy-20260609-133354
```

Summary:

```text
episodes = 11
successes = 0
success_rate = 0.0
cosmos_policy_ckpt = nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
idm_checkpoint = idm_siglip_robocasa_real_real_k16_20260609-112643/best.pt
idm_type = feature
cosmos_native_chunk_size = 32
idm_reference_horizon = 16
```

## Interpretation Trap

For IDM-as-policy:

```text
query_action_chunks = executed IDM action
```

So this evaluation:

```text
IDM(C,P) vs query_action_chunks
```

is mostly a self-check.

The meaningful comparison against Cosmos's own action head uses:

```text
--reference-action-dataset query_cosmos_action_chunks
```

## Current Result

Against Cosmos action reference:

```text
episodes = 11
success_episodes = 0
idm_model_vs_selected_step_l2/mean = 0.8422
idm_model_vs_selected_step_cos_dist/mean = 0.4907
idm_model_vs_selected_eef_step_l2/mean = 0.5232
idm_model_vs_selected_eef_step_cos_dist/mean = 0.8525
idm_model_vs_selected_gripper_mismatch_rate/mean = 0.2040
```

Against self reference:

```text
selected_vs_executed_l2/mean = 0.0
```

That self-reference result should not be interpreted as good policy alignment.
