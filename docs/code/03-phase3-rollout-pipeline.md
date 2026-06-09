# Phase 3 Rollout Pipeline

## Purpose

Phase 3 runs policies online in a simulator and records:

```text
C = current observation at query timestep
P = world-model predicted future
R = observed future after executing actions
a = selected/executed policy action chunk
```

The rollout data is then evaluated offline by the canonical evaluator and renderer.

## Canonical Runners

### Cosmos Policy On RoboCasa

```text
experiments/phase3/run_robocasa_cosmos_deviation.py
```

Meaning:

- Cosmos Policy is the executed policy.
- Cosmos also provides `P`.
- `query_action_chunks` is the first 16 actions selected from the native Cosmos action chunk.
- Native RoboCasa Cosmos chunk can be 32 x 7.

### IDM As Policy On RoboCasa

```text
experiments/phase3/run_idm_as_policy_robocasa.py
```

Meaning:

- Cosmos provides `P`.
- IDM predicts `IDM(C,P)`.
- IDM action is executed.
- `query_action_chunks` stores executed IDM action chunks.
- `query_cosmos_action_chunks` stores Cosmos's own action-head chunks for comparison.

### IDM As Policy On LIBERO

```text
experiments/phase3/run_idm_as_policy_libero.py
```

Same conceptual path as RoboCasa, but with LIBERO action/environment conventions.

### Pi / Molmo LIBERO Probe

```text
experiments/phase3/run_libero_policy_cosmos_probe.py
```

Meaning:

- Pi or Molmo executes policy actions.
- Cosmos is queried at the same observation for `P` and Cosmos action head.
- External action is not used as Cosmos action conditioning.

Default cadence:

```text
policy predicts 10 actions
execute 8
requery policy
execute 8
compare 16-step chunk
```

## HDF5 Dataset Contract

Common expected datasets:

```text
primary_images
query_primary_images
query_future_primary_images
query_t
query_action_chunks
query_cosmos_action_chunks optional
actions_7d
actions
proprio / states as runner provides
success metadata
task metadata
```

Important distinction:

```text
primary_images = dense execution frames
query_primary_images = sparse frames where model was queried
```

Canonical videos should use dense execution frames for playback and overlay the active query/chunk.

## Action Conditioning Caveat

For Pi/Molmo/IDM-as-policy probes, the current Cosmos setup is not a true counterfactual generator conditioned on the external action.

Current path:

```text
Cosmos input = current image/proprio/task
Cosmos output = P + Cosmos action head
External policy output = a
```

Not current path:

```text
Cosmos input = current image/proprio/task + external action a
Cosmos output = counterfactual P(a)
```

That distinction matters for the research claim.
