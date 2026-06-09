# 2026-06-08 - Pi/Molmo LIBERO Cosmos Probes

Artifact:

```text
run-artifacts/pi05-molmo-libero-cosmos-proper-20260608
```

## Purpose

Run Pi 0.5 and MolmoAct2 as external policies in LIBERO while querying Cosmos for future/action probes at the same observations.

## Pi 0.5 Summary

```text
episodes = 10
successes = 9
success_rate = 0.9
horizon = 16
policy_prediction_steps = 10
policy_execute_steps = 8
cosmos_ckpt = nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

## MolmoAct2 Summary

```text
episodes = 10
successes = 9
success_rate = 0.9
horizon = 16
policy_prediction_steps = 10
policy_execute_steps = 8
cosmos_ckpt = nvidia/Cosmos-Policy-LIBERO-Predict2-2B
```

## Interpretation

The diagnostic asks:

```text
Does the external policy action agree with the action implied by Cosmos's imagined future?
```

It does not ask:

```text
What future would Cosmos predict if conditioned on the external policy action?
```

## Caveat

These runs use a patch-transformer IDM checkpoint in the available summaries. Compare against MLP baseline only when a matching MLP summary exists.
