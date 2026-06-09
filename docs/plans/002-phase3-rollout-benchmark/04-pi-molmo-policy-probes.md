# 04 - Pi/Molmo Policy Probes

## Purpose

Run an external policy in LIBERO while using Cosmos as a world-model/action-head probe.

```text
external policy = Pi 0.5 or MolmoAct2
world model probe = Cosmos Policy LIBERO checkpoint
diagnostic = IDM(C,P_cosmos) vs external policy action a
```

## Current Artifact

```text
run-artifacts/pi05-molmo-libero-cosmos-proper-20260608
```

Pi 0.5:

```text
episodes = 10
successes = 9
success_rate = 0.9
policy_prediction_steps = 10
policy_execute_steps = 8
horizon = 16
```

MolmoAct2:

```text
episodes = 10
successes = 9
success_rate = 0.9
policy_prediction_steps = 10
policy_execute_steps = 8
horizon = 16
```

## Cadence

The runner assembles a 16-step comparison chunk using two policy prefixes:

```text
predict 10
execute 8
requery
execute 8
```

## Important Caveat

Cosmos is not action-conditioned on the external policy action here. It is asked what future/action it predicts from the current state/task. Therefore:

```text
IDM(C,P_cosmos) vs a_external
```

means:

```text
Does the external policy action agree with the Cosmos-imagined future?
```

It does not mean:

```text
What future would Cosmos predict if forced to take the external action?
```
