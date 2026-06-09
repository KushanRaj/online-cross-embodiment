# 04 - Experiment Design

## Goal

For an action-conditioned world model:

```text
C + a -> P(a)
execute a -> R
compare P(a), R, and IDM(C,P(a))
```

## Base Test

Use a policy action chunk:

```text
a = action chosen by Cosmos/Pi/Molmo/IDM
```

Ask the world model to predict `P(a)`.

Then execute `a` and collect `R`.

Metrics:

- P(a) vs R,
- IDM(C,P(a)) vs a,
- IDM(C,R) vs a,
- P(a) semantic substep correctness,
- task success/failure.

## Counterfactual Test

Change action:

```text
a' = modified action chunk
```

Examples:

- gripper open vs close,
- EEF direction perturbation,
- scaled motion magnitude,
- policy action from another model.

Then inspect whether the generated future changes in the expected way.
