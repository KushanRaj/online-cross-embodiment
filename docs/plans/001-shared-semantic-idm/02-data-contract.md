# 02 - Data Contract

## Base Sample

Each sample should contain:

```text
sample_id
task_name
trajectory_id
t
horizon = 16
current_primary_image
future_primary_image
current_wrist_image optional
future_wrist_image optional
current_proprio optional
action_chunk_16x7
success/failure metadata optional
source
```

Success/failure is metadata. It is not an IDM input.

## Base Source

Base source is real-to-real trajectory windows:

```text
C = observed current frame
F = observed future frame at t + 16
target = executed action chunk t:t+16
```

LIBERO uses demonstration trajectories. RoboCasa uses rollout/dataset trajectories aligned with the Cosmos Policy RoboCasa action convention.

## Generated-Future Sources

Generated futures are ablations, not base:

```text
real -> model
model -> model
model -> real
```

The motivation is to expose the IDM to the distribution of model-generated images. The risk is teaching the IDM to follow model artifacts.

## Reverse Data

Reverse data was discussed:

```text
given future and past frame, predict reverse action
```

This should stay as an ablation. Do not mix reverse samples into the base run without a named experiment.

## Dataset Size Principle

Match the existing LIBERO base scale first. Then expand.

Do not change:

- dataset size,
- horizon,
- encoder,
- model architecture,
- optimizer,
- loss,

in the same comparison claim.
