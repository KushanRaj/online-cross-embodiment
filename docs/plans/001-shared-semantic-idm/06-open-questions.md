# 06 - Open Questions

## Data Quantity

How much trajectory data is enough for a diagnostic-quality IDM?

Need to track:

- number of windows,
- number of tasks,
- train/val split,
- source environments,
- action distribution,
- gripper event frequency.

## Gripper Handling

The gripper channel may be underweighted. Need to decide whether to:

- keep Smooth L1 and analyze mismatch only,
- add gripper weighting,
- add binary gripper classification.

## Generated Future Distribution

Can generated futures be added without teaching the IDM artifacts?

Need staged runs:

1. real-to-real base,
2. real-to-model only,
3. simple mixed real-to-real + real-to-model,
4. model-to-model if there is a clean source.

## Cross-Environment Generalization

Does a LIBERO IDM work on RoboCasa at all? Probably not without action/environment alignment. This should be tested only after both single-environment IDMs are clean.
