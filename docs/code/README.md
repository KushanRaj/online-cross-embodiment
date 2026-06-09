# Code Documentation Index

This folder answers: "where is the code, what does each piece own, and what must not change silently?"

## Files

| # | Doc | Purpose |
|---|-----|---------|
| 01 | [repo-map](01-repo-map.md) | Directory-level map and important tracked/untracked locations. |
| 02 | [idm-pipeline](02-idm-pipeline.md) | IDM data export, feature cache, model, training, W&B, and adapter. |
| 03 | [phase3-rollout-pipeline](03-phase3-rollout-pipeline.md) | Online rollout runners and HDF5 dataset contracts. |
| 04 | [metrics-and-video-contract](04-metrics-and-video-contract.md) | Metrics, aggregate plots, and canonical video renderer. |
| 05 | [remote-ec2-contract](05-remote-ec2-contract.md) | Remote instance, repo paths, env paths, sync, shutdown. |
| 06 | [artifact-layout](06-artifact-layout.md) | Local and remote artifact folder contracts. |

## Read Order For Code Changes

1. Read this README.
2. Read the file for the subsystem you are touching.
3. Open the exact source files named in that subsystem doc.
4. Check [../reflections.md](../reflections.md) for prior mistakes.

## Canonical Rule

If the task is "run the same experiment with a different model/checkpoint," start by swapping the model/checkpoint in an existing runner. Do not write a new pipeline unless the existing HDF5 contract cannot represent the new data.
