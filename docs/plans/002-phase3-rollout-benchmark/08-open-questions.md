# 08 - Open Questions

## Success/Failure Correlation

Current plots do not yet prove a clean separator. Need:

- more rollouts per task,
- balanced success/failure examples,
- task-specific analysis,
- statistical summaries, not only videos.

## Gripper Spike Hypothesis

Hypothesis:

```text
Most large deviations are gripper/contact spikes.
```

Needed tests:

- mark gripper-open intervals,
- compare EEF-only metrics with gripper mismatch,
- inspect training action distribution for gripper events.

## Better World Models

Need to compare:

- Cosmos Policy fine-tuned checkpoints,
- Cosmos 3 Nano/base I2V futures,
- GE-Sim or other action-conditioned futures if runnable,
- possibly DreamZero if runnable later.

## True Counterfactuals

Current probes do not force Cosmos to generate `P(a_external)`. A clean counterfactual experiment needs an action-conditioned future-generation path.
