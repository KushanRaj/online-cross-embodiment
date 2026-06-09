# 01 - Clean Benchmark

## Required Data Per Query

Every query point should support:

```text
C = current observation
P = world-model predicted future at horizon k
R = observed future at t + k
a = selected/executed policy action chunk
optional a_model = model's own action head
```

For this project, base `k = 16`.

## Required Files

Each run should produce:

```text
summary.json
episodes.csv
rollout_data/*.hdf5
deviation/summary.json
deviation/online_cosmos_deviation.csv
deviation/diagnostics_success_failure/*.png
prediction_metric_videos/*.mp4
index.html if locally reviewed
```

## Required Run Metadata

Record:

- environment,
- policy,
- world model/checkpoint,
- IDM checkpoint/type,
- horizon,
- action dimension,
- task list,
- success count,
- reference action dataset,
- preprocessing convention,
- policy execution cadence.

## Required Interpretation

For any claim, state:

```text
Which action was executed?
Which future was predicted?
Which action dataset was used as reference?
Was Cosmos conditioned on the external action or only current state/task?
```
