# Adding A New Policy

## Goal

Add a policy to Phase 3 without changing the evaluation contract.

## Required Outputs

The runner must produce HDF5 fields compatible with:

```text
experiments/phase3/evaluate_online_cosmos_deviation.py
experiments/phase3/plot_deviation_diagnostics.py
experiments/phase3/make_prediction_metric_videos.py
```

At minimum:

```text
primary_images
query_t
query_future_primary_images
query_action_chunks
actions_7d
success metadata
task metadata
```

If Cosmos action head is also available:

```text
query_cosmos_action_chunks
```

## Policy Cadence

State how the policy is executed:

```text
predict N
execute M
replan cadence
how 16-step chunks are assembled
```

For Pi/Molmo current cadence:

```text
predict 10
execute 8
requery
execute 8
```

## Do Not Change

- metric names,
- HDF5 dataset meanings,
- video layout,
- horizon,
- reference action dataset,

without documenting the deviation before running.
