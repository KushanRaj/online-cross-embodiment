# Metrics And Video Contract

## Canonical Evaluator

Use:

```text
experiments/phase3/evaluate_online_cosmos_deviation.py
```

It reads rollout HDF5 files and writes:

```text
online_cosmos_deviation.csv
summary.json
```

It supports:

```bash
--reference-action-dataset query_action_chunks
--reference-action-dataset query_cosmos_action_chunks
```

Use `query_cosmos_action_chunks` when the executed policy is IDM and the goal is to compare IDM policy with Cosmos's own action head.

## Metrics

Visual future metric:

```text
SigLIP(P) vs SigLIP(R)
```

Action metrics:

```text
IDM(C,P) vs a
IDM(C,R) vs a
IDM(C,P) vs IDM(C,R)
```

Report:

- chunk L2,
- per-step L2,
- per-step cosine distance,
- EEF per-step L2,
- EEF per-step cosine distance,
- gripper step L1,
- gripper sign mismatch rate.

## Cosine Rule

Flattened 16x7 cosine is legacy and should not drive interpretation.

Preferred:

```text
for each step:
  cosine(action_step_pred, action_step_ref)
then average across the chunk
```

For EEF:

```text
cosine(pred_step[:6], ref_step[:6])
```

## Canonical Plotter

Use:

```text
experiments/phase3/plot_deviation_diagnostics.py
```

Expected plot features:

- success/failure color coding,
- aggregate and per-task views where possible,
- L2, cosine, EEF, gripper mismatch,
- no hidden metric substitutions.

## Canonical Video Renderer

Use:

```text
experiments/phase3/make_prediction_metric_videos.py
```

Required video layout:

```text
left: dense execution frame
middle: Cosmos predicted future for active query chunk
right: metric traces
caption: success/failure, task, and active frame/chunk
```

Trace styling:

- past/current trace in red,
- active dot or active chunk in green,
- future trace in light gray,
- green band marks the chunk being executed now.

Do not replace this with:

- query-frame-only execution video,
- separate execution and prediction columns without the metric trace,
- new metric names with old labels.

## Known Renderer Bug To Avoid

Bad version:

```text
execution panel used query_primary_images
```

This made the execution look frame-by-frame/sparse.

Correct version:

```text
execution panel uses primary_images
active query/chunk is overlaid on dense execution time
```
