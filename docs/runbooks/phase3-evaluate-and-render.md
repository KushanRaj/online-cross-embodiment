# Phase 3 Evaluate And Render Runbook

## Inputs

You need:

```text
rollout_data/*.hdf5
IDM checkpoint
IDM type
encoder
horizon
reference action dataset
```

## Step 1 - Evaluate

Use:

```text
experiments/phase3/evaluate_online_cosmos_deviation.py
```

Important flag:

```bash
--reference-action-dataset query_action_chunks
```

or:

```bash
--reference-action-dataset query_cosmos_action_chunks
```

Use `query_cosmos_action_chunks` for IDM-as-policy when comparing against Cosmos's native action head.

## Step 2 - Plot

Use:

```text
experiments/phase3/plot_deviation_diagnostics.py
```

Expected:

- success/failure color coding,
- L2,
- per-step cosine,
- EEF cosine,
- gripper mismatch.

Do not make L2-only aggregate plots if the user asked for all metrics.

## Step 3 - Render Videos

Use:

```text
experiments/phase3/make_prediction_metric_videos.py
```

Expected layout:

```text
left = dense execution frame
middle = predicted future for active chunk
right = metric trace
green = active dot/chunk
red = history/current line
gray = future trace
```

Do not rewrite this layout.

## Step 4 - Static Review Page

If an HTML index is needed, point it at the generated videos and plots. Do not change the generated video layout to make HTML easier.

## Non-Phase-3 Live-Control Bundles

SO101/MolmoAct2 live-control bundles do not follow the Phase 3 HDF5 layout. They
store camera JPGs and `action_log.jsonl` files under:

```text
external/molmoact2-so101/runs/<run-name>/
```

or, after download:

```text
run-artifacts/molmoact2_bundle_20260615-133148/local_runs/
```

Do not adapt the Phase 3 renderer by copying it into a run artifact folder. If
the SO101 camera/action-trace video is needed again, move the temporary bundle
renderer into tracked SO101/MolmoAct2 tooling first, then run that tracked tool.

## Sanity Checklist

- Does the video use dense `primary_images`, not sparse query frames?
- Does the plot include the metric the user requested?
- Is success/failure color-coded?
- Is the reference action dataset correct?
- Does `summary.json` record checkpoint and IDM type?
