# Artifact Layout

## Local Run Artifacts

Local downloaded artifacts live under:

```text
run-artifacts/<run-name>/
```

This directory is ignored by git. Do not commit generated videos, HDF5 files, or large run outputs.

## Common Files

Typical run folder:

```text
summary.json
episodes.csv
run_manifest.json
rollout_data/*.hdf5
deviation*/online_cosmos_deviation.csv
deviation*/summary.json
deviation*/diagnostics_success_failure/*.png
prediction_metric_videos*/prediction_metrics_*.mp4
index.html
```

## Recent Artifact Anchors

LIBERO Cosmos expanded:

```text
run-artifacts/phase3-cosmos-libero10-expanded-20260603-161058
```

Pi/Molmo LIBERO probes:

```text
run-artifacts/pi05-molmo-libero-cosmos-proper-20260608
```

RoboCasa Cosmos Policy:

```text
run-artifacts/robocasa-cosmos-deviation-20260609-112643
```

RoboCasa IDM as policy:

```text
run-artifacts/robocasa-idm-policy-20260609-133354
```

Cosmos 3 Nano smoke tests:

```text
run-artifacts/ec2-cosmos3-nano-libero-i2v-smoke-20260604
run-artifacts/cosmos3-nano-robocasa-i2v-smoke-20260608
run-artifacts/cosmos3-nano-robocasa-i2v-long-400f-direct-20260608-133338
```

## Artifact Use In Claims

When summarizing a run, cite:

- local artifact folder,
- remote source path if available,
- checkpoint,
- policy,
- task suite,
- success count,
- reference action dataset,
- IDM checkpoint/type,
- horizon.

Do not claim a metric without checking the run's `summary.json`, because some runs use different IDM types or reference action datasets.
