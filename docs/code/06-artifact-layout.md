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

SO101 / MolmoAct2 live-control bundle:

```text
run-artifacts/molmoact2_bundle_20260615-133148
```

This bundle was downloaded from the local SO101 laptop run plus EC2 logs. It is
not a Phase 3 HDF5/Cosmos deviation artifact. The raw live-control runs are
mirrored under:

```text
run-artifacts/molmoact2_bundle_20260615-133148/local_runs/
```

Each raw run may contain:

```text
*_in0.jpg
*_in1.jpg
action_log.jsonl
```

`*_in0.jpg` and `*_in1.jpg` are the two preprocessed camera inputs saved during
the local orchestrator run. `action_log.jsonl` records policy chunks, target
commands sent to the robot, and hold commands during model/robot timing gaps.

The bundle also contains:

```text
run-artifacts/molmoact2_bundle_20260615-133148/ec2_logs/
run-artifacts/molmoact2_bundle_20260615-133148/rendered/
```

`ec2_logs/` is the remote policy-server log dump. `rendered/` contains derived
review artifacts generated after download:

```text
camera_side_by_side_4fps.mp4
camera_with_action_trace_4fps.mp4
chunks.csv
targets.csv
summary.json
```

The main long live-control review video currently lives at:

```text
run-artifacts/molmoact2_bundle_20260615-133148/rendered/remote_ec2_so101_hardware_20260615_131850/camera_with_action_trace_4fps.mp4
```

Important: the renderer used for this bundle was created inside the artifact
folder as a temporary recovery script. It is not the canonical renderer. If this
view is needed again, promote it into the tracked SO101/MolmoAct2 codebase and
reuse it from there rather than creating another bundle-local script.

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
