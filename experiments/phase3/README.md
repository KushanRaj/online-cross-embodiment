# Phase 3 Clean Rollout Benchmark

This folder tracks the clean task benchmark for the prediction-vs-reality
deviation experiment.

The goal is to keep the benchmark runnable on EC2 without putting datasets,
checkpoints, rollout videos, or generated futures in git.

## What Is Tracked

- `tasks/phase3_clean_tasks.yaml` - selected tasks and paper-backed evidence.
- `scripts/setup_phase3_remote.sh` - clone/check environment repos and write the
  task manifest into the EC2 robotics data root.
- `scripts/download_libero_phase3_remote.sh` - remote-only LIBERO dataset
  downloader for the LIBERO-10 track.
- `scripts/download_robocasa_cosmos_remote.sh` - remote-only RoboCasa/Cosmos
  dataset downloader.
- `scripts/setup_robotwin_remote.sh` - remote-only RoboTwin repo/data setup.
- `scripts/map_robotwin_selected_remote.sh` - remote-only RoboTwin metadata
  mapper and selected-shard downloader.
- `scripts/setup_robolab_remote.sh` - remote-only RoboLab repo setup and
  selected-task manifest writer.
- `scripts/verify_phase3_remote.sh` - non-evaluation readiness check.
- `tasks/robotwin_selected_mapping.json` - exact RoboTwin selected-task shard
  manifest generated from `lerobot/robotwin_unified` metadata.

## Remote Contract

Default EC2 layout:

```text
/home/ubuntu/robotics/repos
/home/ubuntu/robotics/data
/home/ubuntu/robotics/data/phase3
```

The setup scripts are safe to run on the EC2 instance. They do not run
policies, rollouts, training jobs, or evaluations.

## Tracks

1. Common clean track: `LIBERO-10`.
2. Cosmos-native clean track: RoboCasa task families.
3. Fast-WAM / LingBot-native clean track: RoboTwin tasks.
4. GE-native clean track: GE/RoboTwin overlap and GE paper tasks.
5. RoboLab probe track: Isaac Lab / Isaac Sim tasks for later stress testing.

Perturbations are not the core Phase 3 benchmark. They stay as a secondary
extension after the clean benchmark has success, moderate, and failure cases.

## RoboTwin Selected-Shard Mapping

The selected RoboTwin Phase 3 task set maps to 13 task labels, 158 unique
metadata/data/video files, and `22.42 GiB` of selected download from
`lerobot/robotwin_unified`. The remote mapping manifest lives at
`/home/ubuntu/robotics/data/robotwin/phase3_selected_mapping.json`; the tracked
copy is `tasks/robotwin_selected_mapping.json`.

The mapping script defaults to metadata-only mode. Set `DOWNLOAD=1` only after
reviewing the selected paths and storage estimate.

## Remote Download Status

As of 2026-06-02 on the EC2 host:

- LIBERO-10 is downloaded at `/home/ubuntu/robotics/data/libero/libero_10`
  with 10 HDF5 files, about `13G`.
- Selected RoboCasa/Cosmos families are downloaded at
  `/home/ubuntu/robotics/data/robocasa-cosmos-policy` with 542 non-cache files,
  about `5.6G`.
- Selected RoboTwin shards are downloaded at
  `/home/ubuntu/robotics/data/robotwin/selected_phase3` with 158 non-cache
  files, about `23G`.
- RoboLab is cloned at `/home/ubuntu/robotics/repos/RoboLab`, about `13G`.
  Selected probe tasks are recorded at
  `/home/ubuntu/robotics/data/robolab/phase3_selected_tasks.txt`.

No policy rollouts, evaluations, training jobs, Isaac runtime setup, or `uv sync`
were run by these setup/download steps.
