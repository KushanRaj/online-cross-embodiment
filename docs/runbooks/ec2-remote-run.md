# EC2 Remote Runbook

## Instance

```text
instance_id = i-09d3df96cbc593e1a
region = ap-south-1
```

## Remote Paths

```text
repo = /home/ubuntu/robotics/repos/online-cross-embodiment
runs = /home/ubuntu/robotics/runs
cosmos_env = /home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python
```

## Standard Flow

1. Edit locally.
2. Commit locally.
3. Start EC2 only after the run plan is clear.
4. Pull/sync code on EC2.
5. Run remote command.
6. Download artifacts into `run-artifacts/`.
7. Verify summaries/videos locally.
8. Shut down EC2.

## Run Plan Template

Before starting, write:

```text
run name:
policy:
world model/checkpoint:
IDM checkpoint:
environment:
task list:
horizon:
output remote path:
output local path:
expected runtime:
shutdown command:
```

## Shutdown

If the user asks for shutdown after completion, make it explicit in the remote command or monitor. Do not assume a long training job will shut down unless the command was configured to do so.

## Download

Use the existing download scripts when possible:

- `experiments/phase3/scripts/download_libero_phase3_remote.sh`
- `experiments/phase3/scripts/download_robocasa_cosmos_remote.sh`

If a new artifact family is created, make the download script match the existing local `run-artifacts/<run-name>/` pattern.
