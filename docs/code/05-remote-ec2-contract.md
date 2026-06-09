# Remote EC2 Contract

## Default Instance

```text
instance_id = i-09d3df96cbc593e1a
region = ap-south-1
ssh = ssh -i ~/.ssh/kushan-harbor.pem ubuntu@ec2-65-2-38-11.ap-south-1.compute.amazonaws.com
```

Instance state should normally be stopped when no remote work is running.

## Remote Paths

Repo:

```text
/home/ubuntu/robotics/repos/online-cross-embodiment
```

Runs:

```text
/home/ubuntu/robotics/runs/
```

Cosmos Policy repo/env:

```text
/home/ubuntu/robotics/repos/cosmos-policy
/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python
```

Checkpoints:

```text
/home/ubuntu/robotics/checkpoints/
```

## Workflow

1. Edit code locally.
2. Commit locally when the unit is coherent.
3. Sync or pull code on EC2.
4. Run remote-heavy data/model/sim work.
5. Download outputs to `run-artifacts/`.
6. Verify summaries/videos locally when needed.
7. Shut down EC2.

## Shutdown Rule

If a remote run is started, the run plan must state:

- expected runtime,
- output directory,
- whether artifacts need downloading,
- shutdown command or backup automation.

Do not leave the instance running by default.

## Remote Python Rule

Remote Cosmos jobs should usually use the Cosmos env:

```bash
/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python <script>
```

Local scripts should use local `uv`, not this remote env.
