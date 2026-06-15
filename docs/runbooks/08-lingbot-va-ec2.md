# LingBot-VA EC2 Setup

Use this when running LingBot-VA on the robotics EC2 box for LIBERO rollouts or base image-to-video-action probes.

## Remote State

Host:

```bash
ssh -i ~/.ssh/kushan-harbor.pem ubuntu@13.233.128.151
```

Paths:

```text
repo:        /home/ubuntu/robotics/repos/lingbot-va
venv:        /home/ubuntu/robotics/repos/lingbot-va/.venv
base ckpt:   /home/ubuntu/robotics/checkpoints/lingbot/lingbot-va-base
LIBERO ckpt: /home/ubuntu/robotics/checkpoints/lingbot/lingbot-va-posttrain-libero-long
logs:        /home/ubuntu/robotics/logs
runs:        /home/ubuntu/robotics/runs
```

The remote checkout has local setup patches:

- `wan_va/modules/model.py` tolerates missing `flash_attn` when `attn_mode="torch"`.
- `wan_va/configs/va_demo_cfg.py` points to the base checkpoint.
- `wan_va/configs/va_libero_cfg.py` points to the LIBERO posttrain checkpoint.
- checkpoint transformer configs are set to `attn_mode: torch`.
- `evaluation/libero/launch_client.sh` exports the LingBot and LIBERO paths.
- `evaluation/libero/client.py` has a local `write_json` fallback so LeRobot is not required just for result JSONs.

## Base I2VA Probe

The staged SO-101 input folder is:

```text
/home/ubuntu/robotics/inputs/lingbot-so101-pen-box
```

It contains the required demo camera keys:

```text
observation.images.top.png
observation.images.wrist.png
```

Run:

```bash
cd /home/ubuntu/robotics/repos/lingbot-va
. .venv/bin/activate
PYTHONPATH=/home/ubuntu/robotics/repos/lingbot-va \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port 29581 \
  --tee 3 \
  -m wan_va.wan_va_server \
  --config-name demo_i2av \
  --save_root /home/ubuntu/robotics/runs/lingbot-base-so101-i2va
```

Smoke output from setup:

```text
remote: /home/ubuntu/robotics/runs/lingbot-base-so101-i2va-smoke-20260615-121950/demo.mp4
local:  /Users/kushanraj/lossfunk-residency/run-artifacts/lingbot-setup-20260615/lingbot_base_so101_i2va_demo.mp4
```

## LIBERO Rollout

Start the LingBot LIBERO server:

```bash
cd /home/ubuntu/robotics/repos/lingbot-va
. .venv/bin/activate
PYTHONPATH=/home/ubuntu/robotics/repos/lingbot-va:/home/ubuntu/robotics/repos/LIBERO \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port 29582 \
  --tee 3 \
  -m wan_va.wan_va_server \
  --config-name libero \
  --port 29056 \
  --save_root /home/ubuntu/robotics/runs/lingbot-libero
```

In a second shell, run the client:

```bash
cd /home/ubuntu/robotics/repos/lingbot-va
. .venv/bin/activate
PYTHONPATH=/home/ubuntu/robotics/repos/lingbot-va:/home/ubuntu/robotics/repos/LIBERO \
python evaluation/libero/client.py \
  --libero-benchmark libero_10 \
  --port 29056 \
  --test-num 50 \
  --task-range 0 10 \
  --out-dir /home/ubuntu/robotics/runs/lingbot-libero-rollouts
```

Readiness checks completed on 2026-06-15:

- Base I2VA generated a `demo.mp4`.
- LIBERO client `--help` imports through LIBERO and robosuite.
- LIBERO server loaded the posttrain checkpoint and listened on `0.0.0.0:29056`; the smoke run was stopped by timeout after readiness.

Do not stop the EC2 instance if another policy server or user-owned GPU job is active.
