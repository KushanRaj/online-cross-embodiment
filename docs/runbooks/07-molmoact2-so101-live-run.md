# Runbook: SO-101 MolmoAct2 Live Inference (EC2 + Local Camera/Robot)

## Scope
This runbook captures the exact setup used for:
- EC2-hosted MolmoAct2 policy server
- Local SO-101 execution client with RealSense/wrist camera
- Artifact capture (`*_in0.jpg`, `*_in1.jpg`, `action_log.jsonl`, `inference.log`)

## Environment assumptions
- SO-101 connected to local USB serial: `/dev/tty.usbmodem5A7A0558171`
- Scene cam source index: `0`
- Wrist/top cam source index: `1`
- Project path (local): `/Users/kushanraj/lossfunk-residency`
- Repo paths on EC2:
  - Repo: `/home/ubuntu/robotics/repos/molmoact2-so101`
  - Logs: `/home/ubuntu/robotics/logs`
  - Policy venv: `/home/ubuntu/robotics/repos/PolaRiS/.venv`
- SSH key: `~/.ssh/kushan-harbor.pem`
- Instance ID: `i-09d3df96cbc593e1a`

## 1) EC2 bootstrap
```bash
aws ec2 start-instances --region ap-south-1 --instance-ids i-09d3df96cbc593e1a
aws ec2 wait instance-running --region ap-south-1 --instance-ids i-09d3df96cbc593e1a
IP=$(aws ec2 describe-instances --region ap-south-1 --instance-ids i-09d3df96cbc593e1a --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "$IP"
```

## 2) Sync repo code on local side
(If clone already exists, run pull only)
```bash
cd /Users/kushanraj/lossfunk-residency
cd /Users/kushanraj/lossfunk-residency/external/molmoact2-so101
git pull
```

## 2a) Pre-flight checks (recommended)
```bash
# confirm SO-101 is connected and correct serial
python - <<'PY'
import glob
print('\\n'.join(sorted(glob.glob('/dev/tty.usbmodem*') + sorted(glob.glob('/dev/cu.usbmodem*') + sorted(glob.glob('/dev/tty.usbserial*') + sorted(glob.glob('/dev/cu.usbserial*'))))))
PY
ls -1 /dev/{tty,cu}.usbmodem* /dev/{tty,cu}.usbserial*

# confirm camera indexes (scene/wrist) match your setup
python - <<'PY'
import cv2
for i in [0, 1]:
    cap = cv2.VideoCapture(i)
    print(i, 'open?', cap.isOpened())
    if cap.isOpened():
        ok, frame = cap.read()
        print(' frame ok=', ok, 'shape=', None if not ok else frame.shape)
    cap.release()
PY
```

## 3) Start policy server on EC2
```bash
ssh -i ~/.ssh/kushan-harbor.pem -o StrictHostKeyChecking=no ubuntu@$IP 'cd /home/ubuntu/robotics/repos/molmoact2-so101
/home/ubuntu/robotics/repos/PolaRiS/.venv/bin/python tools/policy_server.py --backend molmo --host 127.0.0.1 --port 8008 --device cuda --dtype bfloat16 > /home/ubuntu/robotics/logs/molmoact2-so101-policy-server.log 2>&1 &
 echo $! > /home/ubuntu/robotics/logs/molmoact2-so101-policy-server.pid'

ssh -i ~/.ssh/kushan-harbor.pem -o StrictHostKeyChecking=no ubuntu@$IP 'curl -sf http://127.0.0.1:8008/health'
```

## 4) Start local SSH tunnel
```bash
ssh -N -L 8008:127.0.0.1:8008 -o StrictHostKeyChecking=no -i ~/.ssh/kushan-harbor.pem ubuntu@$IP
```
Use `-f` to background if needed.

## 5) Start local inference run
```bash
cd /Users/kushanraj/lossfunk-residency
RUN_DIR="external/molmoact2-so101/runs/remote_ec2_so101_live_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

env PYTHONUNBUFFERED=1 PYTHONPATH=external/molmoact2-so101 \
  .venv-so101/bin/python external/molmoact2-so101/inference.py \
  --policy-backend remote-http \
  --remote-url http://127.0.0.1:8008/predict_chunk \
  --remote-timeout 30 \
  --follower-port /dev/tty.usbmodem5A7A0558171 \
  --scene-source 0 \
  --scene-flip none \
  --wrist-source 1 \
  --wrist-flip none \
  --prompt 'pick up the pen and put it in the box' \
  --chunk-timestamp arrival \
  --min-query-period 0.1 \
  --actions-per-chunk 30 \
  --exec-hz 10 \
  --max-step-deg 3 \
  --smooth-alpha 0.7 \
  --save-frames-dir "$RUN_DIR"
```

## 6) Useful runtime checks
- `tail -f $RUN_DIR/inference.log`
- `tail -f $RUN_DIR/action_log.jsonl`
- EC2 policy log: `/home/ubuntu/robotics/logs/molmoact2-so101-policy-server.log`
- Policy health from local: `curl -i http://127.0.0.1:8008/health`

## 7) Stop and cleanup (after artifact capture)
```bash
# local
pkill -f 'external/molmoact2-so101/inference.py'
pkill -f 'ssh -N -L 8008:127.0.0.1:8008'

# verify
pgrep -af 'external/molmoact2-so101/inference.py' || true
pgrep -af 'ssh -N -L 8008:127.0.0.1:8008' || true
```

## 8) Artifact harvest sequence (all logs + runs + images + policy logs)
```bash
IP=$(aws ec2 describe-instances --region ap-south-1 --instance-ids i-09d3df96cbc593e1a --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p run-artifacts/molmoact2_bundle_${TS}/local_runs
mkdir -p run-artifacts/molmoact2_bundle_${TS}/ec2_logs
rsync -a --delete external/molmoact2-so101/runs/ run-artifacts/molmoact2_bundle_${TS}/local_runs/
rsync -a -e "ssh -i $HOME/.ssh/kushan-harbor.pem -o StrictHostKeyChecking=no" "ubuntu@$IP:/home/ubuntu/robotics/logs/" "run-artifacts/molmoact2_bundle_${TS}/ec2_logs/"
tar -czf run-artifacts/molmoact2_bundle_${TS}.tar.gz run-artifacts/molmoact2_bundle_${TS}
sha256sum run-artifacts/molmoact2_bundle_${TS}.tar.gz > run-artifacts/molmoact2_bundle_${TS}.tar.gz.sha256

# quick completeness check: include action log JSONLs and inference logs
find run-artifacts/molmoact2_bundle_${TS}/local_runs -name '*.jsonl' -type f
find run-artifacts/molmoact2_bundle_${TS}/local_runs -name '*.jpg' -type f | wc -l
find run-artifacts/molmoact2_bundle_${TS}/ec2_logs -name '*.log' -type f | wc -l
```

### Example bundle for this run
- `run-artifacts/molmoact2_bundle_20260615-133148/`
- `run-artifacts/molmoact2_bundle_20260615-133148.tar.gz`
- `run-artifacts/molmoact2_bundle_20260615-133148.tar.gz.sha256`

## 9) Known gotchas and fixes
1. `Connection refused` in inference logs
   - Usually missing local tunnel or tunnel dropped.
   - Recreate: kill old tunnel and restart step 4.

2. `[Follower] Could not connect ... device disconnected`
   - Camera/USB contention on serial port.
   - Ensure only one process owns `/dev/tty.usbmodem...`, reseat/replace cable, retry step 5.

3. Policy server occasionally logs `BrokenPipeError`
   - Usually client timeout/disconnect while request in progress.
   - Ensure tunnel is healthy and keep inference loop alive.

4. Duplicate AV foundation warnings
   - Non-blocking warnings from local cv2/av; not fatal in observed runs.

## 10) EC2 shutdown
```bash
aws ec2 stop-instances --region ap-south-1 --instance-ids i-09d3df96cbc593e1a
aws ec2 describe-instances --region ap-south-1 --instance-ids i-09d3df96cbc593e1a --query 'Reservations[0].Instances[0].State'
```

Notes:
- In some cases the instance can remain in `stopping` briefly before final `stopped` transition.
- Re-check state before any further EC2 actions.
