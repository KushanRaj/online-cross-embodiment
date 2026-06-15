# 07 - LIBERO Non-Panda Policy Adapters

## Goal

Run LIBERO-trained policies in the same LIBERO task scene while replacing the
default Franka/Panda arm with another single-arm robosuite robot.

This is the clean version of the current question:

```text
same LIBERO task
same task prompt
same camera names
same 7D OSC_POSE action interface
different robot model
```

It is not a claim that the checkpoint is trained for the target robot. It is a
test of how far the shared EEF-style control contract gets us before robot
geometry, reset pose, camera occlusion, proprio distribution, and controller
calibration break the policy.

## Adapter Added

Shared helper:

```text
experiments/cross_embodiment/libero_robot_adapter.py
```

What it does:

- registers `MountedJaco`, `MountedKinova3`, `MountedUR5e`, and `MountedIIWA`
  aliases because LIBERO tabletop tasks internally wrap robot names as
  `Mounted{Robot}`;
- creates `OffScreenRenderEnv` with explicit `robots=[robot]`,
  `controller="OSC_POSE"`, and the canonical LIBERO cameras;
- handles LIBERO initial states for non-Panda robots by copying object joint
  state from the Panda benchmark env by joint name while leaving the target
  robot in its native reset pose.

Why this matters:

LIBERO stored initial states are full Panda simulator state vectors. Directly
calling `env.set_init_state(panda_state)` on UR5e/Jaco/Kinova/IIWA fails because
the robot qpos/qvel widths differ.

## Smoke Result

Local OSC_POSE instantiation worked for:

```text
Panda
Jaco
Kinova3
UR5e
IIWA
```

Sawyer failed on this local MuJoCo stack with:

```text
mesh volume is too small: robot0_head_1
```

## Policy Runners

Pi/Cosmos shared probe:

```text
experiments/phase3/run_libero_policy_cosmos_probe.py
```

Supported policy values:

```text
--policy cosmos
--policy pi05
--policy molmo
```

Robot/controller knobs:

```text
--robot UR5e
--controller OSC_POSE
```

For native Cosmos-style execution inside this HDF5-compatible runner:

```bash
python experiments/phase3/run_libero_policy_cosmos_probe.py \
  --policy cosmos \
  --task-suite-name libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --robot UR5e \
  --policy-execute-steps 16 \
  --out-dir /home/ubuntu/robotics/runs/phase3/cosmos-libero-ur5e-smoke
```

For Pi:

```bash
python experiments/phase3/run_libero_policy_cosmos_probe.py \
  --policy pi05 \
  --pi-host 127.0.0.1 \
  --pi-port 8010 \
  --task-suite-name libero_spatial \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --robot UR5e \
  --out-dir /home/ubuntu/robotics/runs/phase3/pi05-libero-ur5e-smoke
```

Native Cosmos eval patch:

```text
vm-patches/cosmos/libero/run_libero_eval.py
vm-patches/cosmos/run_libero_eval.py
```

These now expose:

```text
robot: str = "Panda"
controller: str = "OSC_POSE"
```

Use this path when we specifically want Cosmos's original evaluation loop,
including its value-based best-of-N path.

## Caveats

- The action space is still 7D OSC_POSE, so this is an EEF-controller transfer
  test, not native joint-space transfer.
- The policy's proprio input distribution changes because `robot0_gripper_qpos`,
  `robot0_eef_pos`, and `robot0_eef_quat` now come from the target robot.
- The initial robot pose is the target robot's reset pose, not Panda's saved
  benchmark pose.
- Camera names are preserved, but robot geometry and occlusion can still shift
  what the model sees.
