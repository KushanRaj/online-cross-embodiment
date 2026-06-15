# 05 - DROID-To-SO101 EEF Adapter

## Goal

Test the weakest useful cross-embodiment bridge:

```text
DROID policy output
  -> DROID FK / source EEF trajectory
  -> target scene/base transform
  -> SO101 IK / controller
  -> SO101 joint targets
```

This is not expected to be a strong zero-shot policy. It is a diagnostic for
whether the source policy's intended end-effector motion is useful once the
native joint/action decoder is removed.

## Fastest Candidate Model

Use `allenai/MolmoAct2-DROID` first.

Reasons:

- Existing PolaRiS patch already loads this checkpoint.
- It uses direct Transformers inference, no OpenPI websocket server.
- The current adapter has the exact expected input shape:

```text
images:
  [external_cam, wrist_cam]

state:
  7 Franka/DROID arm joints + 1 gripper value

output:
  8D DROID action chunk
```

Existing reference file:

```text
vm-patches/polaris/molmoact2_droid_client.py
```

Second candidate: `pi0.5-DROID` / OpenPI.

Reasons it is slower to wire:

- Requires an OpenPI websocket policy server.
- Existing adapter treats public DROID output as joint velocity and converts it
  to joint position before PolaRiS execution.
- More moving parts before we even reach the EEF bridge.

Existing reference file:

```text
vm-patches/polaris/droid_jointvel_to_pos_client.py
```

## Expected Camera Setup

For DROID-aligned policies, assume:

```text
external / right / third-person camera
wrist / hand camera
robot state
task text
```

Current MolmoAct2-DROID PolaRiS adapter maps:

```text
obs["splat"]["external_cam"] -> right_image
obs["splat"]["wrist_cam"]    -> wrist_image
obs["policy"]["arm_joint_pos"] + gripper -> state
```

For the SO101 live setup, the closest available mapping is:

```text
scene camera / laptop / phone side view -> external_cam
top or wrist-like camera                -> wrist_cam
```

This is a major caveat. If the second camera is not physically wrist-mounted,
the policy is seeing an out-of-distribution wrist view. A scene-only or
duplicated external view can be tested, but that changes the model input
contract and should be labeled.

## Required Adapter Pieces

### 1. DROID Policy Server

Run MolmoAct2-DROID on EC2 as an HTTP policy server, analogous to the existing
SO101 MolmoAct2 server, but with:

```text
repo_id = allenai/MolmoAct2-DROID
norm_tag = franka_droid
state_dim = 8
action_dim = 8
```

### 2. Source State Proxy

The DROID policy expects Franka/DROID joint state. On SO101 we do not have real
Franka joints.

Minimal first proxy:

```text
maintain a simulated/source Franka state
initialize it to a neutral DROID pose
after each DROID action, update the source state
feed that source state back to the next policy query
```

This makes the source policy internally consistent, even though the real robot
is SO101. It should be logged explicitly.

### 3. DROID FK

Convert the DROID/Franka source joint trajectory to an EEF trajectory:

```text
q_droid[t] -> T_source_eef[t]
delta_T_source[t] = inv(T_source_eef[t-1]) @ T_source_eef[t]
```

The initial implementation should be dry-run only until the Franka convention,
joint ordering, units, gripper convention, and FK model are verified.

### 4. Frame Transform

Map source EEF deltas into SO101 target frame:

```text
delta_T_source -> delta_T_target
```

For a first smoke test, use a manually configured axis/sign/scale mapping.
Later, use calibrated camera/table coordinates.

### 5. SO101 IK / Controller

Convert target EEF deltas into SO101 joint targets:

```text
current_so101_joints + desired_delta_T_target -> target_so101_joints
```

This must enforce:

- joint limits,
- max per-tick delta,
- workspace limits,
- gripper limits,
- stop-on-IK-failure behavior.

## Initial Safety Mode

First run should be:

```text
camera + policy + DROID action
  -> source FK
  -> target EEF delta
  -> SO101 IK
  -> log only, no torque
```

Only after inspecting trajectories:

```text
execute with max_step_deg <= 1-2 degrees
execute_horizon small
manual e-stop ready
```

## Paper Grounding

This setup follows the project reading:

- Robustness benchmark: model performance depends on action representation,
  action dimension, chunk size, control frequency, camera assumptions.
- GE/Genie Envisioner: world priors can transfer, but action decoders are not
  reused directly across embodiments; new action heads/adapters are required.
- Cosmos 3: action vectors require an embodiment config; without it, numbers
  do not have universal meaning.
- LPS: planning wrappers still depend on embodiment-specific policies.

So this adapter is a diagnostic baseline, not proof of true universal action.

