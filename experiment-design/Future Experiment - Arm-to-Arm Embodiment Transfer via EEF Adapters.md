---
type: future-experiment
topic: robotics-world-models
status: parked
created: 2026-05-28
source_papers:
  - "Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking"
related:
  - "[[Robotics World Models]]"
---

# Future Experiment - Arm-to-Arm Embodiment Transfer via EEF Adapters

## Why This Is Parked

This is a useful future direction, but it is not the immediate experiment. The current focus remains inference-time planning / future-prediction signals for existing VLA and WAM policies. This note preserves the arm-to-arm transfer idea for later.

## Thesis

Arm-to-arm transfer should separate:

```text
task/world reasoning
+ observation interface
+ kinematic action interface
+ dynamics/control execution
```

The hypothesis is that model reasoning and task understanding may transfer better than actuator conditioning. Failures across arms may come less from "the model does not know what to do" and more from "the model's action/control interface is tied to the source embodiment."

## Any2Any Analogy

Any2Any for humanoid whole-body tracking decomposes cross-embodiment transfer into:

```text
kinematic alignment:
  make source and target observations/actions mean the same thing

dynamics adaptation:
  freeze most pretrained policy
  train small LoRA/PEFT residuals in dynamics-sensitive modules
```

For arm manipulation, the equivalent would be:

```text
kinematic alignment:
  convert all robot observations/actions into a shared EEF/task-space convention

dynamics adaptation:
  adapt only the action/control residual needed by the target arm
```

## Shared Arm Interface

Use end-effector action space as the common semantic control layer:

```text
single arm:
  delta EEF pose = dx, dy, dz, droll, dpitch, dyaw
  gripper = open/close or continuous openness

dual arm:
  left EEF 7D + right EEF 7D
```

Source robot joint logs can be converted into this format by forward kinematics:

```text
q_t, q_t+1
  -> FK(q_t), FK(q_t+1)
  -> delta EEF transform
```

The model should learn:

```text
image + proprio + language -> EEF action chunk
```

rather than:

```text
image + proprio + language -> source-robot joint deltas
```

## Observation Adapter

The observation adapter maps a target robot's raw observations into the exact input contract used during source training.

Example source model contract:

```text
image_primary:
  224x224 RGB, fixed camera convention

image_wrist:
  224x224 RGB wrist view, if used

proprio:
  EEF pose in model frame
  gripper openness in [0, 1]
  optional joint state

language:
  task instruction
```

Target robot raw observations may differ:

```text
camera resolution / crop / viewpoint
missing wrist camera
TCP pose in world frame instead of robot base frame
binary gripper state instead of continuous gripper state
different proprio normalization
```

The adapter handles:

```text
camera resize/crop/order
frame transforms
EEF/tool-frame convention
gripper normalization
proprio normalization
prompt formatting
```

## Action / Control Adapter

The action adapter makes the model's canonical EEF action executable on the target arm:

```text
model EEF delta action
  -> target robot IK / operational-space controller
  -> joint commands or target poses
```

It absorbs:

```text
joint limits
IK failures
controller gains
latency
velocity limits
tool-frame offsets
gripper timing
```

## Dataset

Base training data:

```text
source-arm trajectories
  observations: RGB views, proprio, language
  actions: converted to EEF delta chunks
```

Candidate sources:

```text
DROID / Franka-style data
LIBERO / robosuite trajectories
RoboMimic / MimicGen
PolaRiS exported rollouts
small teleop dataset
```

Target adaptation data, if used:

```text
10-50 target-arm demos or rollouts
same task family
converted to same EEF action convention
```

## Evaluation Ladder

Start narrow and widen only after failure modes are clear:

```text
Level 0:
  same robot, same camera, same controller

Level 1:
  same robot, changed camera/task perturbations

Level 2:
  single-arm source -> single-arm target
  e.g. Franka -> UR5e / Kinova

Level 3:
  single-arm -> dual-arm
  introduces arm assignment and coordination
```

## Baselines

Compare:

```text
A. source-trained model, no adaptation
B. source-trained model + calibration only
C. frozen model + small residual EEF adapter
D. frozen backbone + fine-tuned action head
E. target-trained-from-scratch baseline, if feasible
```

## Interpretation

```text
A fails, C/D recover:
  reasoning/perception transferred; actuator conditioning was weak

A and C/D fail:
  observation/task/world reasoning did not transfer cleanly

A works:
  EEF-space transfer is already strong

C works with tiny data:
  embodiment gap is mostly controller/dynamics calibration

D beats C strongly:
  target action semantics require deeper action-head adaptation
```

## Later Connection To Current Thesis

This future experiment should be framed as a separate test of:

```text
what transfers across embodiments:
  task/world latent prior
  visual reasoning
  EEF action semantics
  actuator-specific control
```

It should not distract from the current near-term experiment, which is about whether predicted futures or semantic latent scores can improve or diagnose inference-time planning.
