# 06 - WidowX-To-LIBERO EEF Transfer

## Goal

Run a non-Franka single-arm policy inside LIBERO by converting its intended
end-effector action into LIBERO's Franka end-effector control space.

The useful test is:

```text
source policy trained for WidowX / Bridge-style setup
  -> source EEF-style action
  -> LIBERO/Franka EEF adapter
  -> closed-loop LIBERO rollout
```

This is not expected to solve LIBERO zero-shot. The point is to test whether
the source policy's semantic action direction survives once we remove the
native robot/action contract.

## Candidate Policies

### A. X-VLA-WidowX

Primary candidate.

Useful properties:

- single-arm non-Franka source embodiment,
- public checkpoint: `lerobot/xvla-widowx` or `2toINF/X-VLA-WidowX`,
- trained/fine-tuned on BridgeData/WidowX-style manipulation,
- continuous flow-matching action output,
- paper explicitly standardizes actions around EEF pose representations,
- soft prompts are designed to encode camera/embodiment/domain heterogeneity.

The X-VLA paper is especially relevant because it treats robot identity,
camera layout, proprioception, and action convention as part of the policy
conditioning problem, not just a final action-head problem.

Important paper details:

```text
pretraining: 290K episodes
sources: AGIBOT, DROID, RoboMind-Franka, RoboMind-UR, RoboMind-Dual-Franka, RoboMind-Agilex
robots: seven hardware setups across five arm types
action representation: EEF xyz + 6D rotation + binary gripper
camera examples:
  RoboMind-Franka: top view, 30 Hz
  RoboMind-UR: top view, 30 Hz
  DROID: left/right view + wrist, 15 Hz
  AGIBOT: head + wrist, 30 Hz
  Agilex: head + wrist, 30 Hz
```

This makes X-VLA-WidowX the cleanest source policy for our single-arm
non-Franka-to-LIBERO test.

### B. OpenVLA-WidowX / Bridge

Comparison candidate.

Useful properties:

- public checkpoint: `openvla/openvla-7b`,
- BridgeData V2 / WidowX is an intended out-of-box domain,
- inference can request Bridge unnormalization with:

```python
action = vla.predict_action(..., unnorm_key="bridge_orig")
```

Expected action is a 7-DoF Bridge/WidowX action.

Why it is useful:

- it is a classic token-decoded VLA baseline,
- it gives a contrast to X-VLA's continuous flow-matching and soft-prompt setup,
- source robot is also single-arm non-Franka.

Caveat:

OpenVLA's action is tied to Bridge action discretization and unnormalization.
That makes it more brittle and less cleanly embodiment-aware than X-VLA, but
that brittleness is itself informative.

## Shared LIBERO Adapter

Both policies should use the same adapter and rollout harness.

```text
LIBERO observation
  -> source-policy image/proprio/prompt adapter
  -> source policy action
  -> WidowX/Bridge action interpretation
  -> LIBERO EEF action
  -> env.step(...)
```

Minimum input mapping:

```text
LIBERO agentview -> source primary image
LIBERO wrist     -> optional auxiliary image if the policy expects it
task text        -> source prompt format
LIBERO state     -> source proprio proxy or omitted if checkpoint permits
```

The first implementation should keep camera handling simple:

- use `agentview` as the primary/source camera,
- do not silently rotate/crop/flip differently per model,
- log raw and model-input frames for every query,
- test duplicated/blank wrist only as labeled ablations.

## Action Adapter Modes

Define these modes before looking at success/failure videos.

### Mode 1 - Direct 7D Passthrough

Interpret source action as:

```text
dx, dy, dz, droll, dpitch, dyaw, gripper
```

Then clip to LIBERO's safe action range.

This is the dumbest baseline. If it works at all, the source and target action
contracts are surprisingly aligned.

### Mode 2 - Source-Stat Scaled EEF

Use source action statistics to rescale translation, rotation, and gripper
before passing to LIBERO.

This tests whether the source policy has useful metric direction but wrong
scale.

### Mode 3 - Direction-Only EEF

Normalize the EEF translation/rotation direction and execute with fixed step
sizes.

This tests whether the policy knows "where to move" even if magnitude and
calibration are wrong.

If only this mode works, the result still supports the thesis that reasoning or
task direction may transfer better than low-level action calibration.

## Absolute vs Delta EEF Issue

The biggest adapter detail is whether the source action should be treated as:

```text
absolute EEF target
```

or:

```text
delta EEF command
```

X-VLA paper language suggests many downstream benchmarks use absolute EEF.
LIBERO execution is usually delta EEF under OSC control.

So for X-VLA we may need:

```text
source absolute target
  -> source/shadow current EEF
  -> source delta
  -> LIBERO delta
```

For OpenVLA Bridge, verify the exact Bridge action convention before claiming
EEF equivalence.

## What To Measure

Do not rely only on success.

Log per step:

- raw frames,
- model-input frames,
- source prompt,
- source proprio proxy,
- raw source action,
- adapted LIBERO action,
- clipping/saturation flags,
- gripper open/close,
- LIBERO reward/success,
- current EEF pose if available.

Rollout diagnostics:

- does it reach toward the correct object,
- does it grasp at a meaningful time,
- does it move the object after grasp,
- does it freeze/no-op,
- does it saturate one axis,
- does camera/prompt ablation change behavior.

## Prompt And Camera Ablations

Use the same ablations for both policies:

- canonical LIBERO prompt,
- Bridge-style rewritten prompt,
- wrong-object prompt,
- generic prompt like `pick up the object`,
- primary image only,
- primary image duplicated as auxiliary,
- wrist/auxiliary blanked if the policy supports it.

The goal is to separate:

```text
semantic task understanding
camera/domain mismatch
action translation mismatch
```

## Expected Outcomes

Likely:

- neither policy solves LIBERO cleanly without adaptation,
- X-VLA-WidowX produces smoother and more interpretable EEF motion,
- OpenVLA-WidowX may be more brittle because Bridge action tokens and
  unnormalization are tied to the source domain,
- useful signal appears in reaching direction, gripper timing, and sensitivity
  to prompt/camera perturbations before it appears in full task success.

## Scientific Framing

This experiment tests:

```text
Can a single-arm non-Franka policy produce task-relevant EEF intent in a
Franka/LIBERO environment when executed through a shared EEF adapter?
```

It does not test:

- native LIBERO performance,
- pure zero-shot universal robot control,
- whether WidowX and Franka are interchangeable,
- whether a model can solve a new camera setup without adaptation.

## Best First Run

Run the exact same task slice with both models:

```text
X-VLA-WidowX + Mode 1/2/3 adapter
OpenVLA-WidowX + Mode 1/2/3 adapter
```

Use a small LIBERO task set with visible reaching/grasping behavior and inspect:

- action traces,
- EEF paths,
- camera frames,
- success/failure videos.

Only after this should we decide whether to add a small learned residual adapter
or prompt/soft-prompt adaptation.
