---
type: paper-note
topic: robotics-world-models
created: 2026-05-26
paper: "Do World Action Models Generalize Better than VLAs? A Robustness Study"
arxiv: "2603.22078"
local_pdf: "papers/2603.22078.pdf"
local_text: "papers/2603.22078.txt"
---

# Do WAMs Generalize Better than VLAs - Robustness Benchmark Design

## Why This Note Exists

This paper is useful less as a result to cite and more as a benchmark-engineering template. It shows how to structure a controlled robustness suite for robot policies:

1. choose one closed-loop benchmark environment,
2. keep the task set fixed,
3. define perturbation dimensions as independent config branches,
4. run each model on clean + one perturbation dimension at a time,
5. report success/progress per perturbation class,
6. record model interface details: action representation, action dimension, chunk size, and control frequency.

This maps directly to our planned PolaRiS/MolmoSpaces experiments.

## Core Benchmark Pattern

The paper evaluates policies using:

- clean baseline,
- camera perturbation,
- robot initial-state perturbation,
- language perturbation,
- lighting perturbation,
- background perturbation,
- image noise perturbation,
- object/layout distractor perturbation.

The important design choice is that each perturbation branch activates exactly one perturbation dimension while keeping everything else clean. This makes the failure attribution much cleaner than one combined stress-test.

Template:

```text
for task in tasks:
  for model in models:
    run clean config
    run camera config
    run robot_state config
    run language config
    run light config
    run background config
    run image_noise config
    run layout config
```

The metric is success rate in their benchmarks. For our PolaRiS setup, use both task success and partial progress/rubric criteria because success is sparse.

## LIBERO-Plus Setup

LIBERO-Plus is the single-arm benchmark side of the paper.

Important properties:

- Simulator: MuJoCo / robosuite.
- Robot: Franka Panda, 7 DoF.
- Arms: single arm.
- Cameras: third-person + wrist.
- Image resolution: 256 x 256.
- Native action space: 7D delta end-effector action.
- Control mode: OSC / delta end-effector pose.
- Control frequency: 10 Hz.
- Base tasks: 40 tasks across 4 suites.
- Training demos: 50 per task.
- Total trajectories: 22,400.
- Distractor objects: 416.

What this gives us:

- a clean single-arm manipulation benchmark,
- standard VLA/WAM checkpoints,
- easy comparison to prior LIBERO numbers,
- perturbation axes that are already accepted by the community.

What it does not give us:

- DROID-style joint-position control,
- direct compatibility with DreamZero-DROID or MolmoAct2-DROID,
- direct PolaRiS/DROID real-to-sim alignment.

## RoboTwin 2.0-Plus Setup

RoboTwin 2.0-Plus is their bimanual benchmark extension.

Important properties:

- Simulator: SAPIEN / ManiSkill3.
- Robot: Aloha-AgileX, 14 DoF.
- Arms: dual arm / bimanual.
- Cameras: head + two wrist cameras.
- Image resolution: 320 x 240.
- Native action space: 14D joint positions.
- Control mode: joint position control.
- Control frequency: 25-30 Hz.
- Base tasks: 50 collaborative tasks.
- Training demos: 50 clean + 500 randomized.
- Total trajectories: 27,500.
- Distractor objects: 731 across 147 categories.

This is useful as a model for how to extend an existing simulator with perturbation configs.

## Perturbation Taxonomy

The RoboTwin-Plus appendix gives the most implementation-ready version of the perturbation suite.

### Clean

No perturbation. This is the reference condition.

### Sensor Noise

One noise type per episode, cycled deterministically:

- motion blur,
- Gaussian blur,
- zoom blur,
- fog,
- glass blur.

Applied to RGB observations at render time for both head and wrist cameras.

### Lighting

Perturb:

- RGB light tint,
- light direction,
- material specular strength / shininess,
- shadows on/off.

This tests photometric robustness without changing task geometry.

### Camera

Perturb:

- camera distance,
- spherical pose / azimuth / elevation,
- yaw / pitch / roll.

In their main config, camera spherical-position perturbation is disabled by default to avoid instability. This is important: camera perturbation can accidentally change observability or break benchmark fairness.

### Robot Initial State

Perturb:

- initial joint angles,
- gripper initial state.

This tests whether the policy can recover from a slightly different body configuration.

### Background

Perturb:

- wall/floor texture or color,
- table material properties,
- roughness/metallic/tint.

This is a visual-domain perturbation, not a dynamics perturbation.

### Object Layout

Perturb:

- number of task-irrelevant distractors,
- target object position,
- target object yaw.

This is close to what we want for task generalization: same instruction, same main task, but changed object context.

### Language

Use pre-generated instruction variants:

- irrelevant conversational wrapping,
- common-sense rewording with object descriptions and verb synonyms,
- goal-state or outcome description instead of imperative instruction.

This gives a principled way to test whether failure is visual/control-related or instruction-following-related.

## Model Evaluation Pattern

They evaluate models only where runnable checkpoints and official repositories exist. This is a practical rule we should copy.

For each model, record:

- benchmark environment,
- checkpoint source,
- whether the model was finetuned on the benchmark,
- action representation,
- action dimension,
- action chunk size,
- control frequency,
- observation/camera assumptions.

Their model-interface table matters because raw success rates are not comparable unless the action/control conventions are tracked.

Examples from the paper:

- pi0 / pi0-FAST: 7D delta end-effector, chunk size 50.
- pi0.5: absolute end-effector or joints, 18-19D, chunk size 50.
- OpenVLA-OFT: 7D delta end-effector, chunk size 8, 3-10 Hz.
- VLA-JEPA: 7D delta end-effector, chunk size 7.
- Cosmos-Policy: native 7D or 14D action space, action encoded as latent frames, chunk size 16.
- GE-Act: absolute end-effector poses, chunk size 54, 30 Hz.
- LingBot-VA: combined absolute end-effector and joint angles, chunk size 4, 50 Hz.

This table is a warning for our work: model failures can come from action-space mismatch, not just policy weakness.

## What They Did With Data

They separate:

- embodied pretraining data,
- embodied post-training data,
- task-specific finetuning trajectories.

For benchmark evaluation, they mainly use public/released checkpoints trained or finetuned for the relevant benchmark distribution.

Key lesson:

```text
Do not compare "base model zero-shot" against "task-finetuned model" without labeling it.
```

For our PolaRiS runs, this matters because:

- `pi05_droid_jointpos_polaris` is not raw base pi0.5-DROID; it is DROID policy plus 90% DROID + 10% PolaRiS random sim cotraining for 1k steps.
- base `pi05_droid` is closer to raw DROID behavior, but we are forcing it through the PolaRiS joint-position client.
- `MolmoAct2-DROID` is DROID-aligned but not necessarily PolaRiS-cotrained.

So each result table must explicitly label training/domain status.

## What Their Results Say For Experiment Design

The headline is not enough. The useful pattern is:

- WAMs often survive lighting, noise, and layout perturbations better than VLAs.
- WAMs still struggle with camera viewpoint and robot initial-state changes.
- pi0.5 can be very strong when trained on enough diverse robot/web data.
- Fast-WAM shows that video priors alone are not enough; task-specific data diversity still matters.
- WAMs are slower, often because visual-state denoising dominates inference.

Experiment implication:

```text
Separate visual perturbations from geometric/control perturbations.
```

If a WAM is robust to noise but fails under camera or robot-state shift, that supports the idea that its video prior helps appearance robustness but does not automatically solve action translation or embodiment geometry.

## How To Port This To PolaRiS

PolaRiS should be treated as our DROID-style closed-loop benchmark.

Current PolaRiS interface in our setup:

- Robot: DROID-like Franka/Panda.
- Cameras: external camera + wrist camera.
- Raw frames: 720 x 1280 RGB.
- Model frames for OpenPI: padded 224 x 224 per view.
- Action space: 8D joint-position/gripper action.
- Control: joint-position-like DROID policy client.
- Current task tested: `DROID-FoodBussing`.
- Candidate tasks: `DROID-FoodBussing`, `DROID-MoveLatteCup`, `DROID-TapeIntoContainer`.

### PolaRiS Clean Config

Use the existing task initial condition and prompt.

Log:

- task name,
- prompt,
- model name/checkpoint,
- seed/initial condition id,
- rollout video,
- success,
- progress,
- action chunks,
- proprio,
- model input frames,
- model output actions,
- predicted futures if available.

### PolaRiS Perturbation Branches

Mirror the paper's seven branches:

| Branch | PolaRiS version |
| --- | --- |
| clean | current initial condition and canonical prompt |
| camera | perturb external camera pose, wrist camera pose, or crop/resize framing |
| robot_state | perturb initial joint positions or gripper state |
| language | paraphrase, distractor phrase, goal-state instruction, wrong object prompt |
| lighting | change scene lights/material exposure if available in USD/Isaac |
| background | change table/floor/wall material or color |
| noise | blur/fog/noise/darken model input images before policy call |
| layout | move task objects, add distractor objects, change target object yaw |

The easiest first four are:

1. language,
2. image noise/input corruption,
3. camera crop/framing,
4. object layout.

Lighting/background may require deeper USD/Isaac scene edits.

### PolaRiS Model Table

Every result row should include:

| Model | Checkpoint | Training/domain status | Obs | Action | Chunk | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pi05_droid_jointpos_polaris | OpenPI PolaRiS checkpoint | DROID + 10% PolaRiS sim cotrain | external+wrist+state | 8D joint/gripper | OpenPI chunk | near-ID |
| pi05_droid | OpenPI base DROID checkpoint | DROID, no PolaRiS cotrain | external+wrist+state | 8D through client | OpenPI chunk | possible action/control mismatch |
| MolmoAct2-DROID | allenai/MolmoAct2-DROID | DROID-style model | external+wrist+8D state | 8D DROID action | model chunk | rollout adapter |
| DreamZero-DROID | DreamZero checkpoint if runnable | DROID WAM | image/state/action history | video+action chunk | TBD | predicted futures available |
| Cosmos-Policy/LIBERO | optional parallel track | LIBERO WAM | LIBERO obs | 7D EEF | 16 | controlled WAM baseline, not PolaRiS-native |

## Repository/Experiment Structure To Copy

Use config-driven evaluation:

```text
configs/
  polaris/
    clean.yaml
    perturb_camera.yaml
    perturb_robot_state.yaml
    perturb_language.yaml
    perturb_light.yaml
    perturb_background.yaml
    perturb_noise.yaml
    perturb_layout.yaml

scripts/
  run_polaris_eval.py
  summarize_results.py
  dump_policy_inputs.py

results/
  <date>/<model>/<task>/<branch>/<seed>/
    rollout.mp4
    eval_results.csv
    policy_inputs/
    actions.npy
    obs_meta.json
    predicted_futures/
```

Each branch should change one thing. Combined stress tests come later.

## What We Should Not Copy Blindly

The paper is a robustness leaderboard. Our goal is inference-time planning diagnostics.

So we should copy:

- perturbation taxonomy,
- config branch structure,
- model-interface table,
- per-branch success/progress reporting.

But we should add:

- predicted future vs actual future,
- semantic latent distance,
- action recoverability / inverse dynamics probe,
- predicted-progress vs actual-progress gap,
- prompt/image ablation effects on future prediction before action outcome.

## Immediate PolaRiS Experiment Template

Start small:

```text
Task:
  DROID-FoodBussing

Models:
  pi05_droid_jointpos_polaris
  pi05_droid
  MolmoAct2-DROID

Branches:
  clean
  language_paraphrase
  language_wrong_object
  image_noise_or_blur
  camera_crop
  layout_target_pose

Seeds:
  3 per branch
```

Report:

- mean progress,
- success count,
- first-contact time,
- object-specific rubric progress,
- action norm statistics,
- saved exact 224 x 224 model input frames,
- videos.

If DreamZero or Cosmos is added, also report:

- predicted future frames,
- semantic latent distance to actual future,
- predicted-progress vs actual-progress,
- whether failure was reasoning/prediction or action translation.

## Open Questions

- Can PolaRiS perturb camera poses cleanly without breaking rendering/camera calibration?
- Are object pose perturbations exposed through initial-condition JSON, USD edits, or environment reset arguments?
- Can we add distractor objects without editing the scene USD manually?
- Should image perturbations happen before resize/pad or after exact 224 x 224 policy input creation?
- For fair model comparison, should every model receive the same perturbed 224 x 224 frames or its native preprocessing?
- Which perturbation branches should count as near-ID versus OOD for DROID-trained models?
- How do we separate action-space mismatch from robustness failure for base `pi05_droid`?

## Links

- [[Robotics World Models]]
- [[Reconstruction or Semantics - Semantic Latents for Robotic World Models]]
