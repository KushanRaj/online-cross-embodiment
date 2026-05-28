# DreamZero / pi0.5 / MolmoAct2 Experiment Setup Plan

## Goal

Run pure-inference closed-loop experiments for DROID-style robot foundation models:

- DreamZero-DROID
- pi0.5-DROID / OpenPI DROID-style policy
- MolmoAct2-DROID

The immediate goal is not training. The goal is to produce structured rollout evidence:

1. vanilla policy rollouts for "vibes but logged"
2. prompt and image ablations
3. predicted-vs-actual diagnostics where the model exposes predicted visual futures

## Environments

### PolaRiS

Use PolaRiS as the near-ID / DROID-aligned closed-loop simulator.

Why:

- DROID-style Franka setup
- real-to-sim reconstructed tasks
- explicit PolaRiS path for pi0.5-style DROID joint-position policy
- likely most natural first target for DreamZero-DROID-style evaluation

First tasks:

- `DROID-FoodBussing`
- `DROID-MoveLatteCup`
- `DROID-TapeIntoContainer`

### MolmoSpaces

Use MolmoSpaces as the broader DROID-compatible generalization simulator.

Why:

- DROID-style manipulation interface
- broader object/scene/task distribution
- relevant to MolmoAct2-DROID and pi0.5-DROID benchmark comparisons
- useful for OOD prompt/object/camera perturbations

First task families:

- pick
- pick-and-place
- open / close
- pick-and-place-color or pick-and-place-next-to

## First Experiment Matrix

Treat PolaRiS as near-DROID and MolmoSpaces as broader DROID-compatible OOD.

| Model | PolaRiS | MolmoSpaces |
| --- | --- | --- |
| DreamZero-DROID | primary near-ID target | adapter feasibility spike |
| pi0.5-DROID | supported baseline path first | broader OOD target |
| MolmoAct2-DROID | adapter feasibility spike | primary benchmark target |

## Tier 1: Vanilla Rollouts

For each successful model/environment adapter:

- run 3 seeds per task
- save rollout video
- save success/progress
- save task name and prompt
- save observations, actions, proprio, and timestamps
- for DreamZero, save predicted future video/action chunks when accessible

Initial minimum:

- 2 PolaRiS tasks
- 2 MolmoSpaces tasks
- 3 models where adapters work
- vanilla prompt only

## Tier 2: Inference-Only Ablations

Prompt ablations:

- canonical instruction
- paraphrase
- wrong object absent from scene
- wrong object present as distractor
- verb swap
- underspecified instruction

Image ablations:

- mask wrist camera
- mask third-person camera
- shuffle camera order
- crop target object
- blur/noise
- stale previous frame
- darken lighting
- occlude gripper

Object/task ablations:

- same verb, new object
- same object, new verb
- same task, changed pose
- composition of known primitives
- language-object mismatch, e.g. cloth visible but prompt says apple

## Diagnostic Claim

Central hypothesis:

> Reasoning / world-model prediction may be stronger than action translation.

For DreamZero-style models, bucket failures by predicted future vs actual rollout:

| Predicted future | Actual execution | Interpretation |
| --- | --- | --- |
| good | good | policy works |
| good | bad | action translation / control gap |
| bad | bad | reasoning / world-model gap |
| bad | good | visual prediction not faithful to action |

Primary metric:

```text
intent_action_gap = predicted_progress - actual_progress
```

## VM Setup Target

VM:

```text
<redacted public EC2 hostname>
GPU: NVIDIA L40S, ~46GB VRAM
RAM: ~124GB
Disk: ~485GB root volume
```

Directory layout:

```text
~/lossfunk-residency        # repo copy and setup scripts
~/robotics                  # external cloned engines/repos
~/robotics/checkpoints      # model checkpoints
~/robotics/data             # simulator datasets/assets
~/robotics/logs             # install and rollout logs
~/robotics/results          # rollout outputs
```

Setup order:

1. system inventory: GPU, disk, Python, CUDA
2. install base tools: git-lfs, uv, Miniforge/conda if needed
3. clone PolaRiS
4. download PolaRiS-Hub assets
5. run PolaRiS random-policy smoke test
6. run supported pi0.5/OpenPI PolaRiS baseline
7. clone/setup MolmoSpaces
8. run MolmoSpaces minimal smoke test
9. download only required model checkpoints, not all variants blindly
10. build unified rollout logging wrapper

## Stop/Start Discipline

The VM can be stopped after setup. Compute billing stops, but EBS storage remains billed. Keep downloads on the persistent root/EBS volume and avoid instance-store-only paths.

After a working setup:

- save exact commands in this repo
- snapshot or AMI the instance
- keep model checkpoints in `~/robotics/checkpoints`
- keep environment assets in `~/robotics/data`

## VM Setup Status

Last updated: 2026-05-22.

Completed on the AWS VM:

- verified GPU: NVIDIA L40S, ~46 GB VRAM
- verified RAM/disk: ~124 GB RAM, ~485 GB root volume
- installed base tools: `git-lfs`, `uv`, `hf`, ffmpeg/runtime graphics libraries
- cloned PolaRiS with submodules under `~/robotics/repos/PolaRiS`
- rewrote GitHub SSH submodule URLs to HTTPS, so GitHub keys are not required
- installed PolaRiS via `uv sync`
- added local PolaRiS packaging override for `flatdict` build dependency
- downloaded PolaRiS-Hub assets to `~/robotics/data/PolaRiS-Hub` and symlinked into the repo
- accepted Isaac/Omniverse EULA noninteractively via `OMNI_KIT_ACCEPT_EULA=YES`
- installed missing headless runtime libraries: `libglu1-mesa`, `libxt6`
- registered PolaRiS `FakeClient` for smoke testing
- ran fake-policy `DROID-FoodBussing` rollout:
  - output: `~/robotics/results/polaris-fake-smoke/episode_0.mp4`
  - output: `~/robotics/results/polaris-fake-smoke/eval_results.csv`
  - result: `success=False`, `progress=0.0`
- installed OpenPI under `~/robotics/repos/PolaRiS/third_party/openpi`
- downloaded and loaded `pi05_droid_jointpos_polaris`
- ran real pi0.5 PolaRiS rollout on `DROID-FoodBussing`:
  - output: `~/robotics/results/polaris-pi05-foodbussing-smoke/episode_0.mp4`
  - output: `~/robotics/results/polaris-pi05-foodbussing-smoke/eval_results.csv`
  - result: `success=False`, `progress=0.6666666666666666`
- downloaded and loaded base `pi05_droid` checkpoint:
  - config: `pi05_droid`
  - checkpoint: `gs://openpi-assets/checkpoints/pi05_droid`
  - note: this is the non-PolaRiS co-trained DROID checkpoint available in the OpenPI tree; there is no registered `pi05_droid_jointpos` base config in this checkout
- ran base `pi05_droid` through the same PolaRiS `DroidJointPos` client on `DROID-FoodBussing`:
  - output: `~/robotics/results/polaris-pi05-base-foodbussing-smoke/episode_0.mp4`
  - output: `~/robotics/results/polaris-pi05-base-foodbussing-smoke/eval_results.csv`
  - local copy: `run-artifacts/polaris-pi05-base-foodbussing-episode_0.mp4`
  - local copy: `run-artifacts/polaris-pi05-base-foodbussing-eval_results.csv`
  - result: `success=False`, `progress=0.0`
- added a PolaRiS rollout client for `allenai/MolmoAct2-DROID`:
  - local patch: `vm-patches/polaris/molmoact2_droid_client.py`
  - remote file: `~/robotics/repos/PolaRiS/src/polaris/policy/molmoact2_droid_client.py`
  - registered client name: `MolmoAct2Droid`
  - maps PolaRiS `external_cam`, `wrist_cam`, and 8D proprio state into MolmoAct2 `predict_action`
  - uses `norm_tag="franka_droid"` and continuous 8D DROID joint-position actions
- pinned PolaRiS root `transformers==4.57.2` for MolmoAct2 remote code compatibility
- added a small MolmoAct2 tokenizer shim because the checkpoint exposes `extra_special_tokens` as a list while current `transformers` expects a dict
- downloaded and loaded `allenai/MolmoAct2-DROID`
- ran real MolmoAct2 PolaRiS rollout on `DROID-FoodBussing`:
  - output: `~/robotics/results/polaris-molmoact2-foodbussing-smoke/episode_0.mp4`
  - output: `~/robotics/results/polaris-molmoact2-foodbussing-smoke/eval_results.csv`
  - local copy: `run-artifacts/polaris-molmoact2-foodbussing-episode_0.mp4`
  - local copy: `run-artifacts/polaris-molmoact2-foodbussing-eval_results.csv`
  - result: `success=False`, `progress=0.16666666666666666`

Important observed interface:

```text
PolaRiS action space: Box(-inf, inf, (1, 8), float32)
Action terms:
  arm: 7
  finger_joint: 1

Policy observation group:
  arm_joint_pos: (7,)
  gripper_pos: (1,)

DroidJointPos image inputs:
  obs["splat"]["external_cam"]
  obs["splat"]["wrist_cam"]
```

Current VM footprint:

```text
~/.cache/openpi                       ~12 GB
~/.cache/huggingface                  includes MolmoAct2-DROID checkpoint
~/robotics/repos/PolaRiS              ~25 GB
~/robotics/data/PolaRiS-Hub           ~1.7 GB
root disk used                        ~115 GB / 485 GB
```

No policy/eval server was left running after the smoke tests; GPU was clear after stopping the pi0.5 server and after the MolmoAct2 rollout.

## FoodBussing Rollout Comparison

Last updated: 2026-05-25.

Environment:

```text
DROID-FoodBussing
Prompt: Put all the foods in the bowl
Rollout length: 450 env steps
Metric: PolaRiS success/progress from eval_results.csv
```

| Model/checkpoint | Prior smoke progress | New 3-run progress | New mean | 4-run mean | Successes |
| --- | ---: | --- | ---: | ---: | ---: |
| `pi05_droid_jointpos_polaris` | 0.6667 | 0.3333, 0.1667, 0.6667 | 0.3889 | 0.4583 | 0/4 |
| `pi05_droid` base | 0.0000 | 0.0000, 0.0000, 0.0000 | 0.0000 | 0.0000 | 0/4 |
| `allenai/MolmoAct2-DROID` | 0.1667 | 0.1667, 0.1667, 0.1667 | 0.1667 | 0.1667 | 0/4 |

Local rerun artifacts:

```text
run-artifacts/rerun3/polaris-pi05-polaris-foodbussing-rerun3.csv
run-artifacts/rerun3/pi05-polaris/episode_0.mp4
run-artifacts/rerun3/pi05-polaris/episode_1.mp4
run-artifacts/rerun3/pi05-polaris/episode_2.mp4

run-artifacts/rerun3/polaris-pi05-base-foodbussing-rerun3.csv
run-artifacts/rerun3/pi05-base/episode_0.mp4
run-artifacts/rerun3/pi05-base/episode_1.mp4
run-artifacts/rerun3/pi05-base/episode_2.mp4

run-artifacts/rerun3/polaris-molmoact2-foodbussing-rerun3.csv
run-artifacts/rerun3/molmoact2/episode_0.mp4
run-artifacts/rerun3/molmoact2/episode_1.mp4
run-artifacts/rerun3/molmoact2/episode_2.mp4
```
