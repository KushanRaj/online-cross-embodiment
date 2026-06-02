# LIBERO Language-Adherence Runbook

Date: 2026-05-29

## Core Question

For the same LIBERO scene and initial state, does changing only the language instruction change the policy's behavior?

This is a context-faithfulness probe, not a benchmark run.

## Why Start With LIBERO

LIBERO is cleaner than PolaRiS for this specific probe:

- standardized tasks,
- known success predicate,
- already running for Pi, MolmoAct2, Cosmos, FastWAM, and GE-Act,
- same simulated embodiment across models,
- easier to hold initial state fixed.

But LIBERO has one caveat: the simulator success predicate remains the canonical BDDL task. Counterfactual prompts must therefore be scored from rollout video/action traces, not from the binary success flag alone.

## Model Priority

| Priority | Model | Why |
| --- | --- | --- |
| 1 | OpenPI `pi05_libero` | Clean VLA language-conditioned baseline. |
| 1 | `MolmoAct2-LIBERO` | Clean VLA/action-reasoning baseline. |
| 2 | Cosmos Policy LIBERO | Planning/value-style WAM baseline. Verified runner path can compute T5 embeddings on demand for prompt strings that are not already in the cache. |
| 3 | GE-Act | Useful WAM-style comparison. Verified runner passes the prompt string into the T5/text-conditioning path, but first rerun with full-horizon video logging because the existing artifact has only 9 frames. |
| 3 | FastWAM `libero_uncond_2cam224` | Verified language-conditioned runner path. LIBERO `task.language` is wrapped in `DEFAULT_PROMPT`, encoded by the loaded text encoder, then passed into the video and action denoising paths. `uncond` here means the FastWAM base variant is not action/future-conditioned, not that language is disabled. |

## First Task

Use the task already run across all models:

```text
Suite: libero_spatial
Task id: 0
Canonical instruction: Pick up the black bowl between the plate and the ramekin and place it on the plate.
```

This task is useful because it contains:

- object identity: black bowl,
- distractor object: ramekin,
- spatial relation: between,
- target relation: on the plate.

## Prompt Ladder

### P0 Canonical

```text
Pick up the black bowl between the plate and the ramekin and place it on the plate.
```

### P1 Paraphrase

```text
Take the black bowl that is between the plate and the ramekin, and put it onto the plate.
```

Purpose: tests wording robustness without changing task semantics.

### P2 Expanded Object/Relation

```text
Find the black bowl positioned between the plate and the ramekin. Grasp that black bowl and place it on top of the plate.
```

Purpose: tests whether explicit object and spatial grounding changes approach behavior.

### P3 Step-Decomposed

```text
Move the gripper above the black bowl located between the plate and the ramekin. Lower the gripper, grasp the black bowl, lift it without touching the ramekin, move over the plate, lower the bowl onto the plate, and release it.
```

Purpose: tests whether procedural language changes grasp timing, path, or collision behavior.

## Negative / Counterfactual Prompts

Run these only after P0-P3.

These are split into two groups. The first group should still look like normal manipulation language and should be close enough to the training distribution to be meaningful. The second group is deliberately OOD and tests whether the learned task prior overwhelms explicit language.

### In-Distribution Conflicts

#### C1 Wrong Object, Valid Manipulation

```text
Pick up the ramekin and place it on the plate.
```

What to look for: does the first-contact object switch from black bowl to ramekin?

#### C2 Wrong Target Relation, Valid Manipulation

```text
Pick up the black bowl and place it next to the plate, not on it.
```

What to look for: does the policy still execute the canonical "on plate" behavior?

#### C3 Same Objects, Reversed Target

```text
Pick up the plate and place it near the black bowl.
```

What to look for: does the policy try to interact with the plate, or does it fall back to picking the bowl?

#### C4 Distractor Avoidance

```text
Pick up the black bowl without touching the ramekin, then place the bowl on the plate.
```

What to look for: does the path or approach direction change relative to the canonical prompt?

#### C5 Other-Task Prompt From Same Suite

Use a canonical instruction from another LIBERO spatial/object task whose referenced object is visible or plausible in the current scene.

What to look for: does the model attempt the requested alternate behavior, freeze/degrade, or execute the current scene's canonical task prior?

### Out-of-Distribution / Inhibitory Prompts

#### O1 Approach-Only

```text
Move near the black bowl.
```

What to look for: does the model merely approach the referenced object, or does the learned pick-place prior continue into grasping and placement? This avoids putting the inhibitory phrase "do not pick it up" into the same prompt.

#### O2 Explicit No-Op / Inhibition

```text
Do not pick up anything. Keep the gripper away from the objects.
```

What to look for: does the policy hold position, move without grasping, or execute a learned pick-place prior?

## Second Task Selection

After task 0, choose one task from each category:

| Suite | Desired task type | Why |
| --- | --- | --- |
| `libero_spatial` | another relation-heavy pick/place | Tests whether task 0 was special. |
| `libero_object` | object identity task with distractors | Best for wrong-object language tests. |
| `libero_goal` | goal-state task | Tests whether longer goal descriptions help/hurt. |
| `libero_10` | longer horizon | Tests decomposition, but only after simpler tasks work. |

When the VM is up, dump exact names:

```bash
cd ~/robotics/repos/PolaRiS/third_party/openpi
.venv/bin/python - <<'PY'
from libero.libero import benchmark

for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    bench = benchmark.get_benchmark_dict()[suite]()
    print(f"\n{suite}")
    for i in range(bench.get_num_tasks()):
        task = bench.get_task(i)
        print(i, task.language)
PY
```

## First Run Matrix

Keep it small:

| Model | Prompts | Seeds / init states |
| --- | --- | --- |
| `pi05_libero` | P0, P1, P2, P3 | 3 |
| `MolmoAct2-LIBERO` | P0, P1, P2, P3 | 3 |

Then:

| Model | Prompts | Seeds / init states |
| --- | --- | --- |
| `pi05_libero` | N1, N2, N3 | 3 |
| `MolmoAct2-LIBERO` | N1, N2, N3 | 3 |

Only after that, add Cosmos/GE/FastWAM.

Cosmos, GE-Act, and FastWAM now have VM-side prompt override support patched into their LIBERO eval scripts:

- Cosmos: `cosmos_policy/experiments/robot/libero/run_libero_eval.py`
- GE-Act: `experiments/eval_libero.py`
- FastWAM: `experiments/libero/eval_libero_single.py` via `EVALUATION.prompt_override=...`

All patched files passed `py_compile` on the VM.

## What To Log

Each rollout directory should contain:

```text
metadata.json
rollout.mp4
eval.log
actions.npy
proprio.npy
agentview_frames/
wrist_frames/
policy_inputs/
```

`metadata.json` should include:

```json
{
  "suite": "libero_spatial",
  "task_id": 0,
  "canonical_task_language": "...",
  "prompt_variant_id": "P2",
  "prompt": "...",
  "model": "MolmoAct2-LIBERO",
  "seed_or_init_state": 0,
  "success": true,
  "episode_steps": 80
}
```

## Metrics

Primary:

- success,
- episode length,
- first contacted object,
- first lifted object,
- final object placement,
- whether the goal object matches the prompt.

Action-level:

- action L2 distance from canonical prompt at same timestep,
- gripper close timestep,
- first large movement direction,
- cumulative path length,
- collision/contact events if exposed by robosuite.

Language-adherence labels:

| Label | Meaning |
| --- | --- |
| `adheres` | Behavior changes correctly with prompt. |
| `canonical_prior` | Ignores prompt and performs canonical task. |
| `degraded` | Longer prompt worsens behavior without a meaningful semantic shift. |
| `ambiguous` | Video/action evidence not enough to classify. |

## Expected Outcomes

The likely informative outcomes:

1. P1/P2 behave the same as P0: model is robust to paraphrase.
2. P3 performs worse: procedural language overloads the policy rather than helping.
3. N1/N2 still perform canonical task: model is weakly language faithful and follows visual/task prior.
4. FastWAM does not change under prompt: confirms unconditioned or weakly language-conditioned behavior.
5. Cosmos/GE-Act become valid WAM-side prompt-sensitivity tests now that the eval runners accept prompt overrides.

## Practical Next Step

Next VM session:

1. Dump LIBERO task names.
2. Patch the Pi and Molmo LIBERO runners to accept a prompt override.
3. Run `libero_spatial` task `0`, prompts P0-P3, 3 init states, Pi + Molmo only.
4. Copy videos/logs locally.
5. Decide whether negative controls are worth running before expanding to other tasks.
