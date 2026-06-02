# PolaRiS and LIBERO Language-Adherence Probes

Date: 2026-05-29

## Goal

Measure whether VLA behavior changes when only the language input changes, while keeping the scene, embodiment, camera contract, and initial state fixed.

The first pass should not be a benchmark leaderboard. It should be a controlled input-sensitivity probe:

- same model,
- same task,
- same seed / initial state,
- same camera and proprio inputs,
- only prompt granularity changes.

## PolaRiS Task Shortlist

Confirmed from our existing setup notes:

| Task | Why it is useful | Current status |
| --- | --- | --- |
| `DROID-FoodBussing` | Multi-object cleanup task; good for testing whether language can bias object order and subgoal progress. | Already run with Pi and MolmoAct2. |
| `DROID-MoveLatteCup` | Single dominant object; cleaner language-adherence test because object ambiguity is lower. | Candidate listed in prior PolaRiS robustness note; verify manifest on VM before running. |
| `DROID-TapeIntoContainer` | Container insertion task; useful for testing spatial and contact-sensitive instructions. | Candidate listed in prior PolaRiS robustness note; verify manifest on VM before running. |

Next time the VM is up, dump the exact PolaRiS task registry before running:

```bash
cd ~/robotics/repos/PolaRiS
python - <<'PY'
import pkgutil
import polaris
print("polaris", polaris.__file__)
# Then inspect the local task/benchmark registry in the installed checkout.
PY
```

## PolaRiS FoodBussing Prompt Ladder

Use the same initial condition and run each prompt for each model:

- `pi05_droid_jointpos_polaris`
- `pi05_droid` with our adapter, if still useful as a failure/control baseline
- `MolmoAct2-DROID`

### Level 1: Canonical

```text
Put all the foods in the bowl.
```

This is the base prompt already captured in `run-artifacts/policy-input-dump/README.md`.

### Level 2: Object-Enumerated

```text
Put each food item into the bowl. Pick up the ice cream cone, the battery-shaped item, and the other loose food items one by one, and place them inside the bowl.
```

Purpose: test whether naming visible objects changes target order, reaching direction, and first-contact object.

### Level 3: Step-Decomposed

```text
Move toward the nearest loose food item first. Align the gripper above it, lower carefully, close the gripper, lift it clear of the table, move over the bowl, lower into the bowl, open the gripper, and repeat this process for the remaining food items until all food items are inside the bowl.
```

Purpose: test whether procedural language affects approach style, gripper timing, and repeated subgoal behavior.

### Optional Negative Controls

Use these after the three-level ladder:

```text
Put only the ice cream cone in the bowl and leave the other items where they are.
```

```text
Do not put anything in the bowl. Move near the bowl and stop.
```

```text
Put the food items next to the bowl, not inside it.
```

These tell us whether the model is using language or just executing a learned cleanup prior.

## PolaRiS Metrics

For every rollout, save:

- success and progress from PolaRiS,
- rollout video,
- exact external and wrist model-input frames,
- language prompt,
- first object contacted,
- first object lifted,
- first object dropped into the bowl,
- gripper close/open timing,
- per-step action chunks,
- action delta between prompt levels for the same seed.

For FoodBussing, the most useful manual rubric is:

| Substep | Score |
| --- | --- |
| reaches a food item | 1 |
| grasps a food item | 1 |
| lifts it | 1 |
| moves it toward bowl | 1 |
| releases in/near bowl | 1 |
| repeats for another object | 1 |

## LIBERO Task Shortlist

Use LIBERO as the parallel, cleaner sim track. It gives us standardized tasks and many released policies, but it is not the same embodiment/interface as PolaRiS.

| Suite / task type | Why it is useful | Prompt test |
| --- | --- | --- |
| `libero_spatial`, task id `0`: pick up the black bowl between the plate and the ramekin and place it on the plate | Already run across Cosmos, FastWAM, GE-Act, Pi, and MolmoAct2. Good first language-sensitivity baseline. | Spatial relation adherence: "between", "on the plate", wrong target relation. |
| `libero_spatial`, another spatial-relation pick/place task | Tests whether the result is not specific to the black-bowl task. | Vary relation language: left/right/between/near. |
| `libero_object`, object-identity task | Tests object-name grounding more directly than spatial grounding. | Canonical vs enumerated vs wrong object. |
| `libero_goal`, goal-conditioned task | Tests whether prompt wording changes goal interpretation when objects are similar. | Short goal vs decomposed goal. |
| `libero_10`, a longer-horizon task | Harder but more relevant for planning claims. | Step decomposition may help or hurt; useful stress test after simpler suites. |

When the VM is up, dump exact LIBERO task names:

```bash
cd ~/robotics/repos/PolaRiS/third_party/openpi
.venv/bin/python - <<'PY'
from libero.libero import benchmark
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    bench = benchmark.get_benchmark_dict()[suite]()
    print("\\n", suite)
    for i in range(bench.get_num_tasks()):
        task = bench.get_task(i)
        print(i, task.language)
PY
```

## LIBERO Prompt Ladder

For the already-run `libero_spatial` task id `0`:

### Level 1: Canonical

```text
Pick up the black bowl between the plate and the ramekin and place it on the plate.
```

### Level 2: Object and Relation Expanded

```text
Find the black bowl that is positioned between the plate and the ramekin. Grasp that black bowl and place it on top of the plate.
```

### Level 3: Step-Decomposed

```text
Move the gripper above the black bowl located between the plate and the ramekin. Lower the gripper, grasp the black bowl, lift it without touching the ramekin, move over the plate, lower the bowl onto the plate, and release it.
```

### Negative Controls

```text
Pick up the ramekin and place it on the plate.
```

```text
Pick up the black bowl and place it next to the plate, not on it.
```

```text
Move near the black bowl.
```

## Recommended First Run Matrix

Start small and comparable:

| Environment | Task | Models | Prompts | Seeds |
| --- | --- | --- | --- | --- |
| PolaRiS | `DROID-FoodBussing` | Pi PolaRiS checkpoint, MolmoAct2-DROID | 3 prompt levels | 3 |
| LIBERO | `libero_spatial` task `0` | Pi LIBERO, MolmoAct2-LIBERO, Cosmos, FastWAM, GE-Act | 3 prompt levels | 3 |

Then add negative controls only after the clean prompt ladder runs.

## Next Priority After LIBERO Saturation

The first LIBERO language sweep saturated on canonical-compatible prompts: all five models were at 100% for P0-P3/C4 on three trials. That means LIBERO task 0 is still useful for bad-prompt persistence, but it is weak evidence for whether prompt granularity improves task performance.

The next run should therefore prioritize PolaRiS `DROID-FoodBussing`:

| Prompt band | Goal | Expected signal |
| --- | --- | --- |
| Level 1 canonical | Baseline task completion | Does the policy do the task at all? |
| Level 2 object-enumerated | Object-order and target grounding | Does first-contact / first-lift object change? |
| Level 3 step-decomposed | Procedural control | Does gripper timing, reach path, or repeat behavior improve? |

For PolaRiS, raw success is not enough. The review should score partial progress with the FoodBussing rubric above, then plot prompt granularity versus substep score. This is the better place to test whether richer language improves performance.

## What To Compare

The key comparison is not just success rate. For each model, compare:

- Does first-contact object change with object-enumerated prompts?
- Does gripper timing change with step-decomposed prompts?
- Does action direction change immediately after prompt perturbation?
- Does the model ignore long instructions and execute the same action chunk?
- Does the model become worse when the prompt is too procedural?
- Are WAM-style policies less language-sensitive than direct VLAs?

The expected failure mode is that models follow the scene prior more than the prompt. That is still useful evidence.
