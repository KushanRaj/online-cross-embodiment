---
type: progress-report
topic: context-faithfulness-world-models
status: presentation-draft
created: 2026-05-29
---

# Progress Report - Context Faithfulness and World Model Rollouts

## One-Line Thesis

We are testing whether robot policies actually obey their input context, and whether world-model futures can expose or improve action selection beyond ordinary benchmark success.

The working hypothesis is:

```text
If success stays high even when the language asks for a different behavior, the model is likely following a scene/task prior rather than the prompt.
If predicted futures diverge from observed futures before failure, world models can become a planning or diagnostic signal.
```

## What We Set Up

### Environments

| Environment | Role | Status |
| --- | --- | --- |
| LIBERO / robosuite | Clean standardized benchmark for apples-to-apples policy comparison | Running |
| PolaRiS / DROID-SIM | DROID-style closed-loop environment for harder transfer/action-interface tests | Running |
| Cosmos LIBERO demo data | Source for real-vs-model future-pair dataset | Smoke dataset generated |

### Models / Policies

| Model | Type | Current status |
| --- | --- | --- |
| Pi 0.5 LIBERO | VLA baseline | Runs on LIBERO |
| MolmoAct2 LIBERO | VLA/action baseline | Runs on LIBERO |
| Cosmos Policy LIBERO Predict2 2B | World-action model / future-prediction policy | Runs on LIBERO |
| GE-Act LIBERO spatial checkpoint | World-action model | Runs on LIBERO |
| FastWAM LIBERO release checkpoint | Video co-training WAM-style policy | Runs on LIBERO |
| Pi 0.5 DROID base | DROID VLA baseline | Runs in PolaRiS but fails FoodBussing |
| Pi 0.5 PolaRiS/JAX-adapted checkpoint | PolaRiS-aligned Pi checkpoint | Runs in PolaRiS and can succeed |
| MolmoAct2 DROID | DROID VLA/action baseline | Runs in PolaRiS but weak on FoodBussing |

## Smoke Test Result: All LIBERO Policies Can Run

Task:

```text
LIBERO suite: libero_spatial
Task id: 0
Instruction: pick up the black bowl between the plate and the ramekin and place it on the plate
```

Single-trial smoke results:

| Model | Result |
| --- | ---: |
| Cosmos Policy | 1 / 1 success |
| FastWAM | 1 / 1 success |
| GE-Act | 1 / 1 success |
| Pi 0.5 LIBERO | 1 / 1 success |
| MolmoAct2 LIBERO | 1 / 1 success |

Artifact note: rollout videos are listed in `run-artifacts/world-model-rollouts-20260529/README.md`.

Takeaway: the evaluation stack is real. We can now run the same LIBERO task across direct VLAs and WAM-style models.

## LIBERO Language-Adherence Probe

We then changed only the language prompt while keeping the LIBERO task/scene fixed.

Prompt ladder:

| ID | Type | Example |
| --- | --- | --- |
| P0 | Canonical | Pick up the black bowl between the plate and the ramekin and place it on the plate. |
| P1 | Paraphrase | Take the black bowl that is between the plate and the ramekin, and put it onto the plate. |
| P2 | Expanded relation | Find the black bowl positioned between the plate and the ramekin. Grasp that black bowl and place it on top of the plate. |
| P3 | Step-decomposed | Move the gripper above the black bowl... grasp... move over the plate... release it. |
| C1-C3 | Conflicting but plausible prompts | Pick up the ramekin / move the plate / put bowl next to the plate. |
| O1-O2 | OOD or inhibitory prompts | Move near the black bowl / Do not pick up anything. |

Aggregate result over 3 trials per prompt:

| Model | Valid prompt success | Bad-prompt canonical persistence | Prompt sensitivity |
| --- | ---: | ---: | ---: |
| Pi 0.5 | 100% | 93% | 7% |
| MolmoAct2 | 100% | 80% | 20% |
| Cosmos | 100% | 100% | 0% |
| GE-Act | 100% | 87% | 13% |
| FastWAM | 100% | 93% | 7% |

Interpretation:

- All models are perfect on canonical-compatible prompts.
- Most models keep succeeding even on prompts that should not satisfy the canonical task.
- This is not automatically "good" performance. On bad prompts, success is suspicious because the simulator success predicate still checks the original canonical task.
- The useful signal is prompt sensitivity, not raw success.

Main early read:

```text
On this saturated LIBERO task, policies appear heavily anchored to the scene/task prior.
The language prompt changes behavior only weakly, especially for Cosmos and FastWAM.
```

Manual review priority:

- MolmoAct2 C3 failure,
- Pi 0.5 O2 failure,
- MolmoAct2 O2 failure,
- GE-Act O2 failure,
- FastWAM C2 failure.

These are the cases where prompt changes actually disrupted canonical success and may show real language adherence or policy destabilization.

## PolaRiS FoodBussing Result

Task:

```text
DROID-FoodBussing
Prompt: Put all the foods in the bowl.
```

Initial 450-step runs:

| Model | Episodes | Success | Progress |
| --- | ---: | ---: | ---: |
| Pi 0.5 DROID base | 3 | 0 / 3 | 0.00, 0.00, 0.00 |
| MolmoAct2 DROID | 3 | 0 / 3 | 0.17, 0.17, 0.17 |
| Pi 0.5 PolaRiS checkpoint | 3 | 0 / 3 | 0.33, 0.17, 0.67 |

900-step runs:

| Model | Success | Progress |
| --- | ---: | ---: |
| Pi 0.5 DROID base + velocity adapter | 0 / 1 | 0.00 |
| MolmoAct2 DROID | 0 / 1 | 0.00 |
| Pi 0.5 PolaRiS checkpoint | 1 / 1 | 1.00 |

Interpretation:

```text
PolaRiS is much more diagnostic than LIBERO for our immediate question.
The base DROID checkpoint is not simply "a little worse"; it appears action-interface/domain misaligned.
The PolaRiS-adapted checkpoint can solve the task when given enough horizon.
```

This supports the action-grounding thesis:

- The model may understand the scene semantically.
- The failure can still come from camera contract, action representation, proprio/state encoding, control frequency, or embodiment-specific decoding.
- A small translation layer may help, but naive velocity/joint conversion was not enough in the current FoodBussing run.

## World-Model Future Dataset Work

We generated Cosmos-style real/model future-pair artifacts.

Current dataset smoke:

| Dataset | Model | Horizon | Count |
| --- | --- | ---: | ---: |
| `cosmos-real-model-smoke` | Cosmos Policy LIBERO Predict2 2B | 16 | 20 real/model future pairs |
| `cosmos-real-model-1k` | Cosmos Policy LIBERO Predict2 2B | sampled LIBERO demos | larger 1k-style artifact present |

What each sample gives us:

- current external frame,
- current wrist frame,
- real future external/wrist frame,
- model-predicted future external/wrist frame,
- task language,
- LIBERO action-space metadata.

Why this matters:

```text
This is the seed of the "prediction honesty" experiment:
compare predicted future versus observed future in semantic/action-relevant latent space.
```

The next step is not pixel reconstruction scoring alone. We should score future mismatch using DINO/SigLIP/V-JEPA/action-relevant probes.

## Paper-Level Insights Feeding The Experiment

### Reconstruction Versus Semantic Latents

The latent-space papers converge on one useful point:

```text
High-fidelity visual reconstruction is not the same as action-relevant state understanding.
```

For robotics, the key retained information is:

- object identity,
- object pose,
- hand/object relation,
- affordance,
- contact/progress state,
- action recoverability.

This motivates evaluating predicted futures in semantic latent space, not just pixels.

### Cosmos Policy

Cosmos is the clearest planning inspiration because it trains a value function and uses imagined futures/action chunks at inference.

Important takeaway:

```text
Planning requires a value/scoring interface. Future prediction alone is not enough.
```

For our work, Cosmos suggests two experiments:

- counterfactual action futures: try candidate actions and compare future states;
- value/progress scoring: predict whether a future is closer to task completion.

### VLA-JEPA

VLA-JEPA adds a latent visual-dynamics head to a Pi-style VLA setup.

Takeaway:

```text
World-model supervision can be used as auxiliary training, even if the future generator is discarded at inference.
```

This is a useful middle ground between direct VLA behavior cloning and full inference-time planning.

### FastWAM

FastWAM is useful because it does not generate future frames at inference. It uses video/world prediction during training to improve action behavior.

Takeaway:

```text
World-model training can help representations, but LIBERO results suggest robot data and task alignment still dominate.
```

In our prompt sweep, FastWAM was mostly insensitive to bad language prompts, which makes it useful as a "strong task-prior" comparison.

### LingBot-VA

LingBot is the most explicitly causal/sequential architecture among the papers we reviewed: language, video future prediction, and action prediction are modularized.

Takeaway:

```text
It is conceptually attractive for causal rollout, but likely too slow and over-engineered for our immediate experiments.
```

## Current Core Findings

1. Clean LIBERO task success is saturated.

All five policies can solve the same LIBERO spatial task. This is useful for infrastructure validation but weak as a research result.

2. Prompt perturbation is a better signal than raw success.

Bad prompts with continued success indicate canonical task-prior persistence. That is exactly what we want to measure.

3. PolaRiS exposes the real bottleneck.

Base Pi and Molmo do not transfer cleanly to FoodBussing. The PolaRiS-adapted Pi checkpoint performs much better, especially with a longer horizon.

4. World-model rollouts are now available enough for a first diagnostic.

Cosmos can produce model futures on LIBERO. We can begin measuring future honesty against real rollout futures.

5. The immediate claim should be modest and concrete.

We should not claim that WAMs plan better yet. We can claim that we have a working testbed to measure:

- context faithfulness,
- task-prior persistence,
- action-interface failures,
- future-prediction honesty.

## Recommended Presentation Framing

### Slide 1: Claim

```text
Benchmark success does not prove context faithfulness.
We need to test whether policies change behavior when only the prompt changes.
```

### Slide 2: Setup

```text
Same scene + same initial state + same embodiment + changed language only.
Models: Pi 0.5, MolmoAct2, Cosmos, GE-Act, FastWAM.
Environments: LIBERO for clean comparison, PolaRiS for DROID-style transfer stress.
```

### Slide 3: LIBERO Result

```text
All models get 100% on valid prompt variants.
Most models also keep succeeding on bad/conflicting prompts.
This suggests canonical task-prior persistence.
```

Plot:

```text
run-artifacts/presentation-20260529/libero_prompt_sensitivity_scatter.svg
```

Use this table:

| Model | Valid prompts | Bad prompts | Sensitivity |
| --- | ---: | ---: | ---: |
| Pi 0.5 | 100% | 93% | 7% |
| MolmoAct2 | 100% | 80% | 20% |
| Cosmos | 100% | 100% | 0% |
| GE-Act | 100% | 87% | 13% |
| FastWAM | 100% | 93% | 7% |

### Slide 4: PolaRiS Result

```text
PolaRiS exposes action-interface/domain alignment failures that LIBERO hides.
Base DROID policies struggle; the PolaRiS-adapted Pi checkpoint is meaningfully better.
```

Plot:

```text
run-artifacts/presentation-20260529/polaris_foodbussing_progress_scatter.svg
```

Use this table:

| Model | 450-step success | 450-step progress | 900-step success |
| --- | ---: | ---: | ---: |
| Pi 0.5 DROID base | 0 / 3 | 0.00 avg | Not successful with adapter |
| MolmoAct2 DROID | 0 / 3 | 0.17 avg | 0 / 1 |
| Pi 0.5 PolaRiS checkpoint | 0 / 3 | 0.39 avg | 1 / 1 |

### Slide 5: Why World Models

```text
Future prediction gives us a way to ask:
what did the policy/world model think would happen, and did reality agree?
```

Current status: Cosmos can produce real/model future pairs on LIBERO, so we can now build the first prediction-honesty probe.

### Slide 6: Next Experiment

```text
Run prompt ladders on PolaRiS FoodBussing and score partial progress:
reach, grasp, lift, move-to-bowl, release, repeat.
Then compare action divergence and future-prediction mismatch across prompt variants.
```

### Problem

Modern VLAs and WAMs often score well on standard tasks, but it is unclear whether they are obeying the current context or replaying a learned task prior.

### Method

Hold the scene, embodiment, and initial state fixed. Change only the prompt. Then measure:

- task success,
- action divergence,
- first contacted object,
- gripper timing,
- rollout video behavior,
- predicted future mismatch.

### Early Result

On LIBERO task 0, all models succeed on valid prompts, but most also succeed on conflicting prompts. This suggests strong canonical task-prior persistence.

On PolaRiS FoodBussing, action-interface/domain alignment dominates: base DROID checkpoints fail, while the PolaRiS-adapted checkpoint can complete the task with enough horizon.

### Research Direction

Use world-model predicted futures as an honesty signal:

```text
Does the model's imagined future change when the prompt/action changes?
Does imagined-vs-real future mismatch predict failure before the final success label?
Can that mismatch become a reward/value signal for planning or adaptation?
```

## Immediate Next Steps

1. Finish visual review of the LIBERO bad-prompt failures.
2. Re-run PolaRiS FoodBussing prompt ladder:
   - canonical,
   - object-enumerated,
   - step-decomposed,
   - approach-only negative control.
3. Score PolaRiS with partial-progress labels, not just binary success.
4. Plot prompt type versus:
   - success/progress,
   - first object contacted,
   - first object lifted,
   - action divergence from canonical rollout.
5. Build the semantic future-mismatch probe from Cosmos real/model future pairs.
6. Add one non-saturated LIBERO task from `libero_object` or `libero_goal` to avoid task-0 overfitting.

## Artifacts

| Artifact | Path |
| --- | --- |
| LIBERO prompt-sensitivity plot | `run-artifacts/presentation-20260529/libero_prompt_sensitivity_scatter.svg` |
| PolaRiS FoodBussing progress plot | `run-artifacts/presentation-20260529/polaris_foodbussing_progress_scatter.svg` |
| LIBERO language sweep review | `run-artifacts/libero-language-sweep-20260529-1120/language_sensitivity_review.md` |
| World-model rollout smoke tests | `run-artifacts/world-model-rollouts-20260529/README.md` |
| PolaRiS 900-step videos/results | `run-artifacts/polaris-900-20260528/` |
| PolaRiS 3-episode reruns | `run-artifacts/rerun3/` |
| Cosmos future-pair datasets | `run-artifacts/idm-datasets/` |
| Experiment plan | `experiment-design/polaris-libero-language-adherence-plan.md` |
| LIBERO runbook | `experiment-design/libero-language-adherence-runbook.md` |
