---
type: experiment-design
topic: robotics-world-models
status: corrected-draft
created: 2026-06-02
---

# Phase 3 Rollout Task Catalogue

## Core Experiment

Phase 3 is about prediction-vs-reality mismatch for models that can produce a future.

For each rollout window:

```text
C = current observation
P = decoded predicted future from the model/world model
R = actual observed future after execution
a = action chunk executed / proposed
```

The measurements are:

```text
semantic_prediction_error      = d(Encoder(P), Encoder(R))
actual_action_recovery_error   = || IDM(C, R) - a ||
imagined_action_consistency    = || IDM(C, P) - a ||
residual_correction_magnitude  = || IDM(P, R) ||
```

The model-private latent is not the evaluation interface.

Canonical pipeline:

```text
model future latent
  -> model decoder
  -> RGB future P
  -> shared semantic encoder
  -> IDM / semantic metrics
```

This removes the model-specific latent-space problem from Phase 3.

## Models In Scope

The relevant model class is:

```text
policy / WAM / simulator that can provide both action and future prediction,
or can take a candidate action and generate a future.
```

| Family | How it gives `P` | Current interpretation |
| --- | --- | --- |
| Cosmos Policy 2 / Predict2 | Native future images, wrist future images, value, action chunk. | Already verified in our LIBERO setup. |
| Cosmos 3 / Cosmos Framework | Current release advertises policy and forward-dynamics/future-observation generation, including DROID/UMI support. | New candidate; verify when VM is up. |
| Fast-WAM variants | Direct Fast-WAM skips future at inference, but Joint/IDM/future-visualization paths can expose future video. | Need to run the future branch, not only direct Fast-WAM. |
| GE stack | GE-Act is the action decoder; GE-Base/GE-Sim provide video/future generation. | Treat "GE" as valid only when future video is returned. |
| DreamZero | Joint action + future-video WAM. | Conceptually valid; not first until runnable. |

Pi/Molmo/OpenPI-style VLAs are not the core model for this phase unless paired with an external action-conditioned world model:

```text
C + a_from_vla -> external_WM -> P
execute a_from_vla -> R
```

That secondary setup is useful, but the claim becomes:

```text
external world model scoring of VLA actions
```

not:

```text
the VLA's own prediction honesty
```

## Actual Bottleneck

The hard part is not defining the metric anymore. The hard part is building a task/outcome distribution with:

```text
successes
partial / moderate successes
failures
```

without accidentally measuring:

```text
wrong camera convention
wrong action API
bad decoder wiring
one task overfit
one perturbation artifact
one model's native benchmark advantage
```

So task selection must be empirical, not just chosen by intuition.

## Discovery Tree

The discovery order should be:

```text
1. Environment
2. Tasks inside that environment
3. Which future-capable model variants can run there
4. Empirical outcome bucket per model-task pair
5. Balanced final task set
```

Do not start from the model alone. A model can be strong in one environment and weak in another because the environment changes:

```text
robot embodiment
camera contract
action space
control frequency
task distribution
success predicate
object/layout diversity
```

The first question is:

```text
Where can we run closed-loop rollouts and get P, R, a?
```

The second question is:

```text
Within that environment, which tasks give success, partial success, and failure?
```

## Environment Catalogue

| Environment | What it is | Task families | Outcome expectation | Which models matter first | Status for us |
| --- | --- | --- | --- | --- | --- |
| LIBERO | MuJoCo/robosuite Franka benchmark. | Spatial, Object, Goal, Long; 40 tasks total. | Clean tasks are often saturated; Long and perturbations should create moderate/failure cases. | Cosmos Policy 2/3, Fast-WAM variants, GE future path, pi/Molmo LIBERO as secondary controls. | Already runnable; Cosmos future output verified. |
| LIBERO-Plus style perturbations | Robustness extension/taxonomy over LIBERO. | Same base tasks, perturb camera/robot/language/lighting/background/noise/layout. | Best way to turn saturated LIBERO into balanced outcomes without changing environment. | Same as LIBERO. | Need implement/port perturbation branches locally. |
| RoboTwin / RoboTwin-Plus | SAPIEN/ManiSkill3 bimanual benchmark. | 50+ collaborative dual-arm tasks, clean + randomized. | Naturally harder than LIBERO; good moderate/failure source. | Fast-WAM family first; LingBot-like WAMs if available. | Fast-WAM repo supports RoboTwin eval; not yet set up locally. |
| RoboCasa | Franka kitchen benchmark. | 24 kitchen manipulation tasks. | Less saturated than LIBERO; Cosmos reports lower average than LIBERO, so likely better success/failure spread. | Cosmos Policy. | Not wired locally. Good next environment if Cosmos setup is easy. |
| ALOHA | Real bimanual manipulation setup. | 4 bimanual tasks in Cosmos paper. | Useful but not a cheap simulation source for our immediate task search. | Cosmos Policy. | Not immediate. |
| RoboLab | Simulation benchmark used in Cosmos 3 DROID policy quickstart. | Example: `BananaInBowlTask`; likely multiple task-generalist policy tasks. | Potential new source of DROID/Cosmos3 success/failure. | Cosmos 3 Policy DROID. | New candidate; verify after VM setup. |
| PolaRiS | DROID-like real-to-sim closed-loop environment. | FoodBussing, MoveLatteCup, TapeIntoContainer candidates. | Already shows success, partial, and failure for DROID policies. | DreamZero-DROID, Cosmos3/DROID forward dynamics, external-WM scoring of pi/Molmo actions. | Good outcome distribution; needs future predictor for core Phase 3. |
| MolmoSpaces | Broad simulated DROID-style/multi-robot ecosystem. | Pick, pick-place, open/close, navigation/manipulation. | Good OOD stress source; likely failure-rich. | Molmo/pi as action policies, but core Phase 3 requires DreamZero/Cosmos3/external WM. | Not yet wired. |
| EWMBench | Offline future-video benchmark, not closed-loop environment. | 10 held-out AgiBot-style video tasks with substeps. | Good for future-quality metrics, not policy rollouts. | GE-Base/GE-Sim-style video prediction. | Use as metric template, not task rollout source. |

## Model-Environment Strength Map

This is the table we use to decide where to search for tasks.

| Model family | Strong / likely saturated | Moderate / promising | Weak or not immediate | Task search implication |
| --- | --- | --- | --- | --- |
| Cosmos Policy 2 / LIBERO | Clean LIBERO, especially simple Spatial/Object tasks. | LIBERO Long, LIBERO perturbations, RoboCasa if setup works. | PolaRiS/DROID without Cosmos3 or adapter. | Start with LIBERO task triage, then RoboCasa if available. |
| Cosmos 3 / DROID | RoboLab / DROID-style setup if official quickstart works. | PolaRiS DROID tasks if observations/actions can be bridged. | LIBERO unless a LIBERO policy/checkpoint exists. | Verify RoboLab first; then test PolaRiS FoodBussing/TapeIntoContainer. |
| Fast-WAM future-capable variants | Clean LIBERO likely strong/saturated. | RoboTwin clean/randomized; LIBERO perturbations. | PolaRiS/DROID. | Use LIBERO for sanity, RoboTwin for harder distribution. |
| GE future path / GE-Sim | GE native AgiBot/EWMBench-style prediction tasks; possibly LIBERO if our runner can return video. | LIBERO perturbations if future video can be returned for same action. | PolaRiS/DROID without adaptation. | First verify GE future output, then run LIBERO/RoboTwin-like tasks only if compatible. |
| DreamZero-DROID | DROID/PolaRiS-like settings if runnable. | PolaRiS task perturbations and DROID-style OOD. | LIBERO without adaptation. | Use PolaRiS task distribution after one successful DreamZero rollout. |
| Pi/Molmo action-only | LIBERO/PolaRiS depending checkpoint. | As action sources for external WM scoring. | Core Phase 3 without an external WM. | Controls only; do not use to define core P-vs-R task set. |

## Task Catalogue By Environment

### LIBERO

Task source:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Expected bins:

| Suite | Expected bucket | Why |
| --- | --- | --- |
| `libero_spatial` easy tasks | strong | Existing task 0 is success-saturated across models. |
| `libero_object` with distractors | strong to moderate | Object identity failures are more likely than simple spatial pick-place. |
| `libero_goal` | moderate | Goal-state tasks should expose future-progress mismatch. |
| `libero_10` | moderate to weak | Long-horizon tasks are better for planning/future-prediction failure. |

LIBERO perturbation catalogue:

| Perturbation | Use when | Expected bucket shift |
| --- | --- | --- |
| mild camera crop/framing | clean task is too easy | strong -> moderate |
| wrist occlusion/noise | model relies on wrist/contact view | strong/moderate -> failure |
| robot initial-state shift | action grounding/recovery is the target | strong -> moderate/failure |
| object pose/layout shift | spatial task is too saturated | strong -> moderate |
| distractor object | object grounding is too easy | strong -> moderate/failure |
| language conflict/no-op | context-faithfulness probe | not enough for Cosmos task 0; use after other axes |

Immediate LIBERO triage:

```text
Cosmos first:
  all 40 tasks if feasible, 1 seed each
  otherwise 3 tasks per suite

Record:
  success
  number of replans
  decoded future P
  observed future R
  action chunk a
```

Then choose:

```text
2 strong tasks
2 moderate tasks
2 weak tasks
```

### RoboTwin

Task source:

```text
RoboTwin / RoboTwin-Plus task registry
```

Why it matters:

```text
Fast-WAM reports RoboTwin support and the benchmark is harder than LIBERO.
RoboTwin has randomized task variants, so it is a better source of moderate/failure outcomes.
```

Expected bins:

| Task type | Expected bucket | Why |
| --- | --- | --- |
| clean easy bimanual tasks | strong/moderate | Fast-WAM is reported strong here. |
| randomized bimanual tasks | moderate | Domain randomization should expose robustness differences. |
| contact/cooperation-heavy tasks | moderate/weak | Good for prediction/action mismatch. |
| RoboTwin-Plus perturbations | weak/failure | Best for controlled failure if clean is saturated. |

Use only after:

```text
Fast-WAM future-capable path is verified.
```

### RoboCasa

Task source:

```text
24 Franka kitchen manipulation tasks
```

Why it matters:

```text
Cosmos reports strong but not saturated performance here.
This may naturally produce mixed success/failure without artificial perturbations.
```

Expected bins:

| Task type | Expected bucket |
| --- | --- |
| simple pick/place kitchen tasks | strong/moderate |
| multi-stage kitchen tasks | moderate |
| object/layout OOD kitchen tasks | weak/moderate |

Use if:

```text
Cosmos RoboCasa runner/checkpoints are feasible on the VM.
```

### RoboLab / Cosmos 3

Task source:

```text
Cosmos3-Nano-Policy-DROID official quickstart uses RoboLab,
example task: BananaInBowlTask.
```

Why it matters:

```text
This may give us a newer DROID-style policy/future model stack without adapting older Cosmos Policy 2.
```

Expected bins:

| Task type | Expected bucket |
| --- | --- |
| official quickstart task | strong/instrumentation |
| multiple RoboLab tasks | unknown; discover empirically |
| parallel sub-environments | useful for quickly estimating success distribution |

Use if:

```text
Cosmos3 Docker/server/client setup works on the VM.
```

### PolaRiS

Task source:

```text
DROID-FoodBussing
DROID-MoveLatteCup
DROID-TapeIntoContainer
```

Existing evidence:

```text
FoodBussing already has success, partial progress, and failures for DROID policies.
```

Expected bins:

| Task | Expected bucket | Why |
| --- | --- | --- |
| `DROID-MoveLatteCup` | strong/moderate | Single-object, less ambiguous. |
| `DROID-FoodBussing` | moderate | Multi-object partial progress already observed. |
| `DROID-TapeIntoContainer` | moderate/weak | Container/contact-sensitive. |

Use only for core Phase 3 when one of these is true:

```text
DreamZero-DROID produces P on PolaRiS/DROID
Cosmos3/DROID forward dynamics produces P for PolaRiS observations/actions
another DROID-compatible WM produces P
```

Without that, PolaRiS is still useful but only for action-only baselines.

### MolmoSpaces

Task source:

```text
MolmoSpaces benchmark / DROID-config robot tasks
```

Why it matters:

```text
broader scene/object/task variation than PolaRiS
good OOD distribution once DROID-compatible future prediction is available
```

Expected bins:

| Task type | Expected bucket |
| --- | --- |
| frequent DROID-like pick/place | strong/moderate |
| novel object/layout pick-place | moderate/weak |
| articulated open/close tasks | moderate/weak |
| navigation/mobile tasks | outside immediate manipulation scope |

Use after:

```text
PolaRiS or Cosmos3 DROID future path is working.
```

## Outcome Labels

Use three bins.

| Bin | Definition | Why it matters |
| --- | --- | --- |
| success | simulator success or full rubric completion | Checks low-error cases. |
| partial | progress > 0 but no success, or some manually scored substeps completed | Best signal for diagnosing where prediction/action diverges. |
| failure | no meaningful progress or early wrong-object/wrong-motion behavior | Checks high-error cases. |

For LIBERO, success is available automatically, but partial progress may need manual/rubric review.

For PolaRiS/FoodBussing-style tasks, use progress plus substep rubrics.

## Task Distribution Strategy

Do not start with a hand-picked final task list.

Use a three-stage selection process.

### Stage 0 - Future-Output Verification

For each model variant:

```text
run 1 task
save C, P, R, a
verify decoded future P is visually sane
verify P and R are time-aligned
verify actions are logged in executable units
```

Pass/fail:

| Family | Verification |
| --- | --- |
| Cosmos Policy 2 | Already mostly passed on LIBERO task 0. |
| Cosmos 3 | Verify install + one DROID or LIBERO forward-dynamics query. |
| Fast-WAM Joint/IDM/future branch | Run `visualize_future_video` or equivalent and confirm predicted-vs-actual clips. |
| GE-Sim / GE future generation | Run public GE-Sim inference, then try a LIBERO-aligned observation/action pair. |
| DreamZero | Run one DROID/PolaRiS/sim-eval rollout with generated future MP4 and actions. |

Only models that pass this stage enter core Phase 3.

### Stage 1 - Task Triage Sweep

Run a cheap native-environment sweep to discover task difficulty.

For the LIBERO track:

```text
models:
  Cosmos Policy 2/3
  Fast-WAM future-capable variant
  GE future-capable variant

tasks:
  all available LIBERO suites if feasible
  at minimum: spatial, object, goal, libero_10

seeds:
  1 seed per task for triage
```

Record:

```text
success/failure
partial progress if easy to score
number of replan queries
future frames saved
action chunks saved
```

Bucket each model-task pair:

```text
strong:   succeeds cleanly
moderate: mixed or partial progress
weak:     fails cleanly
```

This is how we avoid guessing which tasks are "hard".

### Stage 2 - Balanced Collection

Select tasks from the empirical buckets.

Target per model family:

```text
2 strong tasks
2 moderate tasks
2 weak tasks
```

Also force category diversity:

```text
spatial relation task
object identity task
goal-state task
long-horizon task
contact-sensitive / container task if available
```

Then run:

```text
3-5 seeds per selected task
clean condition first
```

This gives natural successes, partials, and failures without perturbations.

### Stage 3 - Controlled Perturbations

Only after clean triage, add one perturbation at a time.

Use perturbations to fill outcome gaps:

```text
if a task is too easy -> add mild camera/state/layout perturbation
if a task is too hard -> use easier seed/init or simpler task
if partials are missing -> use object pose/layout perturbations
```

Perturbation axes:

| Axis | Why useful | Confound risk |
| --- | --- | --- |
| camera crop / framing | Tests observation contract and prediction robustness. | Can become pure preprocessing bug. |
| robot initial state | Tests recovery and action grounding. | Can become unreachable start state. |
| object layout / pose | Tests spatial reasoning and world prediction. | Needs valid success predicate. |
| distractor object | Tests object grounding. | Can break scene semantics if unrealistic. |
| image noise / occlusion | Tests semantic encoder and perception sensitivity. | Can be too artificial. |
| language perturbation | Tests context conditioning. | Weak for Cosmos task 0; existing runs saturate. |

Language perturbations should not be the main failure source for Cosmos task 0 because existing local runs show it still succeeds under bad prompts.

## Environment Tracks

### Track A - LIBERO Native Future-Predictor Track

This is the first serious track because Cosmos Policy 2 is already working here.

Candidate suites:

| Suite | Role |
| --- | --- |
| `libero_spatial` | Easy/known sanity and spatial relations. |
| `libero_object` | Object identity and distractors. |
| `libero_goal` | Goal-state matching. |
| `libero_10` | Long-horizon / planning-sensitive failures. |

First triage:

```text
Run Cosmos across all 40 LIBERO tasks with 1 seed if feasible.
If not feasible, run 3 tasks per suite.
```

Then select:

```text
2 easy / 2 medium / 2 hard tasks
```

For Fast-WAM/GE:

```text
only include the future-capable variants/configs
run the same selected tasks after future-output verification
```

### Track B - DROID / PolaRiS Future-Predictor Track

PolaRiS is useful only when paired with a DROID-compatible future predictor.

Candidate future predictors:

```text
DreamZero-DROID
Cosmos 3 / Cosmos Framework DROID forward dynamics
other DROID-compatible action-conditioned WM
```

Candidate tasks:

| Task | Role |
| --- | --- |
| `DROID-FoodBussing` | Multi-object, existing success/failure/partial evidence. |
| `DROID-MoveLatteCup` | Easier single-object control. |
| `DROID-TapeIntoContainer` | Contact/container, likely hard and action-sensitive. |

This track is attractive because PolaRiS already gives mixed outcomes, but it is not ready for core Phase 3 until a future predictor is verified.

### Track C - External-WM Scoring Of VLA Actions

This comes after Track A or B has a working action-conditioned future model.

Setup:

```text
policy: pi0.5 / MolmoAct2 / GE-Act action-only
world model: Cosmos 3 / GE-Sim / other action-conditioned WM
```

For each step:

```text
C -> VLA gives a
C, a -> external WM gives P
env executes a -> R
compare P vs R
```

This is useful for asking:

```text
Can a world model diagnose failures of action-only VLAs?
```

It is not the same as:

```text
Does the VLA itself have honest prediction?
```

## Minimum Useful Dataset

For the first real Phase 3 result:

```text
one environment track
one future-predicting model family
6 selected tasks after triage
3 seeds each
clean condition
then perturb only to fill missing bins
```

Target:

```text
20+ successful rollout windows
20+ partial/moderate rollout windows
20+ failed rollout windows
```

The unit of analysis is not just episode success. It is replan/query windows:

```text
C at query time
P at model-predicted horizon
R at corresponding executed horizon
a action chunk
episode outcome
substep/progress outcome
```

## First VM Actions

1. Dump exact LIBERO task names.
2. Run Cosmos Policy 2 triage:

```text
all LIBERO tasks, 1 seed each
save C, P, R, a
```

3. Summarize task difficulty:

```text
success / partial / failure
per suite
per task
```

4. Pick 6 LIBERO tasks:

```text
2 easy
2 medium
2 hard
cover spatial/object/goal/long
```

5. Re-run selected tasks:

```text
3-5 seeds each
full future/action/observation logging
```

6. In parallel, run future-output verification for:

```text
Fast-WAM future-capable variant
GE-Sim / GE future-generation path
Cosmos 3 DROID/forward-dynamics path
DreamZero-DROID if hardware/API is ready
```

Only after this should we expand beyond Cosmos.
