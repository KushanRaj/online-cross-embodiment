---
type: experiment-design
topic: robotics-world-models
status: groundwork
created: 2026-05-29
---

# Shared Semantic IDM - Prediction Reality Deviation

## Goal

Train one embodiment-specific, model-agnostic inverse dynamics model (IDM) that maps an observed transition to the action chunk that explains it.

```text
IDM(obs_t, obs_t+k, proprio_t, instruction) -> action_t:t+k
```

Then use this IDM as a diagnostic tool:

```text
a_real = IDM(current_observation, actual_future_observation)
a_pred = IDM(current_observation, model_predicted_future)

deviation = a_real - a_pred
```

The core question is whether this deviation signal tracks rollout failure, prompt/image perturbations, or action-translation weakness.

## What We Borrow From Reconstruction Or Semantics

The reconstruction-vs-semantics paper trains IDM probes to test whether different latent spaces preserve action-relevant geometry.

Their IDM setup:

```text
input:
  z_t, z_t+1, ..., z_t+k
  where z_t = frozen_encoder(image_t)

target:
  action chunk a_t:t+k-1

horizons:
  k = 1 and k = 4

loss:
  Smooth-L1

metric:
  Pearson r between predicted and true continuous action dimensions
```

Their architecture:

```text
native encoder patch grid N x D
linear projection into IDM width
spatial + temporal positional embeddings
flattened sequence over (k + 1) frames
k learned CLS/action readout tokens
Transformer blocks
2-layer MLP action head
```

Their data:

```text
BridgeV2
~60K WidowX demonstrations
13 task families
RGB observations
7-DoF end-effector actions
language instructions
```

Their purpose was to compare latent spaces, so they trained one IDM per encoder space.

Our purpose is different. We are not asking which latent space is best for IDM training. We are asking whether a competent IDM can expose action-relevant mismatch between predicted future and observed reality.

Important horizon caveat:

```text
k = 1 and k = 4 came from their probe setup.
They are not automatically the right horizons for our model-generated futures.
```

For our experiment, the IDM horizon must match the future frame/video that the world model actually emits. If Cosmos gives us a predicted future after 16 action steps, the matching model-pair IDM is `IDM_k16`. If a GE LIBERO checkpoint produces a 36-step action chunk and a shorter decoded video clip, the manifest must record the effective frame/action horizon we selected from that decoded clip.

## Our IDM Contract

Use one shared semantic encoder and one IDM for a fixed embodiment/action space.

```text
image_t      -> shared encoder -> z_t
image_t+k    -> shared encoder -> z_future
delta_z      = z_future - z_t

IDM(z_t, z_future, delta_z, proprio_t, instruction, source_id) -> action chunk
```

The IDM is:

```text
embodiment-specific
model-agnostic
shared-semantic-feature based
trained on actual and generated transitions
```

The IDM is not:

```text
one model per DreamZero/Cosmos/GE-Act private latent
cross-embodiment
raw-pixel-first
the policy itself
```

## Decoder Requirement

Every world model must expose predicted RGB futures before it can participate in this experiment.

```text
model-specific future latent
  -> model-specific decoder
  -> predicted RGB future frame/video
  -> shared semantic encoder
  -> IDM input latent
```

The shared IDM never consumes DreamZero, Cosmos, or GE-Act private latents directly. It consumes semantic features extracted from decoded RGB futures.

This keeps the IDM comparable across models.

## Training Data

Each training example needs a known action label.

```text
image_t
image_t+k
proprio_t
instruction
source_id
optional model_id
target action chunk a_t:t+k
```

Possible buckets:

| Bucket | Input transition | Target action | Purpose |
| --- | --- | --- | --- |
| real -> real | observed current, observed future | executed action | Learn real action geometry. |
| sim -> sim | simulator current, simulator future | simulator action | Cheap volume; same action convention if using LIBERO/robosuite. |
| real -> model | observed current, model-predicted future | model-generated/planned action | Expose IDM to generated futures from real contexts. |
| model -> model | model-generated current/context, model-generated future | model-generated/planned action | Expose IDM to same-to-same generated distribution. |
| model -> real | model-predicted future, observed current/future | reversed or corrective action label | Ablation for whether reverse/correction supervision helps or hurts. |

Do not overbuild the mixture first. The first version should prove that the IDM works on actual trajectories. Generated futures and reverse rows are separate ablations, not part of the default training set.

Frame convention:

```text
Cosmos model-native convention:
  raw LIBERO HDF5 / live sim frame -> flipud

GE model-native convention:
  raw LIBERO HDF5 / live sim frame -> rot180
```

This is a repo/checkpoint convention, not a physics claim. Cosmos regenerated/evaluated LIBERO with a vertical flip to correct renderer orientation. GE's LIBERO wrapper rotates frames 180 degrees with the code comment "match train preprocessing." Therefore Cosmos and GE images can look mirrored relative to each other even when both are correct for their own checkpoint.

For model inference:

```text
feed each model its own native convention
```

For the Phase 1 real-to-real IDM:

```text
use one canonical LIBERO transform for every current/future frame in the manifest
```

For shared IDM / cross-model metrics later:

```text
canonicalize C, P, and R together before feature extraction
never transform only the generated frame
```

Manifests should carry per-frame transform fields rather than silently baking this into the dataloader.

Default first mix:

```text
100% actual transitions
0% generated transitions
0% reverse transitions
```

## Data Scale

The paper uses about 60K BridgeV2 demonstrations for the broader world-model setup. We do not need that scale for the first IDM.

First serious run:

```text
10K-50K transition windows
start with horizons that match available generated futures
single embodiment/action space
```

Current horizon policy:

| Data source | Horizon use | Why |
| --- | --- | --- |
| real -> real sanity data | k = 1 first, k = 4 second | k = 1 verifies local action geometry; k = 4 tests whether a later semantic transition still recovers an action chunk. |
| Cosmos real -> model | k = 16 in the current export | Cosmos emits a later predicted future/action chunk, so `IDM_k16` is the faithful comparison. |
| GE-Act diagnostic real -> model | use the released config's recorded horizon | The current LIBERO config reports an action chunk horizon of 36; decoded video has its own frame count, so metadata must record the selected future frame. |
| GE-Base / GE-Sim real -> model | record model-specific effective horizon | Do not assume k = 1 unless the model actually predicts the next control step/frame. |

Approximate collection routes:

```text
LIBERO/robosuite:
  100-500 trajectories
  ~100 usable transition windows per trajectory

DROID/Bridge-style logged data:
  sample 10K windows first
  scale to 50K if validation is unstable
```

Generated-transition data should be added only after the real transition IDM has a reasonable validation signal.

## Real To Model Data

The next data bucket should be `real -> model`.

For a sampled LIBERO demonstration state:

```text
input:
  real obs_t
  real proprio_t
  instruction

model query:
  Cosmos / GE-Sim-style action-conditioned future model predicts:
    model future obs_t+k
    model action chunk a_model_t:t+k
    optional value/confidence

stored row:
  real obs_t
  model obs_t+k
  a_model_t:t+k
  model_id
  optional real obs_t+k reference
```

This is the important bridge distribution. It exposes the IDM to decoded world-model futures while keeping the current context anchored in real/demo data.

For GE specifically, separate the modules:

```text
GE-Act:
  use for C -> a_model

GE-Base:
  use for C -> P_model when testing instruction-conditioned prediction honesty

GE-Sim:
  use for C, a_model -> P_model when testing action-conditioned futures

GE-Act return_video=True:
  diagnostic only unless the decoded future quality is empirically acceptable
```

The GE paper says GE-Act action pretraining disables video generation and supervises only the action decoder on ground-truth action trajectories. So GE-Act's decoded video branch is something we can test from the released pipeline, but GE-Sim is the paper-defined component for action-conditioned future video.

Do not query every LIBERO window at first. Use a sampled subset:

```text
smoke: 20-100 query states
first useful run: 1K-5K query states
larger run: only if validation says generated futures matter
```

## Reverse Data Ablation

Reverse rows should be an explicit ablation, not part of the default training mix.

For a forward row:

```text
obs_a, obs_b -> action_chunk
```

the approximate reverse row is:

```text
obs_b, obs_a -> reverse(action_chunk)
```

For LIBERO 7D end-effector delta actions, the first approximation is:

```text
reverse([dx, dy, dz, drx, dry, drz, grip])
  = [-dx, -dy, -dz, -drx, -dry, -drz, -grip]
```

For multi-step chunks, reverse the order too:

```text
[a_t, a_t+1, ..., a_t+k-1]
  -> [reverse(a_t+k-1), ..., reverse(a_t+1), reverse(a_t)]
```

This is useful but not physically exact. Contact dynamics, grasping, collisions, and object motion are not generally reversible. The gripper dimension is especially suspect because "undoing" a grasp is not always the negative of the close command.

Recommended ablations:

| Variant | Training mix | Why |
| --- | --- | --- |
| base | real -> real only | Checks whether IDM recovers action geometry from true transitions. |
| generated | real -> real + real -> model | Checks whether generated futures make the IDM robust to world-model outputs. |
| reverse-real | base + reversed real -> real | Tests whether reverse supervision helps or injects nonphysical labels. |
| reverse-generated | generated + reversed real -> model as model -> real | Tests whether correction-style supervision helps deviation scoring. |

Current decision: do not include reverse rows in the base training run. The main result should report reverse data separately. If it improves deviation correlation, it becomes a useful training trick. If it worsens validation or makes gripper predictions unstable, we drop it.

## Model Size And Budget

Freeze the visual encoder and cache features. Do not backprop through V-JEPA/DINO/SigLIP in the first run.

Target IDM:

```text
10M-50M trainable parameters
patch projection to width 512
4-6 Transformer layers
8 attention heads
MLP ratio 4
k learned readout tokens
2-layer action head
```

Budget on 48 GB L40:

```text
feature extraction:
  depends on encoder and frame count

IDM training on cached features:
  30 minutes to 2 hours target
```

The useful constraint is not GPU memory; it is keeping the training loop and data export simple enough that the result is inspectable.

## Encoder Choice

Start with the easiest strong semantic encoder available on the VM.

Candidate order:

```text
1. DINOv2 or SigLIP
2. V-JEPA / V-JEPA2 if setup is already practical
3. reconstruction encoder as a later ablation
```

The reconstruction-vs-semantics paper motivates semantic latents because action recoverability and task progress are often better preserved than in pixel-reconstruction latents. We do not need to reproduce the full encoder sweep first.

## Phase 1 - IDM Training

The first phase is only about learning a reliable inverse dynamics coordinate system.

Training input:

```text
C = current observation
F = future observation at horizon k
q = proprio/state at current step
l = language instruction
```

Training target:

```text
a = action chunk executed or generated between C and F
```

Model:

```text
IDM(C, F, q, l) -> a
```

Start with real transitions:

```text
C = real obs_t
F = real obs_t+k
a = executed action chunk a_t:t+k
```

Then add generated transitions as a separate training variant:

```text
C = real obs_t
F = model predicted obs_t+k
a = model generated/planned action chunk
```

The first goal is not control. The first goal is that the IDM learns a stable, action-relevant representation of visual transitions.

## Phase 2 - IDM Evaluation And Calibration

The IDM must pass basic recoverability tests before we trust deviation plots.

Validate on held-out actual transitions:

```text
action MSE / Smooth-L1
Pearson r over continuous action dimensions
EEF direction cosine similarity
gripper accuracy
k = 1 and k = 4 separately
```

Validate on generated-transition holdout:

```text
does predicted action stay numerically stable?
does source_id/model_id prevent distribution collapse?
does the IDM overfit generated artifacts?
```

If actual-transition validation is weak, deviation plots are meaningless.

Important calibration check:

```text
IDM(real obs_t, real obs_t+k) ~= executed action chunk
```

This comparison is not the experiment result. It is the anchor that tells us whether the IDM is usable. If this fails, any later "correction" or "mismatch" signal is just model error from the IDM itself.

## Phase 3 - Prediction Reality Experiment

At each rollout window, define the three-frame triangle:

```text
C = current frame / observation
P = predicted or expected future from the world model
R = actual observed future from the environment
a = action chunk actually executed or generated by the model
```

There are three transitions:

```text
C -> R : what actually happened
C -> P : what the model thought would happen
P -> R : residual between imagined outcome and reality
```

The central point is that `P vs R` alone is not enough. We want to know whether the mismatch is action-relevant, semantically meaningful, and predictive of failure.

### Experiment Metric 1 - Semantic Prediction Error

Compare the predicted future and observed future in a frozen representation space.

```text
z_P = Encoder(P)
z_R = Encoder(R)

semantic_prediction_error = distance(z_P, z_R)
```

This asks:

```text
Did the world model predict the same task-relevant future state that actually happened?
```

This is the reconstruction-vs-semantics axis. Pixel error may be dominated by texture, lighting, camera artifacts, or small render differences. Semantic encoders should better capture object state, robot-object relation, and task progress.

Report alongside:

```text
pixel error
LPIPS/perceptual error if easy
semantic encoder distance
```

### Experiment Metric 2 - Actual Action Recovery

Run the IDM on the real transition:

```text
a_real = IDM(C, R)

actual_action_recovery_error = ||a_real - a||
```

This asks:

```text
Can the IDM recover the action that actually produced the observed transition?
```

This is mostly an IDM sanity check. It is useful because it tells us whether `C -> R` is represented well enough to trust the IDM. It is not by itself the main novelty, because the true action label is already known.

### Experiment Metric 3 - Imagined Action Consistency

Run the IDM on the model-imagined transition:

```text
a_pred = IDM(C, P)

imagined_action_consistency_error = ||a_pred - a||
```

This asks:

```text
Is the future the model imagined consistent with the action it planned/executed?
```

This is a key internal-consistency test. If the model outputs action `a`, but the predicted future `P` implies a very different action under the IDM, then the model's action head and world prediction are not aligned.

Possible interpretations:

| Pattern | Interpretation |
| --- | --- |
| low semantic error, low imagined consistency error | Prediction and action are internally aligned and reality matches. |
| high semantic error, low imagined consistency error | Model is internally consistent but wrong about the world. |
| low semantic error, high imagined consistency error | Future may look plausible, but action translation is inconsistent. |
| high semantic error, high imagined consistency error | World prediction and action grounding are both unreliable. |

### Experiment Metric 4 - Residual Correction Magnitude

Run the IDM on the residual transition from imagined future to observed future:

```text
a_residual = IDM(P, R)

residual_correction_magnitude = ||a_residual||
```

This asks:

```text
How large is the action-space residual between where the model thought it would land and where reality landed?
```

This should be treated carefully. `P` is model-generated, not necessarily a physically valid state. Therefore `IDM(P, R)` is not automatically a real executable correction action.

The safe first interpretation is:

```text
IDM(P, R) = action-space residual diagnostic
```

not:

```text
IDM(P, R) = deployable correction policy
```

If this residual correlates with failure, time-to-failure, or perturbation severity, then we have evidence that prediction-vs-reality mismatch can be mapped into an action-relevant coordinate system.

## Main Diagnostic Readout

For each rollout step, compute:

```text
semantic_prediction_error      = d(Encoder(P), Encoder(R))
actual_action_recovery_error   = || IDM(C, R) - a ||
imagined_action_consistency    = || IDM(C, P) - a ||
residual_correction_magnitude  = || IDM(P, R) ||
```

Then compare these against:

```text
success / failure
time-to-failure
prompt perturbation
image perturbation
in-domain vs out-of-domain task
model identity: DreamZero vs Cosmos vs GE-Act
```

The strongest first result would be:

```text
semantic prediction error or residual correction magnitude rises before failure,
and/or imagined action consistency separates successful from failed rollouts better than pixel error.
```

## What Each Comparison Is For

| Comparison | IDM call | Primary use | Caveat |
| --- | --- | --- | --- |
| current -> actual | `IDM(C, R)` | Calibrate IDM against known executed action. | Not novel by itself. |
| current -> predicted | `IDM(C, P)` | Test whether model future is consistent with model action. | Depends on decoded future quality. |
| predicted -> actual | `IDM(P, R)` | Measure action-space residual between imagination and reality. | Diagnostic first, not guaranteed executable correction. |
| predicted vs actual semantic features | `d(E(P), E(R))` | Test semantic/task-state mismatch. | Encoder choice matters. |

## Expected Claims

Weak claim:

```text
semantic-space prediction error is a better failure signal than raw pixel reconstruction error.
```

Stronger claim:

```text
an IDM can convert prediction-vs-reality mismatch into an action-relevant residual.
```

Strongest claim:

```text
world-model reasoning can be decent while action translation is weak, and imagined-action consistency exposes this gap.
```

We should not claim online adaptation or closed-loop correction until residual vectors are tested as actual interventions.

## Logging Contract

For every transition or rollout window, save:

```text
run_id
model_id
source_id
environment
task_id
instruction
obs_t_rgb
observed_obs_t+k_rgb
predicted_obs_t+k_rgb
proprio_t
executed_or_generated_action_chunk
success/failure/progress label if available
decoder used
shared encoder used
feature cache path
```

The `decoder used` field matters because decoded RGB is the bridge from model-private latent space into the shared IDM space.

## First Implementation Milestone

1. Pick one embodiment/action space.
2. Export 10K actual transition windows.
3. Encode frames with one frozen semantic encoder.
4. Train the IDM for k=1.
5. Train a separate IDM for k=4 if k=1 is stable.
6. Report held-out action recoverability.
7. Add one generated-future source as a diagnostic/ablation, not as part of the base model.
8. Plot real-vs-predicted implied action deviation on rollouts.

## Open Decisions

- Which first environment: LIBERO/robosuite or PolaRiS?
- Which first shared encoder is already easiest on the EC2 VM?
- Which model gives decoded predicted futures first: Cosmos, GE-Act, or DreamZero?
- Should `source_id` be enough, or do we need explicit `model_id` embeddings?
- Should the output be raw action chunk or normalized action chunk in environment units?
- Should gripper be modeled as regression or classification?

## Next Track - RoboCasa IDM Dataset

The next dataset direction is RoboCasa, but it should be treated as a separate sanity pass rather than mixed into LIBERO immediately.

The reason is simple:

```text
LIBERO is small and toy-like.
RoboCasa has richer household scenes, more contact variation, and more visual diversity.
```

If the same IDM-style deviation signal still correlates with failure in RoboCasa, the claim becomes less likely to be a LIBERO artifact.

RoboCasa manifest shape:

```text
sample_id
task_name
trajectory_id
t
horizon
current_primary_image
future_primary_image
current_wrist_image optional
future_wrist_image optional
current_proprio
action_chunk_horizon_x_D
success/failure if available
source = robocasa_real_real
```

Later, add model futures:

```text
source = robocasa_real_model_cosmos
current_real_image
future_model_image
action_chunk_from_model
model_id
decoder/preprocessing_used
```

First RoboCasa experiment:

```text
1. Export RoboCasa real -> real IDM windows.
2. Train a RoboCasa-only MLP IDM baseline.
3. Run Cosmos RoboCasa future predictions if the model gives plausible images.
4. Compute IDM(C, P) vs policy action.
5. Compare success/failure and gripper/contact spikes.
6. Only then try LIBERO + RoboCasa mixed IDM.
```

Expectation should stay grounded:

```text
RoboCasa will add distribution shift and preprocessing bugs first.
The first goal is dataset sanity, not a stronger headline result.
```

## Next Track - LIBERO Policy Deviation Across Policies

The immediate LIBERO comparison should go back to the MLP IDM baseline first, then swap policies while keeping the world-model future source fixed.

Policy set:

```text
1. MLP IDM baseline / current IDM diagnostic policy
2. pi0.5 policy
3. Molmo policy
```

World-model future source:

```text
Cosmos predicted future P
```

For each policy rollout window:

```text
C = current observation
P = Cosmos predicted future
a_policy = action chunk from the active policy

score = distance(IDM(C, P), a_policy)
```

This keeps the question clean:

```text
Given the same imagined future, how aligned is each policy's action with the action implied by the IDM?
```

The π0.5 and Molmo setup should not change the metric logic. They only replace the source of `a_policy`.

Required outputs:

```text
execution video
Cosmos predicted future video/frame
IDM(C, P) vs a_policy trace
success/failure label
task and prompt
```

The comparison should be run on the same LIBERO tasks where possible, because otherwise the deviation plots become a mixture of policy quality and task distribution.

Core readout:

```text
Does the IDM-world-model mismatch separate failures from successes for each policy?
Does one policy have systematically higher imagined-action inconsistency than another?
Do prompt/task perturbations increase mismatch before visible failure?
```

This is the concrete bridge from the current Cosmos-policy experiments to π0.5 and Molmo.

### LIBERO Policy Chunk Contract

For π0.5 and Molmo, do not force a native 16-step policy rollout. Use their chunked-policy behavior and aggregate two committed prefixes into one Cosmos-aligned 16-step comparison.

Rollout contract:

```text
At t:
  query Cosmos once -> P16
  query policy -> 10 actions
  execute first 8 actions

At t + 8:
  query policy again -> 10 actions
  execute first 8 actions

Now define:
  a_policy_16 = first 8 from policy query 1 + first 8 from policy query 2

Metric:
  compare IDM(C_t, P16) vs a_policy_16
```

Then repeat:

```text
At t + 16:
  query Cosmos again -> next P16
  query policy -> next 10 actions
  execute first 8
  query policy again
  execute first 8
```

This gives Cosmos and the IDM their native `k = 16` comparison while keeping π0.5 and Molmo close to their normal receding-horizon control style.

Do not compare Cosmos `P16` against only the first policy action. Do not silently pad π0.5's 10-action chunk to 16. The useful object is the actually executed 16-action policy chunk assembled from two 8-step commitments.
