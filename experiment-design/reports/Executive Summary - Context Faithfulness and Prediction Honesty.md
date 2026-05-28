---
type: research-report
topic: robotics-world-models
status: populated-draft
created: 2026-05-28
updated: 2026-05-28
---

# Executive Summary - Context Faithfulness and Prediction Honesty

## Central Question

```text
Can predicted futures / world-model signals tell us whether a policy is honestly conditioned on its context,
and can that signal help diagnose or improve action selection?
```

The literature now points to a sharper version:

```text
large policies often have enough semantic/world knowledge,
but action grounding, context conditioning, and contact execution remain weak.
```

So the immediate research target is not "train a bigger VLA" or "prove WAMs are always better." It is:

1. measure whether a policy's action is faithful to image, language, and proprio context;
2. measure whether predicted/latent futures are honest about what will happen;
3. test whether mismatch in semantic/action-relevant latent space predicts failure or suggests correction.

## The Big Pattern

The strongest reported policies tend to have the most robot data and the best domain alignment. That is visible across pi0/pi0.5, MolmoAct2, OpenVLA/OFT, Octo, GR00T, LingBot-VA, and DreamZero.

On clean LIBERO, many methods are near saturated. The leaderboard does not by itself tell us whether planning helped. Once robustness perturbations appear, rankings change, and broad-data models like pi0.5 stay surprisingly strong. That suggests data breadth and deployment alignment are doing a lot of the work.

This is the first grounding point:

```text
do not interpret high benchmark performance as proof of inference-time reasoning.
```

## What WAMs Actually Give Us

World-action models are not one thing.

| Use of future prediction | Example | What it proves |
| --- | --- | --- |
| Future prediction as auxiliary training loss | Fast-WAM | Video/world prediction can improve representations even if no future is generated at test time. |
| Future prediction as diagnostic signal | DreamZero, GE-Base/EWMBench | Generated futures can expose model belief and maybe predict failure. |
| Future prediction as planning interface | Cosmos Policy, GE-Sim-style action-conditioned model | Candidate actions can be scored through imagined futures and values. |

This distinction matters because our thesis is about planning and action selection. A pretty future video is not enough. The model must either:

- change its future prediction when context changes,
- predict failure before the rollout fails,
- or score alternative actions in a causally meaningful way.

DreamZero is powerful for prediction honesty, but it jointly generates action and future. It is not the cleanest action-counterfactual simulator. GE-Sim and Cosmos are cleaner for causal action counterfactuals, and they are smaller / more plausible to run than DreamZero in parameter-memory terms. The caveat is not model size; it is whether we have the released checkpoints, inference code, and environment wrapper wired to the exact robot/task interface.

## Why Reconstruction Versus Semantics Matters

The reconstruction-vs-semantics paper, VLA-JEPA, and CLAP are all pushing toward the same insight:

```text
pixel reconstruction is not the same as action-relevant understanding.
```

A robot does not need every texture correct. It needs to preserve:

- object identity,
- object pose,
- hand/object relation,
- affordance,
- contact/progress state,
- substep completion,
- action recoverability.

That explains why VLA-JEPA predicts in a frozen semantic latent space rather than pixel space, and why CLAP tries to align visual transitions with executable robot action latents. It also tells us what metric to use:

```text
compare predicted and observed futures in DINO / SigLIP / V-JEPA / action-relevant latent space,
not only in pixels.
```

## How This Shapes Our Experiment

The experiment should be framed as context faithfulness and prediction honesty, not generic benchmark chasing.

### Experiment A: Context Faithfulness

Run pi0.5-DROID and MolmoAct2-DROID in PolaRiS first, MolmoSpaces next.

Perturb:

- prompt paraphrase,
- wrong-object prompt,
- absent-object prompt,
- visual mask/crop,
- distractor insertion,
- proprio perturbation if adapter allows.

Measure:

- success/progress,
- action delta similarity,
- target direction change,
- gripper timing,
- exact 224x224 input frames,
- rollout divergence.

This answers whether direct VLAs are actually using the context they are given.

### Experiment B: Semantic Mismatch Probe

Use DROID and rollout logs.

Train small frozen-encoder probes:

- inverse dynamics,
- progress prediction,
- failure prediction,
- residual correction.

Use semantic encoders:

- DINOv2,
- SigLIP,
- V-JEPA2 if setup is practical.

This answers whether semantic latent movement contains enough signal to diagnose action quality.

### Experiment C: Future Honesty

Run once DreamZero/Cosmos/GE-Sim-class future generation is available.

Measure:

- predicted future versus observed future,
- semantic mismatch over time,
- substep completion mismatch,
- whether mismatch predicts rollout failure earlier than final success labels,
- whether action-conditioned futures change under candidate actions.

This is where the planning thesis becomes direct.

## What The Benchmarks Contribute

LIBERO gives us common comparisons, but not the main claim.

LIBERO-Plus gives us perturbation design.

PolaRiS gives us the DROID-style closed-loop evaluation contract.

MolmoSpaces gives us broader out-of-domain object/environment stress.

EWMBench gives us a world-model metric template: scene consistency, EEF path alignment, temporal alignment, dynamics, semantic substep completion, and diversity.

So the benchmark stack should be:

```text
DROID offline -> per-state/probe sanity
PolaRiS -> in-domain-ish DROID closed-loop behavior
MolmoSpaces -> broader DROID-like stress
EWMBench metrics -> predicted future evaluation
LIBERO-Plus -> perturbation vocabulary
```

## What Public Discussion Adds

Public review/forum/GitHub discussion mostly points to interface failures:

- camera calibration,
- image crops,
- state tokens,
- action normalization,
- control frequency,
- action decoder conventions,
- contact timing.

This supports a practical rule:

```text
before interpreting a model failure as weak reasoning,
prove the observation/action contract is correct.
```

It also supports the deeper thesis: users often report that models can reach the right area or understand the object but fail final insertion/contact. That is exactly the gap between high-level semantic competence and low-level actuator conditioning.

## Compute Reality

The 48 GB L40 is enough for:

- pi0.5 inference,
- MolmoAct2 inference,
- PolaRiS sweeps,
- MolmoSpaces attempts,
- OpenVLA/Octo-style baselines,
- frozen semantic encoders,
- latent probes,
- small residual/scorer models.

It is not enough for:

- full DreamZero-14B inference comfortably,
- DreamZero training,
- MolmoAct2 training,
- Cosmos full fine-tuning,
- GE-Base/GE-Sim training,
- CLAP-scale reproduction.

Use 2x H100/80GB or better only when DreamZero-14B specifically, or another model that actually exceeds the L40 runtime envelope, becomes the blocker. For Cosmos/GE-Sim-class models, the likely blocker is setup/checkpoint/interface availability, not raw VRAM.

## Current Position

The strongest immediate path is:

1. Stop treating success rate alone as the result.
2. Treat action sensitivity under context perturbations as the first result.
3. Treat semantic latent mismatch as the second result.
4. Treat world-model future mismatch as the third result once DreamZero/Cosmos/GE-style generation is available.

The key hypothesis is:

```text
if a model's semantic future/progress estimate is good but its action fails,
then the bottleneck is action translation/conditioning;
if the semantic future itself is wrong,
then the bottleneck is world/context understanding.
```

That gives us a clean way to prove or disprove the thesis quickly.

## Next Steps

1. Script PolaRiS context-faithfulness sweeps for pi0.5-DROID and MolmoAct2-DROID.
2. Save exact model input frames and action vectors for every step.
3. Build a DROID/PolaRiS latent-probe dataset.
4. Train DINO/SigLIP/V-JEPA progress and inverse-dynamics probes.
5. Add DreamZero future generation when multi-GPU hardware is available; add Cosmos/GE-Sim-style action-conditioned futures earlier if usable checkpoints and wrappers are available.
6. Use MolmoSpaces as the broader OOD stress test once the PolaRiS loop is clean.
