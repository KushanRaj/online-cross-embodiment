---
type: research-report
topic: robotics-world-models
status: populated-draft
created: 2026-05-28
updated: 2026-05-28
---

# Feasible Experiments and Compute Report

## Core Thesis

We want to test context faithfulness and prediction honesty:

```text
Does model behavior honestly track changes in instruction, image, proprio, action, and predicted-vs-actual future?
```

The current 48 GB L40 is enough for the most relevant immediate experiments. It is not enough for frontier model training or comfortable DreamZero-14B inference.

## What Fits On 48 GB L40

| Workload | 48 GB L40? | Notes |
| --- | --- | --- |
| pi0.5-DROID inference | Yes | Already ran in PolaRiS. |
| pi0.5 small fine-tune / LoRA / frozen-backbone probe | Likely yes | Use small batches, frozen modules, or LoRA. Full PI-style training is out of scope. |
| MolmoAct2-DROID inference | Yes | DROID checkpoint is around 22 GB; bf16 inference should fit. |
| MolmoAct2 full training | No | Paper-scale training uses 32-64 H100s. |
| MolmoAct2 action-head/adaptor experiment | Possible, with code maturity risk | Better to begin with inference and probes. |
| VLA-JEPA eval/inference | Likely yes | Full reproduction uses 8 A100s; serialized eval or small probes should fit. |
| VLA-JEPA-style latent probe | Yes | Freeze encoders and train small latent/action/progress modules. |
| DreamZero-14B inference | Not on one L40 | Official repo indicates multi-GPU distributed inference, tested on GB200/H100, minimum 2 GPUs. |
| DreamZero training/fine-tuning | No | 14B DiT updates, 100k steps, batch 128 in paper. |
| Cosmos Policy inference / released-checkpoint use | Likely easier than DreamZero if checkpoints and wrappers are available | 2B class; action-conditioned planning interface is more useful for counterfactuals. |
| Cosmos Policy full fine-tune | No | 8-64 H100s depending setting. |
| GE-Sim inference / released-checkpoint use | Likely easier than DreamZero if checkpoints and wrappers are available | GE-Base is 2B class and GE-Act decoder is 160M; the blocker is integration/release state, not parameter size. |
| Fast-WAM full training | Probably no | 6B model; use released checkpoints if practical. |
| Fixed-encoder semantic probes | Yes | Best immediate training target. |

## Compute Reference Points

| Method | Reported compute | What it means |
| --- | --- | --- |
| DreamZero | 14B main; 100k steps batch 128; official distributed inference requires multi-GPU. | Rent 2x H100/80GB or better for serious DreamZero inference. Do not train. |
| Cosmos Policy | 2B model. LIBERO full fine-tune: 40k steps on 64 H100s for 48h; ALOHA uses 8 H100s for 48h. | Inference/released-checkpoint use should be much more tractable than DreamZero; full fine-tuning is the expensive part. |
| MolmoAct2 | Pretrain 200k on 64 H100s, about 5,760 GPUh; fine-tunes about 1,150-2,300 GPUh. | Inference yes; training no. |
| pi0.5 | 280k pretrain + 80k post-train, large PI data pipeline. | Inference/small adaptation yes; full reproduction no. |
| VLA-JEPA | 50k pretrain, 30k sim, 20k real; 8 GPUs, batch 256. | Full reproduction no; latent-probe variant yes. |
| CLAP | Act-VAE 190h, VD-VAE 380h, NTP 3,800h, RF 2,000h on A100-80G estimates. | Conceptual method; too expensive to reproduce. |
| OpenVLA | 64 A100s for 14d, about 21.5k A100h. | Use checkpoints/adapters only. |
| Octo | TPU v4-128 for about 14h; fine-tunes on A5000 24GB in about 5h. | Lightweight baseline candidate. |

## Immediate Experiment 1: Context Faithfulness Sweep

Run this first on the L40.

```text
Models:
  pi0.5-DROID
  pi05_droid_jointpos_polaris
  MolmoAct2-DROID

Environment:
  PolaRiS first
  MolmoSpaces next if adapter is stable

Tasks:
  FoodBussing
  one object pickup/place task
  one distractor-heavy task

Conditions:
  clean prompt
  paraphrase prompt
  wrong-object prompt
  absent-object prompt
  target masked/cropped
  distractor masked/cropped
  changed proprio if adapter permits

Metrics:
  success / progress
  action delta similarity
  target direction change
  gripper timing
  rollout divergence
  exact model input frames
```

This directly tests whether the policy is conditioned on the context it is supposedly using.

## Immediate Experiment 2: Semantic Prediction / Reality Mismatch

For pi0.5 and MolmoAct2, we do not yet have a world prediction. Start with semantic progress and inverse dynamics over observed rollouts:

```text
Data:
  DROID subset
  PolaRiS rollouts
  MolmoSpaces rollouts if available

Encoders:
  DINOv2
  SigLIP
  V-JEPA2 if setup is practical

Signals:
  z_t
  z_t+k
  instruction embedding
  action chunk
  progress/success label

Train:
  small inverse-dynamics probe
  small progress classifier
  optional residual/correction predictor
```

When DreamZero or another future-generating model is available, replace `z_t+k observed` with `z_t+k predicted` and measure predicted-vs-observed mismatch.

## Immediate Experiment 3: Future-Generation Honesty

Requires a future-generating model:

| Model | Can run now? | Test |
| --- | --- | --- |
| DreamZero | Needs bigger/multi-GPU VM | Compare generated future to observed PolaRiS/DROID rollout; see whether mismatch predicts failure. |
| Cosmos Policy | Smaller than DreamZero; depends on released checkpoints/setup | Sample candidate actions, predicted futures, values; test whether chosen action is actually better. |
| GE-Sim | Smaller than DreamZero; depends on released checkpoints/setup | True action counterfactual: force action A versus B, compare predicted future. |
| Fast-WAM | If checkpoints usable | Negative control: no test-time future, but video-trained representations. |

## Best Training Run We Can Execute Now

Train a small scorer/residual model over frozen rollout logs.

| Input | Target |
| --- | --- |
| current obs latent, instruction latent, model proposed action, predicted future latent/video if available, actual next obs latent | predict progress/failure, semantic substep completion, or residual correction action |

Why this is the right size:

- It directly tests whether semantic latent mismatch is useful.
- It avoids retraining pi0.5/Molmo/DreamZero.
- It uses the L40 efficiently.
- It creates a measurable bridge from world-model/reasoning signals to action correction.

The concrete version of this is now [[Shared Semantic IDM - Prediction Reality Deviation]]:

```text
train one embodiment-specific, model-agnostic IDM
use decoded world-model futures -> shared semantic encoder -> IDM
compare action implied by observed future against action implied by predicted future
```

This follows the reconstruction-vs-semantics paper's IDM probe recipe, but changes the purpose. Their IDM tests which latent space preserves action geometry. Our IDM is a fixed diagnostic instrument for plotting prediction-reality deviation.

## Can We Train pi0.5 In A VLA-JEPA Style?

Not as full VLA-JEPA. The full recipe trains a VLM, latent world model, and action head with SSV2 + DROID on multi-GPU hardware.

A useful approximation:

```text
freeze pi0.5 / OpenPI backbone
extract hidden states or visual-action context
freeze DINO/SigLIP/V-JEPA target encoder
train small latent future predictor:
  current latent + language/action context -> future semantic latent
train small action/progress head:
  predicted future latent -> progress/failure/residual action
```

This is not a claim that we reproduced VLA-JEPA. It is a controlled test of the same idea: semantic latent future prediction as an auxiliary or diagnostic signal.

## Counterfactual Levels

| Level | Models | Question |
| --- | --- | --- |
| Context counterfactual | pi0.5, MolmoAct2, OpenVLA, Octo | If I change prompt/image/proprio, does the action change appropriately? |
| Prediction counterfactual | DreamZero-style WAM | If the model imagines a future, does that future match reality and predict failure? |
| Action counterfactual | GE-Sim/Cosmos-style action-conditioned models | If I force candidate action A vs B, does the predicted future change causally and does the scorer choose the better action? |

## Experiment Log Requirements

For every rollout, save:

- raw camera frames,
- exact resized/cropped model inputs,
- prompt string,
- proprio/state vector,
- checkpoint name,
- adapter name,
- action vector before and after scaling,
- control rate,
- environment task ID and seed,
- rollout video,
- success/progress labels,
- optional predicted future video/latent when available.

Without this, failures will be impossible to classify.

## Recommended Order

1. Finish the PolaRiS context-faithfulness sweep for pi0.5 and MolmoAct2.
2. Dump exact model input frames for every condition.
3. Train semantic latent inverse-dynamics/progress probes from DROID/PolaRiS.
4. Move to MolmoSpaces as the broader stress environment.
5. Rent 2x H100/80GB only when DreamZero-14B future generation is the blocker.
6. If Cosmos/GE-Sim checkpoints and wrappers are usable, prioritize them before DreamZero for action-counterfactual futures because they are smaller and structurally cleaner.
7. Add predicted-vs-observed future mismatch once DreamZero/Cosmos/GE-Sim-class model is available.
