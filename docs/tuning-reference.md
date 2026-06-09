# Tuning Reference

This file tracks the experiment knobs that change the meaning of a run.

## Horizon

Current base:

```text
k = 16
target = 16 x 7 action chunk
```

Why:

- Cosmos LIBERO/RoboCasa diagnostics use a 16-step future.
- A one-step IDM is not aligned with a 16-step world-model future.
- Longer horizons such as 32 or 128 can be explored, but they answer a different question.

Interpretation:

- `k=1`: immediate local dynamics, easier and often less useful for Cosmos-style P16.
- `k=16`: current base diagnostic.
- `k=32/128`: longer-horizon action plan recovery; useful only if training data and eval are explicitly matched.

## Encoder

Current base:

```text
google/siglip-base-patch16-224
```

Why it was used:

- Easy pooled visual features.
- Mentioned as good enough in related semantic/reconstruction experiments.
- Fast enough for MLP IDM training and online eval.

Candidate future encoders:

- DINOv3
- V-JEPA 2
- SigLIP variants

Rule:

All encoder swaps must use the same data split first. Do not change encoder and data mixture in the same claim.

## IDM Architecture

Current baseline:

```text
FeatureIDM
pooled C/P visual features + optional proprio -> 16 x 7 action chunk
```

Attempted larger model:

```text
PatchTransformerIDM
patch-token visual features -> action chunk
```

Important:

The patch transformer is a model-choice deviation. It may be useful, but it should not silently replace the baseline.

## Action Target

Base target:

```text
7D manipulation action
```

RoboCasa execution caveat:

```text
IDM predicts 7D
environment may require 12D
runner pads with fixed mobile/base tail: [0, 0, 0, 0, -1]
```

Do not feed success labels to the IDM. Success is evaluation metadata only.

## Loss

Base:

```text
Smooth L1 over K x 7
```

Known concern:

Gripper signals can be under-emphasized because the gripper is one channel and contact/open-close events may be sparse. Keep gripper mismatch plots even if the training loss stays simple.

Future ablation:

- weighted gripper loss,
- separate gripper classification head,
- per-step discount loss for long horizon.

These are not base until explicitly chosen.

## Optimizer And Resume

The current training code previously restored model weights without restoring full AdamW optimizer state in at least one resume path. That produced visible loss kinks because the resumed run used good weights with a fresh optimizer state.

This is not evidence of model failure, but it is not a true continuous training curve.

For publishable training curves:

- save and restore optimizer state,
- save scheduler state,
- log `train/lr`,
- log global step consistently,
- avoid mixing resumed segments in a single claim unless the resume semantics are documented.

## Policy Cadence

Pi/Molmo LIBERO probe cadence:

```text
policy predicts 10 actions
execute 8
requery
execute 8
assemble 16-step action chunk for comparison
```

Reason:

The policy and Cosmos diagnostic need a 16-step comparison window, but the external policy naturally replans in chunks. This is a compromise, not a statement that all policies natively execute 16 open-loop actions.

## Metrics

Use:

- SigLIP P-vs-R L2 and cosine,
- `IDM(C,P) vs a` chunk and per-step metrics,
- `IDM(C,R) vs a` chunk and per-step metrics,
- `IDM(C,P) vs IDM(C,R)` as an auxiliary residual,
- EEF-only L2 and cosine,
- gripper L1 and sign mismatch.

Avoid as main:

- flattened 16x7 cosine.

## Preprocessing

Rule:

`C`, `P`, and `R` must be in the same image convention before shared IDM comparison.

Known caveat:

Cosmos-prepared LIBERO frames can appear flipped relative to raw simulator frames. The correct comparison is not "flip generated image only"; it is "use the same prepared convention for all three images."
