# Plan 001 - Shared Semantic IDM

**Status:** Active baseline and ablation program.  
**Purpose:** Train an embodiment-specific, model-agnostic IDM that converts visual future pairs into action chunks.

## Files

| # | Doc | Why It Exists |
|---|-----|---------------|
| 01 | [hypothesis](01-hypothesis.md) | What the IDM is supposed to measure and what it is not. |
| 02 | [data-contract](02-data-contract.md) | Exact samples, sources, and action targets. |
| 03 | [training-contract](03-training-contract.md) | Architecture, encoder, horizon, loss, logging, and deviations. |
| 04 | [diagnostics](04-diagnostics.md) | How to evaluate whether the IDM is useful. |
| 05 | [ablation-queue](05-ablation-queue.md) | Future variants without polluting the base run. |
| 06 | [open-questions](06-open-questions.md) | Things not settled yet. |

## One-Sentence Summary

Train `IDM(C,F,proprio) -> 16x7` on real trajectory windows, then use it to ask whether a model's imagined future implies the same action as the policy actually took.
