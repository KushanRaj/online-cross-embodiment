# Online Cross-Embodiment Research Notes

This repo is the public research notebook for the cross-embodiment / world-action-model literature review and experiment planning.

The main thesis we are tracking:

- modern VLAs and WAMs often have decent semantic/planning priors,
- the weak point is frequently action conditioning / actuator translation,
- world-model prediction error may be useful as an actionable correction signal,
- perturbation-free success rates are not enough; we care about context faithfulness, prediction honesty, and causal action sensitivity.

## Where Things Live

- `paper-catalog.md` - compact catalog of the papers in this thread.
- `10_SOURCES/Papers/` - paper-level notes copied from Obsidian.
- `20_CONCEPTS/ML/Robotics/Reports/` - larger synthesis reports: training methodology, benchmarks, datasets, compute, experiments, and public-review notes.
- `30_MOCs/` - maps of content / navigation notes.
- `docs/planning/` - working experiment notes and older planning artifacts.
- `docs/slack/` - Slack-facing mirror artifacts, including the paper-list CSV draft.

## What Is Intentionally Not Tracked

The local working folder contains PDFs, screenshots, rollout videos, VM setup scripts, temporary extraction files, and scratch notes. Those are intentionally ignored by git so the public repo stays readable and does not accidentally publish bulky or private material.

If a paper PDF is needed, use the citation/link in `paper-catalog.md` or the corresponding source note rather than committing the PDF.

## Current Research Direction

The immediate experiment family is not "train the biggest VLA." It is:

1. Run native/in-distribution policies in their comfortable environments.
2. Add prompt, image, object, and state perturbations.
3. Compare predicted future against observed reality.
4. Measure residuals in semantic/JEPA-style latent spaces, not only pixels.
5. Test whether an inverse-dynamics-style residual model can map "predicted future vs actual future" into an actionable correction signal.

DreamZero is useful because it jointly predicts action and future. GE-Sim / Cosmos-style action-conditioned simulators are cleaner for causal counterfactuals. Pi/Molmo-style VLAs are useful for instruction sensitivity and action-token/action-expert comparisons.
