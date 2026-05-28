# Online Cross-Embodiment Research Notes

This repo is the public research notebook for the cross-embodiment / world-action-model literature review and experiment planning.

The main thesis we are tracking:

- modern VLAs and WAMs often have decent semantic/planning priors,
- the weak point is frequently action conditioning / actuator translation,
- world-model prediction error may be useful as an actionable correction signal,
- perturbation-free success rates are not enough; we care about context faithfulness, prediction honesty, and causal action sensitivity.

## Where Things Live

- `Robotics World Models.md` - the top-level map of content.
- `papers/` - paper-level notes plus the paper catalog.
- `pdfs/` - local paper PDFs.
- `experiment-design/reports/` - larger synthesis reports: training methodology, benchmarks, datasets, compute, experiments, and public-review notes.
- `docs/planning/` - working experiment notes and older planning artifacts.
- `docs/slack/` - Slack-facing mirror artifacts, including the paper-list CSV draft.

## Current Research Direction

The immediate experiment family is not "train the biggest VLA." It is:

1. Run native/in-distribution policies in their comfortable environments.
2. Add prompt, image, object, and state perturbations.
3. Compare predicted future against observed reality.
4. Measure residuals in semantic/JEPA-style latent spaces, not only pixels.
5. Test whether an inverse-dynamics-style residual model can map "predicted future vs actual future" into an actionable correction signal.

DreamZero is useful because it jointly predicts action and future. GE-Sim / Cosmos-style action-conditioned simulators are cleaner for causal counterfactuals. Pi/Molmo-style VLAs are useful for instruction sensitivity and action-token/action-expert comparisons.
