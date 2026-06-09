# Online Cross-Embodiment Research Notes

This repo is the public research notebook for the cross-embodiment / world-action-model literature review and experiment planning.

The main thesis we are tracking:

- modern VLAs and WAMs often have decent semantic/planning priors,
- the weak point is frequently action conditioning / actuator translation,
- world-model prediction error may be useful as an actionable correction signal,
- perturbation-free success rates are not enough; we care about context faithfulness, prediction honesty, and causal action sensitivity.

## Where Things Live

- `docs/INDEX.md` - first stop for future agents; links repo map, experiment
  plans, operating rules, and casebook.
- `index.html`, `about.html`, `blog/`, `posts/`, `assets/` - the public GitHub Pages site and blog skeleton.
- `Robotics World Models.md` - the top-level map of content.
- `papers/` - paper-level notes plus the paper catalog.
- `pdfs/` - local paper PDFs.
- `experiment-design/reports/` - larger synthesis reports: training methodology, benchmarks, datasets, compute, experiments, and public-review notes.
- `experiments/` - tracked experiment code. Edit locally, sync/pull to EC2, run against remote data/checkpoints there.
- `docs/planning/` - working experiment notes and older planning artifacts.
- `docs/slack/` - Slack-facing mirror artifacts, including the paper-list CSV draft.

## Experiment Workflow Rules

Do not run project Python through bare system Python or the Codex bundled
runtime. For local experiment scripts, initialize and use a `uv` environment
first, then run through that environment. This keeps plotting, analysis, and
training utilities reproducible instead of depending on whatever Python happens
to be available on the machine.

## Current Research Direction

The immediate experiment family is not "train the biggest VLA." It is:

1. Run native/in-distribution policies in their comfortable environments.
2. Add prompt, image, object, and state perturbations.
3. Compare predicted future against observed reality.
4. Measure residuals in semantic/JEPA-style latent spaces, not only pixels.
5. Test whether an inverse-dynamics-style residual model can map "predicted future vs actual future" into an actionable correction signal.

DreamZero is useful because it jointly predicts action and future. GE-Sim / Cosmos-style action-conditioned simulators are cleaner for causal counterfactuals. Pi/Molmo-style VLAs are useful for instruction sensitivity and action-token/action-expert comparisons.

## GitHub Pages

The site is plain static HTML/CSS for now. In GitHub, enable Pages from the `main` branch root to publish it. The blog is intentionally flat until the writing structure becomes clearer.
