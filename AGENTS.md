# Agent Entry Point

Start with `docs/INDEX.md`.

Do not begin by scanning random experiment files. The index points to the repo
map, experiment plans, canonical rollout/eval/plot code, and the operating
rules created from previous mistakes in this project.

Minimum read order before changing experiment code:

1. `docs/INDEX.md`
2. `docs/repo-map.md`
3. `docs/experiment-plans.md`
4. `docs/operating-rules.md`
5. The specific source files named by the repo map for the task at hand.

Hard rules:

- Reuse existing rollout, evaluation, diagnostic, and video-rendering code
  unless you can state exactly why it cannot support the task.
- Do not silently change experiment semantics, metrics, horizons, reference
  action datasets, preprocessing, or policy execution cadence.
- Local Python work must use the repo `uv` environment, not bare system Python.
- Remote-heavy data, model, and simulator jobs run on EC2; code should be edited
  locally and then synced or pulled remotely.
- If EC2 is started, stop it when the requested remote work is complete.

