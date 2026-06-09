# Behavior Casebook

## 2026-06-09 - Phase 3 Experiment Contract

### Idea

The Phase 3 pipeline should preserve one behavioral invariant:

```text
rollout HDF5 -> evaluator -> diagnostic plots -> canonical videos
```

New policies or environments may change how HDF5s are produced, but they should
not change the downstream contract unless the change is deliberate and
documented.

### Positive Cases

- A new RoboCasa IDM-as-policy runner can be added if it writes the same core
  fields consumed by `evaluate_online_cosmos_deviation.py`.
- A new policy can be compared by selecting the correct reference action dataset:
  `query_action_chunks` for the executed policy, or
  `query_cosmos_action_chunks` for Cosmos's native action head.
- A new video set should use `make_prediction_metric_videos.py` so execution
  frames, predicted futures, and metric traces stay visually comparable.

### Negative Cases

- Do not write a fresh plotter when the canonical plotter only needs a new CSV
  or reference-action selection.
- Do not drive execution-video playback from sparse query frames; use dense
  `primary_images`.
- Do not treat flattened 16x7 cosine as the main cosine metric. Use per-step
  cosine, especially EEF-only cosine.
- Do not interpret IDM-as-policy self-reference metrics as evidence that the
  policy agrees with Cosmos. That comparison needs `query_cosmos_action_chunks`.

### Imaginary Future Breakages

If a video looks frame-by-frame instead of continuous:

- likely cause: the renderer used `query_primary_images` instead of
  `primary_images`;
- inspect: `experiments/phase3/make_prediction_metric_videos.py`;
- useful test: verify the left panel advances through every dense rollout frame.

If `IDM(C,P) - a` is near zero for IDM-as-policy and seems too good:

- likely cause: `a` is `query_action_chunks`, which is the IDM's own executed
  chunk;
- inspect: evaluator `--reference-action-dataset`;
- useful test: rerun with `query_cosmos_action_chunks`.

If success/failure colored plots collapse into one class:

- likely cause: the rollout set has only successes or only failures;
- inspect: `episodes.csv` and `summary.json`;
- useful test: pick mixed task seeds or run more trials.

### Evidence So Far

- RoboCasa Cosmos-as-policy artifacts:
  `run-artifacts/robocasa-cosmos-deviation-20260609-112643`.
- RoboCasa IDM-as-policy artifacts:
  `run-artifacts/robocasa-idm-policy-20260609-133354`.
- LIBERO Pi/Molmo probe artifacts:
  `run-artifacts/pi05-molmo-libero-cosmos-proper-20260608`.

### Open Questions

- Which action-conditioned world model path should support true external-action
  counterfactual video generation?
- Should the base IDM remain feature MLP for fast iteration, or should the patch
  transformer become the default once logging and comparisons are stable?
- How much generated-future data should be mixed into IDM training without
  causing the IDM to chase model artifacts?

## 2026-06-09 - Agent Operating Discipline

### Idea

Future agents should start from the repo's own documentation and preserve
experiment semantics instead of treating each request as a blank-slate coding
task.

### Positive Cases

- Start with `docs/INDEX.md`, then the source files named by
  `docs/repo-map.md`.
- Make narrow patches to canonical code.
- Commit coherent code/doc changes.
- Use the `uv` env for local Python.
- Shut down EC2 after remote work.

### Negative Cases

- Do not silently change architecture, optimizer, metrics, horizon, or training
  data and reveal it later.
- Do not invent new plot/video layouts when the user asked for the existing
  setup.
- Do not use local system Python for project scripts.

### Imaginary Future Breakages

If an agent cannot find where code lives:

- likely cause: it skipped `docs/repo-map.md`;
- inspect: `AGENTS.md` and `docs/INDEX.md`;
- useful test: ask it to name the canonical rollout/eval/video files before
  editing.

If a remote run finishes but artifacts are missing locally:

- likely cause: the run was not downloaded before shutdown or the local
  `run-artifacts/` path was not created;
- inspect: remote `/home/ubuntu/robotics/runs/` and local `run-artifacts/`;
- useful test: compare remote `summary.json` against local extraction.

### Evidence So Far

- Multiple Phase 3 plotting/video iterations were needed because code was
  rewritten instead of reused.
- The repo now has `AGENTS.md`, `docs/INDEX.md`, `docs/repo-map.md`,
  `docs/experiment-plans.md`, and `docs/operating-rules.md`.

### Open Questions

- Should every future run append a short run note under `docs/runs/`, or are
  `run_manifest.json` files enough for now?

