# Documentation Index

This is the first file to read in this repo.

The repo has two jobs:

1. Track the literature and thesis around world models, VLAs, planning, and
   action translation.
2. Run controlled experiments around prediction honesty: compare model-predicted
   futures, observed futures, and policy actions through an IDM diagnostic.

## 1. Understand The Repo

- `README.md` - public-facing summary and high-level research direction.
- `OBSIDIAN.md` - how the repo mirrors the original Obsidian research notes.
- `docs/repo-map.md` - exact map of important directories and code files.
- `papers/paper-catalog.md` - paper inventory.
- `Robotics World Models.md` - top-level research map/MOC.

Use `docs/repo-map.md` when asking:

- Where is the rollout code?
- Where is the diagnostic code?
- Where is the plotting/video code?
- Which scripts run locally vs on EC2?
- What artifacts are produced and where are they expected to live?

## 2. Experiment Plans

- `docs/experiment-plans.md` - current experiment phases and what each phase
  means.
- `experiment-design/Shared Semantic IDM - Prediction Reality Deviation.md` -
  core IDM thesis and data contract.
- `experiment-design/Phase 3 Clean Rollout Benchmark.md` - task suites and
  benchmark rationale.
- `experiment-design/Phase 3 Rollout Task Catalogue.md` - task-level catalogue.
- `experiments/idm/README.md` - IDM training/data contract.
- `experiments/phase3/README.md` - Phase 3 clean rollout setup and remote data.

Use these when asking:

- What are we trying to show?
- What does `C`, `P`, `R`, or `a` mean?
- What is Phase 1 vs Phase 3?
- Which environments/tasks are in scope?
- What is the correct action/reference dataset for a metric?

## 3. Operating Rules And Grievances

- `docs/operating-rules.md` - practical rules for agents working in this repo.
- `docs/behavior-casebook.md` - dated cases of experiment behavior, failure
  modes, and RCA handles.

These files exist because repeated issues slowed the project down:

- rewriting code instead of reusing canonical runners,
- changing plot layouts and metrics without asking,
- mixing up selected action, executed action, Cosmos action, and IDM action,
- forgetting preprocessing/horizon caveats,
- running local scripts outside the `uv` env,
- starting EC2 jobs without a shutdown/download plan.

## 4. Current Core Experiment Vocabulary

For the prediction-reality deviation experiments:

```text
C = current observation
P = model-predicted future observation
R = real observed future observation
a = action chunk selected/executed by the policy under test
IDM(C, P) = action chunk implied by the world model future
IDM(C, R) = action chunk implied by the observed future
```

Important distinction:

```text
query_action_chunks
  action chunks selected/executed by the policy for this rollout

query_cosmos_action_chunks
  Cosmos Policy's own action-head prediction at the same query point
```

For Pi/Molmo/IDM-as-policy probes, Cosmos futures currently come from Cosmos's
own current-observation/task-conditioned generation path. The external policy
action is not fed into Cosmos as action conditioning.

## 5. Fast Start For Future Agents

Before doing new experiment work:

```bash
pwd
git status --short
sed -n '1,220p' docs/INDEX.md
sed -n '1,260p' docs/repo-map.md
sed -n '1,260p' docs/operating-rules.md
```

Then read only the relevant experiment source files named in the repo map.

