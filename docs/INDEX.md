# Documentation Index

This is the first file to read in this repo.

The repo has two jobs:

1. Track the literature and thesis around world models, VLAs, planning, and action translation.
2. Run controlled experiments around prediction honesty: compare model-predicted futures, observed futures, and policy actions through an IDM diagnostic.

This index is intentionally layered like an Obsidian vault. Do not flatten it into one report. Use the category indexes first, then open the leaf documents relevant to the task.

## Core Reference

| Doc | Purpose |
|-----|---------|
| [design-philosophy.md](design-philosophy.md) | The thesis: planning priors are useful, action translation is weak, and prediction honesty can be measured. |
| [evolution.md](evolution.md) | Chronological record of how the project moved from paper reading to LIBERO/RoboCasa rollouts. |
| [tuning-reference.md](tuning-reference.md) | Experiment knobs: horizons, encoders, IDM architecture, metrics, preprocessing, and policy cadence. |
| [reflections.md](reflections.md) | User directives, grievances, and process rules mined from the working sessions. |
| [next-challenges-and-open-questions.md](next-challenges-and-open-questions.md) | The unresolved research and engineering questions. |

## Code And Repo Maps

| Doc | Purpose |
|-----|---------|
| [code/README.md](code/README.md) | Code documentation index. |
| [code/01-repo-map.md](code/01-repo-map.md) | Directory and source-file map. |
| [code/02-idm-pipeline.md](code/02-idm-pipeline.md) | IDM data, cache, model, training, and adapter pipeline. |
| [code/03-phase3-rollout-pipeline.md](code/03-phase3-rollout-pipeline.md) | Online rollout runners and HDF5 contracts. |
| [code/04-metrics-and-video-contract.md](code/04-metrics-and-video-contract.md) | Canonical metrics, plots, and video renderer behavior. |
| [code/05-remote-ec2-contract.md](code/05-remote-ec2-contract.md) | Remote paths, EC2 discipline, and shutdown rules. |
| [code/06-artifact-layout.md](code/06-artifact-layout.md) | Local and remote run artifact layout. |

Compatibility wrappers:

| Legacy Doc | New Home |
|------------|----------|
| [repo-map.md](repo-map.md) | Summary wrapper for [code/](code/README.md). |
| [experiment-plans.md](experiment-plans.md) | Summary wrapper for [plans/](plans/README.md). |
| [operating-rules.md](operating-rules.md) | Summary wrapper for [reflections.md](reflections.md) and [runbooks/](runbooks/README.md). |
| [behavior-casebook.md](behavior-casebook.md) | Historical behavior notes; new cases should point to [research/session-2026-06-idm-rollouts](research/session-2026-06-idm-rollouts/README.md). |

## Plan Bundles

Each plan bundle is a small repository of micro-documents. Read the bundle README first.

| # | Bundle | Status | What It Captures |
|---|--------|--------|------------------|
| 001 | [Shared Semantic IDM](plans/001-shared-semantic-idm/README.md) | Active | Train an embodiment-specific, model-agnostic IDM over semantic future pairs. |
| 002 | [Phase 3 Rollout Benchmark](plans/002-phase3-rollout-benchmark/README.md) | Active | Run policies online and score prediction-action consistency against success/failure. |
| 003 | [Counterfactual Action-Conditioned World Models](plans/003-counterfactual-action-conditioned-wm/README.md) | Planned | Move from joint policy probes to true action-conditioned future generation. |
| 004 | [Cross-Embodiment Future](plans/004-cross-embodiment-future/README.md) | Parked | Arm-to-arm EEF adapter transfer and later morphology transfer. |

## Research Sessions

| Session | Purpose |
|---------|---------|
| [2026-06 IDM Rollouts](research/session-2026-06-idm-rollouts/README.md) | Conversation-mined research record for the IDM/Cosmos/LIBERO/RoboCasa work. |

## Experiment Records

| Doc | Run / Evidence |
|-----|----------------|
| [experiments/2026-06-03-libero-cosmos-expanded.md](experiments/2026-06-03-libero-cosmos-expanded.md) | LIBERO Cosmos Policy expanded run and corrected diagnostics. |
| [experiments/2026-06-08-pi-molmo-libero-probes.md](experiments/2026-06-08-pi-molmo-libero-probes.md) | Pi 0.5 and MolmoAct2 LIBERO probes with Cosmos futures. |
| [experiments/2026-06-09-robocasa-cosmos-policy.md](experiments/2026-06-09-robocasa-cosmos-policy.md) | RoboCasa Cosmos Policy run with RoboCasa IDM diagnostics. |
| [experiments/2026-06-09-robocasa-idm-as-policy.md](experiments/2026-06-09-robocasa-idm-as-policy.md) | RoboCasa IDM-as-policy run and the correct Cosmos-reference interpretation. |

## Runbooks

| Doc | Use When |
|-----|----------|
| [runbooks/README.md](runbooks/README.md) | Choosing the right operational recipe. |
| [runbooks/local-python.md](runbooks/local-python.md) | Running local scripts without violating the `uv` rule. |
| [runbooks/ec2-remote-run.md](runbooks/ec2-remote-run.md) | Starting/syncing/running/stopping EC2 work. |
| [runbooks/phase3-evaluate-and-render.md](runbooks/phase3-evaluate-and-render.md) | Evaluating rollouts and rendering canonical videos. |
| [runbooks/adding-a-new-policy.md](runbooks/adding-a-new-policy.md) | Adding Pi/Molmo/new policy probes without changing metrics. |
| [runbooks/adding-a-new-environment.md](runbooks/adding-a-new-environment.md) | Adding RoboCasa-style environment support. |

## Literature And Reports

| Location | Purpose |
|----------|---------|
| [papers/paper-catalog.md](../papers/paper-catalog.md) | Paper inventory. |
| [papers/](../papers/) | Individual paper notes. |
| [experiment-design/reports/](../experiment-design/reports/) | Larger synthesis reports: datasets, benchmarks, training methodology, compute, public review. |
| [Low-Level Controller Data Requirements](../experiment-design/reports/Low-Level%20Controller%20Data%20Requirements.md) | Focused retrieval note for how much robot observation-action data each model used to make the controller work. |
| [experiment-design/](../experiment-design/) | Older but still useful experiment notes and task catalogues. |

## Minimum Read Order Before Experiment Work

1. [reflections.md](reflections.md)
2. [code/README.md](code/README.md)
3. The specific plan bundle for the task.
4. The relevant runbook.
5. The exact source files named by the runbook.

Do not start by scanning random files. The repo already has canonical runners, evaluators, renderers, and artifact contracts.
