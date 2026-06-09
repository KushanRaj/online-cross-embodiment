# Operating Rules And Grievances

This file encodes the repeated project lessons that future agents must follow.

## Prime Before Acting

Before editing or running experiments, read:

1. `docs/INDEX.md`
2. `docs/repo-map.md`
3. `docs/experiment-plans.md`
4. This file
5. The exact source files named by the repo map.

Do not rely on memory alone.

## Do Not Rewrite Working Experiment Code

Repeated failure:

```text
Instead of using existing rollout/eval/video code, a new script was written
with different semantics or worse visualization.
```

Rule:

- Reuse canonical code first.
- If a new runner is needed, preserve the existing HDF5 contract so current
  evaluators and videos work unchanged.
- If a metric/plot/video layout needs to change, state the reason and preserve
  the old path unless explicitly replacing it.

Canonical files:

- `experiments/phase3/evaluate_online_cosmos_deviation.py`
- `experiments/phase3/plot_deviation_diagnostics.py`
- `experiments/phase3/make_prediction_metric_videos.py`

## Preserve Metric Meaning

Do not rename or reinterpret these silently:

```text
C = current observation
P = model-predicted future
R = observed future
a = selected/executed policy action chunk
```

Key HDF5 datasets:

```text
query_action_chunks
  selected/executed policy action chunks

query_cosmos_action_chunks
  Cosmos action-head chunks at the same query observation

actions_7d
  dense executed 7D actions

actions
  environment action; may be padded, e.g. RoboCasa 12D
```

For IDM-as-policy, evaluating against `query_action_chunks` is mostly a
self-check because `query_action_chunks` is the IDM action. To compare IDM with
Cosmos's native policy head, evaluate with:

```bash
--reference-action-dataset query_cosmos_action_chunks
```

## Preserve Horizon Meaning

The horizon is not cosmetic.

If the world model predicts a 16-step future, use a 16-step IDM comparison. Do
not evaluate a one-step IDM because it is easier.

Current base horizon:

```text
k = 16
target = 16 x 7 action chunk
```

RoboCasa Cosmos policy native output:

```text
32 x 7 action chunk
first 16 actions used for IDM/reference diagnostics
```

## Preserve Preprocessing Meaning

Preprocessing is model/checkpoint-specific.

Examples already observed:

- Cosmos LIBERO/RoboCasa paths use prepared frames from their wrapper, including
  flip conventions.
- GE paths may rotate frames differently.

Rule:

- Feed each model the image convention it was trained/evaluated with.
- For shared IDM comparisons, canonicalize `C`, `P`, and `R` together.
- Do not transform only the generated future frame.
- Write preprocessing notes into summaries when a run depends on it.

## Use `uv` Locally

Local Python commands must use the repo `uv` env:

```bash
uv venv .venv
.venv/bin/python <script.py>
```

Do not run project Python through bare system Python or an unrelated bundled
runtime.

Remote Cosmos jobs use the remote Cosmos env unless the remote script says
otherwise:

```text
/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python
```

## Remote Work Discipline

Default EC2 instance:

```text
i-09d3df96cbc593e1a
region ap-south-1
```

Default remote repo:

```text
/home/ubuntu/robotics/repos/online-cross-embodiment
```

Rules:

- Edit locally first.
- Sync or pull to EC2.
- Run data/model/sim jobs remotely.
- Download artifacts to `run-artifacts/`.
- Stop EC2 when finished.
- If a job is long, set a backup shutdown before starting.

## Commit Code Changes

Code and documentation changes should be committed after a coherent unit of
work. Generated artifacts should not be committed.

Before final:

```bash
git status --short
```

## Be Explicit About Deviations

If implementation diverges from the plan, say it before running:

- different model architecture,
- different horizon,
- different encoder,
- different action target,
- different reference action dataset,
- different preprocessing,
- different policy execution cadence,
- different plot/video layout.

This project has already lost time because these deviations were discovered
after the fact.

