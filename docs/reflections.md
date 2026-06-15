# Reflections And Operating Directives

These notes are mined from the current project session and the repo artifacts through 2026-06-09. No separate exported transcript was found inside the repo, so this file records the durable lessons from the working conversation and generated evidence.

## Prime Before Acting

Future agents must read:

1. [INDEX.md](INDEX.md)
2. [code/README.md](code/README.md)
3. The relevant plan bundle in [plans/](plans/README.md)
4. The relevant runbook in [runbooks/](runbooks/README.md)
5. The exact source files named there

Do not start by grepping random scripts and inventing a new runner.

## Do Not Rewrite Working Code

Repeated grievance:

```text
There was already code to do the rollout/evaluation/video.
Why did you write something new?
```

Rule:

- Reuse canonical scripts first.
- If a new runner is needed, preserve the existing HDF5 schema.
- If a new plot is needed, keep the old canonical plot path working.
- If the canonical code is insufficient, document the exact missing input/output before adding code.

Canonical current files:

- `experiments/phase3/evaluate_online_cosmos_deviation.py`
- `experiments/phase3/plot_deviation_diagnostics.py`
- `experiments/phase3/make_prediction_metric_videos.py`

SO101/MolmoAct2 live-control artifacts are not Phase 3 HDF5 artifacts. If a
camera/action-log renderer is useful, put it in tracked SO101/MolmoAct2 tooling
and reuse it. Do not create a fresh `render_*.py` inside each downloaded
`run-artifacts/` bundle.

## No Silent Experimental Drift

The user must be told before changing:

- IDM architecture,
- encoder,
- horizon,
- optimizer,
- learning rate,
- loss,
- action target,
- preprocessing,
- reference action dataset,
- policy execution cadence,
- plot/video layout,
- task suite,
- checkpoint.

Architectural deviations are especially sensitive. The project moved from MLP IDM to patch-transformer IDM without enough explicit discussion. That should not repeat.

## Local Python Rule

Local Python must use the repo `uv` environment. Do not run project Python with bare system Python.

Correct local shape:

```bash
uv venv .venv
.venv/bin/python <script.py>
```

Remote Cosmos jobs use the remote Cosmos environment when required:

```text
/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python
```

## Be Precise About Action Datasets

The following names are not interchangeable:

```text
query_action_chunks
  action chunks selected/executed by the active policy in the rollout

query_cosmos_action_chunks
  Cosmos action-head chunks at the same query observation

actions_7d
  dense 7D executed manipulation actions

actions
  environment actions, possibly padded or wider than 7D
```

For IDM-as-policy runs, `query_action_chunks` is the IDM action, so comparing `IDM(C,P)` to `query_action_chunks` is a near self-check. To compare IDM policy against Cosmos's native policy, use `query_cosmos_action_chunks` as the reference action dataset.

## Preserve Video Layout

The canonical video should show:

- dense execution frame,
- Cosmos predicted future image for the active chunk,
- metric traces with red history and green active chunk/dot/band,
- task caption and success/failure status.

Do not replace this with a sparse query-frame-only video. The sparse playback bug happened because the renderer used query frames instead of dense execution frames.

## Metric Lessons

Do not use flattened 16x7 cosine as the main interpretation metric. It hides per-step structure.

Preferred diagnostics:

- per-step action L2 averaged across the chunk,
- per-step action cosine distance averaged across the chunk,
- EEF per-step L2,
- EEF per-step cosine distance,
- gripper mismatch rate,
- SigLIP P-vs-R visual similarity.

L2 has units in action-space scale; cosine distance is unitless and high means lower directional alignment.

## EC2 Discipline

Default instance:

```text
i-09d3df96cbc593e1a
region ap-south-1
```

Rules:

- Do not start EC2 without a concrete run plan.
- Sync local code to remote before running.
- Download artifacts before shutdown when the user needs videos/plots.
- Stop the instance when remote work is complete.
- For long runs, set a backup shutdown.

## Documentation Standard

The user rejected shallow macro-docs. Future docs should be:

- micro-documents grouped by category,
- indexed at every folder level,
- explicit about task, inputs, outputs, and caveats,
- grounded in run artifacts and source paths,
- written so the next agent knows what already exists before creating anything new.
