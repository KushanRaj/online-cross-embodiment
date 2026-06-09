# Repo Map

This file is the navigation map for code and research artifacts.

## Top-Level Structure

```text
papers/              paper notes and paper catalogue
pdfs/                local paper PDFs
experiment-design/   experiment ideas, phase notes, synthesis reports
docs/                repo navigation, planning docs, operating rules
experiments/         tracked experiment code
run-artifacts/       local downloaded run outputs; ignored by git
vm-patches/          older VM patch snippets and one-off scripts
```

`run-artifacts/` is where downloaded videos, HDF5 rollout dumps, summaries, and
plots live locally. Do not commit generated run outputs.

## Research Notes

- `Robotics World Models.md` - top-level MOC.
- `papers/paper-catalog.md` - paper list.
- `papers/*.md` - distilled paper notes.
- `experiment-design/reports/*.md` - larger synthesis reports.
- `docs/planning/*.md` - older planning notes.

## IDM Code

Base path:

```text
experiments/idm/
```

Important files:

- `experiments/idm/README.md` - IDM data/training contract. Read before
  touching IDM data or model code.
- `experiments/idm/data/manifest.py` - manifest parsing and image/frame loading.
- `experiments/idm/data/patch_windows.py` - LIBERO/demo transition-window
  construction.
- `experiments/idm/data/cache_features.py` - pooled/frozen encoder feature cache.
- `experiments/idm/data/cache_patch_windows.py` - patch-token feature cache.
- `experiments/idm/model/idm.py` - `FeatureIDM` and `PatchTransformerIDM`.
- `experiments/idm/model/train_idm.py` - main IDM training loop and W&B logging.
- `experiments/idm/model/adapter.py` - online inference adapter used by Phase 3
  rollout/eval scripts.
- `experiments/idm/model/evaluate_cosmos_deviation.py` - older/offline deviation
  evaluator. Prefer Phase 3 evaluator for current online rollout artifacts.
- `experiments/idm/model/backfill_wandb.py` - W&B backfill utility.

Remote data/query scripts:

- `experiments/idm/remote/export_libero_demo_windows.py`
- `experiments/idm/remote/export_robocasa_rollout_windows.py`
- `experiments/idm/remote/export_rollout_windows.py`
- `experiments/idm/remote/query_cosmos_real_to_model.py`
- `experiments/idm/remote/query_ge_act_real_to_model.py`
- `experiments/idm/remote/query_ge_base_real_to_model.py`

Remote orchestration scripts:

- `experiments/idm/scripts/run_libero_idm_remote.sh`
- `experiments/idm/scripts/run_mixed_horizon_idm_remote.sh`
- `experiments/idm/scripts/run_h128_idm_remote.sh`
- `experiments/idm/scripts/run_robocasa_idm_remote.sh`
- `experiments/idm/scripts/sync_to_remote.sh`

## Phase 3 Rollout, Evaluation, Plotting

Base path:

```text
experiments/phase3/
```

Canonical rollout runners:

- `experiments/phase3/run_robocasa_cosmos_deviation.py`
  - Runs RoboCasa with Cosmos Policy as the executed policy.
  - Saves Cosmos futures and action chunks for IDM deviation diagnostics.
- `experiments/phase3/run_idm_as_policy_robocasa.py`
  - Runs RoboCasa with the IDM as the executed policy.
  - Cosmos supplies `P`; IDM predicts the action chunk from `IDM(C, P)`.
  - Stores Cosmos's native action-head chunk separately.
- `experiments/phase3/run_idm_as_policy_libero.py`
  - LIBERO analogue of IDM-as-policy.
- `experiments/phase3/run_libero_policy_cosmos_probe.py`
  - Runs Pi 0.5 or MolmoAct2 as the executed LIBERO policy while querying
    Cosmos for a future/action probe at the same observation.
- `experiments/phase3/run_cosmos3_robocasa_i2v_smoke.py`
  - Cosmos 3 Nano image-to-video smoke test, not the main policy path.

Canonical evaluators and diagnostics:

- `experiments/phase3/evaluate_online_cosmos_deviation.py`
  - Reads online rollout HDF5 files.
  - Computes SigLIP P-vs-R similarity and IDM action-difference metrics.
  - Supports `--reference-action-dataset`.
- `experiments/phase3/plot_deviation_diagnostics.py`
  - Produces success/failure colored aggregate plots.
  - Includes chunk L2, per-step L2, EEF cosine, EEF L2, and gripper mismatch.
- `experiments/phase3/make_prediction_metric_videos.py`
  - Canonical video renderer.
  - Layout: execution frame, Cosmos predicted future for active chunk, metric
    traces with red history and green active chunk.
  - Do not replace this layout casually.

Setup/task files:

- `experiments/phase3/README.md`
- `experiments/phase3/tasks/phase3_clean_tasks.yaml`
- `experiments/phase3/tasks/robotwin_selected_mapping.json`
- `experiments/phase3/scripts/download_libero_phase3_remote.sh`
- `experiments/phase3/scripts/download_robocasa_cosmos_remote.sh`
- `experiments/phase3/scripts/run_robocasa_idm_cosmos_remote.sh`

## Artifact Layout

Local downloaded runs use:

```text
run-artifacts/<run-name>/
```

Common files inside a run:

```text
summary.json
episodes.csv
run_manifest.json
rollout_data/*.hdf5
deviation*/online_cosmos_deviation.csv
deviation*/summary.json
deviation*/diagnostics_success_failure/*.png
prediction_metric_videos*/prediction_metrics_*.mp4
```

Remote runs usually live under:

```text
/home/ubuntu/robotics/runs/
```

Default remote repo path:

```text
/home/ubuntu/robotics/repos/online-cross-embodiment
```

Default remote Cosmos env:

```text
/home/ubuntu/robotics/repos/cosmos-policy/.venv/bin/python
```

## Current Local Run Artifacts To Know

These names are useful anchors when debugging recent results:

- `run-artifacts/robocasa-cosmos-deviation-20260609-112643`
  - RoboCasa with Cosmos Policy as policy.
- `run-artifacts/robocasa-idm-policy-20260609-133354`
  - RoboCasa with IDM as policy, using Cosmos future `P`.
- `run-artifacts/pi05-molmo-libero-cosmos-proper-20260608`
  - LIBERO Pi/Molmo policy probes with Cosmos future/action probes.
- `run-artifacts/phase3-cosmos-libero10-expanded-20260603-161058`
  - Expanded LIBERO Cosmos run.

Do not assume these are committed; they are local artifacts.

