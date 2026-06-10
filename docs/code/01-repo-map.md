# Repo Map

## Top-Level Structure

```text
AGENTS.md            agent entrypoint; points to this documentation system
README.md            public-facing repo summary
docs/                indexed documentation hierarchy
experiment-design/   older Obsidian-style experiment notes and reports
experiments/         tracked experiment code
papers/              paper notes and paper catalogue
pdfs/                local PDFs
run-artifacts/       downloaded run outputs; ignored by git
vm-patches/          older VM patches / scratch scripts
```

## Research Notes

Important tracked notes:

- `Robotics World Models.md`
- `papers/paper-catalog.md`
- `papers/*.md`
- `experiment-design/reports/*.md`
- `experiment-design/*.md`
- `docs/planning/*.md`

The docs in this folder do not replace those notes. They index the active experimental program and make the repo navigable.

## IDM Code

Base path:

```text
experiments/idm/
```

Important source files:

- `experiments/idm/README.md`
- `experiments/idm/data/manifest.py`
- `experiments/idm/data/patch_windows.py`
- `experiments/idm/data/cache_features.py`
- `experiments/idm/data/cache_patch_windows.py`
- `experiments/idm/model/idm.py`
- `experiments/idm/model/train_idm.py`
- `experiments/idm/model/adapter.py`
- `experiments/idm/model/evaluate_cosmos_deviation.py`
- `experiments/idm/model/backfill_wandb.py`

Remote/export helpers:

- `experiments/idm/remote/export_libero_demo_windows.py`
- `experiments/idm/remote/export_robocasa_rollout_windows.py`
- `experiments/idm/remote/export_rollout_windows.py`
- `experiments/idm/remote/query_cosmos_real_to_model.py`
- `experiments/idm/remote/query_ge_act_real_to_model.py`
- `experiments/idm/remote/query_ge_base_real_to_model.py`

Remote orchestration:

- `experiments/idm/scripts/run_libero_idm_remote.sh`
- `experiments/idm/scripts/run_mixed_horizon_idm_remote.sh`
- `experiments/idm/scripts/run_h128_idm_remote.sh`
- `experiments/idm/scripts/run_robocasa_idm_remote.sh`
- `experiments/idm/scripts/sync_to_remote.sh`

## Phase 3 Code

Base path:

```text
experiments/phase3/
```

Current runners:

- `experiments/phase3/run_robocasa_cosmos_deviation.py`
- `experiments/phase3/run_idm_as_policy_robocasa.py`
- `experiments/phase3/run_idm_as_policy_libero.py`
- `experiments/phase3/run_libero_policy_cosmos_probe.py`
- `experiments/phase3/run_cosmos3_libero_i2v_smoke.py`
- `experiments/phase3/run_cosmos3_robocasa_i2v_smoke.py`

Current eval/render:

- `experiments/phase3/evaluate_online_cosmos_deviation.py`
- `experiments/phase3/plot_deviation_diagnostics.py`
- `experiments/phase3/make_prediction_metric_videos.py`

Task/config files:

- `experiments/phase3/tasks/phase3_clean_tasks.yaml`
- `experiments/phase3/tasks/robotwin_selected_mapping.json`

Remote scripts:

- `experiments/phase3/scripts/download_libero_phase3_remote.sh`
- `experiments/phase3/scripts/download_robocasa_cosmos_remote.sh`
- `experiments/phase3/scripts/run_robocasa_idm_cosmos_remote.sh`
- `experiments/phase3/scripts/setup_phase3_remote.sh`
- `experiments/phase3/scripts/setup_robolab_remote.sh`
- `experiments/phase3/scripts/setup_robotwin_remote.sh`

## Local Artifacts

Local downloaded artifacts:

```text
run-artifacts/<run-name>/
```

These are intentionally not committed. Use [06-artifact-layout.md](06-artifact-layout.md) to know what to expect inside them.
