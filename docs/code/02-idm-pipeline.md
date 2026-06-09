# IDM Pipeline

## Purpose

The IDM maps visual transition pairs to action chunks:

```text
IDM(C, F, proprio) -> K x 7 action chunk
```

In this project, the IDM is an instrument for asking whether a future image implies the same action as a policy actually took.

## Data Contract

A training sample should contain:

```text
sample_id
task_name
trajectory_id
t
horizon
current_primary_image
future_primary_image
current_wrist_image optional
future_wrist_image optional
current_proprio optional
action_chunk_Kx7
success/failure metadata optional
source
```

Success/failure metadata is not an IDM input.

## Current Base Training Setup

Baseline:

```text
encoder = google/siglip-base-patch16-224
idm_type = feature
model = FeatureIDM
horizon = 16
target = 16 x 7
loss = Smooth L1
input = pooled C/P features + proprio when available
```

The baseline is intentionally simple. It should be the first comparison point before larger architectures.

## Larger IDM Attempt

There is also:

```text
PatchTransformerIDM
```

This uses patch-token features rather than only pooled features. It may be useful, but it is an architecture change. Future agents must not silently substitute it for the MLP baseline.

## Important Source Files

Data:

- `experiments/idm/data/manifest.py`
- `experiments/idm/data/patch_windows.py`
- `experiments/idm/data/cache_features.py`
- `experiments/idm/data/cache_patch_windows.py`

Model/training:

- `experiments/idm/model/idm.py`
- `experiments/idm/model/train_idm.py`
- `experiments/idm/model/adapter.py`

Remote data export:

- `experiments/idm/remote/export_libero_demo_windows.py`
- `experiments/idm/remote/export_robocasa_rollout_windows.py`

## W&B Logging Expectations

Minimum expected logs:

- train loss,
- validation loss,
- learning rate,
- global step,
- validation action metrics where available,
- media examples when configured.

If a new training path logs fewer metrics than the baseline, treat it as incomplete instrumentation rather than acceptable simplification.

## Horizon Notes

Base horizon is 16. Longer horizons such as 32 or 128 are separate ablations.

For a mixed-horizon model, the model must be described explicitly:

```text
Does it always output 128?
Which prefix is evaluated for k=16, k=32, or k=128?
Is the data mixed in one training run or trained one horizon at a time?
```

The desired mixed-horizon idea was one model trained on mixed data, not one independent model at a time.

## Reverse Data

Reverse future-to-past action recovery is an ablation. It is not part of the base run.

Potential reverse sample:

```text
IDM(F, C) -> reverse action chunk
```

This could expose temporal symmetry, but it could also confuse the forward diagnostic. Keep it separate.
