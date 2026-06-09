# 03 - Training Contract

## Base Model

Current baseline:

```text
FeatureIDM
input = pooled SigLIP(C), pooled SigLIP(F), optional proprio
output = 16 x 7 action chunk
loss = Smooth L1
```

Encoder:

```text
google/siglip-base-patch16-224
```

## Approved Deviation: Proprio

The related paper's IDM probe was visual-latent-only. This project allows proprio as input because the diagnostic is embodiment-specific and the goal is not to compare latent spaces in isolation.

Record whether proprio was used in every run summary.

## Architecture Deviations

Patch-transformer IDM is a deviation from the base. It should be logged as:

```text
idm_type = patch
encoder = google/siglip-base-patch16-224
input = patch tokens / adapted C-P window
```

It should not silently replace the MLP baseline.

## Logging Contract

W&B should log:

- `train/lr`,
- `train/batch_loss`,
- validation loss,
- action recovery metrics,
- global step,
- epoch or pass count,
- examples/media when configured.

If a new path logs only train loss and lr, it is under-instrumented.

## Checkpoint Contract

Training checkpoints should include:

- model state,
- optimizer state,
- scheduler state if any,
- config,
- global step,
- encoder,
- horizon,
- action dimension.

If optimizer state is not restored, resumed loss curves are not true continuous training curves.
