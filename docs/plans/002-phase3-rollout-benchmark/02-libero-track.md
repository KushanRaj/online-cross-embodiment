# 02 - LIBERO Track

## Why LIBERO

LIBERO is the first runnable environment for the diagnostic. It is smaller and more toy-like than RoboCasa, but it has demonstrations and compatible policy checkpoints.

## Current Important Run

Artifact:

```text
run-artifacts/phase3-cosmos-libero10-expanded-20260603-161058
```

Corrected deviation summary:

```text
episodes = 14
success_episodes = 12
tasks 4 and 8 had repeated success/failure attempts
checkpoint = idm_siglip_libero_downloaded_real_real_k16/best.pt for original MLP eval
later patch-transformer eval also exists
```

## What It Showed

Early interpretation:

- failures existed on tasks 4 and 8,
- metric spikes often seemed gripper/contact related,
- the diagnostic was visually interesting but not yet a clean separator.

## Caveats

- LIBERO may be too narrow to prove generality.
- More balanced failure cases are needed.
- Use per-step cosine, not flattened 16x7 cosine.
- Do not infer that the metric is useless just because aggregate lines overlap; inspect task/phase-specific traces.
