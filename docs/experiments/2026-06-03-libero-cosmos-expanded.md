# 2026-06-03 - LIBERO Cosmos Expanded

Artifact:

```text
run-artifacts/phase3-cosmos-libero10-expanded-20260603-161058
```

## Purpose

Run Cosmos Policy on LIBERO-10 with extra attempts on tasks that had interesting success/failure behavior, then compute prediction-action deviations.

## Summary

Corrected deviation summary:

```text
n = 248 query points
episodes = 14
success_episodes = 12
checkpoint = idm_siglip_libero_downloaded_real_real_k16/best.pt in original MLP eval
horizon = 16
```

Tasks 4 and 8 have repeated runs including failures.

## Important Lessons

- Videos suggested deviations often spike around gripper/contact moments.
- Per-step cosine replaced flattened chunk cosine as the useful action-angle metric.
- The canonical video layout became execution frame + predicted future + metric trace.

## Caveats

- LIBERO is narrow.
- Failures are sparse.
- Some later analyses use the patch-transformer IDM, so compare summaries carefully.
