# 04 - Diagnostics

## IDM Quality

For the IDM itself, inspect:

- validation Smooth L1,
- per-dimension action error,
- EEF error,
- gripper error,
- per-step cosine distance,
- Pearson/correlation if logged,
- sample visualizations.

Do not judge only train loss.

## Downstream Usefulness

The IDM is useful if the derived diagnostic is interpretable:

```text
IDM(C,P) vs a
IDM(C,R) vs a
IDM(C,P) vs IDM(C,R)
```

Useful outcomes:

- failure episodes have distinct patterns,
- spikes correspond to semantically meaningful phases,
- gripper/contact spikes can be separated from EEF drift,
- signal is stable across tasks and environments.

## Known Caveats

IDM-as-policy can be misleading:

```text
IDM(C,P) is optimized to recover actions from futures.
It is not necessarily trained as a stable closed-loop controller.
```

Self-evaluation in IDM-as-policy is trivial if the reference action is the IDM action. Compare against Cosmos action head if asking whether the IDM policy agrees with Cosmos.
