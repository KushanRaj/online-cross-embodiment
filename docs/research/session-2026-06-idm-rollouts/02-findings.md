# 02 - Findings

## Finding 1 - The Diagnostic Is Plausible But Not Proven

The videos and metrics show interpretable deviations, especially around contact and gripper movement. This is enough to justify deeper experiments, but not enough to claim a robust failure predictor.

## Finding 2 - Action Reference Matters

For normal policy runs:

```text
query_action_chunks = selected/executed policy action
```

For IDM-as-policy:

```text
query_action_chunks = IDM action
query_cosmos_action_chunks = Cosmos action-head reference
```

Confusing these changes the meaning of the plots.

## Finding 3 - The Video Renderer Is Part Of The Experiment

The useful video is not just a visualization. It is how the metric is interpreted.

Required layout:

```text
dense execution frame + active predicted future + metric trace
```

Sparse query-frame playback made the result misleading.

## Finding 4 - Gripper/Contact Needs Separate Treatment

Large spikes often appeared around gripper state changes. This might be real signal or a data/loss artifact.

Need:

- EEF-only metrics,
- gripper mismatch rate,
- action distribution analysis,
- more videos from successes and failures.

## Finding 5 - The Current Cosmos Probe Is Not A Full Counterfactual

Pi/Molmo/IDM external actions are not fed into Cosmos as arbitrary action conditioning. Current probes compare external action to the future Cosmos imagines from current state/task.

This is still useful, but the stronger causal experiment remains open.
