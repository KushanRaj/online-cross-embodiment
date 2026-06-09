# 04 - Caveats

## Caveat 1 - Some Runs Use Different IDMs

Not all summaries use the same IDM.

Examples:

- early LIBERO MLP IDM,
- later patch-transformer IDM snapshots,
- RoboCasa MLP IDM.

Always check `summary.json`.

## Caveat 2 - Some Runs Use Different Reference Actions

`query_action_chunks` and `query_cosmos_action_chunks` answer different questions.

The RoboCasa IDM-as-policy run has both self-reference and Cosmos-reference deviation summaries. The self-reference summary is not the meaningful policy comparison.

## Caveat 3 - Cosmos Future Is Not Always External-Action-Conditioned

For Pi/Molmo probes:

```text
Cosmos future is generated from current observation/task,
not from the external policy action chunk.
```

## Caveat 4 - Preprocessing Can Flip Frames

Cosmos-prepared frames may differ from raw simulator frames. For metrics, compare frames that share the same preprocessing convention.

## Caveat 5 - Success Counts Are Sparse

Several runs use one episode per task. This is not enough for strong statistical claims.
