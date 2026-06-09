# 02 - Current Cosmos Limitation

Current Pi/Molmo/IDM-as-policy probes do not feed the external policy action into Cosmos as action conditioning.

Current:

```text
Cosmos input = current image/proprio/task
Cosmos output = P + Cosmos action-head chunk
external policy output = a_external
```

The external action affects the next Cosmos query only because the simulator state changes after execution.

Not current:

```text
Cosmos input = current image/proprio/task + a_external
Cosmos output = P(a_external)
```

This limits causal claims.
