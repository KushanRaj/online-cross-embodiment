# 03 - Artifact Map

## LIBERO Cosmos Expanded

```text
run-artifacts/phase3-cosmos-libero10-expanded-20260603-161058
```

Use for:

- LIBERO Cosmos Policy success/failure examples,
- corrected deviation plots,
- canonical prediction metric videos.

Known summary:

```text
episodes = 14
success_episodes = 12
tasks 4 and 8 have repeated runs including failures
```

## Pi/Molmo LIBERO Probes

```text
run-artifacts/pi05-molmo-libero-cosmos-proper-20260608
```

Use for:

- Pi 0.5 as external policy,
- MolmoAct2 as external policy,
- Cosmos future probe at same observation,
- policy cadence 10 predict / 8 execute / 8 execute.

Known summary:

```text
Pi 0.5: 9/10 successes
MolmoAct2: 9/10 successes
```

## RoboCasa Cosmos Policy

```text
run-artifacts/robocasa-cosmos-deviation-20260609-112643
```

Use for:

- RoboCasa Cosmos Policy as executed policy,
- RoboCasa IDM diagnostics,
- 11 task household manipulation set.

Known summary:

```text
episodes = 11
successes = 4
success_rate = 0.3636
policy_ckpt = nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
```

## RoboCasa IDM As Policy

```text
run-artifacts/robocasa-idm-policy-20260609-133354
```

Use for:

- IDM as executed policy,
- Cosmos future P,
- comparison against Cosmos action head through `query_cosmos_action_chunks`.

Known summary:

```text
episodes = 11
successes = 0
```

## Cosmos 3 Nano Smoke Tests

```text
run-artifacts/ec2-cosmos3-nano-libero-i2v-smoke-20260604
run-artifacts/cosmos3-nano-robocasa-i2v-smoke-20260608
run-artifacts/cosmos3-nano-robocasa-i2v-long-400f-direct-20260608-133338
```

Use for:

- visual inspection of base I2V future quality,
- deciding whether base Cosmos 3 is useful as a world model in these environments.
