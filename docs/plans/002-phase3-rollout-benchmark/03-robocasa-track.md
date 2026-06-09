# 03 - RoboCasa Track

## Why RoboCasa

RoboCasa adds richer household manipulation and contact variation. It helps answer:

```text
Is prediction-action inconsistency only a LIBERO artifact?
```

## Cosmos Policy RoboCasa Run

Artifact:

```text
run-artifacts/robocasa-cosmos-deviation-20260609-112643
```

Summary:

```text
mode = cosmos_policy_robocasa_deviation
policy_ckpt = nvidia/Cosmos-Policy-RoboCasa-Predict2-2B
episodes = 11
successes = 4
success_rate = 0.3636
policy_native_chunk_size = 32
idm_reference_horizon = 16
```

Tasks:

```text
CloseDrawer
CoffeeServeMug
CoffeeSetupMug
OpenDrawer
TurnOffMicrowave
TurnOffSinkFaucet
TurnOffStove
TurnOnMicrowave
TurnOnSinkFaucet
TurnOnStove
TurnSinkSpout
```

## Action Convention

The IDM target is 7D. RoboCasa environment action may be wider. The runner pads the manipulation action for execution.

Do not train the IDM on success labels.

## Caveats

- The first RoboCasa run is one trial per task.
- Need more repetitions for success/failure correlation.
- Ensure the RoboCasa IDM checkpoint is used for RoboCasa diagnostics.
- Check whether the Cosmos Policy RoboCasa dataset/action convention matches the executed environment convention before stronger claims.
