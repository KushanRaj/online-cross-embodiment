# Experiment Plans

This file is the planning spine. Detailed research notes live in
`experiment-design/`, but this page should be enough to orient future work.

## Core Thesis

The bet is not that we should train a larger policy immediately.

The bet is:

```text
world models / VLAs often encode useful semantic planning priors
but action translation and context-conditioned control are weak
```

Therefore we test whether predicted-future mismatch can become an actionable
signal:

```text
current C
model-predicted future P
observed future R
policy action chunk a

compare:
  P vs R
  IDM(C, P) vs a
  IDM(C, R) vs a
  IDM(C, P) vs IDM(C, R)
```

## Phase 1 - IDM Training

Goal:

```text
train IDM(C, F, proprio) -> 16 x 7 action chunk
```

Baseline:

- Feature MLP IDM.
- Encoder: `google/siglip-base-patch16-224`.
- Action target: first 7 manipulation/action dims.
- Horizon: `k = 16`.
- Source: real-to-real trajectory windows.

LIBERO baseline:

- Uses LIBERO demonstration trajectories.
- Action space is 7D end-effector/gripper action.

RoboCasa baseline:

- Uses RoboCasa/Cosmos rollout-style dataset trajectories.
- IDM target is still 16 x 7.
- RoboCasa env execution pads 7D with fixed mobile-base tail:

```text
[0, 0, 0, 0, -1]
```

That padding is not part of the IDM target.

Important:

- Success labels are metadata only; do not feed them into the IDM.
- Reverse future-to-past data is an ablation, not a base run.
- Generated futures are ablations unless a run explicitly says otherwise.

## Phase 2 - Offline Diagnostics

Goal:

```text
given stored C, P, R, and a, compute diagnostic metrics
```

Metrics currently used:

- SigLIP `P` vs `R` feature L2 and cosine.
- Chunk L2: `IDM(C,P) - a`.
- Chunk L2: `IDM(C,R) - a`.
- Chunk L2: `IDM(C,P) - IDM(C,R)`.
- Per-step action L2.
- Per-step EEF L2.
- Per-step action cosine distance.
- Per-step EEF cosine distance.
- Gripper L1 and gripper sign mismatch rate.

Cosine caveat:

- Flattened 16x7 cosine is legacy only.
- The useful cosine is per-step cosine averaged across the chunk, especially
  EEF-only cosine.

## Phase 3 - Online Rollout Evaluation

Goal:

```text
run a policy in a simulator
query a world-model future P during rollout
plot whether IDM/world-model mismatch aligns with policy failure
```

Current environments:

- LIBERO / robosuite.
- RoboCasa / robosuite.

Current policy tracks:

- Cosmos Policy as policy.
- Pi 0.5 as policy, with Cosmos as future/action probe.
- MolmoAct2 as policy, with Cosmos as future/action probe.
- IDM as policy, using Cosmos future `P` as the action target.

Important action-conditioning caveat:

For Pi/Molmo/IDM-as-policy probes, Cosmos is currently not given the external
policy action as an action-conditioned future input. Cosmos receives:

```text
current prepared image(s)
current proprio
task text
```

and returns:

```text
Cosmos future P
Cosmos action-head chunk
optional value
```

The executed Pi/Molmo/IDM action affects the next Cosmos query only indirectly,
through the next observed simulator frame.

This means the current diagnostic is:

```text
Does the external policy action agree with the action implied by Cosmos's
imagined future?
```

It is not yet:

```text
Given this external action chunk, what counterfactual future does Cosmos render?
```

That latter experiment needs a true action-conditioned generation API/checkpoint.

## Phase 4 - Counterfactual Futures

Future goal:

```text
feed arbitrary action chunks into an action-conditioned world model
render counterfactual futures
score whether the action-conditioned future matches observed consequences
```

Candidate model families:

- GE-Sim / action-conditioned simulator path.
- Cosmos action-conditioned future path, if exposed cleanly.
- Other WAMs that explicitly condition generated futures on action chunks.

Do not conflate this with the current Cosmos Policy joint action/future probe.

## Phase 5 - Cross-Environment / Cross-Embodiment Extensions

Near-term:

- Train RoboCasa-only IDM.
- Compare LIBERO-only, RoboCasa-only, and mixed IDM.
- Check whether the IDM signal survives richer household tasks.

Later:

- Arm-to-arm EEF adapter experiments.
- Cross-embodiment action-space normalization.
- Humanoid/mobile manipulation is out of scope until arm-to-arm transfer is
  well specified.

See:

- `experiment-design/Future Experiment - Arm-to-Arm Embodiment Transfer via EEF Adapters.md`
- `experiment-design/Shared Semantic IDM - Prediction Reality Deviation.md`

