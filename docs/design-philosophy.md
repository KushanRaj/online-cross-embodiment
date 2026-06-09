# Design Philosophy

## Core Bet

The working thesis is:

```text
World models and VLAs often contain useful semantic/planning priors.
The weak link is usually action translation and context-conditioned control.
```

The project is not trying to prove that one policy is simply better on a leaderboard. It is trying to ask whether the gap between imagined future and observed future can become a useful signal for adaptation, failure prediction, or later RL.

## Why Prediction Honesty Matters

The central diagnostic is:

```text
C = current observation
P = model-predicted future
R = real observed future
a = policy action chunk
```

The useful comparisons are:

```text
P vs R
IDM(C, P) vs a
IDM(C, R) vs a
IDM(C, P) vs IDM(C, R)
```

If a model's imagined future implies an action different from the action it actually takes, the model may be semantically competent but actuator-conditioning weak. If the imagined future stays semantically wrong for a task, the issue may be world-model/task grounding rather than action decoding.

## Why IDM Exists Here

The IDM is not the thesis by itself. It is an instrument.

It turns a pair of frames into an action-space statement:

```text
Given current frame C and future frame F,
what action chunk would have caused this transition?
```

Then the project can ask:

```text
Does the action implied by the world model's imagined future match the policy action?
Does the action implied by observed reality diverge when the rollout fails?
Do these divergences spike around gripper/contact events?
```

## What Counts As Evidence

Strong evidence:

- The same metric separates success/failure across multiple task families.
- The signal survives richer environments such as RoboCasa, not only LIBERO.
- The signal is not only gripper-open spikes.
- The signal is stable across world models and policies.
- True action-conditioned counterfactual futures improve interpretability.

Weak evidence:

- A single pretty video.
- A single task where failures have larger L2.
- A metric that depends on flattened 16x7 cosine.
- A self-check where IDM-as-policy is compared to its own executed action.

## What This Repo Should Optimize For

This repo should optimize for reproducible, inspectable experiments:

- Preserve raw HDF5 rollout data.
- Preserve summaries and diagnostics.
- Render videos with the canonical layout.
- Keep action/reference dataset semantics explicit.
- Keep every architecture or preprocessing deviation written down before claiming results.

The first version can be simple. It cannot be ambiguous.
