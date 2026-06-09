# 01 - Hypothesis

## Core Hypothesis

The IDM should act as an action-space probe for future images:

```text
If the world model predicts future P from current C,
then IDM(C,P) is the action the future image implies.
```

Compare that to the policy action:

```text
IDM(C,P) vs a
```

If they diverge, the policy may be taking actions that do not match its own imagined future or the world model's imagined future.

## What This Does Not Prove Alone

This does not prove:

- the IDM is a good policy,
- the world model is causally correct,
- the policy would improve if corrected by the IDM,
- cross-embodiment transfer works.

It only gives a diagnostic axis:

```text
future/action consistency
```

## Why Model-Agnostic

The IDM should not depend on DreamZero/Cosmos/GE internal latents for the first pass. Decode or render futures to pixel space, encode with a shared visual encoder, and compare in a common feature space.

This keeps the diagnostic from becoming:

```text
one IDM per model latent space
```

The current target is:

```text
one IDM per embodiment/action convention
many world models can be probed through decoded futures
```

## Why Embodiment-Specific

The output is still an action chunk. A LIBERO 7D end-effector action and a RoboCasa padded action convention are not the same execution interface.

Therefore:

- LIBERO IDM is trained for LIBERO action convention.
- RoboCasa IDM is trained for RoboCasa/Cosmos 7D manipulation target and padded at execution time when needed.

Embodiment-free action translation is a later research track.
