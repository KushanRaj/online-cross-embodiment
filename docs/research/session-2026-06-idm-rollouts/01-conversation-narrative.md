# 01 - Conversation Narrative

## Literature Phase

The work began by reading and comparing DreamZero, IVLR, LPS, RLT/Pi, Molmo, GE-Act/GE-Sim, Cosmos Policy, FAST WAM, LingBot, and semantic-vs-reconstruction IDM work.

The recurring idea was:

```text
models may reason or imagine well,
but the action translation layer is fragile.
```

This led to the question:

```text
Can predicted-vs-observed future mismatch become a signal?
```

## Environment Phase

The project looked at DROID, PolaRiS, LIBERO, MolmoSpaces, and RoboCasa.

The important distinction was established:

```text
dataset trajectories cannot support arbitrary closed-loop rollouts
sim environments can
```

LIBERO became the first environment because it was available and lightweight. RoboCasa became the richer follow-up.

## IDM Phase

The first IDM was trained from real-to-real trajectory windows:

```text
current frame + future frame + proprio -> 16 x 7 action chunk
```

The baseline used pooled SigLIP features and an MLP. Longer-horizon and patch-transformer variants were explored, but the baseline remains the clean reference.

## Online Rollout Phase

The key loop became:

```text
run policy online
query Cosmos future P at selected timesteps
observe real future R
compute IDM(C,P) vs policy action a
render videos and aggregate plots
```

LIBERO expanded runs gave early success/failure examples. RoboCasa then tested whether the signal survives richer household/contact tasks.

## Documentation Phase

The user rejected shallow docs and requested a browser-challenge-style hierarchy:

- top-level index,
- micro-documents,
- plan bundles,
- runbooks,
- explicit grievances and operating rules.

This folder is the result.
