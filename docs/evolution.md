# Evolution

This is the chronological record of how the project reached the current experimental setup. It is intentionally written as project memory, not as a polished paper.

## 1. Paper Reading And Thesis Formation

The early focus was literature triage:

- DreamZero: joint future/action modeling and prediction honesty, but not the cleanest action-counterfactual simulator.
- Pi / Pi 0.5 / RLT: strong VLA-policy family; questions around expert data, online/off-policy RL, and RL-token adaptation.
- LPS and IVLR: planning over latent traces and batching plan/exploit phases.
- Molmo / MolmoAct: instruction/vision sensitivity and DROID-style policy rollout questions.
- GE-Act / GE-Sim / EWM Bench: action-conditioned world-model evaluation, predicted progress, semantic substep completion, and scene consistency.
- Cosmos Policy: world model plus decision-transformer-style policy and value/search components.
- Reconstruction vs semantic latent-space work: IDM probes as a way to compare latent spaces and priors for action recovery.

The high-level synthesis became:

```text
Reasoning and semantic priors seem useful.
Action translation is where these systems become brittle.
We need diagnostics that expose that mismatch without requiring full RL.
```

## 2. Environment And Benchmark Scope

The project considered DROID, PolaRiS, LIBERO, MolmoSpaces, and RoboCasa.

The important distinction was:

```text
dataset = fixed trajectories; no arbitrary rollout
environment = simulator or real robot; policy can take new actions and observe consequences
```

LIBERO was chosen first because it was runnable and had demonstrations for IDM training. RoboCasa was added because LIBERO is too small and toy-like; RoboCasa has more household/contact diversity.

## 3. Phase 1 IDM Training

The first practical instrument was an IDM:

```text
IDM(C, F, proprio) -> action chunk
```

The baseline was a feature MLP over pooled SigLIP features, using real-to-real demonstration windows. The useful base horizon became `k = 16`, because Cosmos futures and diagnostics operate around a 16-step future. Longer-horizon IDM experiments were explored, including mixed 16/32/128 horizon data and a patch-transformer IDM.

Important correction:

The patch-transformer IDM was an architectural deviation from the simpler MLP setup. Future architecture changes must be stated before running because the user explicitly wants to approve model architecture, loss, horizon, optimizer, and data changes.

## 4. Phase 3 LIBERO Online Rollouts

The first useful online loop was:

```text
run Cosmos Policy in LIBERO
at query timesteps, store C, P, R, and a
evaluate IDM(C,P) vs a and IDM(C,R) vs a
render videos with prediction + metric trace
```

The expanded LIBERO run showed mixed success/failure on tasks 4 and 8 and gave early videos where spikes often seemed to align with gripper/contact moments.

Metric corrections happened here:

- Flattened 16x7 cosine was judged wrong for the main chart.
- Per-step cosine averaged across the chunk became the better action-angle diagnostic.
- EEF-only metrics and gripper mismatch were added.

## 5. Pi 0.5 And Molmo LIBERO Probes

The next question was whether external policies could be compared against Cosmos's future/action probe:

```text
Pi or Molmo executes actions.
Cosmos observes the same current state and predicts P plus its own action head.
IDM scores whether the external action agrees with the Cosmos-implied future.
```

Key caveat:

The external Pi/Molmo action is not currently fed into Cosmos as action conditioning. Cosmos is queried from current observation, proprio, and task text. The external action only affects the next query indirectly through the next observed frame.

## 6. RoboCasa Extension

RoboCasa was added to test whether the diagnostic survives richer household manipulation.

The RoboCasa setup used:

- Cosmos Policy RoboCasa checkpoint: `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B`
- RoboCasa IDM trained on 7D action targets.
- Environment execution padded 7D policy/IDM actions into the 12D RoboCasa action convention.

The first RoboCasa Cosmos run produced 4 successes out of 11 tasks. The IDM-as-policy run produced 0 successes out of 11 tasks, but it was still useful as a diagnostic of how far a pure world-model-to-IDM policy is from an executable policy.

## 7. Current State

The repo now has:

- an IDM data/training pipeline,
- online LIBERO/RoboCasa rollout runners,
- canonical diagnostics and video rendering,
- run artifacts for Cosmos, Pi, Molmo, and IDM-as-policy probes,
- enough evidence to justify deeper, better-controlled experiments.

The next move is not to invent new plotting code. The next move is to use the existing contracts cleanly and decide which research question the next run answers.
