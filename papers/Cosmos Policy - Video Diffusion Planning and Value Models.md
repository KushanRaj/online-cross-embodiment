---
type: paper-note
topic: robotics-world-models
created: 2026-05-28
paper: "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning"
arxiv: "2601.16163"
local_pdf: "pdfs/Cosmos-Policy.pdf"
repo_review: "docs/planning/cosmos-policy-review.md"
---

# Cosmos Policy - Video Diffusion Planning and Value Models

## Core Takeaway

Cosmos Policy is a strong planning reference because it turns a pretrained video diffusion model into three related objects:

- policy,
- action-conditioned world model,
- value model.

The central trick is to make actions, proprioception, future observations, and values look like video latent frames. Once they are on the same latent canvas, the video diffusion model can denoise different parts of the trajectory under different conditioning masks.

The important interface is:

```text
current state + candidate action chunk -> predicted future state -> predicted value
```

That is the missing piece in many WAM/VLA papers. A direct VLA can act, and a WAM can sometimes imagine, but Cosmos gives an inference-time scorer for candidate actions.

## Decision Transformer Analogy

The paper has a Decision Transformer flavor:

- trajectory-like data,
- returns or values inside the sequence,
- supervised learning over logged experience,
- action choice tied to future return.

But Cosmos does not condition actions on a desired return in the classic Decision Transformer way. It samples candidate actions, predicts futures and values, then reranks by predicted value.

Better shorthand:

```text
behavior cloning + action-conditioned world model + Monte Carlo value labels + best-of-N planning
```

It is not SAC/PPO-style online actor-critic RL.

## Latent Injection

Cosmos-Predict2 uses a video tokenizer. RGB video becomes continuous latent frames. Cosmos Policy injects non-image modalities by reserving latent-frame slots:

```text
action chunk -> normalize -> duplicate across latent H x W x C
proprio      -> normalize -> duplicate across latent H x W x C
value        -> normalize -> duplicate across latent H x W x C
```

There is no separate action encoder or value head in the clean version of the architecture. The low-dimensional vectors are packed into latent-frame tensors.

The blank placeholder images are slot allocators. They are not semantically important. They let the normal video tokenizer produce a fixed latent layout, after which selected latent slots are overwritten by action/proprio/value tensors.

## Sequence Structure

The practical context is short:

```text
history: one current observation at time t
future: one future observation at time t + K
```

For a multi-camera robot, one logical observation may contain several image views, but the model is not consuming a long history window inside one forward pass.

The learned transition is:

```text
s, a -> s'
```

That is why it is useful for counterfactual planning. It can compare what happens under different candidate action chunks from the same current state.

## Training Objectives

The latent sequence is:

```text
s, a, s', V(s')
```

Training uses different conditional masks:

```text
policy objective:
  clean: s
  denoise/predict: a, s', V(s')

world-model objective:
  clean: s, a
  denoise/predict: s', V(s')

value objective:
  clean: s, a, s'
  denoise/predict: V(s')
```

Planning refinement shifts weight toward world/value modeling:

```text
45% world
45% value
10% policy
```

The 10% policy objective is best read as grounding / anti-forgetting. The actual planning gain mostly comes from better future-state and value prediction, not from a directly optimized actor.

## Planning Loop

Cosmos has two modes.

Direct policy:

```text
condition on s
generate action chunk
execute action
```

Planning:

```text
1. Use the base policy checkpoint to sample candidate action chunks.
2. Use the refined planning checkpoint to predict future states for each candidate.
3. Predict values for those futures.
4. Execute the action with the highest predicted value.
```

This dual-checkpoint setup makes sense because planning refinement mostly improves the judge, not the proposal policy.

## Data And Compute

Important pattern:

```text
train/fine-tune per target platform
collect/evaluate rollout outcomes on that platform
plan on that same platform
```

The paper is not showing pure cross-embodiment transfer of a value function.

Main settings:

- **LIBERO**: Franka, 4 suites x 500 demos = 2,000 demos, 40K steps, 64 H100s for about 48h.
- **RoboCasa**: Franka kitchen manipulation, 24 tasks x 50 human demos = 1,200 demos, 45K steps, 32 H100s for about 48h.
- **ALOHA**: two ViperX arms, 185 real demos across 4 tasks, 50K steps, 8 H100s for about 48h.
- **ALOHA planning refinement**: 648 rollout episodes, including prior policy rollouts and additional Cosmos rollouts.

## Generalization Interpretation

The generalization is meaningful but bounded:

- same robot or target platform,
- same action space,
- same broad task family,
- new objects, layouts, visual styles, and initial conditions.

It does not prove:

- cross-embodiment value transfer,
- new control regimes,
- unrelated task-family transfer,
- sim-trained value transfer directly to real robot.

This is still useful for our work because it gives a clean recipe for adding planning to WAMs without pretending the value model is universally portable.

## Inference Cost

Direct action inference is already heavier than standard VLA action heads, and planning is much heavier.

The planning path uses best-of-N search, multiple future predictions, and value ensembles. This is why the paper is conceptually important but not an obvious low-latency deployment recipe.

## Research Hooks

Most useful experiments for us:

- compare predicted future state against actual state after executing the same action,
- evaluate counterfactual futures for different candidate actions,
- train a value/progress probe over semantic latents,
- compare Cosmos-style action-conditioned transitions with DreamZero/GE/Fast-WAM style latent futures,
- test whether semantic latents preserve action progress better than reconstruction latents for value prediction.

## Working Interpretation

Cosmos Policy is the cleanest current example of:

```text
video prior -> action-conditioned world model -> value scorer -> planning
```

It is especially relevant because our core problem is not merely generating good actions. It is adding a planning/evaluation interface to models that otherwise behave like exploitation policies.
