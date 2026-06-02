---
type: moc
topic: robotics-world-models
created: 2026-05-25
---

# Robotics World Models

This MOC tracks the research thread around planning, world-action models, semantic latents, VLA adaptation, and using learned world models as evaluators or supervisory signals.

## Source Notes

- [[Reconstruction or Semantics - Semantic Latents for Robotic World Models]]
- [[Dreamer 4 - Training Agents Inside Scalable World Models]]
- [[Cosmos Policy - Video Diffusion Planning and Value Models]]
- [[Cosmos 3 - Omnimodal World Model]]
- [[Do WAMs Generalize Better than VLAs - Robustness Benchmark Design]]
- [[Fast-WAM - Video Co-Training Without Test-Time Future Imagination]]
- [[Genie Envisioner - Cross Embodiment and EWMBench]]
- [[Joint Embedding vs Reconstruction - SSL Inductive Bias]]
- [[LingBot-VA - Causal Autoregressive Video-Action World Model]]
- [[VLA-JEPA - Latent World Model for VLA]]

## Core Questions

- How do we turn exploitation-style VLA or WAM policies into planning-capable agents?
- Which latent spaces preserve action-relevant structure for robotics?
- Can world-model or semantic latent signals replace expert demonstrations or sparse human success labels?
- How should candidate rollouts be scored: pixel reconstruction, semantic progress, inverse dynamics consistency, or task-success probes?
- When does planning in latent space outperform direct action generation?

## Working Thesis

Planning should happen in a latent space that preserves action geometry and task progress, not merely in pixel reconstruction space. Semantic latents from encoders like V-JEPA, DINO, and SigLIP may be better state spaces for candidate rollout scoring than VAE-style reconstruction latents.

The current model taxonomy matters for experiment design:

- **Direct policies** such as pi0.5 and MolmoAct2 map observation + language to action chunks.
- **Joint world-action models** such as DreamZero map observation + language to future video + action, but action is co-generated rather than an explicit intervention.
- **Video-co-trained direct policies** such as [[Fast-WAM - Video Co-Training Without Test-Time Future Imagination]] use future prediction during training but remove explicit future generation at inference.
- **Action-conditioned world simulators** such as GE-Sim map observation + language + candidate action to future video, making them structurally closer to counterfactual planning.
- **Value-scored planners** such as [[Cosmos Policy - Video Diffusion Planning and Value Models]] sample candidate actions, predict future states, and choose by predicted value.
- **Omnimodal world/action foundations** such as [[Cosmos 3 - Omnimodal World Model]] expose reasoning, generation, action-conditioned forward dynamics, inverse dynamics, and policy modes behind one interface, but still require embodiment-specific action contracts for robot control.

The open research question is whether DreamZero-style imagined futures can be made useful for planning without a clean action-conditioned simulator interface.

## Current WAM Taxonomy

```text
pi0.5 / MolmoAct2:
  observation + language -> action

DreamZero:
  observation + language -> future video + action
  joint imagination, not action-as-intervention

Fast-WAM:
  training: observation + language -> future video latent + action
  inference: observation + language -> action
  no value function, no future rollout at test time

LingBot-VA:
  observation/action history + language -> future visual latent
  future visual latent + history -> action
  autoregressive KV-cache, no value function

GE-Sim:
  observation + language + candidate action -> future video
  counterfactual simulator interface

Cosmos Policy:
  observation + candidate action -> future state -> value
  best-of-N planning interface

Cosmos 3:
  policy: image + instruction + embodiment config -> action chunk + rollout video
  inverse dynamics: video + instruction + embodiment config -> inferred action chunk
  forward dynamics: image + action chunk + embodiment config -> future video
```

Working lesson:

```text
future prediction as an auxiliary loss is not the same as planning.
planning needs candidate generation plus a scorer or value function.
causal future prediction is also not the same as planning unless futures are scored or used to choose among candidate interventions.
```

## Experiment Direction

1. Encode robot trajectories using candidate visual encoders.
2. Train inverse dynamics probes on pairs or sequences of latents.
3. Measure whether action information is recoverable from the latent transition.
4. Use a world model or WAM to generate candidate futures.
5. Score candidate action chunks by semantic latent progress, action recoverability, or task-success probes.
6. Compare against base VLA/WAM exploitation without planning.
7. Separate three evaluation levels: visual consistency, action/geometry consistency, and semantic substep progress.
8. Include a Fast-WAM-style ablation: keep action prediction but remove or weaken future-video co-training.
9. Include a Cosmos-style planning ablation: sample multiple action chunks, predict future/progress, and select by score.

Current concrete experiment note:

- [[Shared Semantic IDM - Prediction Reality Deviation]]
- [[Phase 3 Clean Rollout Benchmark]]

## Parked Future Experiments

- [[Future Experiment - Arm-to-Arm Embodiment Transfer via EEF Adapters]]

## Active Reports

- [[VLA-WAM Training Methodology Report]]
- [[Datasets and Environments Report]]
- [[Benchmarks and Robustness Report]]
- [[Public Review and Forum Insights Report]]
- [[Feasible Experiments and Compute Report]]
- [[Executive Summary - Context Faithfulness and Prediction Honesty]]

## Benchmark Structure

Use [[Do WAMs Generalize Better than VLAs - Robustness Benchmark Design]] as the experiment-organization template:

- clean baseline plus one perturbation branch at a time,
- perturb camera, robot initial state, language, lighting, background, image noise, and object layout separately,
- log model interface details before comparing scores: checkpoint, training/domain status, observation contract, action space, action chunk size, and control frequency,
- report success/progress by perturbation branch rather than only aggregate success.

## Related Concepts To Create

- Semantic latent
- Reconstruction latent
- Inductive bias
- Inverse dynamics model
- CEM planning in latent space
- World model as policy evaluator
- VLA exploitation vs planning
