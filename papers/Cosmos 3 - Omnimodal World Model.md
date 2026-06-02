---
type: model-note
topic: robotics-world-models
created: 2026-06-02
model: "NVIDIA Cosmos 3"
release_date: "2026-05-31"
sources:
  - "https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/"
  - "https://research.nvidia.com/labs/cosmos-lab/cosmos3/"
  - "https://www.nvidia.com/en-us/ai/cosmos/"
  - "https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID"
  - "https://github.com/NVIDIA/cosmos"
---

# Cosmos 3 - Omnimodal World Model

## Core Takeaway

Cosmos 3 is best read as a physical-AI world foundation model, not simply a robot policy.

It unifies:

- physical reasoning,
- world/video generation,
- action-conditioned simulation,
- action generation / policy post-training.

The policy use case is downstream of the world model. NVIDIA's own positioning says to post-train the generalized model on specialized camera, task, and embodiment data before using it as a policy backbone.

## Model Interface

Supported input modalities:

- text,
- image,
- video,
- audio / ambient sound,
- action trajectories.

Supported output modalities:

- text,
- image,
- video,
- audio,
- action trajectories.

Representative robotics modes:

```text
policy:
  image + instruction + embodiment config -> action chunk + rollout video

inverse dynamics:
  video + instruction + embodiment config -> inferred action chunk + rollout video

forward dynamics:
  image + action chunk + embodiment config -> future video
```

## Mixture Of Transformers

Cosmos 3 uses a Mixture-of-Transformers architecture with two complementary towers:

- an autoregressive reasoner for discrete token generation and physical reasoning,
- a diffusion generator for continuous multimodal generation.

The reasoner path can be run independently for text/vision reasoning. The generator path is not cleanly an isolated diffusion head in the released pipeline. For non-text generation, the diffusion process is conditioned by the model's internal reasoning / multimodal context path.

Practical interpretation:

```text
reasoner alone:
  text/image/video -> text

generator mode:
  text/image/video/audio/action -> image/video/audio/action
  diffusion denoising runs with model conditioning active
```

So it is misleading to think of Cosmos 3 as "just run the diffusion head." The generator is the denoising part, but its useful behavior depends on the shared omnimodal conditioning stack.

## Action Representation

Actions are numeric trajectory arrays, exposed as JSON in the serving interface.

Conceptually:

```text
action trajectory: (T, D)

T = horizon / frames / action chunk length
D = embodiment-specific action dimension
```

The model card lists compatible action dimensions such as:

- camera motion: 9D,
- autonomous vehicle: 9D,
- egocentric motion: 57D,
- single-arm robot variants: 10D,
- dual-arm robot: 20D,
- AgiBot / humanoid-style setting: 29D,
- UMI: 9D.

The action values are converted into model-ready encoded/tokenized representations through modality-specific preprocessors. This is analogous to tokenization, but for continuous action arrays it is better to say encoded rather than text-tokenized.

## Embodiment Config

The embodiment config tells the model how to interpret the action vector.

Typical serving fields:

```json
{
  "action_mode": "policy",
  "domain_name": "bridge_orig_lerobot",
  "raw_action_dim": 10,
  "action_chunk_size": 16
}
```

Important fields:

- `domain_name`: robot/agent/action convention, such as Bridge, DROID-like robot, AV, or camera pose.
- `raw_action_dim`: number of action values per timestep.
- `action_chunk_size`: how many future timesteps/actions to generate.
- `action_mode`: policy, inverse dynamics, or forward dynamics.

Without this config, the action vector is just numbers. The same numeric array could mean end-effector delta pose, joint targets, gripper state, steering/throttle/brake, camera pose, or dual-arm control depending on the embodiment.

## Training Data

Public model-card numbers for the Cosmos3-Nano-Policy-DROID card:

- total corpus scale: 1.3B data points,
- dataset entries: 393,
- collection period: 2024-2026,
- action generation data: 8M samples,
- image generation data: 767M samples,
- video generation data: 348M samples,
- audio generation data: 139M samples.

The corpus mixes internal, public, synthetic, sensor-conditioned, robotics, autonomous-driving, industrial, indoor, outdoor, and human-activity data.

This matters for interpretation: the base model plausibly learns broad physical and visual priors, but policy competence still depends on the target embodiment/action contract.

## Cross-Embodiment Status

Cosmos 3 exposes a multi-embodiment action interface, but the public release should not be treated as proof of broad zero-shot cross-embodiment control.

Evidence for bounded generality:

- The base model supports multiple action dimensions and domains.
- NVIDIA reports action-policy benchmarks on RoboLab and RoboArena.
- The released `Cosmos3-Nano-Policy-DROID` checkpoint is specifically described as taking DROID visual observations and instructions to generate DROID manipulation actions.
- NVIDIA's product language says to post-train on specialized camera and embodiment data.

Working interpretation:

```text
shared physical priors: likely
shared action/world interface: yes
drop-in DROID -> UR/UMI/AgiBot policy transfer: not established
embodiment-specific post-training/calibration: assume required
```

Reasons cross-embodiment remains hard:

- action dimensions differ,
- coordinate frames differ,
- gripper semantics differ,
- camera layouts differ,
- control frequency and action chunking differ,
- normalization and safety constraints differ,
- dataset coverage is uneven across embodiments.

## Difference From Cosmos 2 / 2.5

Cosmos 2 / 2.5 were closer to a family of separate models:

- Predict for world/video generation,
- Transfer for controlled generation,
- Reason for physical reasoning,
- policy/action workflows around post-training.

Cosmos 3 consolidates these into one omnimodal model family with native text, image, video, audio, and action pathways.

The key shift is:

```text
Cosmos 2 / 2.5:
  separate perception/reasoning/generation/action workflows

Cosmos 3:
  one omnimodal world model with reasoning, generation, simulation, and action support
```

## Research Hooks

Useful questions for this repo:

- Can Cosmos 3 forward dynamics provide cleaner counterfactuals than joint world-action models that co-generate action and future?
- Does action-conditioned video improve action recoverability in semantic latents?
- How much of the action prior transfers across embodiments after only adapter-level calibration?
- Can inverse-dynamics output from video be used as a residual correction signal for VLA failures?
- Where does Cosmos 3 fail under prompt, camera, object, and embodiment perturbations compared with Cosmos Policy, GE-Sim, DreamZero, and Fast-WAM?

## Working Interpretation

Cosmos 3 is an important reference because it moves NVIDIA's Cosmos line from a toolkit of separate world/reason/action components toward one omnimodal model interface.

For our experiments, the most important distinction is:

```text
policy output is useful,
but action-conditioned world simulation is the cleaner planning primitive.
```

If we use Cosmos 3, we should log the embodiment config, action dimensions, chunk size, control frequency, camera layout, and whether the model is being used as:

- direct policy,
- inverse dynamics model,
- forward dynamics simulator,
- synthetic data generator,
- or reasoning model.
