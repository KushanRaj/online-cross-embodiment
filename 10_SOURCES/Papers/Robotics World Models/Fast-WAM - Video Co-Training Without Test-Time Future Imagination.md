---
type: paper-note
topic: robotics-world-models
created: 2026-05-28
paper: "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
arxiv: "2603.16666"
local_pdf: "Fast-VAM.pdf"
local_text: "tmp/pdfs/fast_vam.txt"
---

# Fast-WAM - Video Co-Training Without Test-Time Future Imagination

## Core Takeaway

Fast-WAM asks whether WAMs need explicit future imagination at inference time, or whether most of the gain comes from video prediction as a training signal.

Their answer is:

```text
future-video prediction helps mainly by shaping representations during training;
explicit future-video denoising at inference is often not worth the latency.
```

The method keeps future-video latent prediction during training, but removes the future-video branch at inference.

## Base Model

Fast-WAM is not trained from scratch.

It uses pretrained **Wan2.2-5B**:

- video DiT backbone,
- text encoder / T5,
- video VAE.

On top of that, the paper adds a roughly **1B-parameter action expert DiT**, giving about **6B parameters** total.

The paper repeatedly says the model is trained **without embodied pretraining**. That means no extra large robot pretraining stage like AgiBot/DROID/Open-X. It does not mean no pretraining at all. The broad visual/video prior comes from Wan2.2, and the robot grounding comes from benchmark/task demonstrations.

## Architecture

During training, tokens are grouped into:

```text
clean first-frame latent tokens
noisy future-video latent tokens
noisy action tokens
language embeddings
```

The attention mask is the key design:

```text
future video tokens:
  attend to video branch + clean first frame

action tokens:
  attend to action branch + clean first frame
  cannot attend to future video tokens

clean first-frame tokens:
  do not attend to other tokens
```

This prevents future information from leaking into action prediction while still using future-video prediction as an auxiliary training objective.

At inference:

```text
current frame + language
-> video backbone single pass
-> latent world representation
-> action expert predicts action chunk
```

No future video tokens are instantiated. No future video is generated.

## Training Objective

Fast-WAM uses flow matching for both branches:

```text
action loss:
  denoise action chunk

video loss:
  denoise future video latents from pretrained VAE

total:
  L = L_action + lambda * L_video
```

The video objective is not a planning module. It is representation shaping.

## Variants

The paper compares four versions.

### Fast-WAM

The proposed version:

```text
training: action prediction + future video latent prediction
inference: action prediction only
action tokens cannot attend to future video tokens
```

### Fast-WAM-Joint

Joint WAM-style variant:

```text
future video tokens and action tokens are denoised together
attention is allowed between them
```

This resembles methods where future state and action are modeled inside one shared generative process.

### Fast-WAM-IDM

Inverse dynamics / imagine-then-act variant:

```text
current frame + language
-> generate future video latents
-> predict action conditioned on generated future representation
```

This is conceptually clean but creates a bottleneck: the action model depends on the imagined future being accurate and action-informative. Many actions can lead to similar visual futures, so inverse dynamics can be ambiguous.

### Fast-WAM Without Video Co-Training

Control ablation:

```text
same direct-action inference path
remove future-video prediction loss during training
```

This isolates whether the video objective itself matters.

## Results

LIBERO:

```text
Fast-WAM:                     97.6
Fast-WAM-Joint:               98.5
Fast-WAM-IDM:                 98.0
Fast-WAM without video loss:  93.5
pi0.5:                        96.9
```

RoboTwin:

```text
Fast-WAM:                     91.8
Fast-WAM-Joint:               90.6
Fast-WAM-IDM:                 91.3
Fast-WAM without video loss:  83.8
pi0.5:                        79.8
LingBot-VA:                   92.2
```

Real towel folding:

- base Fast-WAM is much lower latency than Joint/IDM,
- Fast-WAM-IDM has the strongest Fast-WAM-family success rate,
- removing video co-training collapses performance.

## Data Used

The paper fine-tunes on task/benchmark robot data:

- **LIBERO**: 4 suites, each 500 demonstrations over 10 tasks; total 2,000 demos; 20K training steps.
- **RoboTwin 2.0**: 2,500 clean demos + 25,000 randomized demos across 50+ bimanual tasks; 30K steps.
- **Real towel folding**: 60 hours of teleoperated demos on Galaxea R1 Lite; 30K steps.

Interpretation:

```text
broad video prior: Wan2.2
robot/task grounding: benchmark-specific demos
no large embodied pretraining stage
```

## Our Interpretation

The paper does not prove that future conditioning is useless. It proves something narrower and useful:

```text
for these benchmarks, video co-training gives most of the WAM benefit,
while explicit future generation at inference adds latency and only small accuracy gains.
```

LIBERO is close to saturation. The no-video-co-training version still gets 93.5, so robot demonstration data and the pretrained Wan2.2 backbone are doing a lot of the work there.

The bigger signal is:

```text
RoboTwin:        83.8 -> 91.8 with video co-training
real towel task: no-video co-training collapses
```

So the stronger takeaway is:

```text
on easy/saturated imitation benchmarks, robot data dominates;
on harder, randomized, long-horizon, or deformable tasks, video co-training matters more.
```

## Comparison To Cosmos And GE

Fast-WAM is not a planner. It has no value function and no default action-conditioned future rollout at inference.

Comparison:

```text
Fast-WAM:
  present observation -> action
  future-video prediction only trains the representation

Cosmos Policy:
  candidate action -> future state -> value
  supports best-of-N planning

GE-Base:
  observation + instruction -> future video

GE-Act:
  GE-Base latent features -> action

GE-Sim:
  candidate action -> future video
  closer to counterfactual planning
```

Fast-WAM is therefore a useful baseline for testing:

```text
world-model objective without planning
```

Cosmos and GE-Sim are more relevant when the question is:

```text
can a world model score candidate actions?
```

## Research Hooks

Good experiment questions:

- Does video co-training improve inverse dynamics recoverability in latent space?
- Does semantic latent progress improve more than pixel reconstruction quality?
- On our tasks, does removing future-video loss hurt only visual robustness or also action geometry?
- Can a Fast-WAM-style backbone be paired with a Cosmos-style value/progress head?
- Does explicit future rollout help only when the task requires long-horizon reasoning or counterfactual selection?

## Working Note

Fast-WAM is a caution against over-claiming planning from WAMs.

It shows that a model can get strong WAM-like scores while behaving at inference like a direct policy:

```text
present frame -> action chunk
```

That makes it important for our thesis. If we want to claim planning, we need an explicit mechanism for candidate generation and scoring, not merely a future-video auxiliary loss.
