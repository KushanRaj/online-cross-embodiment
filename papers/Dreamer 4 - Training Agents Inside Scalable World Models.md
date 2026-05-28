---
type: paper-note
topic: robotics-world-models
created: 2026-05-27
paper: "Training Agents Inside of Scalable World Models"
short_name: "Dreamer 4"
arxiv: "2509.24527"
local_pdf: "pdfs/Dreamer-4.pdf"
---

# Dreamer 4 - Training Agents Inside Scalable World Models

## Core Claim

Dreamer 4 trains an agent entirely inside a scalable action-conditioned world model. The paper shows that a high-capacity world model can learn accurate enough dynamics from offline Minecraft videos and actions to support imagination training, eventually obtaining diamonds without online environment interaction.

## Why It Matters

This paper gives a concrete recipe for using a world model not only as a video generator or evaluator, but as the training environment for an agent. For robotics, the key idea is that a learned world model can generate counterfactual futures under actions, enabling policy improvement when real interaction is unsafe, slow, or expensive.

## My Understanding

The paper is not just about a better Minecraft model. It is a scalable recipe for imagination training:

1. Train a tokenizer that compresses videos into continuous latents.
2. Train an action-conditioned dynamics model in latent space.
3. Add task, policy, reward, and value heads.
4. Train the policy inside imagined rollouts from the world model.

The major takeaway is that a reward model is needed for imagination training. Dreamer 4 learns a reward head from offline event labels, then uses the world model plus reward/value heads to improve the policy in imagination. This opens an interesting domain of checking: given an action, can the model generate the resulting outcome well enough that the resulting imagined rollout is useful for policy learning?

This relates to DreamZero and other world-action models. Dreamer 4 makes the causal action-to-video relation explicit through action-conditioned dynamics and imagination training. DreamZero learns video and action jointly and implicitly aligns them; it does not use the same explicit reward-model-plus-imagination-training loop.

## Method

Dreamer 4 has three phases:

1. **World model pretraining**
   - Train a causal tokenizer on videos.
   - Train an action-conditioned dynamics model on tokenized videos and optional actions.

2. **Agent finetuning**
   - Insert task tokens into the dynamics transformer.
   - Predict actions, rewards, and values from task/agent tokens.
   - Train behavior cloning and reward heads on offline data.

3. **Imagination training**
   - Roll out the policy inside the frozen world model.
   - Use the learned reward head to annotate imagined trajectories.
   - Train value with TD learning.
   - Train policy with PMPO and a KL to the behavioral cloning prior.

## Tokenizer Understanding

The tokenizer is more interesting than simply patchifying images. Each time step contains:

- image patch tokens,
- learned latent tokens.

The learned latent tokens do not replace image patches at the encoder input. They coexist with the patch tokens and act as bottleneck/query/memory slots that gather information from the patches and modalities. After the encoder, the representation is read out from the latent tokens through a low-dimensional projection and `tanh`. That compressed representation is what the dynamics model consumes and predicts.

For decoding, the compressed representation is projected back up and combined with learned decoder tokens to reconstruct image patches.

So the learnable latents are extra capacity and a compression interface. They let the tokenizer decide what information to keep for future prediction, rather than forcing the dynamics model to operate over raw frame patches.

## Efficient Transformer

The model is still recognizably a transformer, but the efficiency recipe matters:

- block-causal attention over time,
- separate space-only and time-only attention layers,
- temporal attention only every 4 layers,
- grouped-query attention to reduce KV cache size,
- register tokens,
- alternating short and long batch lengths,
- long context training to support long rollouts.

The point is not a radically new transformer block. The point is that the attention topology and compute recipe make long-context, real-time interactive world modeling practical.

## Shortcut Forcing

Dreamer 4 combines diffusion forcing with shortcut models.

Diffusion forcing gives different noise/signal levels to different sequence time steps, which allows the model to train on partially clean and partially noisy temporal contexts.

Shortcut models condition on both signal level and requested step size. This lets the model sample with very few denoising steps. Dreamer 4 uses 4 sampling steps per generated frame for interactive inference.

The paper also switches from velocity prediction to clean representation prediction (`x`-prediction), computes the loss in `x`-space, and applies a ramp loss weight that gives more weight to higher-signal examples. This is important for long rollouts because high-frequency prediction errors can accumulate.

## Results

- Dreamer 4 is the first agent in their setup to obtain diamonds in Minecraft from only offline data.
- It uses 2.5K hours of contractor gameplay and no online environment interaction.
- It substantially outperforms VPT offline baselines while using far less data than VPT's web-video setup.
- In human interaction tests inside the world model, Dreamer 4 completes 14/16 Minecraft tasks, while prior Minecraft world models fail on many object interactions.
- It learns useful action conditioning from relatively small amounts of paired action data, while extracting much of its world knowledge from unlabeled videos.

## Clarifications

The model is more than “a standard transformer with masks.” The transformer block is standard-ish, but the block-causal structure, factorized space/time attention, sparse temporal layers, GQA, and shortcut forcing objective are central to making it fast and stable enough for imagination training.

The tokenizer is reconstruction-trained, not semantic in the V-JEPA/SigLIP sense. It uses masked autoencoding with MSE and LPIPS. Any semantics it learns come indirectly from reconstructing and predicting interactive video, not from language alignment or a JEPA-style target.

DreamZero does not appear to use the same causal tokenizer idea. DreamZero uses a pretrained Wan video diffusion backbone and freezes the text encoder, image encoder, and VAE. Its causal/closed-loop structure mainly comes from autoregressive chunking, attention masks, KV caching, and replacing predicted frames with real observations during inference.

## Relation To Semantic World Model Paper

Dreamer 4 is closer to a reconstruction-latent world model than a semantic-latent world model. Its tokenizer is trained to reconstruct videos and provide latents for dynamics. The semantic world model paper argues that semantic latents like V-JEPA, Web-DINO, and SigLIP may be better for action recoverability and planning than reconstruction latents.

This raises an important question:

What happens if Dreamer 4's learned reconstruction tokenizer is replaced or augmented with a pretrained semantic encoder like V-JEPA?

Potential tradeoff:

- Dreamer 4 tokenizer: better reconstruction, interactivity, and domain-specific compression.
- V-JEPA/SigLIP latent: potentially better action recoverability, task progress, and planning geometry.

For robotics, the right answer may be a hybrid: use a reconstruction tokenizer for rollout generation, but score and plan in semantic latent space.

## Experiment Hooks

### 1. Inverse Dynamics Probe

Train an inverse dynamics model on latent transitions:

- input: `z_t`, `z_t+k` or `z_t:t+k`
- target: action chunk `a_t:t+k`

Compare latents:

- Dreamer-style reconstruction tokenizer latents,
- SigLIP latents,
- V-JEPA or DINO latents,
- DreamZero/WAM internal latents if accessible.

If action is more recoverable from a latent transition, that latent is a better candidate for planning.

### 2. Counterfactual Action Consistency

Given a world-action model:

1. Predict an action.
2. Generate or observe the resulting video.
3. Compare predicted action, actual action, predicted video, and actual video in a shared latent space.

Question: does the model's action latent align with the visual future it produces?

Dreamer 4 makes this explicit through action-conditioned dynamics. DreamZero does it implicitly through joint video-action prediction.

### 3. Planning With Semantic Scoring

Generate candidate action chunks from a VLA or WAM. Roll them forward in a world model. Score the resulting futures in semantic latent space rather than pixel space.

Possible scorers:

- distance to goal latent,
- inverse dynamics consistency,
- success-probe score,
- semantic text-image score,
- task-stage progress.

## Open Questions

- Can a semantic encoder replace Dreamer 4's reconstruction tokenizer without losing interactive rollout quality?
- Should world models generate in reconstruction latents but plan in semantic latents?
- Does DreamZero's joint video-action model have action-video correlation that can be measured with an inverse dynamics probe?
- Can counterfactual action consistency become a training or evaluation signal?
- How domain-specific is Dreamer 4's reward model? If the reward head is trained only from one domain's events, how transferable is imagination training?

## Links

- [[Robotics World Models]]
- [[Reconstruction or Semantics - Semantic Latents for Robotic World Models]]
