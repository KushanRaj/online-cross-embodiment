# Cosmos Policy Review

Local paper: `Cosmos-Policy.pdf`

Extracted text: `tmp/pdfs/cosmos_policy.txt`

External page: https://arxiv.org/abs/2601.16163

OpenReview: https://openreview.net/forum?id=wPEIStHxYH

## Core Takeaway

Cosmos Policy is one of the most relevant papers for our planning thread because it turns a video diffusion model into a unified policy, world model, and value model. The central move is simple: make actions, proprioception, future observations, and value estimates look like video latent frames, then train the pretrained video model to denoise different parts of the sequence under different conditioning masks.

The important idea is not just that it gets strong LIBERO or RoboCasa numbers. The important idea is that video priors can be used to learn an action-conditioned transition model:

```text
current state + candidate action chunk -> predicted future state -> predicted value
```

That gives the model a planning interface. Other VLA and WAM papers often have action generation and sometimes future prediction, but they usually do not have a value function that can score candidate futures at inference time. Cosmos Policy makes the missing critic explicit.

## Decision Transformer Analogy

There is a useful analogy to Decision Transformers, but it should not be overstated.

Similarities:

- training is supervised over trajectory-like data,
- returns or values are part of the sequence,
- policy behavior is learned from logged experience rather than actor-critic updates,
- planning or selection is ultimately tied to predicted return.

Difference:

- Decision Transformer conditions the action model on a desired return.
- Cosmos Policy predicts a future state and value, then uses that value to rerank sampled candidate actions.

So Cosmos is closer to:

```text
behavior cloning + action-conditioned world model + Monte Carlo value labels + best-of-N planning
```

It is not SAC/PPO-style reinforcement learning. The policy is not directly optimized by a Bellman actor loss. Planning improves because sampled actions are scored by a learned future-state/value model.

## Latent Injection

Cosmos-Predict2 is a latent video diffusion model. The Wan/Cosmos video tokenizer maps an RGB video sequence into continuous latent frames:

```text
(1 + T) x H x W x 3 -> (1 + T/4) x H/8 x W/8 x 16
```

The first frame is special: it is encoded without temporal compression. Later frames are compressed in groups of four.

Cosmos Policy uses blank RGB frames as slot allocators. The sequence is first sent through the normal video tokenizer, then selected placeholder latent frames are overwritten with normalized duplicated vectors:

```text
action chunk -> flatten -> normalize to [-1, 1] -> duplicate across H' x W' x C'
proprio      -> normalize -> duplicate across H' x W' x C'
value        -> normalize scalar -> duplicate across H' x W' x C'
```

There is no separate action encoder, proprio encoder, or value head. Non-image modalities are packed directly into latent-frame tensors. At inference, the generated action/value latent is averaged back down and un-normalized.

The blank frames are not semantically important. They exist because the tokenizer has a fixed first-frame and group-of-four compression structure. The blank first frame absorbs the special first-frame slot so that current and future observations can live in similarly structured latent positions. Repeating each static image four times creates one latent frame per logical camera/modal slot.

## Sequence And Context

For a 3-camera robot setup, the latent sequence is:

```text
blank special placeholder
current proprio
current wrist image
current third-person image 1
current third-person image 2
action chunk
future proprio
future wrist image
future third-person image 1
future third-person image 2
future value
```

The temporal context is short:

```text
history: one current observation at time t
future: one future observation at time t + K
```

There is no history window and no multi-step future video rollout inside one forward pass. Each observation can be multi-view, but it is still only one current timestep and one future timestep.

The paper's "causality" is action-conditioned transition modeling, not causal discovery. It learns:

```text
s, a -> s'
```

This is still valuable because it gives a natural counterfactual interface: sample different action chunks from the same current state and compare their predicted futures.

## Training Objectives

The fixed latent sequence is:

```text
s, a, s', V(s')
```

The batch split controls the conditioning mask:

```text
Policy objective, 50%:
clean context: s
noisy targets: a, s', V(s')
learns: p(a, s', V(s') | s)

World-model objective, 25%:
clean context: s, a
noisy targets: s', V(s')
learns: p(s', V(s') | s, a)

Value objective, 25%:
clean context: s, a, s'
noisy target: V(s')
learns: p(V(s') | s, a, s')
```

So the split is not "only action is noisy" or "only value is noisy." It is different conditional modeling over the same latent canvas.

During planning refinement, the model is fine-tuned on rollout data with heavier world/value weighting:

```text
45% world model
45% value model
10% policy objective
```

The 10% policy objective is best interpreted as a grounding or anti-forgetting term, not true reward-improving policy optimization. The rollout policy loss can imitate both successful and failed actions, so the improvement comes mainly from better future-state/value prediction.

## Planning Loop

Cosmos Policy has two deployment modes.

Direct policy:

```text
condition on s
generate a, s', V(s') in parallel
execute a
discard s' and V(s')
```

Model-based planning:

```text
1. Use the base policy checkpoint to sample N candidate action chunks.
2. Use the refined planning checkpoint to predict future states for each action.
3. Predict values for the generated future states.
4. Execute the action whose predicted future has the highest value.
```

The paper uses dual deployment:

```text
base checkpoint    -> policy model for candidate action sampling
refined checkpoint -> planning model for world/value prediction
```

This makes sense once planning refinement is understood as critic/world-model refinement. The base policy remains the behavior/action proposal model, while the rollout-trained checkpoint becomes a better judge of candidate futures.

## Value Function And Rollout Data

The value function is trained from Monte Carlo returns. In sparse reward settings:

```text
intermediate reward = 0
terminal reward in [0, 1]
each transition gets the observed discounted terminal outcome
```

For base training, the value signal comes from demonstrations and failed replay outcomes where available. For planning refinement, the important data is newly collected policy rollout data:

```text
run policy in the target environment
record trajectory
record success/fail or fractional task score
offline fine-tune world/value model
```

This is not online RL. It is also not only static imitation data. It is:

```text
offline imitation -> policy rollout collection -> offline supervised world/value refinement -> inference-time search
```

## Datasets, Compute, And Performance

| Setting | Embodiment | Data used | Training compute | Main result |
| --- | --- | --- | --- | --- |
| LIBERO | Franka arm | 4 suites x 500 demos = 2,000 demos. Successful demos for policy; unfiltered demos for world/value. | 40K gradient steps, 64 H100s, global batch 1920, 48 hours. Full model fine-tuning. Action chunk 16. | 98.5 average success: Spatial 98.1, Object 100.0, Goal 98.2, Long 97.6. |
| RoboCasa | Franka arm | 24 kitchen tasks x 50 human demos = 1,200 demos. They do not use the additional 1,000 MimicGen demos per task. | 45K gradient steps, 32 H100s, global batch 800, 48 hours. Full model fine-tuning. Action chunk 32, execute 16. | 67.1 average success across 24 tasks. 1-step denoising gives 66.4 with 0.16s latency per chunk. |
| ALOHA | Two ViperX 300 S arms | 185 real robot demos across 4 bimanual tasks. | 50K gradient steps, 8 H100s, global batch 200, 48 hours. Full model fine-tuning. Action chunk 50 at 25 Hz. | 93.6 average score across the four tasks; planning gives a 12.5-point average score increase on the two hardest tasks. |
| ALOHA planning refinement | Same ALOHA setup | 648 rollout episodes: 505 prior policy evaluation rollouts plus 143 additional Cosmos Policy rollouts for ziploc. | Offline fine-tuning of a planning checkpoint with 90% world/value batches and 10% policy batches. | Best-of-8 planning with future/value ensembles improves the difficult ALOHA tasks. |

Important caveat: these are per-target-platform fine-tuning setups. Cosmos Policy does not show one model transferring from Franka to ALOHA without target robot data.

## Inference Cost

Direct policy latency:

```text
5 denoising steps, LIBERO/RoboCasa: 0.61s per action chunk on 1 H100
10 denoising steps, ALOHA:          0.95s per action chunk on 1 H100
1 denoising step, RoboCasa:         0.16s per action chunk on 1 H100
```

Planning latency is much heavier:

```text
best-of-8 search
8 parallel H100s
10 denoising steps for action proposals
3 future-state predictions per action
5 value predictions per future state
4.9s total search time
```

This is one reason the paper did not become an obvious deployment recipe despite the strong idea.

## Generalization: What It Does And Does Not Show

Cosmos Policy shows strong within-embodiment, within-task-family generalization:

```text
same robot
same action space
same broad task family
new objects
new layouts
new visual styles
new initial conditions
```

It does not show:

```text
cross-embodiment transfer
new action spaces
new control regimes
completely unrelated task classes
sim-trained value transfer to real robot
```

For RoboCasa, the generalization is meaningful but bounded. The task family is still kitchen manipulation with a Franka arm. The policy is evaluated on unseen object instances and unseen scene styles, but the robot, action space, simulator, and broad task distribution remain fixed.

For ALOHA, OOD means variants such as unseen object arrangements, unseen shirts, an unseen bowl, or an unseen ziploc bag. These are useful real-robot robustness tests, but they are not open-world capability tests.

## Robustness Paper Connection

The WAM-vs-VLA robustness paper classifies Cosmos Policy as a WAM and finds that WAMs are generally robust to some perturbation classes, especially noise, lighting, and layout. On LIBERO-Plus:

```text
pi0.5 total:          85.7
Cosmos-Policy total:  82.2
GE-Act total:         80.3
VLA-JEPA total:       77.9
```

Cosmos is strong, but not universally best. The robustness paper says Cosmos future predictions can denoise corrupted robot-arm motion under noise/light perturbations. However, WAMs still struggle when perturbations change geometry, camera viewpoint, robot initial state, or background patterns in ways that distort predicted futures.

The broader conclusion is:

```text
video priors help robustness,
but training-data diversity remains decisive.
```

pi0.5 can beat Cosmos overall under LIBERO-Plus despite lacking the same explicit video-world-model structure, likely because its training data is more diverse.

## OpenReview Takeaways

Cosmos Policy was accepted as an ICLR 2026 poster. Reviewers were mostly positive after rebuttal. They liked the simplicity and strength of treating actions, future states, and values as latent frames in one video diffusion model.

Main concerns:

- latent injection mechanics were initially unclear,
- inference latency was high,
- OOD evidence needed stronger support,
- long-horizon evidence was not fully convincing,
- transfer to other video backbones such as CogVideoX or Wan was plausible but untested,
- implementation/reproducibility details needed expansion.

The authors addressed these by adding:

- detailed latent injection appendix and Figure 8,
- RoboCasa generalization results,
- extra ablations on auxiliary future/value objectives,
- specific training loss numbers,
- 1-step denoising latency result,
- reproducibility/code/checkpoint commitment.

One thing reviewers did not really force them to justify is the dual-checkpoint planning design or the 10% policy loss during planning refinement. The paper's result supports the planning system as a whole, but it does not cleanly ablate:

```text
base actor + refined planner
vs refined actor + refined planner
vs no 10% policy loss during refinement
```

After clarifying the data flow, this is less concerning than it first seemed. The refined checkpoint is mostly a critic/world-model checkpoint, and the base checkpoint is the stable action proposal distribution.

## Research Hooks For Us

### 1. Predicted-vs-actual future state evaluation

Cosmos suggests a direct evaluation suite:

```text
given s and real action a
predict s'
compare predicted s' to actual observed s'
```

This could be run across:

- GE-Act / GE-Sim,
- Fast-WAM,
- Cosmos Policy if checkpoints run locally,
- DreamZero if feasible,
- our own lightweight world-action setup.

Metrics could include:

- image-space similarity,
- latent-space similarity using SigLIP, DINO, V-JEPA, or Cosmos latents,
- object/contact-state correctness,
- action-conditioned consistency,
- inverse-dynamics recoverability from predicted vs real future states,
- task-progress prediction accuracy.

This would make the "world model as causal predictor" question concrete.

### 2. Counterfactual generation

For a fixed current state:

```text
sample candidate actions a1, a2, ..., an
generate predicted futures s'1, s'2, ..., s'n
score whether the futures reflect plausible action consequences
```

In sim, we can execute selected actions and compare real outcomes to generated outcomes. This gives a direct test of whether the model actually learned action-state causality or merely generates visually plausible futures.

### 3. Planning without a learned value function

Cosmos makes the missing piece in our setup obvious: without a value model or reward model, future generation alone is not enough for planning.

Possible substitutes:

- train a value model from rollout success/failure,
- train a task-progress classifier,
- use a VLM as a coarse reward judge,
- use inverse dynamics consistency as a proxy score,
- use LPS-style latent distance to expert/favorable future states.

### 4. Compare semantic vs reconstruction latents for planning

This connects back to the reconstruction-vs-semantics paper and VLA-JEPA:

```text
Do semantic latents better preserve task progress and action geometry?
Do reconstruction latents better preserve physical contact and fine state details?
Which latent is better for value prediction and counterfactual scoring?
```

Cosmos uses reconstruction/video-generation latents. VLA-JEPA uses semantic V-JEPA latents. A useful experiment is to evaluate the same predicted futures under both kinds of latent metrics.

## Working Interpretation

Cosmos Policy is valuable because it gives a concrete recipe for making planning possible in a video-prior policy:

```text
1. Use video pretraining for dynamics priors.
2. Inject actions, proprioception, and values into the latent video sequence.
3. Train policy, world model, and value prediction through conditional denoising masks.
4. Collect rollout outcomes on the target platform.
5. Refine the world/value model.
6. Use best-of-N action search at inference.
```

The strongest insight for our project is that planning needs a scoring mechanism. World models by themselves create possible futures; value functions or reward models decide which future is useful. Cosmos shows one way to get that value function without building a separate critic architecture, by making value another latent frame in the video diffusion sequence.

The main limitation is that its generalization is narrower than the headline suggests. It is strong within a target platform and task family, especially under visual perturbations, but it does not solve cross-embodiment or zero-robot-data transfer. For our goals, Cosmos is best treated as an inspiration for causal transition/value training, not as evidence that video-prior policies automatically generalize across embodiments.
