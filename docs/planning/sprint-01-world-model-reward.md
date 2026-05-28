# Exploration Sprint #01: World Model Prediction Error as Reward

---

## BEFORE YOU START

### The What-If

"What if the gap between what a world model predicts should happen and what actually happens is a sufficient reward signal for RL fine-tuning of VLAs — replacing expert demonstrations entirely?"

Sub-question: "Does a frozen VLA backbone already encode trajectory quality in its latent space, and can we see that structure before we even build anything?"

### Why this, why now?

Three converging observations from the paper survey:

1. **RLT** showed that a frozen VLA backbone + lightweight RL adapter works — but it still needs a human-provided reward (success/failure). The RL token PCA/t-SNE insight (from Server) suggests the backbone already encodes trajectory quality internally.
2. **LPS** showed that cosine similarity to expert latent states is a strong reward signal — but it requires expert demonstrations. If a world model can replace the expert states with predicted states, the expert data requirement disappears.
3. **DreamZero** showed that joint video+action training creates a model that "knows" what should happen visually. That visual knowledge could be the reward source.

The core tension: **expert data is expensive and doesn't scale. Video/behavioral data is cheap. Can we extract reward from the cheap data?**

### What prompted this?

- DreamZero Q4: video-only cross-embodiment data improved task performance → the video signal carries reward-relevant information
- LPS: cosine similarity to expert states works as reward → but requires experts
- RLT colleague's t-SNE insight: the VLA backbone already separates good from bad trajectories in latent space → reward information might already be there, waiting to be extracted
- The realization that all three papers point at the same thing from different angles

---

## EXPECTATIONS

### What do I expect to observe, and why?

**Day 1 (Probe):**
- I expect PCA/t-SNE of frozen VLA hidden states to show visible separation between successful and failed trajectories. Reasoning: the VLA was trained on demonstrations of successful behavior, so its internal representations should encode what "on track" looks like vs "off track." If the backbone can't distinguish these, it's a bad backbone.
- I expect the separation to be clearest in the later transformer layers (closer to action output) and weaker in early layers (closer to vision input).

**Days 2-3 (Reward signal):**
- I expect world model prediction error (predicted next state vs actual next state) to correlate with task success. Reasoning: on successful trajectories, the world model's predictions should be more accurate (the robot is doing something "normal" that the world model has seen in training data). On failed trajectories, the robot enters unusual states the world model hasn't seen, so prediction error spikes.
- I expect this correlation to be stronger for near-distribution tasks (similar to training) and weaker for far-distribution tasks.

### What would genuinely surprise me?

1. **The VLA latent space shows NO structure.** Good and bad trajectories are indistinguishable in latent space. This would mean the backbone is purely reactive — it doesn't build an internal model of trajectory quality. That would kill Branch 1a (latent self-feedback as reward) and suggest we need external scoring.

2. **World model prediction error is INVERSELY correlated with success.** The model predicts failed trajectories better than successful ones. This could happen if failures are more stereotyped (robot just stops) while successes are more diverse (many ways to succeed). This would mean prediction error can't be naively used as reward — you'd need to invert or transform it.

---

## FRUITFULNESS PRE-CHECK

### If this surprises me, then what?

**If the probe shows strong latent structure:** This opens a direct path to extracting reward from the backbone itself (no separate world model needed). The next sprint would be: train an RLT-style adapter using latent-space-derived reward instead of human reward. If that works, you've removed the human from the RL loop entirely.

**If prediction-error-as-reward works:** This means cheap video/behavioral data can replace expert demonstrations for RL fine-tuning. The downstream research question becomes: how does reward quality degrade as the world model gets worse? There's a spectrum from oracle (sim ground truth) → good learned world model → bad learned world model → video prediction only. Mapping that spectrum is a paper.

**If both work:** You have a new paradigm: pretrain VLA (behavioral cloning on cheap data) → extract reward from world model or backbone latents → RL fine-tune with lightweight adapter → deploy. No expert demos needed anywhere in the pipeline. That's a meaningful contribution to the field.

### Who would care?

- The VLA/robotics community working on data-efficient fine-tuning (RLT, TT-VLA, pi0.6/RECAP authors)
- Anyone working on reward-free RL for robotics
- The LossFunk residency, if this connects to a concrete project proposal

### Does this connect to our research directions?

Yes — task generalization via efficient adaptation. If world-model-as-reward works, it directly addresses the braindump's core question: can we achieve broad task generalization with limited expert data? The world model trained on diverse (non-expert) data provides the reward; the RL adapter provides task-specific fine-tuning.

---

## TIME BOX

| | |
|---|---|
| **Sprint duration** | 3 days |
| **Check-in date** | End of Day 1 (after probe results) |
| **Hard stop date** | End of Day 3 |

**Check-in decision:** If the probe shows no latent structure at all, pivot to testing external video prediction as reward (Branch 1c) instead of internal representations (Branch 1a). If the probe shows structure, proceed to reward signal testing using internal representations.

### What 'done' looks like

**Minimum viable output (Day 1):**
- 2D visualization (PCA or t-SNE) of frozen VLA hidden states for successful vs failed trajectories on at least 2 ManiSkill tasks
- A yes/no answer: does the backbone encode trajectory quality?

**Full sprint output (Day 3):**
- Correlation plot: world model prediction error vs task success across trajectories
- Comparison: prediction error from (a) sim ground truth (oracle), (b) simple learned forward model, (c) VLA internal representations
- A go/no-go decision on whether to graduate this to a full research question

---

## EXPERIMENT DESIGN

### Day 1: The Probe

**Setup:**
1. Get a pretrained VLA checkpoint running on ManiSkill 3 (OpenVLA or pi0.5-DROID — whichever has existing ManiSkill integration)
2. Run the VLA on 2-3 ManiSkill tasks (e.g., PickCube, StackCube, PegInsertionSide)
3. Collect ~100 trajectories per task (mix of successes and failures — may need to inject noise into actions to generate failures)
4. At each timestep, extract hidden states from the last 2-3 transformer layers

**Analysis:**
- PCA/t-SNE reduction to 2D
- Color by: success/failure, timestep within trajectory, task identity
- Look for: clustering, trajectory paths, separation boundaries
- Quantitative: linear probe accuracy (train a simple logistic regression on hidden states to predict success/failure)

**Decision gate:** If linear probe accuracy > 70% on held-out trajectories → proceed to Days 2-3. If < 60% → pivot.

### Days 2-3: Reward Signal Quality

**Setup:**
1. Using the same trajectory dataset from Day 1
2. Train three forward predictors:
   - (a) Oracle: ManiSkill ground truth state at t+1 (this is "free" — it's the simulator)
   - (b) Learned MLP: small network trained on trajectory states to predict s_{t+1} from s_t
   - (c) VLA latent predictor: predict next-timestep VLA hidden state from current hidden state

**Analysis:**
- For each predictor, compute prediction error at each timestep of each trajectory
- Plot cumulative prediction error over time for successful vs failed trajectories
- Compute rank correlation (Spearman) between cumulative prediction error and task success
- Compare the three predictors: which gives the cleanest reward signal?

**Decision gate:** If Spearman correlation > 0.5 for at least one predictor → GRADUATE to Research Question Sharpener. If < 0.3 for all → SHELVE or PIVOT.

---

## ENVIRONMENT & MODEL DECISIONS

**Simulation:** LIBERO via LeRobot. 130 Franka manipulation tasks, 4 suites (Spatial, Object, Goal, LIBERO-100). Most benchmarked VLA environment — nearly every major model reports numbers here. Install: `pip install -e ".[libero]"` + `export MUJOCO_GL=egl`. ~1-2 hours to first eval.

**Primary VLA checkpoint:** SmolVLA (450M params, ~2-4GB VRAM). Reasons: small enough for fast iteration, leaves VRAM headroom for hidden state extraction, reports 82-90% on LIBERO (natural mix of successes and failures without injecting noise). Available via LeRobot.

**Backup VLA checkpoint:** OpenVLA-7B (~16GB VRAM). Better documented internals (Llama-based architecture), more papers compare against it. Use if SmolVLA's architecture makes hidden state extraction difficult.

**Future real hardware:** SO-101 (when available). MolmoAct2 trained on SO-101 data. Validates sim findings on real hardware after the sprint.

## OPEN DESIGN QUESTIONS (resolve before starting)

1. **How to generate failed trajectories?** SmolVLA's 82-90% success rate means ~10-18% natural failures. May need more. Options: (a) inject Gaussian noise into actions, (b) run on out-of-distribution LIBERO tasks it wasn't fine-tuned on, (c) mix in random policy trajectories.

2. **Which LIBERO tasks?** Want 2-3 tasks with non-trivial success rate. LIBERO-Spatial and LIBERO-Object are standard. Pick tasks where SmolVLA success rate is 70-90% for a good success/failure mix.

3. **Hidden state extraction API:** RESOLVED. For OpenVLA: pass `output_hidden_states=True` to forward call → returns all 33 hidden states (embedding + 32 Llama layers), each `[batch, seq_len, 4096]`. For SmolVLA: use PyTorch forward hooks on `model.vlm.language_model.model.layers[i]`. Published baseline: Lu et al. (2025) probed all 33 OpenVLA layers on LIBERO with linear classifiers, got >90% accuracy predicting symbolic states. Gupta et al. (2025) found sparse semantic directions (speed, direction) via FFN activation projection in both OpenVLA and pi0. Use `model.named_modules()` to discover exact layer paths.

---

## BRANCHES NOT PURSUED (future sprints)

**Branch 2: DreamZero + post-hoc search.** Train DreamZero-style model on behavioral data, then at inference generate multiple rollouts and score with world model. This is a separate sprint — it's about adding search to an exploitation-trained model, not about reward signals.

**Branch 1b/1c: External video prediction as reward.** If internal representations (Branch 1a) don't work, the next sprint would test whether an external video model's predictions provide better reward signal. Requires a pretrained video model, which is a separate setup cost.

**Planning vs exploitation deep dive.** The broader question of when search/planning beats exploitation is important but too big for a sprint. Park it as a literature review task.

---

## AFTER THE SPRINT

*(Fill in after Day 3)*

### What did I actually observe?

### Were my expectations violated?

### THE DECISION

☐ **GRADUATE** → Pull out the Research Question Sharpener
☐ **SHELVE** → File notes, move on
☐ **PIVOT** → New what-if:
