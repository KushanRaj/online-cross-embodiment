---
type: research-report
topic: robotics-world-models
status: data-centric-audit
created: 2026-05-28
updated: 2026-05-28
---

# VLA-WAM Training Methodology Report

## Readout

This report is deliberately data-first. The earlier architecture-centric summary was too coarse for the question we actually care about: whether a model's reasoning, world model, and action interface are honest to the observation/language/action context.

The important distinction is not just "VLA versus WAM". It is:

1. What high-quality robot action data did the model see?
2. Did it see only successful expert demonstrations, or also failures, corrections, interventions, and autonomous rollouts?
3. Did raw video come with labels such as language, VQA, boxes, pointing, depth, progress, success, or pseudo-actions?
4. Was future prediction trained as pixel reconstruction, semantic latent prediction, action-conditioned simulation, or joint video-action modeling?
5. Is the final policy direct, latent-planning-based, value-scored, or just representation-regularized by world-model training?

The strongest recurring pattern is simple: broad robot data plus clean action conventions beats clever architecture until robustness or planning is measured carefully. Failure data and action-counterfactual training are rare. Most models are trained on successful demonstrations, then evaluated as if they should recover from mistakes.

## Audit Schema

For every paper, track these fields:

- **Robot trajectory data:** hours, episodes, demos, timesteps, or trajectories with actions.
- **Data quality:** successful-only, filtered, expert, teleop, suboptimal, play, failed, autonomous, intervention, or unknown.
- **Non-robot data:** web images, VQA, captions, human videos, egocentric videos, simulation, generated/neural trajectories.
- **Extra labels:** language, subtask captions, boxes, pointing, depth, success/failure, value/progress, pseudo-actions, object lists.
- **Action/observation contract:** camera count, proprio, action dimension, chunk size, control frequency, end-effector versus joint.
- **Trained modules/objective:** CE, flow matching, diffusion, JEPA latent loss, value loss, action decoder, frozen encoders.
- **Compute and model size:** when reported.
- **Unknowns:** missing quantities that should not be silently assumed.

## Cross-Paper Data Matrix

| Family | Methods | Robot data pattern | Failure/rollout data | Non-robot or raw-video data | Main lesson |
| --- | --- | --- | --- | --- | --- |
| Direct VLAs | pi0, pi0.5, pi0.6, pi0.7, OpenVLA, Octo, MolmoAct2 | Large robot demo mixtures, often filtered/curated. | Explicit mainly in pi0.6/RECAP, RLT, and pi0.7 metadata/rollout mix. | Web VQA/captions/boxes in PI/Molmo; little/no extra web data in OpenVLA/Octo policy training. | Direct policies are only as honest as their observation/action contract and data distribution. |
| Action experts on VLMs | MolmoAct2, CLAP, GR00T, RLT | Robot action data trains a continuous expert or small actor. | RLT and GR00T use online/neural/sim rollouts; CLAP mostly curated expert plus pseudo human labels. | Rich labels matter: depth, object boxes, pseudo-actions, subtask/action tokens. | A separate action expert is not enough; the bottleneck is whether the conditioning representation preserves task/action state. |
| Joint WAMs | DreamZero, LingBot-VA, Fast-WAM | Robot video plus actions; often teleop/demo. | Mostly not explicit, except LingBot has mixed sources and Fast-WAM reports no failure labels. | Web-scale video priors inherited from Wan-style backbones, usually not quantified. | Joint future/action generation gives diagnostic futures but not clean action counterfactuals. |
| Action-conditioned WMs | Cosmos Policy, GE-Sim, Reconstruction or Semantics | Action-labeled robot trajectories, demonstrations, rollouts, or BridgeV2-style datasets. | Cosmos is strongest: unfiltered failures feed WM/value; rollout outcomes feed refinement. GE-Sim explicitly augments with failures/suboptimal control. | Mostly robot-domain; GE explicitly says no internet/sim data in GE training. | Best fit for "what happens if I take action A?" and for planning experiments. |
| Semantic latent / planning | VLA-JEPA, LPS, IVLR, Reconstruction or Semantics, Joint Embedding vs Reconstruction | Often smaller robot sets plus human video or generated traces. | Usually not on-policy RL; LPS uses synthetic/value steering; IVLR uses pseudo-trace supervision. | Human videos, JEPA/DINO/SigLIP-style latents, pseudo captions/traces. | Supports measuring semantic progress rather than pixel reconstruction. |

## Data Scale Overview

This table is sorted roughly by reported data quantity plus variation. It is not a perfect numeric ranking because papers report incompatible units: hours, trajectories, episodes, frames, tokens, demos, and rollouts.

| Rank | Model | Reported data scale | Variation | Main caveat |
| --- | --- | --- | --- | --- |
| 1 | LingBot-VA | About 16k h robot manipulation; 1.4T training tokens. | AgiBot, RoboMind, InternData sim, OXE, UMI, RoboCOIN, internal demos. | Huge robot-hour claim, but source split and failure composition are unclear. |
| 2 | pi0.6 / RECAP | "Tens of thousands of hours" pretraining plus hundreds of online rollouts/corrections per task. | Many robots/tasks plus online RL/correction data. | Exact pretraining breakdown is not reported. |
| 3 | pi0 / PI base | More than 10k h robot data; 903M proprietary timesteps; 7 robot configs; 68 tasks. | Single-arm, dual-arm, OXE/Bridge/DROID. | Failure mix is not quantified. |
| 4 | GR00T N1 | 8,375.7h total: 3,288.8h robot, 2,517h human, 1,742.6h sim, 827.3h neural. | Robot + human egocentric + sim + generated video. | Very broad, but humanoid-oriented and some labels are pseudo-actions. |
| 5 | GE / Genie Envisioner | About 2,967h AgiBot; about 1M dual-arm episodes. | Real robot dual-arm, multi-view, language. | Mostly one robot-data ecosystem; failure mix unclear. |
| 6 | CLAP | About 1,500h curated AgiBot + about 50h Astribot + 90h Ego4D. | Robot + human video + latent/pseudo-action training. | Smaller but targeted and multi-stage. |
| 7 | MolmoAct2 | YAM 720h, SO 184h, DROID 74,604 successful episodes, plus 12.5M ER/multimodal samples. | Robot + VQA + pointing + detection + depth + relabeling. | DROID hours are not directly reported. |
| 8 | DreamZero | About 500h AgiBot + task post-training 12-40h + small video-only transfer data. | Robot video-action plus Wan video prior. | Strong WAM, but less robot-data scale than top tier. |
| 9 | OpenVLA | About 970k OXE robot trajectories. | Many Open X embodiments/datasets. | Hours unknown; curated to specific control/camera assumptions. |
| 10 | Octo | About 800k OXE robot episodes. | 25 curated robot datasets. | Hours unknown; smaller model, less annotation richness. |
| 11 | VLA-JEPA | 76k DROID demos + 220k SSV2 human videos + about 2k LIBERO demos. | Human video + robot demos + latent prediction. | Not huge robot scale, but useful semantic-latent setup. |
| 12 | Reconstruction vs Semantics | About 60k BridgeV2 demos + 30.5k SOAR success/failure episodes. | Action-conditioned WM benchmark data. | Evaluation/latent study, not foundation policy scale. |
| 13 | Fast-WAM | LIBERO 2k demos, RoboTwin 27.5k demos, 60h real towel folding. | Sim + real task data. | No broad embodied pretraining. |
| 14 | Cosmos Policy | LIBERO 2k demos, RoboCasa 50/task, ALOHA 185 demos, 648 rollouts. | Small target data + rollout/value data. | Data-efficient planning paper, not broad pretraining. |
| 15 | LPS | 30-100 target demos, 2k OXE pretrain episodes, 300 sim demos, human play. | Low-data target + cross-embodiment WM pretraining. | Compute not reported; policy remains target-specific. |
| 16 | IVLR | LIBERO/SimplerEnv demos; exact count not reported. | Pseudo-traces from segmentation + VLM captions. | Trace method is interesting; data budget underreported. |
| 17 | RLT | 1-10h demos/task + 400-1000 online episodes/task. | Real online rollouts, failures, interventions. | Adapter method, not foundation pretraining. |

### Data Feature Matrix

| Model | Failure / rollout data | VQA / captions / semantic labels | Human video | Sim data | Neural generated data | Action labels |
| --- | --- | --- | --- | --- | --- | --- |
| LingBot-VA | Unknown / limited. | Task language. | Yes, UMI-style. | Yes. | No explicit. | Yes. |
| pi0.6 / RECAP | Yes, strong. | Subtasks, metadata. | Some family data. | Possibly via broader mix. | No. | Yes. |
| pi0 | Some lower-quality/corrections, not quantified. | Segment/task labels. | No clear. | Open data mix. | No. | Yes. |
| pi0.7 | Yes, mistakes/interventions/rollouts. | Subtasks, metadata, VQA, localization. | Yes. | Likely/mixed. | Subgoal image WM. | Yes. |
| GR00T | Partial; success/validity flags, filtered sim/neural. | Object boxes, task labels. | Yes. | Yes. | Yes, 827h. | Yes + pseudo-actions. |
| GE | GE-Sim includes failure/suboptimal. | Language, EWMBench captions for eval. | No. | No. | No. | Yes. |
| CLAP | Mostly curated; LIBERO failures filtered. | Subtask/action tokens. | Yes, Ego4D. | No main. | No. | Yes + pseudo-actions. |
| MolmoAct2 | Mostly successful/filtered. | Very strong: VQA, pointing, detection, tracking, depth. | Not central. | LIBERO fine-tune. | No. | Yes. |
| DreamZero | Idle filtered; failures unclear. | Language only mostly. | Small human video transfer. | No. | No. | Yes, except video-only transfer. |
| OpenVLA | Unknown. | Language only mostly. | No. | No. | No. | Yes. |
| Octo | Unknown. | Language / goal image. | No. | No. | No. | Yes. |
| VLA-JEPA | No explicit failures. | Language, latent targets. | Yes, SSV2. | LIBERO/SimplerEnv. | No. | Yes for robot. |
| Reconstruction vs Semantics | SOAR success/failure probes. | Language, success probe labels. | No. | No. | No. | Yes. |
| Fast-WAM | Not explicit. | Language. | No. | Yes. | No. | Yes. |
| Cosmos Policy | Yes, very strong. | Value/success labels. | No. | LIBERO/RoboCasa/ALOHA. | No. | Yes. |
| LPS | Pretrain can be messy; target success-only. | Value/simulated latent reward. | Yes, play. | Yes. | No. | Target yes; pretrain optical-flow proxy. |
| IVLR | Unknown. | Very strong pseudo-traces/captions. | No. | LIBERO/SimplerEnv. | No. | Yes. |
| RLT | Yes, online failures/interventions. | Human success/failure reward. | No. | No. | No. | Yes + VLA reference action. |

### Data Scale Readout

If sorted by **reported robot-action hours**, the top tier is probably LingBot-VA, the PI family, GR00T, GE, then CLAP/MolmoAct2/DreamZero. If sorted by **data-mixture design**, GR00T becomes more impressive: it explicitly combines robot, human, simulation, neural video, boxes, and pseudo-actions. MolmoAct2 is less raw-hour-heavy but much richer in VQA/pointing/detection/depth/relabeling supervision.

## Direct VLA And Action-Policy Methods

### pi0

**Parameter scale:** about 3.3B total: PaliGemma 3B plus about 300M action expert.

**Training phases**

1. Internet-scale VLM initialization through PaliGemma.
2. Robot pretraining over more than 10,000 robot hours: 903M proprietary timesteps, split into 106M single-arm and 797M dual-arm timesteps, across 7 robot configurations and 68 tasks.
3. Open robot data is mixed in at about 9.1% mixture weight: OXE/Bridge/DROID.
4. Task post-training/fine-tuning uses curated task data: simplest tasks around 5h, hardest tasks 100h+.

**Data quality and failures**

- pi0 explicitly values diverse, lower-quality behavior during pretraining because corrections and recoveries teach robustness.
- Post-training is higher-quality curated data, but successful-only filtering is not clearly stated.
- Failure composition, number of failed episodes, and retry distribution are not quantified.

**Labels and context**

- Task names and segment annotations are used.
- The paper describes about 2s sub-trajectory labels.
- The model consumes 2-3 RGB images, language, and joint-angle proprioception.

**Actions and objective**

- Continuous 50-step action chunks, max action dimension 18, zero-padded across embodiments.
- Deployed at 20 Hz or 50 Hz depending platform.
- Conditional flow matching trains the action expert over continuous action chunks.
- Inference uses about 10 denoising/integration steps.

**Compute**

- Full training: 700k steps.
- Reported comparisons include 160k parity pretraining for some baselines and 320k for Octo baseline.
- Inference timing on RTX 4090: about 73 ms onboard, 86 ms offboard.

**Unknowns**

- Exact post-training hours by task are not extractable from the local text.
- Exact success/failure composition is not reported.

**Source anchors:** [pi0.pdf](pi0.pdf)

### pi0.5

**Parameter scale:** about 2B VLM plus about 300M action expert.

**Training phases**

1. Discrete-token pretraining using FAST-style action tokens plus multimodal/text objectives.
2. Post-training with a continuous flow-matching action expert.

**Robot data**

- Mobile manipulation: about 400h in about 100 homes.
- Includes mobile manipulation, multi-environment tabletop, cross-embodiment robot data, and OXE-style data.
- Exact ME/CE/OXE quantities are not disclosed.
- Pretraining: 280k gradient steps.
- Post-training: 80k steps.

**Data quality and failures**

- Post-training action data is MM+ME filtered to successful episodes below a fixed length threshold.
- This is important: pi0.5's low-level policy is not mainly learning from failures or autonomous correction data.

**Non-robot and annotation data**

- Web data includes CapsFusion, COCO, Cambrian-7M, PixMo, VQAv2, object localization, and extra indoor/household box data.
- Manual semantic subtask labels are used for MM/ME/CE.
- Relevant bounding boxes are provided before subtask prediction.
- Verbal-instruction demonstrations are included; VI is about 11% of high-level mobile manipulation examples.

**Inputs/actions**

- Four cameras for high-level mobile tasks.
- Wrist plus forward camera for low-level control.
- 18-19 DoF state/action, including target joint/end-effector poses, grippers, base velocity, and torso lift at 50 Hz.

**Objective**

- Pretraining predicts text, object locations, and FAST action tokens.
- Post-training combines CE and flow matching with action expert randomly initialized.

**Unknowns**

- Exact ME/CE/OXE/Web/VI sample counts are not disclosed.
- Number of successful episodes remaining after filtering is not disclosed.

**Source anchors:** [pi05.txt](tmp/pdfs/pi05.txt)

### pi0.6 / RECAP

**Parameter scale:** Gemma 3 4B VLM plus about 860M action expert. Value model uses a smaller Gemma3 VLM.

**Training phases**

1. Large pretraining following the pi0.5 family recipe.
2. Supervised task adaptation on demonstrations.
3. Iterative on-robot RL with autonomous rollouts, corrections, and interventions.

**Robot data**

- Pretraining is described as tens of thousands of hours of demonstrations over many tasks and robots.
- Target RL examples:
  - Laundry: about 300 trajectories per iteration.
  - Box task: about 600 autonomous and 360 intervention trials per iteration.
  - Failure ablation: about 1000 autonomous and 280+378 correction episodes.
  - Cafe: 429 correction and 414 autonomous episodes.

**Data quality and failures**

- This is one of the clearest PI-family shifts toward explicit success/failure/correction data.
- Replay includes demonstrations, autonomous rollouts, expert teleoperated interventions, successful episodes, and failed episodes.
- Corrections are forced to positive advantage.

**Labels**

- Episode-level success label.
- Value/progress labels as negative steps-to-success.
- Advantage indicator.
- Subtasks and metadata language inputs.

**Inputs/actions**

- Up to 4 cameras, language, proprioception.
- Joint angles and gripper commands at 50 Hz.
- Experimental bimanual static robot setup uses 3 cameras plus joint/gripper state.

**Objective**

- Train value function, then an advantage-conditioned VLA.
- Continuous actions are trained by flow matching; discrete FAST/action/text with likelihood.
- Policy and value are fine-tuned from pretrained checkpoints each iteration.

**Unknowns**

- Exact pretraining hours/episodes are not reported.
- Some appendix wording conflicts between "demonstrations" and "autonomous trials" for a box task.

**Source anchors:** [pi0.6.pdf](pi0.6.pdf)

### pi0.7

**Parameter scale:** about 5B VLA plus a BAGEL-based 14B subgoal-image world model.

**Training phases**

1. Generalist VLA training with rich observation history, metadata, subgoals, and flow action expert.
2. Separate high-level policy / world model for subgoal image generation.
3. No task-specific post-training for the main generalization evaluations, as described in the local audit.

**Robot data**

- Exact total robot data is not disclosed.
- The mixture includes demonstrations across many tasks/platforms, autonomous evaluation data, human interventions, and open-source robot datasets.

**Data quality and failures**

- The paper explicitly includes lower-quality demonstrations, failures, successful episodes with mistakes, autonomous rollouts, and RL-specialist/eval data.
- It excludes autonomous data from some generalization-focused evaluation tasks to avoid leakage.

**Non-robot and annotation data**

- Egocentric human video.
- Multimodal web data.
- Object localization, attribute prediction, VQA, text-only prediction.
- Video captioning of in-house robot and web data.
- World model initialized from BAGEL 14B and mixed with image-editing/video datasets.

**Labels and metadata**

- Subtask instructions.
- Subgoal images.
- Episode metadata: speed binned by 500 steps, quality 1-5, mistake labels, and control mode.
- High-quality segmented labels for world model training.

**Inputs/actions**

- Up to 4 camera images.
- Up to 6 history frames.
- Up to 3 subgoal images.
- 448 px image inputs.
- Proprio via linear projection.
- 50 action tokens, executing either 15 or 25 steps depending setting.
- Supports joint and end-effector control modes; 20 Hz on UR5e and 50 Hz on other robots.

**Compute**

- Inference: single H100 for pi0.7/high-level policy.
- Reported action inference ranges from about 38 ms to 127 ms.
- Subgoal generation: 4 H100s, 8-bit matmuls, 25 denoising steps in about 1.25 s.

**Unknowns**

- No total hours/episodes/steps for the full training mixture.
- Training hardware is not disclosed.

**Source anchors:** [pi07.txt](tmp/pdfs/pi07.txt)

### RLT / RL Token

**Model role:** not a world model; a task-specific online RL adaptation layer over a pretrained VLA.

**Training phases**

1. Collect task demonstrations.
2. Fine-tune the base VLA and train the RL token for 2k-10k gradient steps.
3. Freeze the VLA and RL token.
4. Train online off-policy actor-critic from replay using the token/reference action context.

**Data budget**

- Base model is pi0.6, whose pretraining is described broadly as large web/robot/demo data, "tens of thousands of hours".
- Per task: 1-10h teleoperated demonstrations.
- Online RL: 400-1000 episodes, about 15 minutes to 5h of real robot data depending task.
- Reported tasks: screw installation, zip-tie fastening, Ethernet plugging, charger insertion.
- Each task has a fixed language instruction in the experiments, so RLT is not learning broad language-conditioned task diversity during online RL; it is adapting a narrow task phase.
- Critical-phase episodes are short: 5-20 s, about 250-1000 control steps.

**Data quality and failures**

- Replay aggregates VLA warmup, online RL rollouts, and optional human interventions.
- Sparse binary reward is human-labeled success/failure at episode end.
- Operator gives +1 when task is completed.
- This is one of the few methods with explicit on-policy robot interaction in the loop.
- The data is not pure expert data: it contains base-VLA rollouts, improved-policy rollouts, failures, and optional corrections/interventions.
- For screw and zip-tie, training starts only on the critical phase, then moves to full-task training where the base model handles earlier phases and the RL policy takes over at the critical phase.

**Training budget**

- VLA/RL-token adaptation: 2k-10k gradient steps on the single-task demonstration data.
- Online RL actor/critic: trained from scratch after freezing the VLA and RL token.
- Actor/critic size:
  - 2-layer MLP, hidden dim 256 for zip-tie, Ethernet, and charger.
  - 3-layer MLP, hidden dim 512 for screw installation.
- Update-to-data ratio: 5.
- Training samples are densified from robot time: action chunks are subsampled 2 control steps apart, so each second of robot data yields roughly 25 RL training samples.
- The practical budget claim is robot-time, not GPU-time: the paper emphasizes improvement within minutes to a few hours of practice.
- GPU/accelerator budget is not reported.

**Inputs/actions**

- pi0.6 base consumes up to four cameras, language, proprioception, and outputs H=50 actions for 1 s.
- Experiments use two wrist cameras, one base camera, and proprioceptive state.
- 50 Hz, 14D action, chunk C=10, so 140D action chunks.

**Objective**

- RL token is trained as a bottleneck reconstruction of frozen VLA embeddings.
- Actor maximizes Q while staying close to the VLA reference chunk.
- Actor/critic are small MLPs trained from scratch with high update-to-data ratio.

**Budget interpretation**

- RLT is data-efficient because it does **small-model RL** on top of a frozen, pretrained VLA prior.
- It still needs task-specific robot interaction and human success/failure labeling.
- For our thesis, RLT is a good comparison for "can feedback improve action translation?", but it is not an inference-only method and it is not a generic world-model planner.

**Unknowns**

- Exact demonstrations per task are reported only coarsely as 1-10h.
- GPU training compute is not reported.

**Source anchors:** [rlt.txt](tmp/pdfs/rlt.txt)

### OpenVLA

**Parameter scale:** about 7B/7.5B class VLA.

**Training phases**

1. Inherit Prismatic VLM: SigLIP/DINOv2/Llama-family visual-language initialization.
2. VLA robot fine-tune over Open X data.
3. Downstream full fine-tuning or LoRA on target robot tasks.

**Robot data**

- VLA training uses about 970k OXE robot demonstrations/trajectories.
- Raw Open X is larger: more than 70 datasets and more than 2M trajectories.
- Downstream tasks use about 10-150 demonstrations per task.
- Examples: 50 carrot demos, 10 flip-pot demos, 150 move-object demos, 70 knock-object demos, 45 towel demos, 70 wipe-table demos.

**Data quality and failures**

- Data is curated to manipulation datasets with at least one third-person camera and single-arm end-effector control.
- DROID was tried at 10% but removed for the final third due to quality/domain concerns.
- Bridge no-op first transition is filtered.
- Success/failure composition of the OXE mixture is not specified.

**Labels and non-robot data**

- VLM is pretrained/fine-tuned on web image-text data, including LLaVA-style image-text/text samples.
- OpenVLA robot fine-tuning does not do robot+web co-training in the same sense as PI/Molmo.
- Robot action labels are language-instruction plus action trajectory; no extra progress/value/subtask labels.

**Inputs/actions**

- Single image observation.
- 7D action: delta end-effector translation, delta rotation, gripper.
- Each action dimension discretized into 256 bins; final 256 Llama tokens are overwritten as action bins.

**Objective and compute**

- Full VLM fine-tuned end-to-end.
- CE loss on action tokens only.
- Pretraining/fine-tuning: 64 A100s for 14 days, about 21,500 A100-hours, batch 2048.
- Downstream full fine-tuning: 8 A100s for 5-15h per task.
- LoRA: 1 A100 for 10-15h, about 1.4% trainable parameters.

**Unknowns**

- Episode hours are not reported.
- Failure mix is not reported.

**Source anchors:** [openvla.txt](tmp/pdfs/openvla.txt)

### Octo

**Parameter scale:** 93M base / 27M small.

**Training phases**

1. OXE pretraining.
2. Downstream full-model fine-tuning.

**Robot data**

- Pretraining uses about 800k robot demonstrations/episodes from OXE.
- OXE raw corpus is about 1.5M episodes.
- 25 datasets after curation.
- Fine-tuning uses about 100 target trajectories.
- Examples: 120 CMU Baking, 118 Stanford Coffee, 100 Peg, 100 Pick-Up, 115 Coke.

**Data quality and failures**

- Removes datasets with no images, non-delta-end-effector control, highly repetitive data, low-resolution data, and niche setups.
- Down-weights repetitive datasets.
- Subsamples up to 100 random steps per trajectory for shuffling.
- Success/failure composition is not stated.

**Labels and non-robot data**

- No web data in policy pretraining.
- Uses pretrained T5-base language encoder.
- About language annotation: uses language where available and goal-image hindsight relabeling otherwise.

**Inputs/actions**

- Wrist plus third-person cameras.
- Optional proprioception.
- Language or goal-image conditioning.
- Diffusion action chunks.
- Supports new observation and action heads.

**Objective and compute**

- DDPM-style diffusion action objective.
- Full model updated during fine-tuning.
- Frozen T5-base performed best for language.
- Octo-Base: 300k steps, batch 2048, TPU v4-128 pod, about 14h.
- Fine-tune: about 5h on an A5000 24GB.

**Unknowns**

- Dataset hours are not reported.
- Success/failure composition is not reported.

**Source anchors:** [octo.txt](tmp/pdfs/octo.txt)

### MolmoAct2

**Parameter scale:** Molmo2/Qwen3-4B family plus a 36-layer DiT continuous action expert.

**Training phases**

1. Molmo2-ER embodied reasoning specialization.
2. OpenFAST action tokenizer training.
3. VLA pretraining.
4. Flow-action post-training.
5. Embodiment/task fine-tuning.
6. MolmoAct2-Think depth/action extension.

**Molmo2-ER data**

- No robot trajectories at this stage.
- 3.26M embodied-reasoning samples plus a Molmo2 mixture totaling about 12.51M non-robot samples.
- Data includes Image QA, Video QA, Image/video pointing, Video tracking, Captions/Long QA, image detection, embodied QA, multi-image ego-exo, and abstract reasoning.
- Stage 1: 20k steps, sequence length 4,200, batch 64, 2 nodes x 8 H100.
- Stage 2: additional 1.5k steps, sequence length 16,384, 8% NLP retained.

**OpenFAST tokenizer data**

- Trained on 1M one-second action sequences.
- Mix: 30% YAM, 30% SO100/101, 30% DROID, 3.33% each Fractal/BC-Z/Bridge.
- One-second chunks padded to 32D.
- 2048-token action vocabulary.
- Covers absolute joint and delta end-effector conventions.
- Uses percentile normalization to limit outlier effects.

**VLA pretraining robot data**

- Training mixture is 90% robot and 10% multimodal.
- Within robot data: YAM/SO/DROID each about 30%; small academic sources together about 10%.
- YAM: 34.5k demonstrations / 720+ h.
- SO: 38,059 episodes / 19.8M frames / 184h.
- DROID: 74,604 successful episodes / 17,758,044 frames.

**Data quality and filtering**

- DROID subset is explicitly successful-only, with at least one valid language instruction, idle-frame filtering, and no significant pauses.
- SO uses a four-stage structural/eval/license/TOPReward quality gate.
- YAM has strict retry/no-op protocols, but retry/failure counts are not reported.

**Extra labels**

- Language relabeling by Qwen3.5-27B doubles unique labels from 71,121 to 146,485.
- MolmoAct2-Think adds monocular depth annotations for every robotics dataset using Depth Anything V2.
- Depth is a model-generated side channel, not human ground truth.

**Objective and compute**

- VLA pretraining: discrete next-token action prediction; state discretized into 256 tokens; train vision encoder, connector, LLM, and added embeddings.
- VLA pretraining: 200k steps, batch 128, 64 H100, about 5,760 GPU-hours.
- Post-training: continuous flow expert trained with LLM + flow loss; flow gradients are insulated from the VLM through knowledge insulation.
- Post-training: 100k updates, batch 128, 64 H100, about 2,300 GPU-hours.
- Embodiment fine-tuning:
  - YAM: 100k, batch 128, 64 H100, about 2,300 GPU-hours.
  - DROID/SO: 100k, batch 64, 32 H100, about 1,150 GPU-hours.
  - LIBERO: 50k, batch 64, 32 H100, about 1,150 GPU-hours.
- LIBERO full mix: 4 suites x 500 demos = 2,000 demos.

**Unknowns**

- Exact number of sequences sampled during VLA pretraining is not reported.
- Exact robot/non-robot ratio in every ablation is not always clear.
- Depth quality filtering is not quantified.

**Source anchors:** [molmoact2.txt](tmp/pdfs/molmoact2.txt)

### GR00T N1

**Parameter scale:** about 2.2B.

**Training phases**

1. Large mixed pretraining over robot, human, simulation, and neural/generated trajectories.
2. Post-training on single-embodiment/task data.
3. Neural trajectory generation and IDM/pseudo-action augmentation.

**Pretraining data**

- Total reported pretraining data: 8,375.7h.
- Real robot: 262.3M frames / 3,288.8h.
- Human video: 181.3M frames / 2,517h.
- Simulation: 125.5M frames / 1,742.6h.
- Neural/generated: 23.8M frames / 827.3h.

**Robot sources**

- GR-1 teleop: 6.4M frames / 88.4h.
- DROID: 23.1M frames / 428.3h.
- RT-1: 338.4h.
- Language Table: 195.7h.
- Bridge: 111.1h.
- Agibot: 213.8M frames / 1,979.4h.
- RH20T-Robot: 62.5h.

**Human/video sources**

- Ego4D, Ego-Exo4D, Assembly-101, HOI4D, HoloAssist, RH20T-Human, EPIC-KITCHENS.
- Human videos use latent actions rather than ground-truth robot actions.

**Simulation and generated data**

- 540k pretraining simulation demos from 54 source-target pairs x 10k.
- Across pre/post, 780k simulated trajectories, equivalent to about 6,500h, generated in 11h.
- Only successful simulated demonstrations are retained.
- Neural data: about 300k neural trajectories, 827.3h, generated from 88h real GR-1 teleop.
- Video model fine-tuned on 3,000 real samples of 81 frames at 480p.
- Commercial MLLM filters instruction-following failures; failed videos are re-captioned.
- Neural data generation cost: about 105k L40 GPU-hours.

**Labels and quality**

- GR-1 has fine-grained atomic labels and coarse task labels.
- Dataset format supports task descriptions, validity flags, and success indicators.
- Auxiliary object detection labels from OWL-v2 boxes for target objects per frame.
- Sampling ratios and exact filtering of real robot data are not fully reported.

**Objective and compute**

- Uses ground-truth robot actions, latent actions for human video, and IDM pseudo-actions for neural videos.
- Final loss combines flow matching and detection loss.
- 200k pretraining steps, batch 16,384.
- Language component frozen; vision encoder and DiT unfrozen.
- Reported cost: about 50,000 H100 GPU-hours, up to 1024 GPUs.

**Post-training**

- Simulation: 30/100/300 demos per task.
- RoboCasa: 24 tasks.
- DexMG: 9 tasks.
- GR-1: 24 tasks.
- Real robot task-specific data: 15 min to 3h per task.
- Neural post-training co-trains real and neural at 1:1; RoboCasa uses 3k neural trajectories per task; real-world uses 100 per task.

**Unknowns**

- Exact real-world demo counts by task are not always reported.
- Sampling ratios across the massive pretraining sources are not reported.

**Source anchors:** [groot.txt](tmp/pdfs/groot.txt)

### CLAP

**Model role:** latent action pretraining from robot and human videos, followed by VLM/action expert training.

**Training phases**

1. Act-VAE learns a physically executable robot action codebook.
2. VD-VAE aligns video transitions to the action codebook.
3. CLAP-NTP predicts subtask and discrete action tokens.
4. CLAP-RF trains a continuous rectified-flow action expert.
5. Fine-tune on LIBERO or real tasks.

**Pretraining data**

- Curated AgiBot subset: 100k episodes / 1,500h from original about 1M trajectories / about 3,000h.
- Astribot S1: 27k episodes / about 50h at 30 Hz.
- Ego4D manipulation subset from UniVLA: 90h curated trajectories.

**Data quality and filtering**

- AgiBot filtering removes mobile manipulation, cooperative tasks, dexterous hand data, and semantic ambiguity.
- Astribot data is expert VR teleoperation.
- LIBERO fine-tuning filters failure cases.
- Failure composition in pretraining robot data is not explicitly reported.

**Actions and labels**

- Dual-arm 14D action: each arm has 3D end-effector position, 3D Euler orientation, and 1 gripper value.
- Human videos have task annotations but no actions/proprio; pseudo latent actions are inferred.
- CLAP-NTP predicts subtask plus action tokens.

**Objective and compute**

- Act-VAE: 150M params, 100k steps, batch 4096, about 190 A100-80G hours.
- VD-VAE/CLAP alignment: 200M params, 100k steps, batch 256, about 380 A100 hours.
- CLAP-NTP: Qwen3VL-4B tokenizer expanded with action codebook tokens, 150k steps, batch 512, about 3,800 A100 hours.
- CLAP-RF: DiT expert cross-attends to VLM KV cache with stop-grad/knowledge insulation, 80k steps, batch 1024, about 2,000 A100 hours.

**Fine-tuning/evaluation data**

- Real tasks: Pack Doll 200 demos, Fold T-shirt 200 demos, Make Bouquets 100 demos for each of two combinations.
- Human generalization add-on: 300 GoPro human demos.
- LIBERO: 4 suites x 10 tasks x 50 demos = 2,000 demos.

**Unknowns**

- Exact robot/human sampling ratio in CLAP-NTP is absent.
- Act-VAE exact trajectory subset is not separately reported.

**Source anchors:** [clap.txt](tmp/pdfs/clap.txt)

## World-Action And World-Model Methods

### DreamZero

**Parameter scale:** Wan2.1-I2V-14B main backbone; paper also reports smaller variants/ablations.

**Training phases**

1. Start from Wan2.1-I2V-14B, a web-scale video pretrained model; exact web-video quantity is not reported.
2. Robot-domain pretraining on AgiBot G1.
3. Separate Franka variant on DROID.
4. Task post-training on target manipulation domains.
5. Optional video-only cross-embodiment adaptation.

**Robot data**

- AgiBot G1: about 500h teleoperation.
- 7,193 episodes.
- Average episode length 4.4 min.
- Average 42.4 subtasks.
- 22 environments.
- DROID/Franka variant: DROID amount not stated in paper.
- Post-training:
  - Shirt folding: 33h.
  - Fruit packing: 12h.
  - Table bussing: 40h.
  - 50k steps per task.
- Cross-embodiment video-only:
  - 72 trajectories.
  - 9 unseen tasks.
  - 8 demos per task.
  - 20 min YAM or 12 min human video.
  - Co-train 10k steps with 1:1 pretraining mix.
- New YAM adaptation:
  - 55 trajectories.
  - 11 tasks.
  - About 30 min play data.

**Data quality and failures**

- Main AgiBot data is heterogeneous, non-repetitive teleoperation, not a small repeated successful-demo set.
- Idle actions are explicitly filtered.
- No explicit failure/suboptimal/on-policy rollout training in the main DreamZero setup.
- Post-training success/failure filtering is not stated.
- YAM adaptation is play data, not described as successful-only.

**Labels**

- Language instructions.
- AgiBot episodes are coarse multi-task episodes.
- YAM adaptation uses only 11 short global language annotations, one per task.
- No value/progress labels for training; task progress is used for evaluation.

**Inputs/actions**

- Visual context through VAE.
- Language through text encoder.
- Proprio through state encoder.
- Multi-view robot data is concatenated into one frame.
- Relative joint positions by default.
- AgiBot video: 5 FPS, actions 30 Hz, horizon 48.
- DROID: actions 15 Hz, horizon 24.

**Objective and compute**

- Teacher-forced chunk-wise joint flow matching over video and action.
- Pretraining: 100k steps, global batch 128 for AgiBot and DROID.
- Freezes text encoder, image encoder, and VAE.
- Trains DiT blocks plus state/action encoders/decoder.
- Training GPU count is not stated.
- Inference optimization: H100/GB200 speedups; paper reports 5.7 s baseline to 150 ms with Flash on GB200.

**Critical interpretation**

DreamZero is a joint world-action model. It models future video and action together from history/language/proprio. It is not a clean action-conditioned simulator. That makes it excellent for prediction honesty, but weaker for "force action A, then predict outcome" counterfactuals.

**Unknowns**

- Success/failure composition of the 500h AgiBot data.
- DROID hours/episodes.
- Full training compute.

**Source anchors:** [dreamzero.txt](tmp/pdfs/dreamzero.txt)

### Cosmos Policy

**Parameter scale:** Cosmos-Predict2-2B base.

**Training phases**

1. Start from Cosmos-Predict2 video model; base video pretraining quantity is not stated.
2. Fine-tune as policy/world/value model on robot demonstrations.
3. Optional rollout refinement using collected rollout outcomes.

**Robot data**

- LIBERO: 500 demos per suite, 10 tasks x 50 demos.
- RoboCasa: 50 human teleop demos per task, 24 tasks.
- ALOHA: 185 demos total:
  - Put X on plate: 80.
  - Fold shirt: 15.
  - Candies in bowl: 45.
  - Ziploc: 45.
- Rollout refinement: 648 rollouts total, 505 prior evaluation rollouts plus 143 Cosmos rollouts.

**Data quality and failures**

- Policy branch filters unsuccessful demonstrations for LIBERO/RoboCasa.
- World model/value branches use full unfiltered data.
- Initial rollout dataset includes failed demos if they exist.
- LIBERO/RoboCasa replay failures are about 10-20%.
- ALOHA has no failed demos initially.
- Rollout refinement records trajectory plus success/fail or fractional score.

**Labels**

- Task descriptions/language.
- Monte Carlo return labels per transition for value prediction.
- Future proprio/images as auxiliary targets.
- Success/fail/fractional rollout outcomes.

**Inputs/actions**

- Latent-frame injection represents current proprio, multi-camera images, action chunk, future proprio/images, and value inside one diffusion sequence.
- ALOHA input: 14 joint angles, 3 cameras, task description.
- Predicts 50-step action chunks at 25 Hz, covering 2 s.
- Non-image modalities normalized to [-1,1].

**Objective and compute**

- Fully fine-tunes all Cosmos-Predict2 weights.
- Learns direct policy, action-conditioned world/value, and value-from-future modes in one architecture.
- Batch split:
  - Base: 50% policy, 25% world model, 25% value.
  - Rollout refinement: 90% world/value, 10% policy.
- LIBERO: 40k steps, 64 H100, batch 1920, 48h.
- RoboCasa: 45k steps, 32 H100, batch 800, 48h.
- ALOHA: 50k steps, 8 H100, batch 200, 48h.

**Critical interpretation**

Cosmos is one of the cleanest published templates for our planning thesis because it separates direct action from action-conditioned future/value scoring. It is also one of the clearest methods for using failures and rollouts.

**Unknowns**

- Exact base video pretraining quantity.
- Whether the 505 prior rollout pool is balanced across methods/distributions.

**Source anchors:** [cosmos_policy.txt](tmp/pdfs/cosmos_policy.txt)

### Genie Envisioner: GE-Base, GE-Act, GE-Sim

**Parameter scale:** GE-Base/GE-Sim around 2B; GE-Act action decoder around 160M.

**Training phases**

1. GE-Base-MR: multi-resolution robot video adaptation.
2. GE-Base-LF: low-frequency adaptation.
3. GE-Act action pretraining with GE-Base-LF fixed and video generation disabled.
4. Task adaptation: video adaptation plus action specialization.
5. GE-Sim: action-conditioned future-video simulator.

**Core data**

- AgiBot-World-Beta.
- About 1M high-quality dual-arm episodes.
- About 2,967h.
- Human teleoperation.
- Three synchronized cameras: head, left wrist, right wrist.

**Training details**

- GE-Base-MR:
  - 57-frame clips.
  - Random 3-30 Hz sampling.
  - 4 sparse memory frames.
  - 32 A100 for about 7 days.
- GE-Base-LF:
  - 9-frame clips at 5 Hz.
  - 4 memory frames.
  - 32 A100 for about 3 days.
- GE-Act:
  - Uses AgiBot-World-Beta actions.
  - Predicts 54 action steps at 30 Hz.
  - 16 A100 for about 3 days.
  - GE-Base-LF frozen, only action decoder updated.
- Task adaptation:
  - Video adaptation: full AgiBot plus task subset upweighted 10x, 8 A100 for 12h.
  - Action specialization: full model on task-specific data, 8 A100 for 36h.
- Example task data:
  - Agilex: about 1h teleop.
  - Dual Franka: 250 episodes, about 1h.
  - RoboTwin: 200 demos.

**Data quality and failures**

- GE-Base data is called high-quality, but success/failure filtering is not specified.
- GE-Act action pretraining is supervised by ground-truth action trajectories.
- GE-Sim explicitly augments with failures: erroneous executions, incomplete behaviors, and suboptimal control trajectories from human teleop and robot deployments.

**Labels**

- Natural language instructions.
- Multi-view visual observations.
- Structured action policies.
- EWMBench evaluation adds step-level captions for 4-10 atomic sub-actions and manually annotated reference trajectories; those are evaluation labels, not core GE training labels.

**GE-Sim action representation**

- Explicit 7D end-effector pose plus gripper per arm.
- 14D dual-arm action over K steps.
- Uses reference-image style conditioning and spatial pose/image conditions.
- Freezes VAE and CLIP, optimizes remaining generation/conditioning parameters with flow matching.

**Critical interpretation**

GE-Base is instruction-conditioned video prediction. GE-Act is an action decoder on GE latents. GE-Sim is the counterfactual simulator: current observation + instruction + candidate action trajectory -> future video. GE explicitly states it does not incorporate internet-scale or simulation-based data in GE training, relying on AgiBot-World-Beta.

**Unknowns**

- Exact task-specific subset sizes beyond examples.
- Exact GE-Act action representation has some wording ambiguity around "torque" versus end-effector/structured action.
- Success/failure composition of core AgiBot data.

**Source anchors:** [2508.05635.txt](papers/2508.05635.txt)

### Fast-WAM

**Parameter scale:** about 6B total: Wan2.2-5B video DiT plus about 1B action expert.

**Training phases**

1. No embodied pretraining reported.
2. Start from pretrained Wan2.2-5B video backbone.
3. Co-train future video and action prediction on target/sim datasets.
4. Remove future-video branch at inference.

**Robot/sim data**

- LIBERO: four suites, each 500 demos over 10 tasks, 20k steps.
- RoboTwin: 2,500 clean demos plus 25,000 randomized demos over more than 50 tasks, 30k steps.
- Real towel folding: 60h teleoperated demos on Galaxea R1 Lite, 30k steps.

**Data quality and failures**

- No explicit success/failure filtering, value labels, interventions, corrections, or rollout learning are stated.
- Evaluation discusses avoiding repeated trial-and-error corrections, but that is not a training annotation method.

**Inputs/actions**

- Language through T5.
- Visual observations through VAE.
- Multi-camera images concatenated into one image.
- Action horizon h=32.
- Video downsampled 4x to 9 frames per chunk.
- Proprio inputs are not specified clearly.

**Objective and compute**

- Joint flow matching objective: action loss plus weighted video loss.
- AdamW LR 1e-4, mixed precision, gradient clipping 1.0.
- Latency reported on RTX 5090D V2 32GB.
- Training GPU count is not stated.

**Critical interpretation**

Fast-WAM is a key negative control. It uses future video during training but intentionally prevents action tokens from attending to future-video tokens at inference. If it performs well, the gain may be representation regularization, not inference-time planning.

**Unknowns**

- Demo quality/failure filtering.
- Wan2.2 base video pretraining quantity.
- Training compute.

**Source anchors:** [fast_vam.txt](tmp/pdfs/fast_vam.txt)

### LingBot-VA

**Parameter scale:** about 5.3B total, with about 350M action-stream parameters.

**Training phases**

1. Pretrain causal autoregressive video-action model.
2. Post-train on task-specific demos, sometimes as few as 50 demonstrations.
3. Use forward dynamics prediction in post-training.

**Pretraining data**

- About 16k hours robot manipulation data.
- Sources: AgiBot, RoboMind, InternData-A1 simulation, OXE OpenVLA subset, UMI human demos excluding DexUMI, RoboCOIN, plus internal demonstrations.
- Pretrains for 1.4T tokens.

**Post-training data**

- As few as 50 demonstrations, 3k steps at LR 1e-5.
- Alternative recipe: 1k steps at LR 1e-4.
- Real-world six-task deployment: 50 real-world demos for training, 500 fine-tuning steps, LR 1e-4, sequence length 150k.
- RoboTwin: 2,500 clean plus 25,000 randomized demos, 50k steps.
- LIBERO: 500 demos per suite, unsuccessful demos filtered, 4k steps.

**Data quality and failures**

- Pretraining data is preprocessed for format and annotation quality, but failure/suboptimal filtering is not explicit.
- LIBERO explicitly filters unsuccessful demonstrations.
- No value labels or policy rollout training are stated.

**Inputs/actions**

- Universal 30D dual-arm action:
  - Per arm: 7D end-effector pose, 7D joint angles, 1 gripper.
- Wan2.2 causal VAE.
- Views concatenated along width.
- Frozen T5 text encoder.
- Chunk K sampled from 1 to 4.
- RoboTwin video downsampled from 50 Hz to 12.5 Hz; action remains 50 Hz.

**Objective**

- Teacher-forced autoregressive flow matching.
- Uses dynamics loss plus inverse-dynamics/action loss.
- Noisy history augmentation.
- Post-training adds forward dynamics prediction loss.

**Critical interpretation**

LingBot is closer to DreamZero than to a direct VLA, but it is more explicitly causal/action-history-conditioned: future visual latent dynamics and inverse dynamics are both trained. It still does not report a value/planning interface like Cosmos.

**Unknowns**

- Exact amount per source.
- Base Wan internet-video quantity.
- GPU compute.
- Whether 50 real-world demos means per task or total in all experiments.

**Source anchors:** [lingbot_va.txt](tmp/pdfs/lingbot_va.txt)

## Semantic Latent, Planning, And Evaluation Papers

### VLA-JEPA

**Model role:** latent world-model regularization plus direct action policy.

**Training phases**

1. JEPA latent pretraining.
2. Action-head fine-tuning.

**Data**

- Something-Something-v2: 220k human videos.
- DROID: 76k high-quality robot demonstration trajectories.
- LIBERO fine-tuning: about 2k simulated expert demonstrations.
- SimplerEnv fine-tuning: Fractal + BridgeV2.
- Real-world post-training: 100 trajectories across 3 pick/place tasks.

**Data quality and failures**

- Human videos are actionless but language-associated in the dataset.
- DROID is action-labeled high-quality demonstrations.
- LIBERO data is expert-only.
- No failures, reward labels, interventions, or on-policy RL are reported.

**Inputs/actions**

- Multi-view RGB plus language.
- V-JEPA2 state encoder concatenates views.
- Real setup: Franka, Robotiq gripper, two third-person RealSense views plus one wrist view.
- Actions: 7D future horizon with end-effector delta position, delta axis-angle, binary gripper.

**Objective and compute**

- Frozen V-JEPA2 target encoder.
- Latent-space prediction loss plus flow-matching action head.
- Joint loss: action flow loss plus weighted world-model loss.
- 8 A100s, global batch 256.
- 50k pretraining steps, 30k sim fine-tuning, 20k real fine-tuning.

**Critical interpretation**

This is a strong template for our feasible experiment. It suggests testing whether semantic latent prediction captures progress and recoverability better than pixel reconstruction, without needing DreamZero-scale video generation.

**Unknowns**

- Exact SSV2 filtering.
- DROID success/failure composition.
- Whether all fine-tuned modules remain trainable.

**Source anchors:** [vla_jepa.txt](tmp/pdfs/vla_jepa.txt)

### Reconstruction or Semantics

**Model role:** controlled latent-space study, not a policy-training paper.

**Training/evaluation setup**

- Fixes data, action conditioning, optimizer, and DiT transition model.
- Varies encoder/adapter/decoder latent representation.
- Tests whether reconstructive or semantic latents yield better robotic world models.

**Data**

- BridgeV2: about 60k WidowX demonstrations across 13 task families.
- BridgeV2 provides RGB, 7-DoF end-effector actions, and language instructions.
- SOAR: about 30.5k success/failure episodes used for success classification probes.

**Data quality and failures**

- Offline robot demonstrations for world-model training.
- Explicit success/failure labels appear in SOAR for probing, not as DiT training rewards.
- No on-policy RL, interventions, corrections, or value learning for the transition model.

**Inputs/actions**

- RGB observations.
- Optional language.
- Continuous 7-DoF actions.
- DiT ignores language during training.
- Uses H=2 history frames and predicts 8 future frames.

**Objective and compute**

- Encoders are frozen.
- Optional adapters/decoders are fixed during world-model training.
- Only action-conditioned DiT transition model is updated.
- S-VAE adapter is separately pretrained to reconstruct frozen semantic features with KL/pixel losses, then frozen.
- Adapter/decoder: 200 epochs on 4 H100s, about 55h.
- DiT-S: 100 epochs on 4 H100s, 6-7h.
- DiT-L: 80 epochs, about 34h.

**Critical interpretation**

The paper is important because it directly supports the user's thesis that pixel-perfect reconstruction may be the wrong signal. For our experiment, semantic latent mismatch between predicted and observed future may be more useful than RGB error.

**Unknowns**

- BridgeV2 success-only status is not specified.

**Source anchors:** [2605.06388.txt](tmp/pdfs/2605.06388.txt)

### LPS

**Model role:** latent policy steering wrapper around an embodiment-specific policy.

**Training phases**

1. Pretrain an image-based Dreamer-style world model using optical flow as an embodiment-agnostic action proxy.
2. Fine-tune the world model on target robot actions.
3. Train a base diffusion policy.
4. Train a robust value function.
5. Select among policy candidate plans using the value function.

**Data budget**

- Target dataset: small, expert, all-success Franka demos with robot actions.
- Real experiments: 30/50 demos or 60/100 demos depending setting.
- Robomimic: 30-50 expert Franka demonstrations; ablations use 100.
- Pretraining:
  - Sim variant: 3 non-target robots with 100 demos each.
  - Real variant: OXE filtered to 2,000 episodes.
  - Human variant: human play with no specific goals; transitions about 43% of public robot dataset.
  - Mix variant: 8 embodiments.
- Real-world success is reported over 20 trials per task.
- Robomimic reports averages across 3 seeds, with each seed evaluated over hundreds of trials.

**Data quality and failures**

- Cross-embodiment pretraining can include suboptimal/non-success data.
- Target demos are successful expert-only.
- No on-policy real RL is used.
- Value training synthesizes policy-likely states through the world model and adds cosine-similarity reward for deviation from expert latent states.
- The cross-embodiment data is allowed to be messy because it is used to pretrain an embodiment-agnostic world model, not directly clone target-robot actions.
- The target action data remains the scarce resource: LPS still needs successful expert demonstrations on the target Franka embodiment.

**Training budget**

- Reported data budget is much clearer than reported compute budget.
- Target training regime:
  - 30-50 successful expert demos for low-data real/Robomimic settings.
  - 60-100 successful expert demos for larger target-data settings.
  - 100 demos for several ablations.
- Pretraining budgets:
  - Simulation pretraining: 3 non-target robots x 100 demos each = 300 demos.
  - Real robot pretraining: 2,000 OXE episodes after dropping too-short/too-long episodes.
  - Human pretraining: human play transitions equal to about 43% of the public robot dataset.
  - Mixed pretraining: 8 embodiments total: 3 sim robots, 4 real robots, and 1 human embodiment.
- Planning/inference horizon:
  - Prediction/action horizon is usually 16.
  - Ablations test horizons 4, 8, 16, 20, and 24.
- Optimizer steps, wall-clock training time, GPU type/count, and model parameter count are not reported in the extracted paper text.

**Inputs/actions**

- Drops proprioception and embodiment-specific actions during pretraining.
- Uses optical flow encoded to target action dimension.
- Target actions are normalized robot actions/end-effector pose.
- Real setup uses wrist and side cameras.

**Critical interpretation**

LPS does not solve universal action generation. The deployed policy remains embodiment-specific. But it is useful as a planning/control wrapper: sample plans, steer with a latent value, and reduce failure from the base policy.

From a budget perspective, the point is not that LPS removes target data. It converts a **30-100 successful-demo target budget** plus cheap cross-embodiment/human/sim pretraining data into better plan selection at inference. It is closer to "use a world model to spend compute on selecting from candidate actions" than to "learn a new universal robot policy."

**Unknowns**

- Exact human-play hours.
- OXE composition after filtering.
- Dreamer architecture size.
- Reward labeling beyond binary success.
- Training steps, hardware, and wall-clock time.

**Source anchors:** [lps.txt](tmp/pdfs/lps.txt)

### IVLR

**Parameter scale:** Show-o2 1.5B family.

**Training phases**

1. Generate a full interleaved visual-language trace from initial observation/instruction.
2. Execute closed-loop actions conditioned on cached trace, instruction, and current observation.

**Data**

- Uses LIBERO and SimplerEnv-WidowX simulated benchmarks.
- Extracted text does not report exact training demo counts.
- Pseudo-traces are constructed from demonstrations and used only for training, not oracle evaluation.

**Trace annotations**

- Robot datasets lack traces, so demos are segmented with UVD.
- Segment endpoint keyframes are selected.
- Qwen3-VL captions each stage.
- Annotation JSON includes:
  - stage id,
  - caption,
  - objects,
  - gripper action,
  - spatial relation,
  - keyframe index.
- Filtering rejects invalid JSON, duplicate/no-change stages, hallucinated objects, and camera-motion-only captions.

**Inputs/actions/objectives**

- Wan2.1 VAE image/keyframe latents.
- Text CE for captions.
- Flow matching for visual keyframes.
- L1 action loss for continuous end-effector actions.
- No modules frozen in the reported setting.

**Compute**

- Trace generation about 10 s on one H20.
- Joint training on 16 H200s.
- LIBERO: 40k steps, about 4h.
- SimplerEnv: 60k steps, about 6h.

**Critical interpretation**

IVLR is the clearest "plan first, execute with correction" template. The key data move is not magic reasoning; it creates pseudo-global traces with segmentation and VLM captioning, then conditions execution on that cached plan plus live observations.

**Unknowns**

- Exact demo counts.
- Success/failure mix.
- Camera count/proprio details.

**Source anchors:** [ivlrr.txt](tmp/pdfs/ivlrr.txt)

### Joint Embedding vs Reconstruction SSL

**Model role:** not robotics, but relevant to latent objective choice.

**Data**

- ImageNet/CIFAR/self-supervised corruption experiments.
- No robot trajectories, actions, rewards, values, or embodiment data.

**Training details**

- ImageNet: batch 256 for 500 epochs.
- BYOL ResNet-50, MAE/DINO ViT-B/16.
- CIFAR-10: ResNet-50, batch 256, 1000 epochs, 5 seeds.

**Critical interpretation**

The core result is conceptual: reconstruction is preferable when irrelevant noise is low magnitude; joint embedding is preferable when irrelevant features are high magnitude. Real robot scenes often contain high-magnitude irrelevant features, so semantic/joint-embedding latent errors may be more faithful than pixel reconstruction errors.

**Source anchors:** [joint_embedding_vs_reconstruction_ssl.txt](tmp/pdfs/joint_embedding_vs_reconstruction_ssl.txt)

## What The Data Says

### 1. Failure Data Is The Exception, Not The Norm

Most VLA/WAM systems are still trained primarily on successful expert demonstrations or curated teleoperation. The explicit exceptions are:

- **pi0.6/RECAP:** autonomous rollouts, failed episodes, corrections, interventions, advantage labels.
- **RLT:** task demos plus online robot RL rollouts and human success/failure labels.
- **Cosmos Policy:** policy learns from successful demos, while world/value modules use unfiltered data including failures; rollout refinement uses outcomes.
- **GE-Sim:** explicitly augments with erroneous, incomplete, and suboptimal trajectories.
- **GR00T:** simulation keeps successful demos, but neural generated data is filtered/re-captioned; dataset format includes success/validity indicators.
- **pi0.7:** explicitly includes lower-quality demos, mistakes, autonomous data, interventions, and rich metadata, but scale is undisclosed.

This matters because "reasoning is good but action translation is weak" is hard to prove if the base models were never trained to observe and recover from failures.

### 2. Raw Video Is Not One Thing

The papers use "video" in very different ways:

- **Web-scale video priors:** Wan/Cosmos/BAGEL-style pretrained backbones. Quantity often not disclosed.
- **Robot-domain video with actions:** DreamZero, GE, LingBot, Fast-WAM, Cosmos, Reconstruction or Semantics.
- **Human actionless video:** VLA-JEPA SSV2, CLAP Ego4D, GR00T human egocentric videos.
- **Generated/neural video:** GR00T neural trajectories.
- **Video with dense semantic labels:** IVLR pseudo-traces, GE EWMBench evaluation captions, MolmoAct2 VQA/pointing/tracking/depth.

So "the model saw video" is not enough. The useful question is whether the video was tied to actions, task progress, labels, or only visual continuity.

### 3. Extra Annotations May Matter As Much As Architecture

MolmoAct2 and pi0.5 are not just "more VLA data". They add VQA, captioning, object localization, pointing/tracking, relabeling, depth, and subtask labels. GR00T adds object boxes, validity/success indicators, generated trajectories, and pseudo-actions. IVLR adds pseudo stage traces.

For our context-faithfulness experiments, these annotation-rich models are important baselines. If they ignore instruction/image perturbations, the issue is not just lack of language grounding data; it may be the action interface or deployment contract.

### 4. Action-Conditioned Models Are The Clean Counterfactual Tools

DreamZero and Fast-WAM produce future/action information, but not in the same way as GE-Sim or Cosmos:

- **DreamZero:** jointly imagines future video and action. Good for prediction honesty.
- **Fast-WAM:** future video helps training but is discarded at inference. Good negative control.
- **GE-Sim:** candidate action trajectory conditions future video. Good for causal counterfactuals.
- **Cosmos:** candidate action, predicted future/value, and best-of-N planning. Good for planning.
- **Reconstruction or Semantics:** action-conditioned DiT over different latents. Good for latent metric design.

This corrects an earlier framing: GE-Sim/Cosmos are not inherently heavier than DreamZero in parameter-memory terms. The friction is checkpoint availability, environment wrappers, and action/observation integration.

### 5. Data Per Parameter Is Usually Not Cleanly Reportable

The papers mix units: hours, demos, episodes, timesteps, tokens, frames, rollouts, and generated trajectories. A few rough anchors:

- **OpenVLA:** about 7B parameters, 970k robot trajectories, about 21.5k A100-hours.
- **Octo:** 93M base, 800k robot episodes, TPU v4-128 for about 14h.
- **MolmoAct2:** 4B-class VLM plus DiT expert, robot mix dominated by YAM 720h, SO 184h, DROID 74,604 successful episodes, plus 12.51M ER/multimodal samples.
- **GR00T:** 2.2B parameters, 8,375.7h mixed pretraining data.
- **DreamZero:** 14B backbone, 500h AgiBot plus task post-training.
- **GE:** 2B-class, about 2,967h AgiBot, no internet/sim data in GE training.
- **Cosmos:** 2B, target-domain demos plus rollouts; base video pretraining quantity not reported.

The honest conclusion is that per-parameter data ratios are only meaningful within a unit family, not across all papers.

## Implications For Our Experiments

### Immediate Context-Faithfulness Tests

Run direct VLAs and WAM policies in their comfortable environments, then perturb one input branch at a time:

- Same image, changed instruction.
- Same instruction, changed image/object crop.
- Same image/instruction, perturbed proprio.
- Same rollout, altered prompt object.
- In-domain object/action versus semantically mismatched object/action.

Track action changes, success, and whether the predicted/semantic future changes in the expected direction.

### Prediction-Honesty Tests

For DreamZero/LingBot/Fast-WAM-style systems:

- Compare predicted future to observed future in semantic latents, not only pixels.
- Measure whether future-error spikes before action failure.
- Compare in-domain tasks versus off-domain perturbations.
- Check whether action prediction and video prediction fail together or decouple.

### Action-Counterfactual Tests

For GE-Sim/Cosmos/Reconstruction-or-Semantics-style systems:

- Hold observation and instruction fixed.
- Force different candidate action chunks.
- Predict future in RGB or semantic latent.
- Score whether the model's future changes causally and plausibly.

This is the clean version of "does the world model understand that this action changes the future?"

### Feasible 48 GB L40 Training Run

Do not attempt full PI/Molmo/DreamZero training. The realistic run is:

1. Use DROID/PolaRiS/MolmoSpaces trajectories or rollouts.
2. Freeze visual encoders: DINO/SigLIP/V-JEPA/Molmo image tower if accessible.
3. Train small probes:
   - inverse dynamics from latent transition to action,
   - progress/success probe,
   - future-latent prediction probe.
4. Compare pixel error versus semantic latent error versus action-recoverability error.
5. Use model-generated future, when available, as input to the same probes.

This is the smallest experiment that tests the central thesis without pretending we can reproduce the full foundation-model training pipelines.

## Open Audit Gaps

- Many papers say "curated" or "high-quality" without reporting failure counts.
- Web/video pretraining quantities are often hidden behind pretrained backbones.
- Exact sampling ratios across robot sources are frequently missing.
- DROID/OXE quality differs by paper because each paper filters/relabels it differently.
- Some models report demos, others hours, others frames, others tokens; cross-paper data-per-parameter comparisons are approximate.
- For IVLR, exact demo counts are missing in the extracted text.
- For pi0.7, the training mixture is qualitatively rich but quantitatively underreported.
