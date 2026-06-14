# Paper Catalog

This catalog groups the local papers by experimental role. The repo does not currently have literal `dataset/` or `engines/` directories; PDFs live under `pdfs/`.

## Dataset / Environment / Benchmark Papers

| Paper | Local file | Why it belongs here |
| --- | --- | --- |
| DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | external: https://arxiv.org/abs/2403.12945 | Core real robot manipulation dataset/platform: 76k trajectories, 350 hours, 564 scenes, 84 tasks. Important for DreamZero, pi0.5-DROID, MolmoAct2-DROID context. |
| MobileEgo Anywhere: Open Infrastructure for Long Horizon Egocentric Data on Commodity Hardware | `pdfs/2605.05945.pdf` | STERA/iPhone Pro capture pipeline for long-horizon egocentric RGB-D, ARKit pose, depth, MANO-style hand annotations, and training-ready exports. Relevant as a commodity capture/data pipeline; the paper does not present a ready real-to-sim environment export stack. |
| MolmoSpaces: A Large-Scale Open Ecosystem for Robot Navigation and Manipulation | `pdfs/MolmoSpaces.pdf` | Sim/eval ecosystem and benchmark. DROID-style Franka manipulation setup, perturbation axes, zero-shot navigation/manipulation tasks. |
| PointWorld: A Multimodal 3D Scene Benchmark for Vision-Language Models | `pdfs/PointWorld.pdf` | 3D scene/benchmark paper, relevant as environment/scene understanding substrate rather than policy engine. |
| Sim2Real-AD | `pdfs/Sim2Real.pdf` | Sim-to-real/autonomous driving dataset/benchmark style paper. Useful mostly as transfer/evaluation reference. |
| exploration sprint | `pdfs/exploration_sprint.pdf` | Local sprint artifact, not a research paper; belongs near experiment planning notes. |

## Engine / Model / Policy / Planning Papers

| Paper | Local file | Primary role |
| --- | --- | --- |
| World Action Models are Zero-shot Policies | `pdfs/dreamzero.pdf` | DreamZero/WAM: joint future video + action generation, zero-shot policy via world action model. |
| Cosmos 3: Omnimodal World Model | external: https://research.nvidia.com/labs/cosmos-lab/cosmos3/; note: `papers/Cosmos 3 - Omnimodal World Model.md` | NVIDIA's 2026 Cosmos release: one omnimodal world model interface for physical reasoning, world/video/audio generation, action-conditioned forward dynamics, inverse dynamics, and policy post-training. Important as a multi-embodiment action-interface reference, but current public evidence does not prove broad zero-shot cross-embodiment control. |
| Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning | `pdfs/Cosmos-Policy.pdf`; review: `docs/planning/cosmos-policy-review.md` | WAM-style Cosmos-Predict2 fine-tuning: actions, future states, proprioception, and values are injected as latent frames so one video diffusion model can act as policy, world model, and value model. Central planning reference for best-of-N action search with rollout-trained value prediction. |
| Fast-WAM: Do World Action Models Need Test-time Future Imagination? | `pdfs/Fast-VAM.pdf` | Wan2.2-based WAM that keeps future-video latent prediction as a training-time auxiliary loss but removes future generation at inference. Useful contrast to Cosmos/DreamZero: strong direct-policy performance suggests video co-training may matter more than test-time imagination for action accuracy, but it does not provide a value/planning interface. |
| LingBot-VA: Causal World Modeling for Robot Control | `pdfs/LingBot-VA.pdf` | Autoregressive IDM-style WAM built on Wan2.2: interleaves video/action tokens, predicts future visual latents, then decodes actions through inverse dynamics while preserving causal history with KV cache. More explicitly causal than Fast-WAM, but much heavier at inference because it keeps future-state generation in the loop. |
| Do World Action Models Generalize Better than VLAs? A Robustness Study | `pdfs/2603.22078.pdf` | Comparative robustness study of WAMs vs VLAs on LIBERO-Plus and RoboTwin 2.0-Plus. Useful for our prompt/image perturbation experiment framing. |
| Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation | `pdfs/2508.05635.pdf` | GE platform: GE-Base video world model, GE-Act action decoder, GE-Sim action-conditioned neural simulator, and EWMBench. Useful for cross-embodiment adaptation design and world-model evaluation metrics. |
| Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models | `pdfs/Reconstruction-or-Semantics-Semantic-WM.pdf` | Controlled study of reconstruction vs semantic latent spaces for action-conditioned robotic world models. Useful for choosing planning/evaluation latents: semantic latents preserve action recoverability, task progress, and policy-relevant structure better than pure pixel-reconstruction latents. |
| MolmoAct2: Action Reasoning Models for Real-world Deployment | `pdfs/MolmoAct2.pdf` | VLA/action reasoning model; DROID and LIBERO variants; OpenFAST/action-token training plus continuous action expert deployment. |
| pi0: A Vision-Language-Action Flow Model for General Robot Control | `pdfs/pi0.pdf` | PI VLA flow-policy baseline. |
| pi0.5: a Vision-Language-Action Model with Open-World Generalization | `pdfs/pi0.5.pdf` | PI open-world generalization VLA; relevant for DROID-style policy comparisons. |
| pi*0.6 | `pdfs/pi0.6.pdf` | PI model iteration; adjacent to RLT/RL-token line. |
| pi0.7 | `pdfs/pi07.pdf` | PI model with world-model/subgoal-image style planning. |
| RL Token: Bootstrapping Online RL with Vision-Language-Action Models | `pdfs/rlt.pdf` | Online off-policy RL using frozen VLA features, RL token, actor-critic, and VLA reference actions. |
| Thinking in Text and Images | `pdfs/IVLRR.pdf` | IVLR: interleaved text/image reasoning traces for long-horizon VLA control. |
| Latent Policy Steering with Embodiment-Agnostic Pretrained World Models | `pdfs/LPS.pdf` | Planning/scoring wrapper over embodiment-specific target policy using pretrained world model/value. |
| OpenVLA | `pdfs/openVLA.pdf` | Open-source VLA baseline. |
| Octo: An Open-Source Generalist Robot Policy | `pdfs/Octo.pdf` | Open-source generalist robot policy trained on large multi-robot trajectory data. |
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | `pdfs/RT-2.pdf` | Tokenized-action VLA foundation model. |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | `pdfs/GRoot.pdf` | Humanoid VLA/foundation model with diffusion action module. |
| Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting | `pdfs/mirage_paper.pdf` | Cross-embodiment transfer by visual robot replacement/cross-painting. |
| Universal Actions for Enhanced Embodied Foundation Models | `pdfs/Universal action representation.pdf` | Universal action tokenization / cross-embodiment action representation. |
| Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking | `pdfs/2605.23733.pdf` | Cross-embodiment humanoid WBT transfer: kinematic alignment plus LoRA-style dynamics adaptation of pretrained whole-body trackers. Useful evidence for separating reusable motion/world priors from embodiment-specific action/dynamics adapters. |
| MEM | `pdfs/MEM.pdf` | Embodied/model-memory style engine paper; needs deeper read before relying on specifics. |
| CEI | `pdfs/CEI.pdf` | Engine/method paper; metadata title only, needs deeper read before use. |
| CLAP | `pdfs/CLAP.pdf` | Engine/method paper; metadata title only, needs deeper read before use. |
| On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning | `pdfs/on the fly vla adaption.pdf` | Test-time RL/adaptation method for VLAs. |
| Flow Policy Gradients for Robot Control | `pdfs/Flow Policy gradient.pdf` | RL/optimization for flow-policy robot control. |
| Flow Matching Policy Gradients | `pdfs/FLow amching policy gradient.pdf` | Flow matching + policy-gradient method. |
| Zero-shot World Models Are Developmentally Efficient Learners | `pdfs/zwm.pdf` | World-model learning reference. |

## JEPA / Latent World-Model Planning Papers

| Paper | Local file | Primary role |
| --- | --- | --- |
| V-JEPA 2 | `pdfs/V-JEPA-2.pdf` | Video JEPA representation/world-model line. |
| Demo-JEPA | `pdfs/Demo-JEPA.pdf` | Demonstration-conditioned JEPA robotics reference. |
| What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? | `pdfs/JEPA-WM-Physical-Planning.pdf` | JEPA rollout/scoring for physical planning; directly relevant to predicted-vs-actual trajectory alignment. |
| Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning | `pdfs/Joint-Embedding-vs-Reconstruction-SSL.pdf` | SSL theory paper comparing reconstruction and joint-embedding objectives. Adjacent support for the latent-choice thesis: reconstruction is attractive under low nuisance variation, while joint embedding has a better inductive bias when high-variance irrelevant factors are present. |
| LeWorldModel | `pdfs/LeWorldModel.pdf` | Latent world model reference. |
| Learning from Reward-Free Offline Data | `pdfs/Reward-Free-Latent-Dynamics-Planning.pdf` | Reward-free latent dynamics planning reference. |
| VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model | `pdfs/VLA-JEPA.pdf`; review: `docs/planning/vla-jepa-review.md` | JEPA-style VLA method with a frozen V-JEPA2 state encoder, Qwen3-VL latent action tokens, an autoregressive latent world model, and a separate DiT-B flow-matching action head. Uses SSV2 human videos plus DROID robot data for pretraining, then LIBERO/SimplerEnv/real robot fine-tuning. |

## Notes

- DROID is not currently downloaded as a local PDF. The arXiv page says the current version is v2, last revised 2025-04-22, with 76k demonstration trajectories / 350 hours across 564 scenes and 84 tasks.
- PolaRiS is not present as a local PDF in this repo under the current filename scan, even though we discussed it earlier from the arXiv link.
- The catalog is a working index. Several short-title PDFs (`CEI`, `CLAP`, `MEM`) need a deeper first-pass read before assigning them to a more precise experiment bucket.
