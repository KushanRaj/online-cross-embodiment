---
type: research-report
topic: robotics-world-models
status: populated-draft
created: 2026-05-28
updated: 2026-05-28
---

# Public Review and Forum Insights Report

## Readout

The public discussion is less about abstract architecture categories and more about deployment interface correctness:

- camera extrinsics and image preprocessing,
- action normalization and action-token decoding,
- control frequency,
- state/proprio conventions,
- dataset format and sharding,
- simulator wrappers,
- contact-rich final execution.

That is directly useful for our thesis. Practitioners repeatedly see policies that look semantically competent but fail in action grounding, timing, or contact. This supports the framing:

```text
the model may reason well enough, but actuator/context conditioning is weak.
```

It does not prove that WAMs are better planners. It does show that our context-faithfulness and prediction-honesty tests are aimed at a real failure surface.

## Cross-Paper Signals

| Signal | Evidence source | What it means for us |
| --- | --- | --- |
| Deployment interface pain dominates. | OpenPI, OpenVLA, DreamZero, Cosmos repos/issues. | Our experiments must log exact model inputs, action scales, camera views, state vectors, and control frequency. Otherwise failures will be uninterpretable. |
| WAMs may be robust but not uniformly better. | Robustness-paper discussion and notes. | We should not claim "WAM = planner". Need mechanism checks: data, latent objective, future prediction, action conditioning. |
| Contact-rich precision remains hard. | Open Robotics pi0.5 reports; RLT framing. | Last-centimeter failures are a good motivation for future RL-token/residual/adaptation work. |
| Semantic competence can coexist with poor action grounding. | Public pi0.5/OpenVLA/OpenPI deployment reports. | Prompt/image/proprio sensitivity should be measured directly, not inferred from success rate. |

## Paper / Model Findings

| Paper / model | Public discussion status | Useful links | Extracted insight |
| --- | --- | --- | --- |
| DreamZero | Workshop/OpenReview page exists; GitHub issues active; Reddit mostly high-level. | [OpenReview](https://openreview.net/forum?id=cd33uUB609), [GitHub](https://github.com/dreamzero0/dreamzero), [GitHub issues](https://github.com/dreamzero0/dreamzero/issues) | Issues around 1-step inference logic, PolaRiS video generation, closed-loop benchmark support, state-token mismatch, YAM data release, and fine-tuning schedule are exactly the world-prediction-vs-rollout-reality surface we care about. |
| OpenVLA | CoRL OpenReview/reviews and active GitHub issues. | [OpenReview](https://openreview.net/forum?id=ZMnD6QZAE6), [GitHub issues](https://github.com/openvla/openvla/issues), [README](https://github.com/openvla/openvla) | Reviewers wanted more sim/fair-comparison coverage and more varied robot setups. Practical pain includes low LIBERO success, LoRA mode collapse, action-tokenizer questions, dataset sharding, and low SimplerEnv success. README warns out-of-box works mainly on training domains and recommends target demos, 5-10Hz data, and avoiding idle actions. |
| pi0.5 / OpenPI | Hacker News, GitHub, Open Robotics, Reddit. | [HN thread](https://news.ycombinator.com/item?id=43764439), [OpenPI issues](https://github.com/Physical-Intelligence/openpi/issues), [Open Robotics thread](https://discourse.openrobotics.org/t/my-first-results-pi0-5-vla-policy/53670) | Practitioner reports include reaching near cable ports but failing insertion. One estimate was about 100 episodes per prompt because ambiguous identical connectors break grounding. This is a last-millimeter grounding failure, not just a high-level semantic failure. |
| pi0.7 / RLT | OpenPI issues, RLT notes, Reddit. | [OpenPI issues](https://github.com/Physical-Intelligence/openpi/issues), [RLT notes](https://davidlxu.github.io/posts/2026/03/rlt-paper-notes/), [RLT Reddit](https://www.reddit.com/r/robotics/comments/1ryt963/physical_intelligence_developed_an_rl_method_for/) | RLT is publicly framed as solving failures where a broad VLA gets close but cannot finish contact-critical tasks. It freezes the VLA and trains a small actor-critic around an RL token/reference action. |
| Cosmos Policy | ICLR OpenReview and official GitHub. | [OpenReview](https://openreview.net/forum?id=wPEIStHxYH), [GitHub](https://github.com/NVlabs/cosmos-policy), [NVIDIA/HF blog](https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control) | Inference can be practical, but training is large: public docs indicate 8x80GB H100 for ALOHA, 32x80GB for RoboCasa, 64x80GB for LIBERO, about 48h. This is a reference architecture, not a one-L40 training target. |
| Fast-WAM | HF paper/model/GitHub; little external review found. | [HF paper](https://huggingface.co/papers/2603.16666), [HF checkpoints](https://huggingface.co/yuanty/fastwam), [GitHub](https://github.com/yuantianyuan01/FastWAM) | The public artifact reinforces the negative-control question: maybe WAM gains come from video modeling during training, not test-time future imagination. |
| Genie Envisioner / GE | OpenReview page and GitHub. | [OpenReview](https://openreview.net/forum?id=fHLtSxDFKC), [GitHub](https://github.com/AgibotTech/Genie-Envisioner), [arXiv](https://arxiv.org/abs/2508.05635) | Reproducibility is limited by release status. GitHub still had TODOs around inference/training code and weights in the public pass. Treat GE as conceptual/evaluation reference unless release matures. |
| MolmoAct2 / MolmoSpaces | Limited independent discussion found. | [MolmoAct2 secondary review](https://www.themoonlight.io/review/molmoact2-action-reasoning-models-for-real-world-deployment), [MolmoSpaces blog](https://allenai.org/blog/molmospaces) | Public material emphasizes deployability and open action reasoning, but there is not much independent setup criticism yet. Our own rollout logs matter more. |
| VLA-JEPA | GitHub/arXiv found; little external review found. | [GitHub](https://github.com/ginwind/VLA-JEPA), [arXiv](https://arxiv.org/abs/2602.10098) | The repo explicitly builds on starVLA and V-JEPA2, making it a practical candidate for the semantic-latent future-prediction thread. |
| WAM robustness paper | Public notes/blog summaries found. | [arXiv](https://arxiv.org/abs/2603.22078), [Wispaper](https://www.wispaper.ai/en/blog/world-action-models-generalize-better-than-vlas-20260324/eng), [Lixin Xu notes](https://davidlxu.github.io/posts/2026/03/wam-vla-robustness-paper-notes/), [SwiftScholar](https://www.swiftscholar.net/paper/69d44b615127ff45a9dee58a) | External critique supports "WAMs are a useful robustness prior", not "WAMs always generalize better." Notes flag non-matched data pipelines, WAM latency, limited geometric robustness, DreamZero exclusion, and background sensitivity. |
| Reconstruction vs Semantics | No substantive public review/forum discussion found in this pass. | Local paper/source note. | Treat the paper itself as primary evidence. |
| Any2Any | arXiv and summary pages found; limited external discussion. | [arXiv](https://arxiv.org/abs/2605.23733), [alphaXiv](https://www.alphaxiv.org/audio/2605.23733) | External framing matches our future-use interpretation: split reusable motion/world prior from embodiment-specific kinematic/dynamics adaptation. |
| PolaRiS | Project/PDF and secondary summaries found. | [PolaRiS PDF](https://polaris-evals.github.io/PolaRiS.pdf), [EmergentMind summary](https://www.emergentmind.com/topics/policy-evaluation-and-environment-reconstruction-in-simulation-polaris) | Public framing is scalable real-to-sim DROID-style policy evaluation. For us, PolaRiS is an environment contract more than a model paper. |

## Evidence Versus Speculation

### Evidence

- OpenVLA reviewers explicitly asked for more sim/fair-comparison coverage and more ablations.
- OpenVLA/OpenPI/DreamZero GitHub issues expose real reproducibility pain around action/token logic, normalization, inference, LIBERO/PolaRiS support, camera calibration, and state handling.
- Public robustness commentary says WAMs are robust but slow, and not uniformly better under geometry shifts.
- Practitioner pi0.5 reports show semantic reaching can work while final insertion fails.

### Speculation

- "WAMs are more honest planners" is not proven publicly. The stronger evidenced claim is that WAMs often produce more robust behavior under visual perturbations, but the mechanism may be video pretraining/data diversity rather than explicit inference-time planning.
- "pi0.5/Molmo ignore language" is not proven. The planned prompt-swap and wrong-object tests are designed to test this directly.
- "Semantic latent mismatch improves control" is not proven. It is a good diagnostic hypothesis supported by reconstruction-vs-semantics, VLA-JEPA, and CLAP, but it still needs our experiment.

## Implications For Our Experimental Setup

1. Log everything that touches the policy contract: raw camera frames, resized/cropped model inputs, prompt string, proprio vector, action vector, action scaling, control rate, model checkpoint, environment task ID, seed, and reset state.
2. Separate semantic failures from actuator failures. A model can select the right object and still fail gripper timing or insertion.
3. Use public setup pain as a checklist before interpreting model quality. A bad camera crop can masquerade as "poor reasoning."
4. Treat released checkpoints carefully. PolaRiS co-trained checkpoints are not the same as raw base pi0.5-DROID; the distinction must stay in tables and filenames.
5. Keep forum claims out of core evidence unless they motivate a test we actually run.

## Open Follow-Ups

- Re-run this public mining later for MolmoAct2/MolmoSpaces as independent users start posting setup reports.
- If GE releases full code/weights, revisit GE-Sim as an action-conditioned counterfactual model.
- Mine specific GitHub issues into a reproducibility checklist if we start adapting OpenVLA, DreamZero, or VLA-JEPA locally.
