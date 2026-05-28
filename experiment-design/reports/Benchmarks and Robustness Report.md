---
type: research-report
topic: robotics-world-models
status: populated-draft
created: 2026-05-28
updated: 2026-05-28
---

# Benchmarks and Robustness Report

## Readout

The clean benchmark picture is misleading. Clean LIBERO is near saturation, so it mainly tells us whether a method is competent under a familiar simulated Franka contract. The useful signal comes from perturbations, out-of-domain task suites, embodiment shifts, and offline future-quality metrics.

The most important cross-paper conclusions are:

1. There is no single "best model." The winner changes with benchmark contract, fine-tuning data, robot embodiment, and checkpoint version.
2. Data scale and domain match explain many leaderboard jumps that look architectural at first glance.
3. WAM/video objectives help, but the papers often use them in different ways: representation learning, auxiliary action learning, action-conditioned simulation, or closed-loop planning.
4. Robustness benchmarks are closer to our question than clean success-rate tables, but our target is narrower: context honesty and action sensitivity.
5. For our experiments, PolaRiS and MolmoSpaces are more useful than clean LIBERO because they expose DROID-style camera/action/prompt brittleness.

## Benchmark Inventory

This is the benchmark/environment map found across the local paper set. PolaRiS and DROID are not present as standalone local PDFs, so those rows are sourced from local papers and setup notes that use or describe them.

| Benchmark / Env | Type | Robot / action-observation contract | Tasks / perturbations / metric | Papers / models using it |
| --- | --- | --- | --- | --- |
| LIBERO | Sim | MuJoCo/robosuite; Franka Panda; 2 cameras; 7D delta EEF OSC at 10 Hz | 40 tasks across Spatial/Object/Goal/Long; success | Cosmos, Fast-WAM, LingBot-VA, VLA-JEPA, MolmoAct2, OpenVLA, pi baselines |
| LIBERO-Plus | Sim robustness | Same LIBERO contract | 7 perturb axes: camera, robot init, language, lighting, background, noise, layout; success | Robustness paper: pi0, pi0.5, OpenVLA-OFT, UniVLA, X-VLA, VLA-JEPA, GE-Act, Cosmos, Fast-WAM |
| RoboTwin 2.0 | Sim | SAPIEN/ManiSkill3; Aloha-AgileX bimanual 14-DoF; 3 cameras; joint-position actions at 25-30 Hz | 50+ dual-arm tasks; clean and randomized; success | Fast-WAM, LingBot-VA, MOTUS, pi0/pi0.5/X-VLA baselines |
| RoboTwin 2.0-Plus | Sim robustness | Same RoboTwin contract | 50 tasks, 7 perturb dimensions, 21 subdimensions, 8 configs/task; success | LingBot-VA, Fast-WAM, MOTUS, pi0.5, X-VLA |
| RoboCasa | Sim | Franka kitchen; GR00T uses 3 RGB views, EEF/base/gripper state, relative EEF/gripper action | 24 kitchen tasks; unseen objects/scenes; success | Cosmos, GR00T |
| DexMG / DexMimicGen | Sim | Bimanual Panda jaw grippers, Panda dexterous hands, GR-1 humanoid dexterous hands | 9 bimanual dexterous tasks; success | GR00T |
| GR-1 Tabletop | Sim | GR-1 humanoid with Fourier dexterous hands; egocentric RGB; arm/hand/waist/neck state | 24 tabletop tasks; average success | GR00T |
| GR-1 Real-World | Real | GR-1 humanoid | Pick-place, articulated, industrial, coordination; partial scoring / success | GR00T |
| EWMBench | Offline WM benchmark | AgiBot-World-Beta multi-view videos, actions, instructions | 10 held-out tasks, 100 videos/task; SceneC, SA, TA, DYN, BLEU, logic, diversity | GE-Base, GE-Sim, Kling, Hailuo, Cosmos, OpenSora, LTX |
| AgiBot G1 | Real | Mobile dual-arm AgiBot G1 | GE: 5 tasks with SR/E2E. DreamZero: 500h across 22 environments with task progress/success | GE-Act, DreamZero, GR00T, pi0/pi0.5, UniVLA |
| Agilex Cobot Magic | Real | Agilex/Aloha-style bimanual teleop | Box folding and cloth folding; about 250 demos/task; success | GE-Act vs GR00T, pi0, UniVLA |
| Dual Franka | Real | Dual Franka platform | Cloth folding; about 250 teleop episodes; success | GE-Act vs GR00T, pi0, UniVLA |
| ALOHA | Real | Two ViperX 300S arms; 3 cameras; 14D proprio; 50-step chunks at 25 Hz | 4 bimanual tasks; ID/OOD; percent completion | Cosmos, Diffusion Policy, OpenVLA-OFT+, pi0, pi0.5 |
| DROID / offline DROID | Offline real robot dataset and deployment setup | Unified Franka; common DROID-style camera/action setup; MolmoAct2 uses 2 exterior + 1 wrist camera, absolute joint pose, 15 Hz chunks | Real-robot trajectories and language; filtered subsets, offline evaluation, real-style OOD evals | MolmoAct2-DROID, DreamZero-DROID, pi0.5-DROID, GR00T-DROID, OpenVLA DROID fine-tunes |
| PolaRiS | Real-to-sim / DROID-aligned closed-loop sim | DROID-style Franka; local setup notes mention 8D joint-position action client for DROID policies | DROID-FoodBussing, DROID-MoveLatteCup, DROID-TapeIntoContainer; local success/progress CSV | Our local pi0.5-DROID / MolmoAct2-DROID runs; target for DreamZero-DROID |
| SimplerEnv | Real-to-sim | Google Robot + WidowX | Google tasks and WidowX manipulation under visual variations; success | VLA-JEPA, LAPA, villa-X, UniVLA, RoboVLMs, GR00T, OpenVLA-OFT, pi0/pi0-FAST |
| MolmoSpaces / MolmoSpaces-Bench | Sim ecosystem | Franka FR3 DROID setup, RB-Y1 mobile manipulator, floating grippers; RGB/depth/proprio/task/action histories | 230k+ scenes; pick/open/close/pick-place/navigation; prompt, joint, lighting, camera occlusion perturbations; sim-to-real correlation | MolmoSpaces paper, pi0/pi0.5/CAP/RING/DualVLN, MolmoAct2-DROID |
| SOAR / BridgeV2 | Offline datasets / WM eval | BridgeV2 WidowX, RGB, 7-DoF EEF actions, language; SOAR success/failure episodes | CEM error, consensus success, Borda, OOD distractor/instruction success | Reconstruction-vs-semantics paper |
| BridgeData V2 WidowX | Real robot eval | WidowX robot, Bridge sink/tabletop environments | 17 OpenVLA tasks; success/partial success | OpenVLA, RT-2-X, Octo, RT-1-X |
| MolmoBot | Sim + real DROID-style eval | Original DROID setup replicated | Single-cycle pick-and-place; 15 real trials/task | MolmoAct2-DROID, pi0.5-DROID, LAP-VLA, X-VLA, MolmoBot |
| RoboEval | Sim | Bimanual Franka | 8 bimanual tasks; outcome + trajectory metrics CT/TL/JPL/CPL/CJ/JJ/collisions/slip | MolmoAct2, Diffusion Policy, GR00T N1.5, X-VLA, pi0.5-DROID |
| Language Table | Sim + qualitative real | 2D tabletop pushing; action as 2D setpoints encoded as text | Sim success | RT-2, BC-Zero, RT-1, LAVA |
| pi0.5 real/mock homes | Real | Mobile manipulators; 4 high-level cameras, wrist/front low-level cameras; 18-19 DoF state/action, 50 Hz chunks | Kitchen/bedroom cleanup; unseen mock/real homes; task progress and instruction following | pi0.5 and ablations |

## Shared Leaderboards

These are the tables where multiple papers or benchmark papers put several methods on a common-enough scale. They are still not always perfectly comparable because checkpoint version and fine-tuning protocol can change.

### Clean LIBERO

Clean LIBERO is useful for sanity checking, but it is too saturated to answer the planning/context-honesty question.

| Rank | Model | Spatial | Object | Goal | Long | Average | Source context |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | **98.5** | LingBot / Fast-WAM tables |
| 1 | Fast-WAM-Joint | 99.6 | 99.4 | 98.2 | 96.8 | **98.5** | Fast-WAM |
| 1 | Cosmos-Policy | 98.1 | 100.0 | 98.2 | 97.6 | **98.5** | Cosmos |
| 4 | MolmoAct2-Think | 100.0 | 99.2 | 100.0 | 93.2 | **98.1** | MolmoAct2 |
| 5 | Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | 97.6 | **98.0** | Fast-WAM |
| 6 | Motus | 96.8 | 99.8 | 96.6 | 97.6 | **97.7** | Fast-WAM table |
| 7 | Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | **97.6** | Fast-WAM |
| 8 | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | **97.2** | VLA-JEPA |
| 8 | MolmoAct2 | 99.0 | 98.0 | 98.2 | 93.6 | **97.2** | MolmoAct2 |
| 10 | OpenVLA-OFT | - | - | - | - | **97.1** | Cosmos / VLA-JEPA baseline tables |
| 11 | GR00T N1.7 | 99.6 | 97.4 | 99.0 | 92.0 | **97.0** | MolmoAct2 table |
| 12 | pi0.5 | 98.8 | 98.2 | 98.0 | 92.4 | **96.9** | Multiple tables |
| 13 | IVLR | 97.8 | 95.8 | 97.0 | 92.4 | **95.5** | IVLR |
| 14 | NORA | - | - | - | - | **94.5** | MolmoAct2 baseline table |
| 15 | pi0 | 96.8 | 98.8 | 95.8 | 85.2 | **94.1-94.2** | Fast-WAM / MolmoAct2 |
| 16 | CLAP-RF | 97.0 | 92.0 | 93.0 | 82.0 | **91.0** | CLAP |
| 17 | SmolVLA | - | - | - | - | **88.8** | CLAP baseline table |
| 18 | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | **76.5** | Fast-WAM / MolmoAct2 |
| 19 | TraceVLA | - | - | - | - | **74.8** | MolmoAct2 |
| 20 | Diffusion Policy | - | - | - | - | **72.4** | Cosmos baseline |
| 21 | LAPA | - | - | - | - | **65.7** | VLA-JEPA / CLAP baselines |

### LIBERO-Plus Robustness

LIBERO-Plus is the stronger table for robustness. It perturbs camera, robot initial state, language, lighting, background, noise, and layout.

| Rank | Model | Original LIBERO | Camera | Robot | Lang. | Light | BG | Noise | Layout | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | pi0.5 | 96.9 | 75.4 | 77.5 | 85.6 | 96.9 | 94.6 | 89.7 | 85.7 | **85.7** |
| 2 | Cosmos-Policy | 98.5 | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | **82.2** |
| 3 | ABot-M0 | 98.6 | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | **80.5** |
| 4 | GE-Act | 94.4 | 60.7 | 77.0 | 77.4 | 95.8 | 86.0 | 90.9 | 80.2 | **80.3** |
| 5 | VLA-JEPA | 97.2 | 64.2 | 67.7 | 88.1 | 91.8 | 93.4 | 65.8 | 83.9 | **77.9** |
| 6 | HoloBrain0-GD | 96.7 | 65.5 | 58.2 | 78.7 | 88.1 | 90.3 | 66.9 | 79.5 | **74.0** |
| 7 | X-VLA | 98.1 | 23.4 | 89.7 | 75.7 | 88.2 | 96.0 | 62.7 | 71.8 | **71.4** |
| 8 | pi0 rerun | 91.3 | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | **69.4** |
| 9 | RIPT-VLA | 97.5 | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | **68.4** |
| 10 | OpenVLA-OFT_m | 97.6 | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | **67.9** |
| 11 | pi0-FAST | 85.5 | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | **61.6** |
| 12 | pi0 | 94.2 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | **53.6** |
| 13 | Fast-WAM | 97.6 | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | **51.5** |
| 14 | UniVLA | 95.2 | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | **42.9** |

Readout: pi0.5, a direct VLA, wins this table. That is important because it means robustness is not reducible to "has a world model." It is more likely a mix of data scale, domain coverage, representation, and deployment alignment.

### Clean RoboTwin 2.0

| Rank | Model | Clean | Randomized | Average |
| --- | --- | ---: | ---: | ---: |
| 1 | LingBot-VA | 92.90 | 91.50 | **92.2** |
| 2 | Fast-WAM | 91.88 | 91.78 | **91.8** |
| 3 | Fast-WAM-IDM | 91.16 | 91.34 | **91.3** |
| 4 | Fast-WAM-Joint | 90.84 | 90.32 | **90.6** |
| 5 | Motus | 88.66 | 87.02 | **87.8** |
| 6 | Fast-WAM without video co-train | 82.76 | 84.80 | **83.8** |
| 7 | pi0.5 | 82.74 | 76.76 | **79.8** |
| 8 | Motus from Wan2.2 | 77.56 | 77.00 | **77.3** |
| 9 | pi0 | 65.92 | 58.40 | **62.2** |

### RoboTwin 2.0-Plus Robustness

| Rank | Model | Original RoboTwin | Camera | Robot | Lang. | Light | BG | Noise | Layout | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LingBot-VA | 92.1 | 28.9 | 36.2 | 87.3 | 89.0 | 91.3 | 80.9 | 87.9 | **74.2** |
| 2 | Fast-WAM | 91.2 | 30.4 | 53.2 | 86.7 | 88.8 | 90.0 | 76.4 | 83.2 | **72.7** |
| 3 | MOTUS | 87.0 | 21.6 | 85.0 | 83.2 | 84.6 | 84.4 | 43.1 | 82.8 | **71.5** |
| 4 | pi0.5 | 78.4 | 45.6 | 27.6 | 74.4 | 49.6 | 71.7 | 64.9 | 56.8 | **58.6** |
| 5 | X-VLA | 65.6 | 23.2 | 65.2 | 64.4 | 63.1 | 58.6 | 49.7 | 34.8 | **53.1** |

Readout: RoboTwin flips the story relative to LIBERO-Plus. LingBot-VA and Fast-WAM look strongest, likely because they are better matched to bimanual RoboTwin and saw randomized sim demonstrations.

### MolmoSpaces / MolmoBot / SO-100

These tables matter for our DROID-style question because they are much less saturated than LIBERO.

| Benchmark | Model | Score | Setting |
| --- | --- | ---: | --- |
| MolmoSpaces avg | StereoVLA | 7.0 | Zero-shot DROID-style checkpoint |
| MolmoSpaces avg | LAP-VLA | 22.2 | Zero-shot DROID-style checkpoint |
| MolmoSpaces avg | pi0-DROID | 23.2 | Zero-shot DROID checkpoint |
| MolmoSpaces avg | pi0.5-DROID | 34.5 | Zero-shot DROID checkpoint |
| MolmoSpaces avg | MolmoAct2-DROID | **37.7** | Zero-shot DROID checkpoint |
| MolmoBot real avg | pi0.5-DROID | 45.2 | 5 real DROID-style tasks, 15 trials/task |
| MolmoBot real avg | MolmoBot | 48.4 | Same |
| MolmoBot real avg | MolmoAct2-DROID | **87.1** | Same |
| SO-100 avg | SmolVLA | 2.3 | SO embodiment checkpoint |
| SO-100 avg | pi0-SO100/101 | 45.3 | SO embodiment checkpoint |
| SO-100 avg | MolmoAct2-SO100/101 | **56.7** | SO embodiment checkpoint |

MolmoSpaces also reports prompt and camera sensitivity: frequent DROID verbs narrow the pi0/pi0.5 gap to within about 1%, while other prompts show a larger gap; pi0.5 drops to 2% when the wrist camera is occluded and 20% when the third-person camera is occluded. This is exactly the kind of context-honesty stress test we should reuse.

### GR00T Native Benchmarks

GR00T is not in the main LIBERO-Plus or RoboTwin-Plus tables. It appears in its own humanoid/native benchmarks and in several later papers as a baseline.

| Benchmark | Setting | BC Transformer | Diffusion Policy | GR00T-N1-2B |
| --- | --- | ---: | ---: | ---: |
| RoboCasa | 100 demos/task | 26.3 | 25.6 | **32.1** |
| DexMG | 100 demos/task | 53.9 | 56.1 | **66.5** |
| GR-1 Tabletop | 100 demos/task | 16.1 | 32.7 | **50.0** |
| Average | 100 demos/task | 26.4 | 33.4 | **45.0** |

| Real GR-1 benchmark | Diffusion Policy 10% | Diffusion Policy full | GR00T 10% | GR00T full |
| --- | ---: | ---: | ---: | ---: |
| Pick-and-place | 3.0 | 36.0 | 35.0 | **82.0** |
| Articulated | 14.3 | 38.6 | 62.0 | **70.9** |
| Industrial | 6.7 | 61.0 | 31.0 | **70.0** |
| Coordination | 27.5 | 62.5 | 50.0 | **82.5** |
| Average | 10.2 | 46.4 | 42.6 | **76.8** |

GR00T is strong against BC/Diffusion Policy baselines in its native setup. That does not imply it beats pi0.5, Cosmos, LingBot, or MolmoAct2 on their reported benchmarks.

### Cosmos / ALOHA / RoboCasa

| Benchmark | Model | Score | Setting |
| --- | --- | ---: | --- |
| RoboCasa | GR00T-N1 | 49.6 | 24 kitchen tasks |
| RoboCasa | UVA | 50.0 | Same |
| RoboCasa | DP-VLA | 57.3 | Same |
| RoboCasa | GR00T + DreamGen | 57.6 | Same |
| RoboCasa | GR00T + DUST | 58.5 | Same |
| RoboCasa | UWM | 60.8 | Same |
| RoboCasa | pi0 | 62.5 | Same |
| RoboCasa | GR00T-N1.5 | 64.1 | Same |
| RoboCasa | Video Policy | 66.0 | Same |
| RoboCasa | FLARE / HAMLET | 66.4 | Same |
| RoboCasa | Cosmos-Policy | **67.1** | Same |
| ALOHA real avg | Diffusion Policy | 33.6 | 4 bimanual tasks |
| ALOHA real avg | OpenVLA-OFT+ | 62.0 | Same |
| ALOHA real avg | pi0 | 77.9 | Same |
| ALOHA real avg | pi0.5 | 88.6 | Same |
| ALOHA real avg | Cosmos-Policy | **93.6** | Same |

### DreamZero Benchmarks

DreamZero is not a standard LIBERO story in the local paper set. It is evaluated in its own AgiBot/DROID regimes.

| Setting | DreamZero | Baselines / comparison | Data / protocol |
| --- | ---: | --- | --- |
| AgiBot G1 seen tasks | 62.2% avg task progress | Best pretrained VLA 27.4% | Scratch and pretrained VLA baselines trained on same 500h AgiBot data |
| AgiBot G1 unseen tasks | 39.5% avg task progress | Pretrained VLA avg 16.3%; from-scratch VLAs <1% | Unseen task evaluation |
| DROID-Franka unseen tasks | 49% progress / 22.5% success | GR00T 31% / 12.5%; pi0.5 33% / 7.5% | DROID-style unseen verbs/tasks |
| AgiBot task-specific post-training | 90.5% avg task progress | pi0.5 pretrained 79.8%; pi0.5 scratch 53.3%; GR00T pretrained 9.8%; GR00T scratch 0.5% | Shirt folding 33h, fruit packing 12h, table bussing 40h, 50k steps |
| Cross-embodiment with 72 video-only trajectories | 38.3 +- 7.6% | +Human2Robot 54.3 +- 10.4%; +Robot2Robot 55.4 +- 9.5% | 8 demos/task, 10k co-training |

Readout: DreamZero's reported advantage is strongest where future/action modeling and domain-specific robot data matter. It is not directly comparable to clean LIBERO leaderboards.

### GE / EWMBench / GE-Sim

| Benchmark | Model | Score / result | What it tests |
| --- | --- | --- | --- |
| EWMBench aggregate | GE-Base | **4.7010** | Scene consistency, motion, semantic substeps, dynamics, logic/diversity |
| EWMBench aggregate | Kling | 3.8698 | Same |
| EWMBench aggregate | Hailuo | 3.4125 | Same |
| EWMBench aggregate | Cosmos | 3.2872 | Same |
| EWMBench aggregate | OpenSora | 3.1392 | Same |
| EWMBench aggregate | LTX | 2.9676 | Same |
| GE red-cylinder ablation | GE-only | E2E 0.81 with S / 0.49 without S; SR 0.64 / 0.26 | Whether generative environment pretraining helps action policy |
| GE red-cylinder ablation | GE + VidAda | E2E 0.89 with S / 0.37 without S; SR 0.76 / 0.37 | GE plus video adaptation |
| GE Agilex fine-grained tasks | GR00T N1 / UniVLA | Text reports 0% success | Cross-embodiment fine-grained deformable/folding tasks |
| GE Agilex / Dual Franka / RoboTwin | GE-Act | Text says GE-Act beats GR00T/pi0/UniVLA on the emphasized setups | Exact figure values were not in the text layer |

GE-Sim is closer than DreamZero to a clean action-counterfactual simulator because it explicitly conditions future video generation on actions. EWMBench itself is not a simulator; it is an offline benchmark of generated futures against held-out robot videos.

### VLA-JEPA / CLAP / LPS / IVLR

| Paper | Benchmark | Best reported result | Important ablation / contrast |
| --- | --- | ---: | --- |
| VLA-JEPA | LIBERO | 97.2 avg | w/o human videos 96.1 |
| VLA-JEPA | LIBERO-Plus | 79.5 avg in its paper; 77.9 in robustness paper | w/o human videos 62.9 |
| VLA-JEPA | SimplerEnv Google | 65.2 avg | w/o human videos 78.4, so human video is not uniformly helpful |
| VLA-JEPA | SimplerEnv WidowX | 57.3 avg | w/o human videos also 57.3 |
| CLAP | Astribot real tasks | CLAP-RF 61.0 avg | pi0.5 60.0, pi0 54.0, UniVLA 35.0 |
| CLAP | Astribot perturbations | CLAP-RF 66.7 | pi0.5 56.7, pi0 46.7, UniVLA 16.7 |
| CLAP | LIBERO | CLAP-RF 91.0 | lower than saturated pi0.5/MolmoAct2/Fast-WAM group |
| LPS | Real Franka 60/100 demos | LPS-mix 73.8 avg | BC 51.2, LPS* 62.5, HPT 13.8 |
| LPS | RoboMimic 50 demos | LPS-mix 63.4 avg | BC 57.3, LPS* 52.9 |
| IVLR | LIBERO | 95.5 avg | w/o trace 68.6, long 37.7, text-only 78.5, vision-only 83.5 |
| IVLR | SimplerEnv-WidowX | 59.4 overall | SpatialVLA 42.7, RoboVLMs 37.5, Octo-small 29.5 |

## Paper-Specific Score Ledger

This ledger is intentionally broad. It records what each paper itself reports, not a clean unified leaderboard.

| Paper | Benchmark / env | Models compared | Metric | Reported score(s) | Main caveat |
| --- | --- | --- | --- | --- | --- |
| pi0 | Real zero-shot robot tasks | pi0, pi0 parity, pi0-small, OpenVLA, Octo | Avg task progress | pi0 about 1.00/0.98/0.90/0.79/0.75 across shirt folding, bussing easy/hard, grocery bagging, toast; baselines mostly near 0 | Plot-read values; task/control regime is pi0-native |
| pi0 | Fine-tuning tasks | pi0 FT, pi0 scratch, DP, Octo, OpenVLA, ACT | Avg task progress | pi0 FT avg about 0.57/0.87/0.88 at 1/5/10h; scratch about 0.39/0.79/0.76; DP about 0.20/0.32/0.39 | Different task-specific data budgets |
| pi0.5 | Real/mock homes | pi0.5 | Task progress | Real homes mostly about 65-95%; mock env about 80-91% | Mobile-manipulation home setup, not LIBERO |
| pi0.5 | Mock-home scaling | pi0.5 with 3/12/22/53/82/104 locations | Avg progress | about 15/47/60/75/66/86%; in-domain no-pretrain about 39%; 104-location no-pretrain about 5% | Strong evidence for location diversity plus pretraining |
| pi0.5 | Prompt/language scaling | pi0.5 by location count | Follow / success | OOD success about 15/13/28/29/54 as locations scale | Text sensitivity improves with environment scale |
| pi0.6 | Real tasks | pi0.6 RECAP vs pretrain/offlineRL variants | Throughput / success | RECAP about 60 succ/hr laundry, 8.4 diverse laundry, 29 espresso, 13.2 box; success about 74-96% depending task | RL/post-training paper, not simple BC |
| pi0.7 | Cross-embodiment shirt folding | pi0.7(GC) vs humans | Progress / success | Humans 90.9% progress / 80.6% success; pi0.7 85.6% / 80.0% | Specific static-bimanual to UR5e transfer |
| OpenVLA | BridgeData V2 WidowX | RT-1-X, Octo, RT-2-X, OpenVLA | Success | Mean 18.5, 20.0, 50.6, 70.6 | OpenVLA-native benchmark |
| OpenVLA | Google robot | RT-1-X, Octo, RT-2-X, OpenVLA | Success | Mean 33.3, 26.7, 78.3, 85.0 | OpenVLA-native benchmark |
| OpenVLA | Franka Tabletop / DROID | DP, Octo, OpenVLA scratch, OpenVLA | Success | Tabletop OpenVLA 67.2; DROID OpenVLA 58.3 | Fine-tuned, not zero-shot |
| Octo | Zero-shot robots | RT-1-X, Octo, RT-2-X | Success | WidowX about 0.52 for Octo; UR5 about 0.70; RT-1 about 0.80 | Plot-read values |
| Octo | Fine-tuning | Scratch, VC-1, Octo | Success | Octo avg 72 vs scratch 20 and VC-1 15 | About 100 demos and 20 trials |
| GR00T | RoboCasa/DexMG/GR-1 Tabletop | BC, DP, GR00T | Success | GR00T avg 45.0 vs DP 33.4 vs BC 26.4 | Native GR00T setting |
| GR00T | Real GR-1 | DP, GR00T | Success | GR00T full avg 76.8 vs DP full 46.4 | Humanoid-native real benchmark |
| MolmoSpaces | MolmoSpaces-Bench | CAP, pi0, pi0-FAST, pi0.5, Paligemma, DualVLN, RING | Zero-shot success | Task-specific plotted values from about 3% to 84% depending model/task | Wide sim ecosystem; not a single unified VLA table |
| MolmoAct2 | MolmoSpaces | StereoVLA, LAP-VLA, pi0, pi0.5, MolmoAct2 | Success | Avg 7.0, 22.2, 23.2, 34.5, 37.7 | DROID checkpoints |
| MolmoAct2 | MolmoBot real | pi0.5, MolmoBot, MolmoAct2 | Success | Avg 45.2, 48.4, 87.1 | Real DROID-style tasks |
| MolmoAct2 | LIBERO | many baselines | Success | MolmoAct2-Think 98.1, MolmoAct2 97.2, GR00T N1.7 97.0, pi0.5 96.9 | Fine-tuned, saturated benchmark |
| DreamZero | AgiBot / DROID | DreamZero, GR00T, pi0.5 | Progress / success | DROID unseen DreamZero 49% / 22.5%, GR00T 31% / 12.5%, pi0.5 33% / 7.5% | DreamZero-native WAM setting |
| Cosmos | LIBERO / RoboCasa / ALOHA | Cosmos vs DP/VLA/GR00T/pi baselines | Success / completion | LIBERO 98.5, RoboCasa 67.1, ALOHA 93.6 | Action-free WM plus policy planning stack |
| Fast-WAM | LIBERO / RoboTwin | Fast-WAM variants, pi baselines, LingBot | Success | LIBERO 97.6, RoboTwin 91.8 | Strong when benchmark training distribution matches |
| LingBot-VA | LIBERO / RoboTwin | LingBot vs pi/MOTUS/X-VLA | Success | LIBERO 98.5, RoboTwin 92.2 | Large cross-embodiment robot data |
| VLA-JEPA | LIBERO / LIBERO-Plus / SimplerEnv | VLA-JEPA and baselines | Success | LIBERO 97.2, LIBERO-Plus 79.5, Google SimplerEnv 65.2, WidowX 57.3 | JEPA representation effect varies by benchmark |
| GE | EWMBench / real cross-embodiment | GE-Base, GE-Act, GE-Sim | WM metrics / success | EWMBench GE-Base 4.7010; GE reports GE-Act beating GR00T/pi0/UniVLA in emphasized real setups | Some real score figures are image-only |
| CLAP | Astribot / LIBERO | CLAP vs pi0.5/pi0/UniVLA | Success | Real CLAP-RF 61.0; perturb CLAP-RF 66.7; LIBERO 91.0 | Smaller real setup, not LIBERO-saturated |
| LPS | Franka / RoboMimic | BC, LPS variants | Success | Real Franka 60/100 demos LPS-mix 73.8 vs BC 51.2 | Planner uses embodiment-specific policy |
| IVLR | LIBERO / SimplerEnv | IVLR variants and baselines | Success | LIBERO 95.5; SimplerEnv-WidowX 59.4 | Latent trace planning, not action-conditioned sim |

## Contradiction And Reconciliation Table

These are the places where a model appears to have incompatible scores across papers. Most are not true contradictions; they are benchmark, checkpoint, or protocol changes.

| Model / claim | Paper A says | Paper B says | Apparent contradiction | Reconciliation / likely cause | What to record for our experiments |
| --- | --- | --- | --- | --- | --- |
| GR00T | GR00T paper: GR00T full gets 76.8 avg on real GR-1 and 45.0 avg on RoboCasa/DexMG/GR-1 sim, beating DP/BC | GE paper: GR00T N1 reportedly gets 0% on Agilex fine-grained folding tasks; DreamZero DROID unseen reports GR00T 31% progress / 12.5% success | GR00T looks strong in one place and weak in another | Different embodiment, action head, task type, fine-tuning protocol, and model version | Do not treat GR00T as a universal baseline number. Specify N1/N1.5/N1.6/N1.7, robot, task, action space, and demos |
| GR00T on LIBERO | VLA-JEPA cites GR00T N1 at 93.9 avg LIBERO | MolmoAct2 cites GR00T N1.7 at 97.0 avg LIBERO | Same model family has different LIBERO scores | Version/checkpoint drift and likely evaluation protocol drift | If comparing to GR00T, use exact checkpoint label and paper source |
| pi0 clean vs robust LIBERO | Clean LIBERO tables: pi0 about 94.1-94.2 avg | LIBERO-Plus: pi0 total 53.6; pi0 rerun original 91.3 and total 69.4 | Strong clean model collapses under perturbations | Clean success hides camera/robot/language/layout brittleness; rerun protocol/checkpoint also changes original score | Clean success is insufficient. Always include perturbation table |
| pi0.5 clean vs robust LIBERO | Clean LIBERO: pi0.5 96.9 | LIBERO-Plus: pi0.5 85.7 total | It drops, but remains best | Broad data helps robustness, but perturbations still bite | pi0.5 is a strong robustness baseline, not just a clean-success baseline |
| pi0.5 RoboTwin | Clean/randomized RoboTwin: pi0.5 79.8 avg | RoboTwin-Plus: pi0.5 original 78.4, total 58.6 | Similar original, much lower total | Plus perturbations expose bimanual/action/camera sensitivity | Include perturbation axes; do not only compare original success |
| pi0.5 across papers | pi0.5 real/mock homes can be around 65-95% progress; LIBERO 96.9; MolmoSpaces-DROID 34.5; local PolaRiS FoodBussing 0/4 success | Same model family appears from excellent to failing | Looks inconsistent | Different checkpoint, deployment adapter, environment, prompt/action contract, and task distribution | Our local failures are evidence about deployment/domain alignment, not a contradiction of pi0.5 paper claims |
| pi0 | pi0 paper real zero-shot progress is high on its selected real tasks | MolmoSpaces shows lower zero-shot manipulation scores; LIBERO tables show 94.2 | High and low scores coexist | Different robot/task/control setups and often different fine-tuning | Separate "pi0-native", "LIBERO fine-tune", and "DROID-style zero-shot" |
| OpenVLA | OpenVLA paper: BridgeData 70.6 and Google robot 85.0 | pi0 paper: OpenVLA near zero on pi0's dexterous/bimanual real tasks; clean LIBERO OpenVLA 76.5 | OpenVLA is both strong and weak | Native benchmark vs out-of-distribution dexterous tasks; action/control mismatch | OpenVLA is sensitive to control contract and fine-tune regime |
| OpenVLA variants | Clean LIBERO OpenVLA 76.5; OpenVLA-OFT 97.1; OpenVLA-OFT_m LIBERO-Plus 67.9; ALOHA OpenVLA-OFT+ 62.0 | Variant names are close but scores differ a lot | Same base family, different adaptation stacks | OFT / OFT_m / OFT+ are not interchangeable | Use full variant name in all tables |
| Octo | Octo paper reports strong zero-shot/fine-tuning gains on its aligned robot settings | OpenVLA paper reports Octo 20.0 BridgeData and 26.7 Google robot in harder generalization tests | Octo looks strong or weak depending paper | Different benchmark difficulty and evaluation alignment | Treat Octo as a pretraining-aligned generalist baseline, not universal |
| Fast-WAM clean vs robust LIBERO | Clean LIBERO: Fast-WAM 97.6 | LIBERO-Plus: Fast-WAM 51.5 | WAM objective does not save robustness | Its LIBERO checkpoint was trained on clean demos; no randomized perturbation coverage | World/video training is not enough without matching perturbation distribution |
| Fast-WAM RoboTwin vs LIBERO-Plus | RoboTwin-Plus: Fast-WAM 72.7, close to LingBot | LIBERO-Plus: Fast-WAM 51.5 | Same method robust in one benchmark, weak in another | RoboTwin checkpoint saw randomized demonstrations; LIBERO checkpoint did not | Use Fast-WAM as evidence that data distribution can dominate architecture |
| LingBot clean vs robust RoboTwin | Clean RoboTwin 92.2 and clean LIBERO 98.5 | RoboTwin-Plus total 74.2 | Strong model still drops under perturbations | Bimanual robustness remains hard even with large robot data | Report both original and perturbed scores |
| Cosmos | Clean LIBERO 98.5, RoboCasa 67.1, ALOHA 93.6, LIBERO-Plus 82.2 | Scores vary by benchmark | Not a contradiction | Different metrics: clean success, kitchen sim success, real completion, perturbation robustness | Cosmos is a strong WAM/policy stack, but not a universal score |
| VLA-JEPA | VLA-JEPA paper: LIBERO-Plus avg 79.5 | Robustness paper: VLA-JEPA total 77.9 | Small mismatch | Rerun/protocol/table rounding | Difference is small enough to treat as same range |
| VLA-JEPA human-video ablation | VLA-JEPA w/o human videos drops on LIBERO-Plus from 79.5 to 62.9 | On SimplerEnv Google, w/o human is 78.4 vs full 65.2 | Human video helps and hurts | Human ego video improves robustness in one setting but can shift representation away from a specific robot-sim distribution | Representation data must match the evaluation domain |
| MolmoAct2 vs pi0.5 | MolmoAct2 beats pi0.5 on MolmoSpaces and MolmoBot | Clean LIBERO has pi0.5 and MolmoAct2 both saturated around 97 | Strong advantage disappears on LIBERO | LIBERO is saturated; DROID-style benchmarks expose the difference | Use MolmoSpaces/PolaRiS for our main Molmo/pi0.5 comparison |
| MolmoAct2 local vs paper | Paper: MolmoAct2-DROID 37.7 MolmoSpaces avg and 87.1 MolmoBot real avg | Local PolaRiS MolmoAct2-DROID run had 0/4 success, 0.1667 progress | Local run looks much worse | Local adapter/task/sample size/setup likely mismatched; one task is not a benchmark | Debug exact model input frames, cameras, prompt, action scaling before interpreting failure |
| DreamZero vs direct VLAs | DreamZero DROID unseen beats pi0.5/GR00T; AgiBot unseen beats pretrained VLA baseline | LIBERO robustness has pi0.5 above WAMs like Cosmos/GE/Fast-WAM | World-action models are not always leaderboard winners | Different benchmarks and different mechanisms; DreamZero's strength is in its own future/action regime, not clean LIBERO | Evaluate DreamZero where its environment/checkpoint is native: DROID/PolaRiS/AgiBot-style |
| GE-Sim / Cosmos vs DreamZero | GE-Sim and Cosmos are cleaner action-conditioned future simulators | DreamZero jointly generates future and action and is heavier to run | "World model" means different things | Action-conditioned counterfactuals are cleaner in GE-Sim/Cosmos; DreamZero is better for prediction/action consistency in its native stack | Pick model by diagnostic: counterfactual action faithfulness vs predicted-action honesty |
| EWMBench vs closed-loop success | GE-Base wins EWMBench aggregate | EWMBench does not execute a policy in a simulator | Future quality is not rollout success | It is an offline future-generation benchmark, not a closed-loop environment | Borrow metrics, but do not claim sim success from EWMBench alone |
| LIBERO vs MolmoSpaces | Clean LIBERO gives many 95-98+ scores | MolmoSpaces gives pi0.5-DROID 34.5 and MolmoAct2-DROID 37.7 | Benchmark difficulty is radically different | LIBERO is benchmark-specific fine-tuning and saturated; MolmoSpaces is broader DROID-style zero-shot stress | Use both: LIBERO for sanity, MolmoSpaces/PolaRiS for useful stress |
| Local PolaRiS vs paper tables | Paper tables show strong pi0.5/MolmoAct2 in some settings | Local FoodBussing runs had repeated mistakes | Looks like underperformance | Could be image viewpoint, prompt/action mismatch, task out-of-domain, or simulation adapter issue | Dump exact 224x224 model frames and compare to DROID/OpenPI docs before drawing model conclusions |

## GR00T Cross-Paper Placement

The answer to "do pi/Molmo/Cosmos test with GR00T?" is: not consistently, but several papers do include it.

| Paper | Does it compare to GR00T? | What it says |
| --- | --- | --- |
| GR00T | Native paper | Compares GR00T mainly to BC Transformer and Diffusion Policy, not to pi0.5/Cosmos/LingBot |
| Cosmos | Yes, on RoboCasa | Cosmos 67.1 vs GR00T-N1 49.6 and GR00T-N1.5 64.1 |
| DreamZero | Yes, on DROID/AgiBot-style settings | DROID unseen: DreamZero 49% progress / 22.5% success vs GR00T 31% / 12.5%; task-specific post-training: DreamZero 90.5% vs GR00T pretrained 9.8% |
| GE | Yes, in cross-embodiment settings | GE reports GR00T N1 at 0% on Agilex fine-grained tasks and worse than GE-Act on emphasized real setups |
| VLA-JEPA | Yes, as baseline in tables | LIBERO GR00T N1 93.9; Google SimplerEnv GR00T 1.4 |
| MolmoAct2 | Yes, on LIBERO | GR00T N1.7 97.0 avg, close to pi0.5 96.9 and MolmoAct2 97.2 |
| pi0 / pi0.5 | Not as a central local comparison | The local extracted pi papers mainly compare pi variants to OpenVLA, Octo, DP/ACT, and internal ablations |

So: GR00T is present in the newer comparison ecosystem, but not as a uniform baseline across all papers. The most honest statement is that GR00T is strong on its native humanoid setup, competitive on saturated LIBERO variants when adapted, but can fail badly under specific cross-embodiment fine-grained manipulation protocols.

## Robustness Axes To Reuse

| Axis | Benchmark source | Our version |
| --- | --- | --- |
| Camera viewpoint | LIBERO-Plus, RoboTwin-Plus, PolaRiS setup failures, MolmoSpaces camera occlusion | Save exact model input frames; perturb external/wrist views; compare action deltas and success |
| Robot initial state / proprio | LIBERO-Plus, RoboTwin-Plus | Change or mask proprio where adapter permits; see whether action compensates |
| Language paraphrase | LIBERO-Plus | Same task, paraphrased prompt; action should remain stable |
| Wrong-object prompt | MolmoSpaces prompt sensitivity, our central question | Same visual state, absent/wrong target prompt; faithful model should not blindly execute original task |
| Visual masking/cropping | MolmoSpaces camera occlusion, robustness paper | Mask target, distractor, wrist, or gripper; action should degrade in interpretable ways |
| Background/lighting/noise | LIBERO-Plus, RoboTwin-Plus | Useful secondary perturbations; less central than object/prompt/proprio conflicts |
| Object layout/distractors | LIBERO-Plus, MolmoSpaces | Add distractors or move target; test grounding |
| Action counterfactual | GE-Sim, Cosmos-style models | For action-conditioned WMs, change action and see if future follows the action |
| Prediction mismatch | DreamZero, EWMBench, reconstruction-vs-semantics | Compare predicted future to observed future with semantic latent metrics, not only pixels |

## What This Means For Our Experiment

The benchmark literature does not make our experiment obsolete. It sharpens it.

The robustness papers ask:

```text
Across benchmark perturbations, do model families generalize better than each other?
```

Our sharper question is:

```text
When the model receives changed or contradictory context, does its action and/or predicted future change in the expected way?
```

That means our first benchmark suite should not be a generic leaderboard. It should be a diagnostic matrix:

| Diagnostic | Model families | Environment | Output |
| --- | --- | --- | --- |
| Input-frame sanity | pi0.5-DROID, MolmoAct2-DROID, DreamZero-DROID if runnable | DROID samples + PolaRiS | Exact 224x224 frames, prompts, proprio, action scale |
| Clean rollout | pi0.5, MolmoAct2, DreamZero | PolaRiS tasks and MolmoSpaces tasks | Success/progress, rollout video, per-step action |
| Prompt counterfactual | pi0.5, MolmoAct2 | PolaRiS/MolmoSpaces | Action change under paraphrase, wrong object, absent object |
| Image counterfactual | pi0.5, MolmoAct2 | PolaRiS/MolmoSpaces | Action change under crop/mask/viewpoint perturbation |
| Prediction honesty | DreamZero, GE-Sim/Cosmos if available | Native supported setup first | Predicted-vs-observed semantic mismatch, plus whether mismatch predicts failure |
| Action counterfactual future | GE-Sim/Cosmos-style model first | Offline/closed-loop supported setup | Does generated future obey changed action? |

## Claims To Make Carefully

We can say:

- clean LIBERO is saturated;
- robustness perturbations change model rankings;
- broad robot data and domain match are major drivers;
- future prediction can be useful as a training signal, diagnostic signal, or planner, but these are different mechanisms;
- semantic/action-relevant latent metrics are more aligned with our question than pixel reconstruction.

We should not yet say:

- WAMs are always better planners;
- DreamZero is the cleanest action-counterfactual simulator;
- pi0.5 or MolmoAct2 ignore language;
- semantic mismatch will necessarily improve control;
- a local PolaRiS failure is model failure before input/action contract checks.

## Next Benchmark Artifacts To Build

1. A PolaRiS/MolmoSpaces run table with columns: model, checkpoint, task, prompt, camera contract, action contract, success, progress, video path, model input frame path.
2. A perturbation table with columns: base prompt/image/proprio, perturbation, action delta norm, semantic action direction, rollout outcome.
3. A future-honesty table with columns: predicted frame/video, observed frame/video, pixel metric, DINO/V-JEPA/SigLIP metric, substep completion, failure timing.
4. A contradiction-aware citation table that always records benchmark, checkpoint, data budget, robot embodiment, and whether the score is clean, robust, zero-shot, or fine-tuned.
