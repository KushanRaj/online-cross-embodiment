---
type: research-record
topic: low-level-controller-data
status: first-pass-audit
created: 2026-06-10
updated: 2026-06-10
---

# Low-Level Controller Data Requirements

## Retrieval Key

If the question is "how much data did these papers need to make the low-level controller work?", answer from this note first.

The scoped question is:

> How much high-quality robot observation-action data did each method use to make the low-level controller work on a target embodiment/environment, and how well did that controller perform?

Do not broaden this into global-reasoner pretraining unless the paper does not cleanly separate the controller from the rest of the model. The target unit is high-cost robot action data: teleop demos, robot trajectories, successful task demos, online robot episodes, or action-labeled simulator episodes when the benchmark controller is simulator-trained.

Preferred normalization:

- Exact reported control timesteps, if available.
- Otherwise `hours * control_hz * 3600`.
- Otherwise demos/episodes only, with a caveat that average trajectory length is missing.

## First-Pass Table

| Model | What Counts As Low-Level Controller Training | Data Used For That Controller | Normalized Control Steps | Reported Performance |
|---|---:|---:|---:|---|
| `pi0` | Flow-matching action expert, about 300M params | More than 10,000h robot data, 903M proprietary timesteps, 68 tasks, 7 robot configs | Exact: `903M` robot timesteps | Strong after task fine-tuning; task data ranges from about 5h for easy tasks to 100h+ for hard tasks |
| `pi0` task post-train | Same action expert adapted to task | About 5h to 100h+ per task | At 20-50Hz: `0.36M-18M+` steps | Enables complex manipulation like laundry, packing, table bussing |
| `pi0.5` | FAST/action-token pretrain plus continuous action expert | About 400h mobile manipulation plus multi-env tabletop and cross-embodiment data; exact split undisclosed | If 50Hz: about `72M` steps for the 400h mobile-manipulation slice | LIBERO avg `96.9`; MolmoSpaces DROID avg `34.5` |
| `MolmoAct2-DROID` | Continuous action expert fine-tuned on DROID | 74,604 successful DROID episodes, 17,758,044 frames/actions | About `17.8M` steps, about 329h at 15Hz | MolmoSpaces avg `37.7`; MolmoBot avg `87.1`; held-out sim avg `20.6` |
| `MolmoAct2-YAM` | Same action expert on YAM | 34.5k demos, 720h | At 30Hz: `77.8M` steps | Strong YAM checkpoint; exact task table should be checked in the figure if needed |
| `MolmoAct2-SO` | Same action expert on SO100/101 | 38,059 episodes, 184h, 19.8M frames | About `19.8M` steps | SO benchmark avg `56.7` |
| `DreamZero` | Joint video-action WAM; not a cleanly separable action expert | AgiBot: 500h, 7.2k episodes; DROID trained separately | AgiBot at 30Hz: about `54M` steps | AgiBot seen-task progress about `62.2`; DROID success `22.5`, progress `49` |
| `DreamZero` task post-train | Same WAM adapted to target tasks | Shirt folding 33h, fruit packing 12h, table bussing 40h | At 30Hz: `1.3M-4.3M` steps/task | Post-training reaches high-success regime; DreamZero strongest on fruit packing |
| `DreamZero` embodiment transfer | Few-shot new robot adaptation | 55 trajectories, about 30min YAM data | At 30Hz: about `54k` steps | Shows few-shot embodiment transfer; exact success table needs figure pass |
| `RLT / pi0.6` | RL token representation plus RL policy on top of pretrained VLA | 1-10h task teleop demos, then 400-1000 online episodes | Demos: `180k-1.8M`; online: `45k-900k`; total roughly `225k-2.7M` steps/task | Screw insertion improves roughly `20% -> 65`; throughput/speed improves up to about 3x |
| `Cosmos Policy LIBERO` | Full Cosmos policy fine-tune, action chunk 16 | 2,000 LIBERO demos, 50 demos/task | Step count not stated; demos only | LIBERO avg `98.5` |
| `Cosmos Policy RoboCasa` | Same policy fine-tune, action chunk 32 | 1,200 human demos, 50 demos/task across 24 tasks | Step count not stated | RoboCasa avg `67.1` |
| `Cosmos Policy ALOHA` | Same policy fine-tune, action chunk 50 | 185 real demos across 4 tasks | At 25Hz but average demo length unstated | Existing note reports about `93.6` avg; planning improves hard tasks |
| `Fast-WAM LIBERO` | About 1B action expert inside 6B WAM | 2,000 LIBERO demos | Step count not stated | LIBERO avg `97.6`; joint/IDM variants `98.0-98.5` |
| `Fast-WAM RoboTwin` | Same | 2,500 clean plus 25,000 randomized demos | Step count not stated | RoboTwin avg about `91.8` |
| `Fast-WAM real towel` | Same | 60h teleop | If 30Hz: about `6.5M` steps | No-video-co-train drops to `10%`; exact main score needs figure pass |
| `LingBot-VA` | Video-action model with action stream/decoder | About 16k hours robot manipulation pretraining | If 50Hz equivalent: up to about `2.9B` steps, but mixed Hz | LIBERO avg `98.5`; RoboTwin about `91-93`; real tasks improve over pi0.5 with 50 demos |
| `LingBot-VA adaptation` | Target robot/task fine-tune | As few as 50 demos | Step count not stated | Real-world tasks show more than 20% improvement over pi0.5 in several settings |
| `GR00T N1` | Flow-matching action policy | 3,288.8h real robot data, 262.3M frames; plus sim/human/neural data | Real robot: `262.3M` frames | RoboCasa 100-demo avg about `32.1`; 300-demo avg about `49.6`; real tasks use 15min-3h/task |
| `OpenVLA` | Autoregressive discretized action-token policy | About 970k OXE robot trajectories | Step count/hours not reported | LIBERO baseline avg `76.5`; OpenVLA-OFT reported about `97.1` |
| `LPS` | Embodiment-specific BC policy; LPS plans over it | 30-100 target demos depending task | Step count not stated | Real-world LPS-mix improves over BC by about 44-70% relative |

## Main Readout

The lowest explicit successful adaptation regimes are:

- `DreamZero` few-shot embodiment adaptation: about 30min / 55 trajectories.
- `RLT`: 1-10h task demos plus 400-1000 online episodes.
- `LingBot-VA`: claims useful adaptation with about 50 demos.
- `LPS`: 30-100 target demos, but the policy remains embodiment-specific and LPS plans over that policy rather than replacing it.

The stronger large systems still rely on tens to hundreds of millions of robot action steps to build the base controller prior. This is the feasibility tension for cross-embodiment work: small adaptation can work after a large controller prior, but the literature rarely shows genuinely cheap controller formation from scratch.

## Caveats

- Some papers report demos but not average trajectory length or control frequency; those rows cannot honestly be converted to control steps.
- Some models do not separate "global reasoner" and "low-level controller." For those rows, the controller data is the robot action data used to train the policy or joint WAM.
- Simulator demos are included only when the reported benchmark controller is trained in sim, such as LIBERO or RoboCasa.
- Exact figure-only performance values should be checked against the PDF figures before citing externally.
