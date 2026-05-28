---
type: research-report
topic: robotics-world-models
status: populated-draft
created: 2026-05-28
updated: 2026-05-28
---

# Datasets and Environments Report

## Readout

The central distinction is offline data versus an interactive environment.

An offline dataset gives us:

- observed images/video,
- proprio/state,
- expert actions,
- language/task labels,
- sometimes success/progress metadata.

It does not let the policy choose a different action and see what happens. For that, we need a simulator or real robot.

For our work:

- DROID is the real-world trajectory distribution.
- PolaRiS is a DROID-like interactive real-to-sim evaluation layer.
- MolmoSpaces is a broader simulated task/ecosystem layer with DROID-like and other embodiments.
- LIBERO is the common controlled benchmark, useful as a perturbation taxonomy but increasingly saturated.
- EWMBench is not a simulator; it is a world-model evaluation suite for predicted videos.

## Dataset And Environment Table

| Dataset / environment | Type | Data / scale | Embodiments and interface | What it enables | What it cannot answer |
| --- | --- | --- | --- | --- | --- |
| DROID | Real offline robot dataset | 76k trajectories, about 350h, 564 scenes, 84 tasks. MolmoAct2 filters 74,604 successful episodes / 17.76M frames. | Franka/DROID setup; typically two exterior RGB cameras plus wrist camera; joint-position/action variants. | Per-state action comparison, visual/proprio/action statistics, in-distribution frame checks, semantic progress probes. | No counterfactual rollouts. If policy action differs from expert, the dataset does not reveal the resulting next frame. |
| Open X-Embodiment / OXE | Real offline multi-dataset corpus | Counts vary by paper/date. OpenVLA reports more than 70 robot datasets / more than 2M trajectories; Octo uses 25 datasets / 800k trajectories; OpenVLA curated about 970k trajectories. | Cross-embodiment robot trajectories across many arms/platforms. | Broad embodiment/action convention pretraining; cross-dataset generalization. | Hard to isolate one clean deployment contract because cameras, controls, action spaces, and task labels differ. |
| AgiBot-World / Beta | Real offline robot dataset | GE reports about 1M instruction-aligned multi-view robot videos / about 2,967h. DreamZero AgiBot setting reports about 500h and 7,193 episodes. | Dual-arm/mobile manipulation settings; GE uses calibrated multi-view observations and 14D dual-arm action trajectories. | Robot-domain video world model training; GE-Base/GE-Act/GE-Sim and DreamZero-style world/action modeling. | Not our immediate DROID/Franka deployment surface. |
| LIBERO | Sim benchmark | Four suites: Spatial, Object, Goal, Long; usually 10 tasks per suite and 50 demos per task. | MuJoCo/robosuite; Franka Panda; third-person + wrist camera; 7D delta end-effector action; OSC control; about 10Hz. | Controlled comparison, task suites, easy success metrics, perturbation extensions. | Too saturated as a main claim; not DROID camera/action convention. |
| LIBERO-Plus | Sim robustness benchmark | Robustness paper reports 22,400 total trajectories and 416 distractors across perturbation branches. | Same base as LIBERO, with camera, robot init, language, light, background, noise, and layout perturbations. | Robustness taxonomy we should copy. | Still a benchmark distribution, not a DROID-like deployment contract. |
| RoboTwin / RoboTwin-Plus | Sim benchmark | RoboTwin-Plus: 50 collaborative tasks; Fast-WAM uses 2,500 clean + 25,000 randomized demos. | SAPIEN/ManiSkill3; Aloha-AgileX 14-DoF bimanual robot; head + two wrist cameras; 14D joint actions; 25-30Hz. | Bimanual and robustness evaluation; harder than LIBERO in some regimes. | Different embodiment/action frequency from DROID/Franka policies. |
| PolaRiS | Real-to-sim interactive evaluation framework | Turns short real-world video scans into interactive sim scenes. Paper uses paired real/sim eval environments, co-training scenes, and about 350 sim demos. | DROID platform: 7-DoF Franka Panda, two ZED RGB cameras plus wrist camera; joint-position action focus. | Closed-loop evaluation of DROID-style policies; prompt/image/context perturbations in an interactive DROID-like world. | Fidelity is limited by reconstruction/simulation quality; not guaranteed to represent all DROID task diversity. |
| MolmoSpaces | Sim ecosystem and benchmark | More than 230k indoor environments, more than 130k objects, more than 42M grasps across 48k interactive rigid/articulated objects; benchmark has 8 tasks. | Simulator-agnostic: MuJoCo, Isaac, ManiSkill. Supports Franka FR3, DROID-config FR3+Robotiq, mobile bimanual RB-Y1, floating grippers. | Broader out-of-distribution and environment-rich evaluation; controlled object/task perturbations. | More like a generated task space than the exact DROID training distribution. Domain status depends on model and checkpoint. |
| SimplerEnv | Real-to-sim eval benchmark | Used for Google Robot and WidowX-style policies. | OXE-style setups; PolaRiS paper notes lack of wrist-camera support blocks many DROID-style policy evaluations. | Useful for OXE/Google/WidowX policy eval. | Less useful for our DROID pi0.5/MolmoAct2/DreamZero path. |
| EWMBench | Offline world-model benchmark | GE paper: 10 tasks, 100 videos/task, 4-10 atomic sub-actions. | Video/world-model evaluation, not a robot control environment. | Predicted-video evaluation: scene consistency, EEF spatial alignment, NDTW temporal alignment, dynamics, VLM substep correctness, diversity. | No closed-loop policy interaction. |
| AMASS | Human motion dataset | Human mocap. | Humanoid/body motion, not robot manipulation. | Future cross-embodiment motion prior work. | Not useful for immediate DROID/PolaRiS manipulation. |
| CARLA | Driving simulator | Online driving environment with camera/BEV observations. | Autonomous driving, not manipulation. | Sim2Real-AD methodology: staged training, observation bridge, action bridge. | Not a manipulation benchmark. |

## Datasets Versus Environments

### Offline Dataset

DROID is a set of recorded trajectories. At each time step we can inspect the state, frames, expert action, and future expert frames. This lets us ask:

- Does the model's action match the expert action?
- Does a visual embedding from frame t predict the future expert frame embedding?
- Does semantic progress in the trajectory correlate with the action?
- Are our policy input frames aligned with the training distribution?

It does not let us ask:

- If the model moves 2cm right instead of the expert's 5cm right, what happens?
- If the gripper closes too early, does the object get bumped?
- Can the policy recover after its own mistake?

### Interactive Environment

PolaRiS, MolmoSpaces, LIBERO, RoboTwin, ManiSkill, Isaac, MuJoCo, and robosuite let a policy choose actions and receive new observations. This lets us test:

- closed-loop success,
- recovery,
- counterfactual prompt/image/object perturbations,
- action timing and contact failures,
- rollout divergence.

The price is that the simulator must match the policy's expected contract: camera views, image preprocessing, control space, control frequency, state normalization, action scaling, and reset/task semantics.

## In-Domain / Out-of-Domain Framing

| Model / checkpoint | More in-domain | More out-of-domain | Notes |
| --- | --- | --- | --- |
| pi0.5-DROID | DROID frames; PolaRiS DROID-like scenes; some MolmoSpaces DROID-config tasks. | LIBERO if not using LIBERO checkpoint; unfamiliar objects/layouts/cameras; non-DROID action conventions. | Strong due to broad data, but deployment can fail if camera/control conventions drift. |
| MolmoAct2-DROID | DROID-style two exterior + wrist camera ordering, absolute joint-pose actions, DROID tasks. | MolmoSpaces tasks with new objects/layouts; PolaRiS if adapter/camera contract is not exact. | It is not enough to "use the weight"; rollout code must match the expected observation/action API. |
| DreamZero-DROID | DROID/Franka-style observations/actions; supported DROID/PolaRiS-like settings if code supports it. | LIBERO unless adapted; MolmoSpaces unless camera/action/state conventions match. | Heavy inference; good for predicted-future honesty once running. |
| GE / AgiBot models | AgiBot dual-arm manipulation and GE benchmark settings. | DROID/PolaRiS without interface adaptation. | GE-Sim is conceptually ideal but not immediately available as a drop-in DROID evaluator. |
| VLA-JEPA | DROID and LIBERO/SimplerEnv reported setups. | PolaRiS/MolmoSpaces until adapters exist. | Useful conceptually for latent/probe experiments even if full policy rollout is not immediate. |
| OpenVLA/OFT | OXE/LIBERO depending checkpoint. | DROID/PolaRiS unless trained/adapted. | Tokenized action conventions can be brittle across environments. |

## What We Should Borrow

### From LIBERO-Plus

Use its perturbation taxonomy:

- camera viewpoint,
- robot initial state,
- language paraphrase,
- wrong or contradictory language,
- lighting,
- background,
- sensor noise/occlusion,
- object layout,
- distractor objects.

### From EWMBench

Use its world-model evaluation idea, but adapt it to our question:

- scene consistency: does the predicted future preserve the workspace?
- EEF path alignment: does predicted/proxy motion track actual robot motion?
- temporal alignment: does progress occur at the right time?
- semantic substep completion: does a VLM judge the relevant subgoal as complete?
- dynamics: are velocity/acceleration patterns plausible?

### From PolaRiS

Use it as a DROID-like closed-loop contract:

- same camera-style policy inputs,
- same Franka/DROID action conventions,
- real-to-sim rank correlation as motivation,
- direct evaluation of DROID policies.

### From MolmoSpaces

Use it as the broader out-of-distribution environment:

- more environments,
- more object diversity,
- more object/task mismatch conditions,
- DROID-config robot support where possible.

## Practical Conclusion

For tomorrow's experiments, the clean split is:

- **DROID subset:** offline sanity and per-state action/latent metrics.
- **PolaRiS:** in-domain-ish closed-loop DROID policy runs.
- **MolmoSpaces:** out-of-domain-ish closed-loop or semi-closed-loop DROID policy stress tests.
- **LIBERO-Plus:** source of perturbation design, not necessarily the main environment.

This lets us avoid pretending that a dataset is an environment, and it makes the claims sharper: DROID tells us whether the model matches historical expert behavior; PolaRiS/MolmoSpaces tell us whether it survives its own actions.
