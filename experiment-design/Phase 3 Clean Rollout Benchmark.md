# Phase 3 Clean Rollout Benchmark

This records the clean task benchmark for the prediction-vs-reality deviation
experiment.

The purpose is not to maximize benchmark score. The purpose is to get a clean
mix of:

- high-alignment rollouts,
- moderate rollouts,
- natural failures,
- model disagreements on the same task.

Perturbations stay tangential for Phase 3. They are useful later, but the first
benchmark should not rely on out-of-domain future prediction.

## Model Scope

DreamZero is out of scope for the immediate run because it is not realistically
runnable in our current setup.

In scope:

- Cosmos Policy / Cosmos future-prediction path,
- GE-Act / GE-Sim where future video is available,
- Fast-WAM future-capable variants,
- LingBot-VA-style video-action models if runnable.

Action-only VLAs such as pi0.5 and Molmo can be controls only if paired with an
external future model. They are not the core Phase 3 subject by themselves.

## Common Clean Track: LIBERO-10

LIBERO is the only clean environment that is paper-backed across Cosmos,
Fast-WAM, and GE-Act. It is also saturated.

Facts:

- Cosmos reports LIBERO Long `97.6`.
- Fast-WAM reports LIBERO Long `95.2`.
- GE-Act reports clean LIBERO original `94.4` in the robustness table.
- The papers do not provide per-task clean LIBERO failures for these models.

Use all ten official `LIBERO-10` tasks:

1. `turn_on_the_stove_and_put_the_moka_pot_on_it`
2. `put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
3. `put_the_yellow_and_white_mug_in_the_microwave_and_close_it`
4. `put_both_moka_pots_on_the_stove`
5. `put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
6. `put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
7. `put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
8. `put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
9. `put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
10. `pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`

Expected signal: mostly high alignment, sparse failures.

## Cosmos-Native Clean Track: RoboCasa

Cosmos reports RoboCasa `67.1` average success over 24 kitchen tasks. The paper
does not report per-task Cosmos policy success. The task-family counts below are
from the official RoboCasa-Cosmos dataset replay stats, not Cosmos policy
per-task scores.

Harder clean task families:

| Task family | Replay success | Expected signal |
|---|---:|---|
| `TurnOffStove` | `39/54` | low to moderate alignment |
| `TurnOnSinkFaucet` | `40/54` | low to moderate alignment |
| `CoffeeSetupMug` | `43/54` | moderate alignment |
| `TurnOnStove` | `44/53` | moderate alignment |
| `CloseDoubleDoor` | `47/54` | moderate alignment |

Easy controls:

| Task family | Replay success | Expected signal |
|---|---:|---|
| `CloseDrawer` | `54/54` | high alignment |
| `CloseSingleDoor` | `54/54` | high alignment |
| `CoffeePressButton` | `54/54` | high alignment |
| `OpenDrawer` | `54/54` | high alignment |
| `TurnSinkSpout` | `54/54` | high alignment |

## Fast-WAM / LingBot Clean Track: RoboTwin

RoboTwin is the best paper-backed clean task-level source for Fast-WAM-family
models. Fast-WAM reports exact per-task clean/randomized scores.

High-alignment controls:

| Task | Fast-WAM clean/randomized |
|---|---:|
| `Grab Roller` | `100/100` |
| `Place Empty Cup` | `100/100` |
| `Stack Blocks Two` | `100/100` |
| `Click Alarmclock` | `100/100` |

Hard/moderate/disagreement tasks:

| Task | Evidence | Expected signal |
|---|---|---|
| `Hanging Mug` | Fast-WAM `58/62`; LingBot `40/28` | low alignment |
| `Open Microwave` | Fast-WAM `62/45`; Joint `3/14`; IDM `54/53`; LingBot `82/86` | disagreement |
| `Turn Switch` | Fast-WAM `61/59`; LingBot `44/45` | low to moderate |
| `Place Can Basket` | Fast-WAM `71/69`; Joint `50/23`; IDM `37/28`; LingBot `81/84` | disagreement |
| `Move Stapler Pad` | Fast-WAM `77/64` | moderate |
| `Press Stapler` | Fast-WAM `90/97`; Joint `52/50`; IDM `50/57`; LingBot `85/82` | disagreement |

Selected data mapping:

- Metadata source: `lerobot/robotwin_unified`.
- Selected labels: 13.
- Selected unique files: 158.
- Selected download size: `22.42 GiB`.
- Remote manifest:
  `/home/ubuntu/robotics/data/robotwin/phase3_selected_mapping.json`.
- Tracked manifest:
  `experiments/phase3/tasks/robotwin_selected_mapping.json`.

## GE Clean Track

GE has named clean tasks, but many values are image-only and GE-Sim does not
report a clean closed-loop task table. Use GE/RoboTwin overlap first if GE
assets are runnable.

RoboTwin overlap:

| Task | GE evidence | Expected signal |
|---|---|---|
| `Grab Roller` | visually near `0.98` SR | high alignment |
| `Handover Mic` | visually near `1.0` SR | high alignment |
| `Lift Pot` | visually around `0.77` SR; paper says slightly behind VLA methods | moderate |
| `Move Can Pot` | visually around `0.85` SR | moderate |

GE-native named tasks:

- `Clean the table`: high alignment.
- `Make a sandwich`: moderate.
- `Pour a cup of tea`: moderate.
- `Heat food in microwave`: moderate.
- `Packing detergent from conveyor`: moderate/dynamic.
- `Grasp red cylinder from table and place into paper cup`: exact table gives
  `0.89` E2E / `0.76` SR with GE-Base init + VidAda + state.

## Remote Setup Record

Tracked setup assets:

- `experiments/phase3/tasks/phase3_clean_tasks.yaml`
- `experiments/phase3/scripts/setup_phase3_remote.sh`
- `experiments/phase3/scripts/download_libero_phase3_remote.sh`
- `experiments/phase3/scripts/download_robocasa_cosmos_remote.sh`
- `experiments/phase3/scripts/setup_robotwin_remote.sh`
- `experiments/phase3/scripts/map_robotwin_selected_remote.sh`
- `experiments/phase3/scripts/verify_phase3_remote.sh`
- `experiments/phase3/tasks/robotwin_selected_mapping.json`

Default EC2 layout:

```text
/home/ubuntu/robotics/repos
/home/ubuntu/robotics/data
/home/ubuntu/robotics/data/phase3
```

The scripts prepare repos, task manifests, and data roots. They do not run
rollouts, evaluations, policy inference, or training.
