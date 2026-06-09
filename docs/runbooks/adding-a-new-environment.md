# Adding A New Environment

## Goal

Add an environment while preserving the Phase 3 diagnostic contract.

## Required Questions

Before implementation:

```text
What is the simulator?
What are the tasks?
What are success conditions?
What observations are available?
What action space does the policy output?
What action space does the environment execute?
What demonstrations/trajectories exist for IDM training?
What preprocessing convention does the policy/world model expect?
```

## RoboCasa Example

RoboCasa required clarifying:

- it is a robosuite/MuJoCo household manipulation environment/task suite,
- Cosmos Policy RoboCasa checkpoint has its own expected action/data convention,
- the IDM predicts 7D manipulation action,
- the environment runner may pad to a wider action vector.

## Data Manifest

Use the IDM sample contract:

```text
sample_id
task_name
trajectory_id
t
horizon
current_primary_image
future_primary_image
current_proprio
action_chunk_16x7
source
```

## Success Labels

Success/failure labels are for evaluation and plotting. They are not IDM inputs.
