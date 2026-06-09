# 03 - Model Candidates

## Cosmos Policy

Useful now:

- runnable,
- has LIBERO and RoboCasa checkpoints,
- produces futures and action-head chunks.

Question:

- Does the available API expose arbitrary action-conditioned future generation?

## Cosmos 3 Nano

Useful for smoke tests:

- can do image-to-video generation,
- base checkpoint can reveal whether generic video priors help.

Limit:

- not necessarily robotics/action-conditioned.

## GE-Sim

Promising because it is explicitly framed as action-conditioned robotic simulation.

Need to resolve:

- exact checkpoint availability,
- input/action conditioning path,
- whether generated futures decode cleanly in our environments.

## DreamZero

Interesting for prediction honesty because it jointly generates action and future.

Limit:

- heavier and less immediately runnable in the current EC2 setup.
- joint action/future generation is not the cleanest arbitrary-action counterfactual.
