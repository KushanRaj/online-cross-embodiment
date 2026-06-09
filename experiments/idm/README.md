# IDM Phase 1 Pipeline

This folder contains the code path for the shared semantic inverse dynamics model experiment.

The intended workflow is:

```text
edit code locally -> push/pull repo on EC2 -> run data/model jobs on EC2
```

Raw datasets, checkpoints, cached features, generated futures, and rollout videos should stay outside git.

## Current Phase

Phase 1 trains and validates an inverse dynamics model on real LIBERO
demonstration transitions:

```text
IDM(C, F, proprio, instruction) -> action chunk
```

where:

```text
C = current observation from the demo
F = future observation from the same demo
```

The base training source is only LIBERO `real -> real` rows. Generated
future rows from Cosmos, GE, or other world models are not part of the
base IDM training set. They are used later for diagnostics or as an
explicit ablation.

## Tracked Code

- `remote/` - EC2-only data export/query scripts.
- `data/manifest.py` - manifest parsing and image loading contract.
- `data/cache_features.py` - frozen visual encoder feature extraction.
- `model/idm.py` - small feature-level IDM.
- `model/train_idm.py` - Phase 1 training and validation.
- `scripts/` - remote sync/run helpers.

## Data Contract

Rows can reference frames in either form:

```text
image_t_path
image_future_path
```

or HDF5 references:

```text
source_file
image_t_hdf5_path
image_t_index
image_t_transform
image_future_hdf5_path
image_future_index
image_future_transform
```

For LIBERO HDF5 rows, the canonical convention is:

```text
raw HDF5 RGB frames -> one manifest-declared transform before encoding
current and future frames in a row must use the same transform
```

The current Phase 1 default is `flipud`, matching the LIBERO convention used
by the Cosmos Policy code path. The transform is stored per frame in the
manifest, so later GE/Cosmos diagnostics can be canonicalized deliberately
instead of silently mixing image conventions.

## RoboCasa Contract

RoboCasa uses the same Phase 1 IDM shape as LIBERO for the first baseline:

```text
IDM(C, F, proprio) -> 16 x 7 action chunk
```

The RoboCasa/Cosmos dataset files we use are rollout-style HDF5 episodes with
top-level arrays such as:

```text
primary_images_jpeg
actions
proprio
```

The action convention is:

```text
dataset action label: first 7 dims only
IDM output:           16 x 7
Cosmos policy output: first 7 manipulation dims
RoboCasa env.step:    12D after appending fixed mobile-base tail [0, 0, 0, 0, -1]
```

The mobile-base padding is only a rollout execution detail. It is not part of
the IDM target or the deviation metric.

For this first RoboCasa MLP run, keep the setup copied from the LIBERO feature
IDM baseline:

```text
source = real -> real RoboCasa-Cosmos trajectories
horizon = 16
encoder = google/siglip-base-patch16-224
model = feature MLP IDM
target = 16 x 7 manipulation action chunk
success_label = metadata only, never an IDM input
```

The selected rollout task families are the Cosmos/RoboCasa Phase 3 set used in
the first run:

```text
CloseDrawer
CoffeeServeMug
CoffeeSetupMug
OpenDrawer
TurnOffMicrowave
TurnOffSinkFaucet
TurnOffStove
TurnOnMicrowave
TurnOnSinkFaucet
TurnOnStove
TurnSinkSpout
```

## Reverse Data

Do not mix future-to-past reverse rows into the base run.

Reverse supervision is an ablation only:

```text
base:          C_t, C_t+k -> action_t:t+k
reverse-real:  C_t+k, C_t -> approximate_reverse(action_t:t+k)
```

This is intentionally separate because reverse labels are not physically exact
under contact, object motion, or gripper state changes. The forward IDM must
first show clean held-out action recovery before reverse data is worth testing.
