# 05 - Ablation Queue

Keep these separate from the base.

## Encoder Ablations

- SigLIP baseline
- DINOv3
- V-JEPA 2

Run on the same data split before claiming encoder differences.

## Architecture Ablations

- Feature MLP IDM
- Patch-transformer IDM
- Smaller/larger MLP
- Separate gripper head

## Data Ablations

- real -> real only
- real -> model added
- model -> model added
- reverse samples added
- LIBERO-only
- RoboCasa-only
- LIBERO+RoboCasa mixed

## Horizon Ablations

- k=16 base
- k=32
- k=128
- mixed-horizon model that always outputs 128 and evaluates prefixes

Do not train separate horizons if the question is whether a single IDM can support multiple horizons.

## Loss Ablations

- Smooth L1 base
- gripper-weighted loss
- EEF/gripper split loss
- discounted long-horizon loss

The user aligned on keeping the first 128 experiment simple: no discount, no curriculum.
