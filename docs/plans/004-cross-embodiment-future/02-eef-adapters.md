# 02 - EEF Adapters

## Observation Adapter

An observation adapter converts the new embodiment's sensors into the model's expected observation format.

Examples:

- camera crop/view convention,
- wrist vs agentview images,
- proprio vector format,
- gripper state normalization,
- task text format.

## Action Adapter

An action adapter maps shared action output into the target robot controller.

Example:

```text
shared EEF delta + gripper -> target-arm controller command
```

This is embodiment-specific.

## Training Data

Possible data:

- demonstrations on source arm,
- demonstrations on target arm if allowed,
- simulated paired tasks,
- no-target-data zero-shot test if the goal is strict transfer.

The current project has not selected this data regime.
