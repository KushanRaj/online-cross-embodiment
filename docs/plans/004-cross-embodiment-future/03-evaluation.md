# 03 - Evaluation

## Clean Evaluation Axis

Separate:

- new task,
- new object,
- new scene,
- new arm,
- new camera,
- new controller.

Do not call all of these "cross embodiment."

## Minimal Arm-To-Arm Test

1. Train/evaluate a model on source arm in EEF space.
2. Keep task/object/scene as close as possible.
3. Swap target arm.
4. Use only observation/action adapters.
5. Measure task success and action/future consistency.

## Relevance To Current IDM Work

The current IDM can eventually become:

```text
Does the future imply an embodiment-free EEF action?
```

But today's IDM still predicts embodiment-specific actions.
