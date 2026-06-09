# 05 - Open Questions

## API Availability

Which runnable checkpoint exposes:

```text
current observation + arbitrary action chunk -> future video/image
```

without requiring major training?

## Action Convention

The model's expected action convention must match the simulator action convention or have a documented adapter.

## Future Horizon

If a model predicts only 16 frames ahead, evaluate against a 16-step IDM. Do not invent one-step labels for a 16-step future.

## Generated Image Quality

If generated futures are too pixelated or semantically wrong, the IDM diagnostic may be measuring world-model OOD rather than policy inconsistency.
