# 05 - Next Actions

## Immediate

1. Keep the MLP IDM as baseline.
2. Use existing Phase 3 evaluator/plotter/video renderer.
3. Run more balanced success/failure episodes in LIBERO and RoboCasa.
4. Add gripper/contact focused analysis without changing base metrics.
5. Compare Pi/Molmo/Cosmos on the same task slices.

## Near-Term

1. Train RoboCasa-only IDM cleanly and document data size.
2. Run RoboCasa repeated trials, not only one per task.
3. Compare LIBERO-only, RoboCasa-only, and mixed IDM if action conventions are clean.
4. Try better/base world models only after a clear input/output contract.

## Later

1. True action-conditioned counterfactual world model.
2. Generated-future IDM training mixture.
3. Cross-embodiment EEF adapter experiment.
