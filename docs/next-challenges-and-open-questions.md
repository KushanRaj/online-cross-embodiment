# Next Challenges And Open Questions

## 1. Does The IDM Signal Really Predict Failure?

Current evidence is suggestive, not decisive.

Open questions:

- Does `IDM(C,P) vs a` separate success/failure better than P-vs-R visual error?
- Are spikes mostly gripper/contact events?
- Does the signal survive across tasks where success/failure is balanced?
- Does the signal survive RoboCasa and other richer environments?

Needed next evidence:

- more success/failure balanced rollouts,
- per-task analysis,
- gripper/contact event tagging,
- confidence intervals rather than only aggregate plots.

## 2. Is Cosmos The Right World Model For This Diagnostic?

Current Cosmos Policy gives useful futures and action-head outputs, but the present setup is not a pure action-conditioned counterfactual simulator.

Open questions:

- Can the available Cosmos API generate future conditioned on arbitrary external actions?
- Can GE-Sim or another action-conditioned WAM produce cleaner counterfactuals?
- Does a base video model such as Cosmos 3 Nano produce useful LIBERO/RoboCasa futures without task-specific fine-tuning?

## 3. What Should The IDM Learn From Generated Futures?

The base IDM is real-to-real:

```text
IDM(real current, real future) -> real action
```

Possible future mixtures:

```text
real -> real
real -> model
model -> model
model -> real
```

The project should keep the base simple before overbuilding synthetic mixtures. Generated-future data is useful only if it improves the diagnostic rather than teaching the IDM to follow world-model artifacts.

## 4. How Much Architecture Is Needed?

The MLP IDM is simple and interpretable. The patch-transformer IDM may be stronger but changes the research claim.

Open questions:

- Is pooled-feature MLP enough for a diagnostic?
- Does patch-token attention improve action recovery or only lower one training metric?
- Is a larger IDM hiding the exact brittleness we want to expose?

## 5. How To Handle Gripper And Contact?

Observed spikes often appear around gripper open/close and contact.

Open questions:

- Are failures correlated with EEF movement, gripper mismatch, or both?
- Should gripper be a separate classification target?
- Should contact stages be labeled in eval videos?

## 6. What Counts As Out Of Distribution?

The project has several OOD axes:

- task text changes,
- object changes,
- visual layout changes,
- simulator/environment changes,
- embodiment/action-space changes.

Do not collapse these into one "OOD" label. Each run should say which axis changed.

## 7. Cross-Embodiment Is Parked, Not Abandoned

The future arm-to-arm plan remains valuable:

```text
shared EEF/action abstraction + observation adapters + embodiment-specific low-level execution
```

But the current priority is prediction-action consistency in runnable environments, not full morphology transfer.
