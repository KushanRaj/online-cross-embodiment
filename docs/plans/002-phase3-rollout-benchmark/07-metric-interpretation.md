# 07 - Metric Interpretation

## P vs R

```text
P = world-model predicted future
R = observed future
```

SigLIP P-vs-R:

- lower L2 means feature closeness,
- higher cosine similarity means feature alignment,
- this is visual/semantic agreement, not action agreement.

## IDM(C,P) vs a

This is the primary policy/world-model consistency metric.

Question:

```text
Does the action implied by the imagined future match the action selected by the policy?
```

High mismatch can mean:

- world-model future is wrong,
- policy action is wrong,
- IDM is wrong,
- all three are off distribution.

Interpret with video.

## IDM(C,R) vs a

This asks whether the observed future implies the action that was actually taken.

It can be useful as an IDM sanity check. It is not always the most interesting mismatch, because the observed future is downstream of the executed action.

## IDM(C,P) vs IDM(C,R)

This is a residual between the action implied by imagined future and observed future.

It is auxiliary. The user specifically asked not to make this the main video plot when `IDM(C,P) vs a` is the intended diagnostic.

## EEF And Gripper Metrics

EEF metrics isolate movement direction/translation/rotation.

Gripper mismatch checks whether open/close sign differs. It matters because many spikes looked gripper/contact-related.

## Query Timestep Axis

`query timestep` means simulator frame/time index where the world model was queried.

Per-episode mean avoids bias where longer episodes accumulate more total error.
