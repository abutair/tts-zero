# Normalizing Flow for VITS2

This chapter builds invertible coupling transformations that map posterior latents into the text-conditioned prior space and reverse that mapping during inference.

Primary reference: `training/models/modules/flow.py`.

## End-to-end pathway

```text
posterior z [B, C, T]
    -> split channels into x0 and x1
    -> predict transform parameters from x0 and speaker
    -> transform x1 while preserving x0
    -> concatenate and flip channels
    -> repeat coupling + flip
z_p [B, C, T]

inference: apply every operation in reverse order
```

## Why VITS2 needs a flow

The posterior is shaped by observed acoustics; the prior is conditioned on text. An invertible flow learns a flexible bridge without discarding information. Training uses `z -> z_p`; inference uses `z_p -> z`.

## Coupling-layer invertibility

Split channels equally, so `C` must be even:

```text
x = concat(x0, x1)
```

Keep `x0` unchanged and predict `m` and `logs` from `x0`:

```text
forward: y1 = m + x1 * exp(logs)
inverse: x1 = (y1 - m) * exp(-logs)
```

The conditioner can be complex and non-invertible because inversion recomputes it from the preserved `x0`.

## Log-determinant

For affine coupling:

```text
log|det J| = sum(logs * mask)
```

The mask excludes padding. This term is required by change-of-variables likelihood calculations.

## The reference is additive

`ResidualCouplingBlock` constructs every coupling with `mean_only=True`. Therefore:

```text
logs = 0
y1 = m + x1
logdet = 0
```

This is an important reference-specific simplification. The code supports affine mode, but the configured block uses additive transforms.

## Identity initialization

The final projection is initialized to zero. Initially `m=0` and `logs=0`, so each new coupling starts as identity. This stabilizes early training and makes forward/reverse tests predictable.

## Flips and flow stacks

A coupling directly transforms only half the channels. `Flip` reverses channel order so later layers transform the other information. Applying `Flip` twice restores the original tensor.

Forward execution visits operations in stored order; reverse execution must visit them in reverse order and invert each coupling.

## Flow conditioner

The coupling conditioner uses a gated WaveNet similar to Chapter 06. It receives the preserved half, mask, and optional speaker vector, then predicts transform statistics.

# Experiment roadmap

## Task 1: Split and concatenate channels

Use an even-channel tensor and prove concatenation restores it exactly.

## Task 2: Hand-designed additive coupling

Choose a simple known `m(x0)`, transform `x1`, derive the inverse, and verify round-trip accuracy.

## Task 3: Extend to affine coupling

Add `logs`, derive the inverse, and calculate a per-sample masked log-determinant.

## Task 4: Test masking

Perturb padded values and prove valid outputs and log-determinants do not change.

## Task 5: Build the conditioning WaveNet

Reuse gated residual/skip reasoning to predict `m` or `[m, logs]` from `x0`.

## Task 6: Verify zero initialization

Inspect the post projection and prove a new coupling starts as identity.

## Task 7: Implement Flip

Reverse channels twice and prove the original order returns.

## Task 8: Stack couplings and flips

Accumulate forward log-determinants and invert operations in reverse order.

## Task 9: Compare mean-only and affine modes

Create a table of output channels, equations, inverse equations, and log-determinants.

## Task 10: Add speaker conditioning

Use the same speaker vector in forward and reverse; explain why changing it breaks round-trip equality.

## Task 11: Compare with the reference

Copy weights into `ResidualCouplingLayer` and `ResidualCouplingBlock`; verify forward/reverse outputs.

## Task 12: Final assertions

Collect invertibility, identity initialization, masking, log-determinant, flip, and reference checks.

# Debugging order

1. Require an even channel count.
2. Preserve `x0` exactly.
3. Use identical `m/logs` in forward and inverse.
4. Negate `logs` in the inverse.
5. Exclude padding from log-determinants.
6. Reverse both operation order and individual transformations.
7. Do not omit flips.

# Questions you must answer

1. Why can the conditioner itself be non-invertible?
2. What guarantees coupling invertibility?
3. Why exponentiate `logs`?
4. Why is the current block's coupling log-determinant zero?
5. Why initialize as identity?
6. Why are channel flips needed?

# Completion criteria

Chapter 07 is complete when you can derive affine forward/inverse/logdet equations, prove block round trips, explain the additive reference choice, and match reference behavior.

Chapter 08 finds the monotonic path aligning text-token prior parameters with acoustic latent frames.
