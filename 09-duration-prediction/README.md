# Duration Prediction for VITS2

This chapter learns how many acoustic frames each text token occupies and turns predicted durations into inference alignment paths.

Primary references: `training/models/modules/duration.py` and `generate_path` in `commons.py`.

## End-to-end pathway

```text
training MAS path [B,1,T_text,T_mel]
    -> durations w [B,1,T_text]

text features [B,C,T_text] + mask + speaker
    -> deterministic log-duration prediction
       or stochastic duration-flow NLL

inference logw -> exp -> length_scale -> ceil
    -> mel length and y_mask
    -> monotonic path [B,T_text,T_mel]
```

## Why predict durations

Training MAS can inspect target acoustics. Inference has no target mel, so the model must decide how long every token lasts before it can expand token-level priors into acoustic time.

## Duration targets

Training durations come from the MAS path:

```text
w = attention.sum(dim=-1)  # mel frames per text token
```

They are nonnegative integers and their valid-token sum equals mel length.

## Deterministic predictor

The deterministic path uses two convolution blocks and a final `1x1` projection to one channel. The reference alternates Conv1d/ReLU/LayerNorm/Dropout and masks after each operation.

Because `LayerNorm(C)` expects channels last, transpose:

```text
[B,C,T] -> [B,T,C] -> LayerNorm -> [B,C,T]
```

It predicts `logw_hat`. Training compares it with:

```text
logw = log(w + epsilon) * mask
```

using masked mean-squared error.

## Why log duration

Durations are positive and vary multiplicatively. Log space makes multiplicative ratios additive. In inference:

```text
w = exp(logw) * mask * length_scale
w_integer = ceil(w)
```

`length_scale > 1` makes speech longer; values below one shorten it.

## Stochastic duration predictor

The SDP models a distribution over rhythm using normalizing flows. Major pieces include:

- detached text conditioning;
- DDSConv context processing;
- posterior flows conditioned on observed durations;
- continuous dequantization using `u = sigmoid(z_u)`;
- `z0 = clamp(w-u)` followed by a log transform;
- prior flows and change-of-variables log-determinants;
- reverse sampling controlled by `noise_scale_w`.

Integer durations are discrete; subtracting learned `u` in `(0,1)` creates positive continuous values suitable for continuous invertible flows.

## DDSConv

Each layer uses a depthwise dilated convolution, LayerNorm, GELU, `1x1` convolution, dropout, and residual addition. Dilation follows `kernel_size ** layer_index` in the reference.

## Generating an inference path

`generate_path` uses cumulative durations. A sequence mask marks all frames before each cumulative endpoint; subtracting the previous cumulative mask isolates the frames assigned to each token.

# Experiment roadmap

## Task 1: Extract durations from toy alignment

Sum over mel frames and verify totals and padding.

## Task 2: Build deterministic inputs

Prepare text features, text mask, and optional speaker projection.

## Task 3: Implement the deterministic conv stack

Handle LayerNorm transposes and mask after each operation.

## Task 4: Compute deterministic loss

Compare target and predicted log durations only at valid tokens.

## Task 5: Explore exp, ceil, and length_scale

Predict exact outputs for several log-duration and speaking-rate values.

## Task 6: Implement generate_path

Use cumulative duration masks and differencing.

## Task 7: Validate generated paths

Check monotonicity, coverage, output length, and recovered durations.

## Task 8: Implement Log and ElementwiseAffine

Derive forward/inverse equations and log-determinants; prove round trips.

## Task 9: Build DDSConv

Track dilation, depthwise grouping, normalization layout, residuals, and masks.

## Task 10: Trace stochastic training

Label `e_q`, `z_q`, `u`, `z0`, `z1`, `logq`, log-determinants, and NLL shapes.

## Task 11: Trace stochastic inference

Reverse flows with controlled `noise_scale_w` and obtain `[B,1,T_text]` log durations.

## Task 12: Compare reference predictors

Match deterministic outputs exactly and stochastic shapes/behavior under controlled seeds.

## Task 13: Final assertions

Collect duration, path, mask, inverse, length-control, and reference checks.

# Debugging order

1. Ensure durations use only valid text tokens.
2. Keep `logw` separate from `w`.
3. Transpose around LayerNorm.
4. Clamp positive values before logarithms.
5. Mask every flow output.
6. Accumulate log-determinants with correct signs.
7. Use `ceil` only when integer inference frame counts are needed.

# Questions you must answer

1. Why cannot inference use MAS?
2. Why predict log duration?
3. What problem does dequantization solve?
4. How do `length_scale` and `noise_scale_w` differ?
5. How do cumulative durations create a path?
6. Why does the stochastic predictor detach text features?

# Completion criteria

Chapter 09 is complete when you can derive duration targets, build deterministic prediction, explain stochastic duration likelihood, and generate valid inference paths.

Chapter 10 expands latent acoustic frames into waveform samples.
