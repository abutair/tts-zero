# Posterior Encoder for VITS2

This chapter converts observed mel features into the acoustic posterior distribution `q(z | mel, speaker)` and samples the latent `z` used during training.

Primary reference: `training/models/modules/posterior.py`.

## Prerequisite

Complete Chapter 05 first. Keep the established `[B, C, T]` convention, explicit lengths, and `[B, 1, T]` masks.

## End-to-end pathway

```text
mel [B, 80, T_mel] + mel lengths [B]
    -> mask [B, 1, T_mel]
    -> 1x1 pre-projection [B, 192, T_mel]
    -> gated WaveNet residual/skip stack
    -> stats [B, 384, T_mel]
    -> m_q, logs_q [B, 192, T_mel]
    -> z = (m_q + epsilon * exp(logs_q)) * mask
```

## Prior versus posterior

The text encoder predicts the prior `p(z | text)`. The posterior observes the target mel and predicts `q(z | mel)`. Training connects them through the flow, alignment, and KL loss. Inference cannot use the posterior because target audio does not exist.

## Reference configuration

```text
mel channels:       80
hidden channels:    192
latent channels:    192
kernel size:        5
dilation rate:      1
layers:             16
speaker dimension:  256
```

The code calculates `dilation = dilation_rate ** layer_index`. With the configured rate `1`, every layer currently has dilation `1`. Do not assume exponentially growing dilation when explaining this implementation.

## Gated WaveNet activation

A convolution produces `[B, 2C, T]`, which is split into equal halves:

```text
a, b = split(x_in)
activation = tanh(a) * sigmoid(b)
```

`tanh` proposes signed content; `sigmoid` gates how much passes. The result returns to `[B, C, T]`.

## Residual and skip paths

All layers except the last produce `2C` channels after the gated activation:

- the first `C` channels update the hidden residual stream;
- the second `C` channels accumulate into the skip/output stream.

The final layer produces only `C` skip channels. Masks are reapplied because convolutions and biases can recreate nonzero padding.

## Speaker conditioning

One speaker vector is projected to `[B, 2*C*n_layers, 1]`. Layer `i` receives its slice:

```text
offset = i * 2C
condition_i = projected[:, offset:offset+2C, :]
```

The singleton time dimension broadcasts over every mel frame.

## Posterior statistics and sampling

The final `1x1` projection outputs `2*C_latent` channels and splits them into mean and log standard deviation:

```text
std      = exp(logs_q)
variance = exp(2 * logs_q)
z        = m_q + epsilon * std
epsilon  ~ Normal(0, 1)
```

This reparameterization keeps randomness in `epsilon`, allowing gradients to reach `m_q` and `logs_q`. Apply the mask after sampling.

# Experiment roadmap

## Task 1: Create mel inputs, lengths, and masks

Use unequal mel lengths. Verify mask sums and label `[B, C_mel, T_mel]`.

## Task 2: Apply the pre-projection

Map mel channels to a small hidden size using `Conv1d(..., kernel_size=1)` and mask the result.

## Task 3: Build the fused gate

Split `2C` channels, implement `tanh(a) * sigmoid(b)`, and inspect its range.

## Task 4: Calculate dilations and receptive field

List the actual reference dilations. Compare rate `1` with a growing-dilation example and calculate both receptive fields.

## Task 5: Build one residual/skip layer

Produce residual and skip parts, update the hidden state, and reapply the mask.

## Task 6: Stack the WaveNet block

Accumulate skip outputs and handle the last layer's different output-channel rule.

## Task 7: Add speaker conditioning

Project once, slice the correct `2C` range per layer, and broadcast across time.

## Task 8: Project posterior statistics

Produce and split `[B, 2*C_latent, T]` into `m_q` and `logs_q`.

## Task 9: Sample with reparameterization

Use controlled epsilon values, verify positive standard deviations, and mask `z`.

## Task 10: Test padding invariance

Change only padded mel values. Valid `m_q`, `logs_q`, and `z` must remain unchanged under controlled randomness.

## Task 11: Compare with the reference

Copy identical weights into `PosteriorEncoder`; use identical random seeds and compare all four outputs.

## Task 12: Final assertions

Collect shape, mask, sampling, padding-invariance, and reference-equivalence checks.

# Debugging order

1. Verify mel layout is `[B, C_mel, T]`.
2. Check dilation and padding together.
3. Split gated channels exactly in half.
4. Slice speaker conditioning using the correct layer offset.
5. Mask residual and final outputs.
6. Control random seeds when comparing sampled `z`.

# Questions you must answer

1. Why is the posterior training-only?
2. What does each half of the gate do?
3. Why use separate residual and skip paths?
4. Why is `logs_q` log-scale rather than log-variance?
5. What does reparameterization make differentiable?
6. What is the actual configured receptive field?

# Completion criteria

Chapter 06 is complete when you can derive every WaveNet shape, explain `q(z|mel)` versus `p(z|text)`, prove padding invariance, and match the reference under identical weights and randomness.

Chapter 07 transforms posterior samples into the text-prior latent space using an invertible normalizing flow.
