# Discriminators and Losses for VITS2

This chapter judges waveform realism at periodic and multi-scale views, then combines adversarial, feature-matching, mel, KL, and duration objectives.

Primary references: discriminator classes in `training/models/modules/decoder.py` and `training/models/losses.py`.

## End-to-end pathway

```text
real y and generated y_hat [B,1,T]
    -> Multi-Period Discriminator: periods 2,3,5,7,11
    -> Multi-Scale Discriminator: original + two pooled views
    -> scores and intermediate feature maps

D loss = real least-squares loss + fake least-squares loss
G loss = adversarial + feature matching + mel + KL + duration
```

## Multi-Period Discriminator

For period `p`, reflect-pad the waveform so time is divisible by `p`, then reshape:

```text
[B,1,T] -> [B,1,T/p,p]
```

Conv2d layers analyze repeated phase positions. Periods `2,3,5,7,11` expose different periodic structures related to pitch and harmonics.

Each discriminator returns a flattened score tensor plus every intermediate feature map.

## Multi-Scale Discriminator

Three scale discriminators inspect:

1. the original waveform;
2. one average-pooled view;
3. a twice-pooled view.

The first uses spectral normalization; the others use weight normalization. Grouped Conv1d layers reduce cost while increasing temporal receptive field.

## Least-squares GAN losses

Discriminator targets:

```text
real score -> 1
fake score -> 0
D = mean((1-D(real))^2) + mean(D(fake)^2)
```

Generator adversarial target:

```text
G_adv = mean((1-D(fake))^2)
```

These terms sum across every MPD and MSD sub-discriminator.

## Feature matching

Compare intermediate real and generated feature maps using L1:

```text
2 * sum(mean(abs(real_feature.detach() - fake_feature)))
```

Detaching real features makes them fixed targets during the generator update. Feature matching supplies dense perceptual guidance even if final adversarial scores are unstable.

## Mel reconstruction

The reference uses:

```text
mel_loss = 45 * L1(mel_generated, mel_target)
```

This strongly anchors generated speech to target spectral content.

## KL loss

The flowed posterior sample `z_p` should be plausible under the aligned text prior. The reference uses `logs_q`, `m_p`, and `logs_p`, masks invalid frames, clamps exponent arguments for stability, sums valid values, and divides by mask sum.

## Alternating optimization

Discriminator step:

```text
wav_hat = generator(...).detach()
update D only
```

Generator step:

```text
wav_hat = generator(...)  # retain graph
compute adversarial + feature + mel + KL + duration
update G only
```

Detaching generated audio only in the discriminator step prevents D backpropagation from modifying G.

# Experiment roadmap

## Task 1: Reshape waveform by period

Reflect-pad toy signals and verify `[B,1,T/p,p]` for several periods.

## Task 2: Build one PeriodDiscriminator

Track every Conv2d and feature-map shape.

## Task 3: Assemble the MPD

Run all five periods and preserve separate score/feature lists.

## Task 4: Build one ScaleDiscriminator

Track grouped Conv1d channels, strides, scores, and feature maps.

## Task 5: Create MSD pooling views

Compare original, once-pooled, and twice-pooled waveform lengths.

## Task 6: Implement discriminator loss

Use hand-designed scores and predict real, fake, and total terms.

## Task 7: Implement generator adversarial loss

Show why a fake score equal to one gives zero generator adversarial loss.

## Task 8: Implement feature matching

Verify real features receive no gradient while generated features do.

## Task 9: Implement mel reconstruction loss

Calculate mean L1 and apply the reference factor `45`.

## Task 10: Implement masked KL loss

Use tiny Gaussian tensors and prove padding does not change the normalized result.

## Task 11: Demonstrate detach behavior

Run toy D and G updates and inspect which parameter gradients exist.

## Task 12: Combine generator objectives

Add adversarial, feature, mel, KL, and duration terms; inspect their relative scales.

## Task 13: Compare with reference losses

Use identical synthetic inputs for exact numerical checks.

## Task 14: Final assertions

Collect discriminator shapes, loss values, gradient ownership, and mask checks.

# Debugging order

1. Pair real and fake outputs from the same discriminator.
2. Reflect-pad before period reshape.
3. Preserve nested feature-map structure.
4. Detach real feature targets.
5. Detach `wav_hat` only for the D step.
6. Normalize KL by valid mask sum.
7. Inspect every loss for finite values and scale.

# Questions you must answer

1. Why do periodic views reveal pitch structure?
2. Why inspect several waveform scales?
3. Which targets define LSGAN?
4. Why detach real feature maps?
5. Why is mel loss heavily weighted?
6. Which gradients should exist during each optimizer step?

# Completion criteria

Chapter 11 is complete when you can trace MPD/MSD shapes, derive every loss, prove correct gradient ownership, and match reference loss values.

Chapter 12 integrates every component into complete training and inference paths.
