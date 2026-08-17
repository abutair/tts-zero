# Digital Audio Foundations for VITS2: Sampling, Frequency, and Sine Waves

This directory contains small experiments used to build the prerequisite knowledge for implementing VITS2 from scratch. The current checkpoint focuses on tensor shapes, digital audio sampling, frequency, and generating a controlled sine wave before returning to STFT and mel-spectrogram extraction.

## Learning approach

Each concept is handled in this order:

1. Understand its physical or mathematical meaning.
2. Predict tensor values and shapes by hand.
3. Implement a tiny isolated experiment.
4. Assert the expected invariants.
5. Connect the result to the VITS2 pipeline.

## 1. Waveform samples

A digital waveform is a sequence of amplitude measurements:

```text
waveform = [0.0, 0.8, 0.2, -0.7]
```

The sample index and amplitude are different quantities:

```text
waveform[2] = 0.2
```

This means that the measurement stored at index `2` has amplitude `0.2`.

Python uses zero-based indexing:

| Measurement | Index | Amplitude |
|---|---:|---:|
| First | `0` | `0.0` |
| Second | `1` | `0.8` |
| Third | `2` | `0.2` |
| Fourth | `3` | `-0.7` |

## 2. Sample rate

The sample rate is the number of waveform measurements taken per second:

```text
sample_rate = samples per second
```

For example:

```text
sample_rate = 4
```

means that four measurements are taken every second. The time separating consecutive samples is:

```text
seconds_per_sample = 1 / sample_rate
```

At four samples per second:

```text
seconds_per_sample = 1 / 4 = 0.25 seconds
```

## 3. Mapping a sample index to time

The timestamp of a sample is:

```text
time = sample_index / sample_rate
```

At four samples per second:

| Index | Time |
|---:|---:|
| `0` | `0.00 s` |
| `1` | `0.25 s` |
| `2` | `0.50 s` |
| `3` | `0.75 s` |

Keep these meanings separate:

- The index identifies which measurement is stored.
- Time says when that measurement occurred.
- Amplitude is the value measured at that time.

## 4. Frequency and cycles

Frequency describes how many complete cycles a wave makes per second:

```text
1 Hz   = 1 cycle per second
2 Hz   = 2 cycles per second
440 Hz = 440 cycles per second
```

For sound, a higher frequency generally produces a higher perceived pitch. Frequency is different from amplitude:

- Frequency controls how quickly the waveform repeats and relates to pitch.
-
- Amplitude controls the waveform's height and relates to loudness.

## 5. Samples per cycle

The number of measurements available for each wave cycle is:

```text
samples_per_cycle = sample_rate / frequency
```

Example:

```text
sample_rate = 12 samples per second
frequency   = 3 cycles per second

samples_per_cycle = 12 / 3 = 4
```

## 6. Cycle progress

At a particular time, the number of cycles that have elapsed is:

```text
cycles_elapsed = frequency * time
```

Example:

```text
frequency = 2 Hz
time      = 0.75 seconds

cycles_elapsed = 2 * 0.75 = 1.5 cycles
```

This represents one completed cycle plus halfway through the next cycle.

## 7. Turning cycle progress into amplitude

One complete cycle corresponds to an angle of `2*pi`. A sine wave's amplitude is therefore:

```text
amplitude = sin(2*pi * cycles_elapsed)
```

Because `cycles_elapsed = frequency * time`, the equivalent formula is:

```text
amplitude = sin(2*pi * frequency * time)
```

Important positions in one sine cycle are:

| Cycle position | Angle | Amplitude |
|---:|---:|---:|
| `0.00` | `0` | `0` |
| `0.25` | `pi/2` | `+1` |
| `0.50` | `pi` | `0` |
| `0.75` | `3*pi/2` | `-1` |
| `1.00` | `2*pi` | `0` |

## 8. Completed toy experiment

The first controlled experiment used:

```text
sample_rate    = 4
frequency      = 1 Hz
waveform_length = 4 samples
```

It produced:

```text
sample indices: [0, 1, 2, 3]
time:           [0.00, 0.25, 0.50, 0.75]
cycles elapsed: [0.00, 0.25, 0.50, 0.75]
amplitude:      [0, +1, approximately 0, -1]
```

A result such as `-8.7423e-08` is effectively zero. It occurs because floating-point arithmetic represents values such as `pi` approximately.

## 9. Why generate an artificial sine wave?

Generating a sine wave is a learning and debugging exercise, not a required operation in the final VITS2 data pipeline.

A known `440 Hz` signal provides ground truth:

```text
known 440 Hz waveform
        -> STFT
        -> strongest energy should appear near 440 Hz
```

If the STFT does not locate that frequency, likely causes include an incorrect sample rate, FFT configuration, tensor shape, or frequency-bin calculation.

This matters because VITS2 ultimately depends on correctly extracted acoustic features:

```text
waveform
  -> STFT
  -> magnitude spectrogram
  -> mel filter bank
  -> log-mel spectrogram
  -> VITS2 acoustic path
```

## 10. Current tensor-shape knowledge

Earlier practice established these conventions:

```text
raw sequence:     [B, T]
encoded features: [B, C, T]
sequence mask:    [B, 1, T]
```

A mask shaped `[B, 1, T]` broadcasts across the channel dimension of `[B, C, T]`. A raw `[B, T]` tensor should normally use a `[B, T]` mask to avoid accidental broadcasting into `[B, B, T]`.

## 11. Completed 440 Hz experiment

The chapter experiment generates a controlled signal using:

```text
sample_rate     = 22,050 samples per second
frequency       = 440 cycles per second
duration        = 1 second
waveform_length = 22,050 samples
```

The data flow is:

```text
sample indices [T]
    -> divide by sample rate
physical time [T]
    -> multiply by frequency
cycles elapsed [T]
    -> sin(2*pi*cycles_elapsed)
amplitude [T]
    -> unsqueeze(0)
batched waveform [1,T]
```

The final tensor has these invariants:

```text
shape:     [1, 22050]
dtype:     float32
amplitude: between -1 and +1
```

The number of measurements per cycle is:

```text
samples_per_cycle = sample_rate / frequency
                  = 22050 / 440
                  = approximately 50.11
```

It is not an integer, so cycle boundaries generally occur between stored samples.

## 12. Why structural assertions are insufficient

Shape, dtype, and range checks cannot prove that a waveform has the intended frequency. For example, this incorrect formula:

```text
sin(pi*cycles_elapsed)
```

produces approximately `220 Hz`, but it still has the correct shape, uses `float32`, and remains inside `[-1,+1]`.

The correct formula is:

```text
sin(2*pi*cycles_elapsed)
```

This demonstrates the difference between:

- Structural invariants: shape, dtype, and range.
- Semantic invariants: whether the data represents the intended physical signal.

## 13. Verifying frequency with upward zero crossings

An upward zero crossing occurs when two adjacent samples satisfy:

```text
current sample < 0
next sample >= 0
```

The comparison uses shifted views of the same amplitude tensor:

```text
current samples: amplitude[:-1]
next samples:    amplitude[1:]
```

A sine wave has approximately one upward zero crossing per cycle. A one-second `440 Hz` signal therefore has approximately `440` crossings.

The generated buffer starts exactly at zero and ends one sample before `1.0` second. Because the initial zero is not preceded by a negative sample and the final boundary is excluded, the discrete comparison normally counts `439` crossings. Accepting a difference of at most one verifies the intended frequency while respecting this boundary convention.

## 14. Chapter completion

Chapter 02 is complete when all of these statements hold:

- Sample indices map correctly to physical time.
- The waveform completes approximately `440` cycles in one second.
- The waveform shape is `[1,22050]`.
- The dtype is `float32`.
- Every amplitude lies inside `[-1,+1]`.
- The upward-zero-crossing count differs from `440` by no more than one.

The next chapter is [STFT and mel features](../03-stft-and-mel-features/README.md). It will test whether the STFT places the controlled waveform's strongest energy near `440 Hz`.

## Related practice files

- [Tensor-foundations experiment](../01-tensor-foundations/experiment.ipynb)
- [Tensor-foundations notes](../01-tensor-foundations/README.md)
- [Chapter 02 experiment](experiment.ipynb)
- [STFT and mel experiment](../03-stft-and-mel-features/experiment.ipynb)
- [STFT and mel notes](../03-stft-and-mel-features/README.md)
