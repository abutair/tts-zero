# VITS2 From-Scratch Practice Notes

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

## Next checkpoint

Generate a one-second `440 Hz` sine wave using:

```text
sample_rate     = 22,050
frequency       = 440 Hz
duration        = 1 second
waveform_length = sample_rate * duration
```

Expected shapes:

```text
sample_indices: [22050]
time:           [22050]
amplitude:      [22050]
batched waveform: [1, 22050]
```

After verifying this waveform, apply the STFT and check whether its strongest frequency bin corresponds approximately to `440 Hz`.

## Related practice files

- `1.1Tensor Shapes, Masks, and Broadcasting.ipynb`
- `1.1Tensor Shapes, Masks, and Broadcasting — Notes.md`
- `1.2-audtio-sampling.ipynb`
- `1.2Audio Sampling & STFT Notes.md`

