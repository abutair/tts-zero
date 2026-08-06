# Audio Sampling and STFT Notes

## 1. What Is the Sample Rate?

The **sample rate** is the number of times an analog sound wave is measured every second.

Example:

```python
sample_rate = 22050
```

This means:

* 22,050 samples represent one second of audio.
* The sample rate is not the waveform itself.
* It tells us how waveform samples map to time.
* It allows us to convert between sample positions and time.

---

## 2. What Is a Sample?

A **sample** is one measurement of an audio signal's amplitude at a specific instant in time.

A waveform is a sequence of samples:

```text
[0.12, 0.25, -0.10, 0.33, ...]
```

Each number is one sample.

The time between consecutive samples is:

```text
seconds_per_sample = 1 / sample_rate
```

For a sample rate of 22,050 Hz:

```text
seconds_per_sample = 1 / 22050
                   = 0.00004535 seconds
                   = 45.35 microseconds
```

---

## 3. Visualizing the Sample Rate

```text
sample_rate = 22050

Time:

0 seconds                                      1 second
|---------------------------------------------------|

Sample index:

0    1    2    3    ...                    22049

Total number of samples: 22,050
```

Therefore:

```text
22,050 samples = 1 second
```

---

# Important Equations

## 4. Audio Duration

```text
duration_seconds = number_of_samples / sample_rate
```

Example:

```text
duration_seconds = 22050 / 22050
                 = 1 second
```

---

## 5. Seconds per Sample

```text
seconds_per_sample = 1 / sample_rate
```

Example:

```text
seconds_per_sample = 1 / 22050
                   = 0.00004535 seconds
                   = 45.35 microseconds
```

---

## 6. Window Duration

An STFT window contains a fixed number of waveform samples.

```text
window_duration = win_length / sample_rate
```

Given:

```python
win_length = 1024
sample_rate = 22050
```

The duration is:

```text
window_duration = 1024 / 22050
                = 0.0464 seconds
                = 46.4 milliseconds
```

Each STFT window therefore analyzes approximately **46.4 milliseconds** of audio.

---

## 7. Hop Duration

The hop length determines how far the analysis window moves before calculating the next STFT frame.

```text
hop_duration = hop_length / sample_rate
```

Given:

```python
hop_length = 256
sample_rate = 22050
```

The duration is:

```text
hop_duration = 256 / 22050
             = 0.0116 seconds
             = 11.6 milliseconds
```

A new STFT frame is produced approximately every **11.6 milliseconds**.

---

## 8. Window Overlap

Neighboring windows overlap when `win_length` is greater than `hop_length`.

```text
overlap = win_length - hop_length
```

Given:

```python
win_length = 1024
hop_length = 256
```

The overlap is:

```text
overlap = 1024 - 256
        = 768 samples
```

---

## 9. Visualizing STFT Windows

```text
Waveform:

|------------------------------------------------------------|

Window 1:
[========================]

Window 2:
         [========================]

Window 3:
                  [========================]
```

Each window:

* Contains 1,024 samples
* Starts 256 samples after the previous window
* Overlaps the previous window by 768 samples

---

## 10. Number of STFT Frames

Ignoring padding, the number of STFT frames is:

```text
number_of_frames =
    1 + floor((waveform_length - win_length) / hop_length)
```

Where:

* `waveform_length` is the total number of waveform samples.
* `win_length` is the number of samples analyzed in each window.
* `hop_length` is the distance between consecutive windows.
* `floor` means rounding down to the nearest whole number.

The number of frames determines the **time dimension** of the spectrogram.

> Some STFT implementations add padding. When padding is enabled, the exact number of frames may be different.

---

## 11. Why Does the STFT Have 513 Frequency Bins?

For real-valued audio, the positive-frequency and negative-frequency parts of the FFT are symmetric.

The negative-frequency half is redundant, so only the non-negative frequencies are normally retained.

For an even value of `n_fft`:

```text
frequency_bins = (n_fft / 2) + 1
```

Given:

```python
n_fft = 1024
```

The number of frequency bins is:

```text
frequency_bins = (1024 / 2) + 1
               = 512 + 1
               = 513
```

The additional `1` includes the zero-frequency, or DC, bin.

---

## 12. From STFT to Mel Spectrogram

An STFT commonly produces a matrix with these dimensions:

```text
frequency_bins x time_frames
```

Example:

```text
513 x 100
```

This means:

* 513 frequency bins
* 100 time frames

A mel filter bank projects the 513 linear-frequency bins into a smaller number of mel-frequency bins.

Example:

```text
STFT:

513 frequency bins x 100 time frames

Mel spectrogram:

80 mel bins x 100 time frames
```

The time dimension remains unchanged.

Only the frequency dimension changes:

```text
513 linear-frequency bins -> 80 mel-frequency bins
```

This happens because the mel filter bank operates on the frequency axis of each time frame.

> Some libraries return spectrograms as `frequency x time`, while others may use `time x frequency`. Always inspect the tensor shape used by the library.

---

## 13. Spectrogram Intuition

A spectrogram can be viewed as an image:

```text
                  Time
                   ->

Frequency     ####################
    ^         ####################
    |         ####################
    |         ####################
```

Typically:

* Rows represent frequency.
* Columns represent time.

Parameter effects:

* Increasing `hop_length` normally produces fewer time frames.
* Decreasing `hop_length` normally produces more time frames.
* Increasing `n_mels` produces more mel-frequency rows.
* Increasing `n_fft` produces more linear-frequency bins.

---

# Questions and Answers

## Q1. How long is a waveform containing 22,050 samples?

Given:

```python
sample_rate = 22050
num_samples = 22050
```

Equation:

```text
duration = num_samples / sample_rate
```

Calculation:

```text
duration = 22050 / 22050
         = 1 second
```

---

## Q2. How much audio does a 1,024-sample window cover?

Given:

```python
win_length = 1024
sample_rate = 22050
```

Equation:

```text
window_duration = win_length / sample_rate
```

Calculation:

```text
window_duration = 1024 / 22050
                = approximately 0.0464 seconds
                = approximately 46.4 milliseconds
```

---

## Q3. How much time separates consecutive STFT frames?

Given:

```python
hop_length = 256
sample_rate = 22050
```

Equation:

```text
hop_duration = hop_length / sample_rate
```

Calculation:

```text
hop_duration = 256 / 22050
             = approximately 0.0116 seconds
             = approximately 11.6 milliseconds
```

---

## Q4. Do Neighboring STFT Windows Overlap?

Yes.

This is because:

```text
win_length > hop_length

1024 > 256
```

The next window begins before the previous window ends.

---

## Q5. How Many Samples Overlap?

Equation:

```text
overlap = win_length - hop_length
```

Calculation:

```text
overlap = 1024 - 256
        = 768 samples
```

---

## Q6. Why Does Converting 513 STFT Bins to 80 Mel Bins Not Change the Time Dimension?

The mel filter bank transforms the frequency values inside every time frame.

It changes:

```text
513 frequency bins -> 80 mel bins
```

It does not create or remove time frames.

The number of time frames is mainly controlled by:

* Waveform length
* Window length
* Hop length
* Padding behavior

The frequency dimension is mainly controlled by:

* `n_fft`
* `n_mels`

---

# Parameters Used

```python
sample_rate = 22050
n_fft = 1024
win_length = 1024
hop_length = 256
n_mels = 80
```

| Parameter     | Meaning                                   | Effect of increasing it                   |
| ------------- | ----------------------------------------- | ----------------------------------------- |
| `sample_rate` | Number of samples recorded per second     | Represents each second using more samples |
| `n_fft`       | Number of points used by the FFT          | Produces more linear-frequency bins       |
| `win_length`  | Number of samples analyzed in each window | Analyzes a longer segment of audio        |
| `hop_length`  | Distance between consecutive windows      | Usually produces fewer time frames        |
| `n_mels`      | Number of mel-frequency bins              | Produces more mel-frequency rows          |

---

# Key Takeaways

* A waveform is a sequence of amplitude samples.
* The sample rate tells us how many samples represent one second.
* Audio duration equals the number of samples divided by the sample rate.
* The STFT divides a waveform into short, usually overlapping windows.
* `win_length` determines how much audio each frame analyzes.
* `hop_length` determines how far the window moves between frames.
* An FFT size of 1,024 produces 513 non-negative frequency bins for real-valued audio.
* A mel filter bank changes the frequency dimension but does not change the number of time frames.
* A mel spectrogram commonly has the logical shape:

```text
mel_bins x time_frames
```
