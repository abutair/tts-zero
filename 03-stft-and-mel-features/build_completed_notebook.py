"""Build the completed, self-contained Chapter 03 experiment notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "experiment.ipynb"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# Chapter 03: STFT and Mel Features for VITS2

This notebook completes the acoustic-feature checkpoint using a controlled
`440 Hz` sine wave. Every important output is predicted, inspected, plotted,
and asserted before moving to the next transformation.

The complete path is:

```text
waveform [B, T_wav]
    -> complex STFT [B, 513, T_frames]
    -> magnitude [B, 513, T_frames]
    -> mel projection [B, 80, T_frames]
    -> log-mel [B, 80, T_frames]
```

This notebook is self-contained. `torchaudio` is optional; the required HTK
mel filter bank is constructed from first principles.
"""
    ),
    code(
        r"""
import math

import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)
torch.set_printoptions(precision=5, sci_mode=False)
"""
    ),
    markdown(
        r"""
## 1. Recreate the controlled 440 Hz waveform

A sample index identifies a measurement. Dividing by the sample rate converts
that index into seconds. Multiplying time by frequency gives elapsed cycles,
and one complete sine cycle corresponds to `2*pi` radians.
"""
    ),
    code(
        r"""
sample_rate = 22_050
tone_frequency_hz = 440.0
duration_seconds = 1

waveform_length = sample_rate * duration_seconds
sample_indices = torch.arange(waveform_length, dtype=torch.float32)
time_seconds = sample_indices / sample_rate
cycles_elapsed = tone_frequency_hz * time_seconds
amplitude = torch.sin(2 * torch.pi * cycles_elapsed)
waveform = amplitude.unsqueeze(0)  # [T] -> [B, T]

current_samples = amplitude[:-1]
next_samples = amplitude[1:]
upward_crossings = (current_samples < 0) & (next_samples >= 0)
upward_crossing_count = upward_crossings.sum().item()

print("Waveform shape:", waveform.shape)
print("Waveform dtype:", waveform.dtype)
print("First timestamp:", time_seconds[0].item(), "seconds")
print("Final timestamp:", time_seconds[-1].item(), "seconds")
print("Minimum amplitude:", waveform.min().item())
print("Maximum amplitude:", waveform.max().item())
print("Upward zero crossings:", upward_crossing_count)
"""
    ),
    code(
        r"""
assert waveform.shape == (1, 22_050)
assert waveform.dtype == torch.float32
assert torch.all(waveform >= -1.0)
assert torch.all(waveform <= 1.0)
assert abs(upward_crossing_count - tone_frequency_hz * duration_seconds) <= 1

print("Controlled-waveform checks passed.")
"""
    ),
    code(
        r"""
# Display only the first 10 ms so individual cycles remain visible.
visible_samples = int(sample_rate * 0.010)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(
    time_seconds[:visible_samples].numpy() * 1000,
    amplitude[:visible_samples].numpy(),
    color="tab:blue",
)
ax.axhline(0, color="0.65", linewidth=0.8)
ax.set(
    title="First 10 ms of the controlled 440 Hz waveform",
    xlabel="Time (ms)",
    ylabel="Amplitude",
    ylim=(-1.1, 1.1),
)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 2. Define and predict the STFT configuration

`win_length` controls how much audio each frame examines. `hop_length` controls
how far the next frame starts. With `center=True`, PyTorch pads approximately
`n_fft // 2` samples on both sides so frame time points align with signal time.

The unit conversion is important:

```text
samples / (samples per second) = seconds
seconds * 1000 = milliseconds
```
"""
    ),
    code(
        r"""
n_fft = 1_024
win_length = 1_024
hop_length = 256
center = True

n_mels = 80
f_min_hz = 0.0
f_max_hz = 8_000.0

window_duration_ms = win_length / sample_rate * 1000
hop_duration_ms = hop_length / sample_rate * 1000
overlap_samples = win_length - hop_length
overlap_percentage = overlap_samples / win_length * 100

frequency_bins = n_fft // 2 + 1
bin_spacing_hz = sample_rate / n_fft

padding_per_side = n_fft // 2 if center else 0
effective_length = waveform_length + 2 * padding_per_side
predicted_frames = 1 + (effective_length - n_fft) // hop_length
predicted_stft_shape = (
    waveform.shape[0],
    frequency_bins,
    predicted_frames,
)

print("Window duration:", window_duration_ms, "ms")
print("Hop duration:", hop_duration_ms, "ms")
print("Overlap:", overlap_samples, "samples")
print("Overlap:", overlap_percentage, "%")
print("Frequency-bin count:", frequency_bins)
print("Bin spacing:", bin_spacing_hz, "Hz")
print("Padding per side:", padding_per_side, "samples")
print("Effective length:", effective_length, "samples")
print("Predicted frames:", predicted_frames)
print("Predicted STFT shape:", predicted_stft_shape)
"""
    ),
    code(
        r"""
assert math.isclose(window_duration_ms, 46.4399093, rel_tol=1e-7)
assert math.isclose(hop_duration_ms, 11.6099773, rel_tol=1e-7)
assert overlap_samples == 768
assert overlap_percentage == 75.0
assert frequency_bins == 513
assert math.isclose(bin_spacing_hz, 21.533203125, rel_tol=1e-12)
assert padding_per_side == 512
assert effective_length == 23_074
assert predicted_frames == 87
assert predicted_stft_shape == (1, 513, 87)
assert 0 <= f_min_hz < f_max_hz <= sample_rate / 2

print("STFT-configuration predictions passed.")
"""
    ),
    markdown(
        r"""
### Predict the tone's FFT-bin location

FFT bins form a discrete frequency ruler. Bin `k` represents
`k * sample_rate / n_fft` Hz. A `440 Hz` tone lies at fractional bin `20.43`,
between bins `20` and `21`.
"""
    ),
    code(
        r"""
fractional_tone_bin = tone_frequency_hz / bin_spacing_hz
predicted_peak_bin = round(fractional_tone_bin)
predicted_peak_hz = predicted_peak_bin * bin_spacing_hz
predicted_peak_error_hz = abs(tone_frequency_hz - predicted_peak_hz)

print("Fractional tone bin:", fractional_tone_bin)
print("Nearest bin:", predicted_peak_bin)
print("Nearest bin center:", predicted_peak_hz, "Hz")
print("Distance from target:", predicted_peak_error_hz, "Hz")

assert predicted_peak_bin == 20
assert predicted_peak_error_hz < bin_spacing_hz
"""
    ),
    markdown(
        r"""
## 3. Construct the Hann analysis window

Abrupt frame boundaries create artificial frequency content. A Hann window
suppresses frame edges and emphasizes the center, reducing spectral leakage.
It must match the waveform's dtype and device.
"""
    ),
    code(
        r"""
hann_window = torch.hann_window(
    win_length,
    periodic=True,
    dtype=waveform.dtype,
    device=waveform.device,
)

print("Hann-window shape:", hann_window.shape)
print("Hann-window dtype:", hann_window.dtype)
print("First value:", hann_window[0].item())
print("Center value:", hann_window[win_length // 2].item())
print("Final value:", hann_window[-1].item())

assert hann_window.shape == (win_length,)
assert hann_window.dtype == waveform.dtype
assert hann_window.device == waveform.device
assert torch.isfinite(hann_window).all()
assert torch.all(hann_window >= 0)
assert math.isclose(hann_window[0].item(), 0.0, abs_tol=1e-7)
assert math.isclose(hann_window[win_length // 2].item(), 1.0, abs_tol=1e-7)
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(hann_window.cpu().numpy(), color="tab:green")
ax.set(
    title="Hann analysis window",
    xlabel="Sample within the frame",
    ylabel="Weight",
    ylim=(-0.05, 1.05),
)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 4. Compute the complex STFT

Each STFT value is complex: it contains both magnitude (strength) and phase
(cycle position). The expected shape is `[B, frequency_bins, time_frames]`.
"""
    ),
    code(
        r"""
complex_stft = torch.stft(
    waveform,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=hann_window,
    center=center,
    pad_mode="reflect",
    normalized=False,
    onesided=True,
    return_complex=True,
)

print("Observed STFT shape:", complex_stft.shape)
print("STFT dtype:", complex_stft.dtype)
print("Is complex:", torch.is_complex(complex_stft))

assert complex_stft.shape == predicted_stft_shape
assert complex_stft.dtype == torch.complex64
assert torch.is_complex(complex_stft)
assert torch.isfinite(complex_stft).all()
"""
    ),
    markdown(
        r"""
## 5. Convert complex values to magnitude

For `z = real + imaginary*j`, magnitude is
`sqrt(real**2 + imaginary**2)`. Magnitude removes phase without changing the
tensor shape. The reference project uses magnitude (`power=1.0`), not squared
power.
"""
    ),
    code(
        r"""
magnitude = complex_stft.abs()

print("Magnitude shape:", magnitude.shape)
print("Magnitude dtype:", magnitude.dtype)
print("Magnitude minimum:", magnitude.min().item())
print("Magnitude maximum:", magnitude.max().item())

assert magnitude.shape == complex_stft.shape
assert magnitude.dtype == torch.float32
assert not torch.is_complex(magnitude)
assert torch.all(magnitude >= 0)
assert torch.isfinite(magnitude).all()
"""
    ),
    markdown(
        r"""
## 6. Detect the controlled 440 Hz tone

The signal is stationary, so averaging magnitude over batch and time gives a
stable frequency summary. The peak-bin estimate is expected to be within one
FFT-bin spacing of `440 Hz`.
"""
    ),
    code(
        r"""
mean_magnitude_by_bin = magnitude.mean(dim=(0, 2))
observed_peak_bin = int(mean_magnitude_by_bin.argmax().item())
observed_peak_hz = observed_peak_bin * bin_spacing_hz
observed_peak_error_hz = abs(tone_frequency_hz - observed_peak_hz)

print("Observed peak bin:", observed_peak_bin)
print("Observed peak frequency:", observed_peak_hz, "Hz")
print("Target frequency:", tone_frequency_hz, "Hz")
print("Absolute error:", observed_peak_error_hz, "Hz")
print("One-bin spacing:", bin_spacing_hz, "Hz")

assert observed_peak_bin in (20, 21)
assert observed_peak_error_hz <= bin_spacing_hz
"""
    ),
    code(
        r"""
plot_start_bin = 15
plot_end_bin = 27
plot_bins = torch.arange(plot_start_bin, plot_end_bin)
plot_frequencies = plot_bins * bin_spacing_hz

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(
    plot_frequencies.numpy(),
    mean_magnitude_by_bin[plot_start_bin:plot_end_bin].numpy(),
    width=bin_spacing_hz * 0.72,
    color="tab:blue",
    alpha=0.8,
)
ax.axvline(tone_frequency_hz, color="tab:red", linestyle="--", label="440 Hz target")
ax.axvline(observed_peak_hz, color="tab:green", linestyle=":", label="Peak-bin center")
ax.set(
    title="Mean STFT magnitude near the controlled tone",
    xlabel="Frequency (Hz)",
    ylabel="Mean magnitude",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 7. Construct an HTK mel filter bank from first principles

FFT bins are evenly spaced in Hz, but human frequency perception is not linear.
The mel scale allocates relatively more resolution to low frequencies.

For the HTK convention:

```text
mel = 2595 * log10(1 + hz / 700)
hz  = 700 * (10 ** (mel / 2595) - 1)
```

The filter bank contains `80` overlapping triangles evaluated at all `513`
FFT-bin frequencies. Its chosen orientation is `[n_mels, frequency_bins]`.
"""
    ),
    code(
        r"""
def hz_to_mel(hz: torch.Tensor | float) -> torch.Tensor:
    hz_tensor = torch.as_tensor(hz, dtype=torch.float32)
    return 2595.0 * torch.log10(1.0 + hz_tensor / 700.0)


def mel_to_hz(mel: torch.Tensor | float) -> torch.Tensor:
    mel_tensor = torch.as_tensor(mel, dtype=torch.float32)
    return 700.0 * (torch.pow(10.0, mel_tensor / 2595.0) - 1.0)


minimum_mel = hz_to_mel(f_min_hz)
maximum_mel = hz_to_mel(f_max_hz)

# Two extra points provide the left boundary of the first triangle and the
# right boundary of the final triangle.
mel_boundary_points = torch.linspace(
    minimum_mel.item(),
    maximum_mel.item(),
    n_mels + 2,
    dtype=torch.float32,
)
hz_boundary_points = mel_to_hz(mel_boundary_points)
fft_bin_frequencies_hz = torch.linspace(
    0.0,
    sample_rate / 2,
    frequency_bins,
    dtype=torch.float32,
)

mel_filter_bank = torch.zeros(n_mels, frequency_bins, dtype=torch.float32)

for mel_index in range(n_mels):
    left_hz = hz_boundary_points[mel_index]
    center_hz = hz_boundary_points[mel_index + 1]
    right_hz = hz_boundary_points[mel_index + 2]

    rising_slope = (fft_bin_frequencies_hz - left_hz) / (center_hz - left_hz)
    falling_slope = (right_hz - fft_bin_frequencies_hz) / (right_hz - center_hz)

    triangle = torch.minimum(rising_slope, falling_slope)
    mel_filter_bank[mel_index] = torch.clamp(triangle, min=0.0)

print("Mel-filter-bank shape:", mel_filter_bank.shape)
print("Minimum filter weight:", mel_filter_bank.min().item())
print("Maximum filter weight:", mel_filter_bank.max().item())
print("All filters contain positive weight:", bool((mel_filter_bank.sum(dim=1) > 0).all()))

assert mel_filter_bank.shape == (80, 513)
assert torch.all(mel_filter_bank >= 0)
assert torch.all(mel_filter_bank <= 1)
assert torch.isfinite(mel_filter_bank).all()
assert (mel_filter_bank.sum(dim=1) > 0).all()
"""
    ),
    code(
        r"""
selected_filter_indices = [0, 10, 30, 50, 79]

fig, ax = plt.subplots(figsize=(10, 4))
for filter_index in selected_filter_indices:
    ax.plot(
        fft_bin_frequencies_hz.numpy(),
        mel_filter_bank[filter_index].numpy(),
        label=f"mel filter {filter_index}",
    )
ax.set(
    title="Selected triangular HTK mel filters",
    xlabel="Frequency (Hz)",
    ylabel="Filter weight",
    xlim=(0, f_max_hz + 300),
    ylim=(-0.02, 1.05),
)
ax.legend(ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
## 8. Project magnitude into mel bands and apply log compression

The matrix multiplication is applied independently to every batch item and
time frame:

```text
[80, 513] x [B, 513, T_frames] -> [B, 80, T_frames]
```

The mel values are clamped before `log` because `log(0)` is negative infinity.
Negative log-mel values are valid; all final values must be finite.
"""
    ),
    code(
        r"""
mel_spectrogram = torch.einsum(
    "mf,bft->bmt",
    mel_filter_bank,
    magnitude,
)
log_mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))

print("Mel-spectrogram shape:", mel_spectrogram.shape)
print("Log-mel shape:", log_mel_spectrogram.shape)
print("Mel minimum:", mel_spectrogram.min().item())
print("Mel maximum:", mel_spectrogram.max().item())
print("Log-mel minimum:", log_mel_spectrogram.min().item())
print("Log-mel maximum:", log_mel_spectrogram.max().item())

assert mel_spectrogram.shape == (1, 80, 87)
assert log_mel_spectrogram.shape == mel_spectrogram.shape
assert mel_spectrogram.shape[-1] == magnitude.shape[-1]
assert mel_spectrogram.dtype == torch.float32
assert torch.all(mel_spectrogram >= 0)
assert torch.isfinite(mel_spectrogram).all()
assert torch.isfinite(log_mel_spectrogram).all()
assert not torch.isnan(log_mel_spectrogram).any()
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

axes[0].imshow(
    magnitude[0].numpy(),
    origin="lower",
    aspect="auto",
    cmap="magma",
    extent=[0, duration_seconds, 0, sample_rate / 2],
)
axes[0].set(
    title="Linear-frequency magnitude spectrogram",
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    ylim=(0, 2_000),
)

image = axes[1].imshow(
    log_mel_spectrogram[0].numpy(),
    origin="lower",
    aspect="auto",
    cmap="magma",
    extent=[0, duration_seconds, 0, n_mels - 1],
)
axes[1].set(
    title="Log-mel spectrogram",
    xlabel="Time (s)",
    ylabel="Mel-filter index",
)
fig.colorbar(image, ax=axes[1], label="Natural-log magnitude")
plt.show()
"""
    ),
    markdown(
        r"""
## 9. Optional comparison with torchaudio

The project reference uses `torchaudio.transforms.MelSpectrogram` with
`power=1.0`, `norm=None`, and `mel_scale="htk"`. If torchaudio is installed,
this cell compares the manual result with that trusted implementation. The
chapter remains fully runnable when torchaudio is unavailable.
"""
    ),
    code(
        r"""
try:
    import torchaudio

    reference_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min_hz,
        f_max=f_max_hz,
        n_mels=n_mels,
        window_fn=torch.hann_window,
        power=1.0,
        normalized=False,
        center=center,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
    )
    reference_mel = reference_transform(waveform)
    maximum_reference_difference = (reference_mel - mel_spectrogram).abs().max().item()

    print("Torchaudio reference shape:", reference_mel.shape)
    print("Maximum absolute difference:", maximum_reference_difference)

    torch.testing.assert_close(
        mel_spectrogram,
        reference_mel,
        rtol=2e-4,
        atol=2e-4,
    )
    print("Manual and torchaudio mel spectrograms agree.")
except (ImportError, OSError) as error:
    print("Optional torchaudio comparison skipped:", error)
"""
    ),
    markdown(
        r"""
## 10. Convert waveform lengths into feature masks

Real batches contain padded waveforms with different valid lengths. Feature
lengths must use the same framing and centering convention as the STFT. The
resulting `[B, 1, T_frames]` mask broadcasts across acoustic channels.
"""
    ),
    code(
        r"""
def make_sequence_mask(lengths: torch.Tensor, maximum_length: int) -> torch.Tensor:
    positions = torch.arange(maximum_length, device=lengths.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).unsqueeze(1)


example_waveform_lengths = torch.tensor([22_050, 16_000], dtype=torch.long)
feature_lengths = 1 + (
    example_waveform_lengths + 2 * padding_per_side - n_fft
) // hop_length
feature_mask = make_sequence_mask(feature_lengths, predicted_frames)

print("Waveform lengths:", example_waveform_lengths)
print("Feature lengths:", feature_lengths)
print("Feature-mask shape:", feature_mask.shape)
print("Valid feature positions:", feature_mask.sum().item())

assert feature_lengths.tolist() == [87, 63]
assert feature_mask.shape == (2, 1, 87)
assert feature_mask.dtype == torch.bool
assert feature_mask.sum() == feature_lengths.sum()
assert feature_mask[0, 0].all()
assert feature_mask[1, 0, :63].all()
assert not feature_mask[1, 0, 63:].any()
"""
    ),
    markdown(
        r"""
## 11. Final chapter verification

Structural checks prove shapes, dtypes, ranges, and finiteness. Semantic checks
prove that the known `440 Hz` signal is actually detected near the expected
frequency. Both kinds are necessary.
"""
    ),
    code(
        r"""
expected_shapes = {
    "waveform": (1, 22_050),
    "complex_stft": (1, 513, 87),
    "magnitude": (1, 513, 87),
    "mel_spectrogram": (1, 80, 87),
    "log_mel_spectrogram": (1, 80, 87),
}

observed_shapes = {
    "waveform": tuple(waveform.shape),
    "complex_stft": tuple(complex_stft.shape),
    "magnitude": tuple(magnitude.shape),
    "mel_spectrogram": tuple(mel_spectrogram.shape),
    "log_mel_spectrogram": tuple(log_mel_spectrogram.shape),
}

assert observed_shapes == expected_shapes
assert torch.is_complex(complex_stft)
assert torch.all(magnitude >= 0)
assert torch.all(mel_spectrogram >= 0)
assert torch.isfinite(log_mel_spectrogram).all()
assert observed_peak_error_hz <= bin_spacing_hz
assert magnitude.shape[-1] == mel_spectrogram.shape[-1]

for name, shape in observed_shapes.items():
    print(f"{name:22s} -> {shape}")

print()
print("All Chapter 03 checks passed.")
"""
    ),
    markdown(
        r"""
## Conclusion

The controlled waveform has been converted into a finite model-ready log-mel
tensor while preserving the time-frame dimension:

```text
[1, 22050] -> [1, 513, 87] -> [1, 80, 87]
```

The STFT detected the `440 Hz` tone near bin `20`, the manually constructed HTK
mel bank compressed `513` linear-frequency bins into `80` perceptual bands, and
valid acoustic-frame masks were derived from waveform lengths.

Chapter 03 is complete. The next chapter should begin text processing and the
text-encoder prior path.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT} with {len(cells)} cells")

