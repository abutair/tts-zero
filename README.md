# VITS2 From-Scratch Learning Path

This directory contains focused experiments and notes for rebuilding VITS2 while understanding each dependency before integrating the full model.

## Current chapters

| Chapter | Topic | Notes | Experiment |
|---:|---|---|---|
| 01 | Tensor shapes, masks, and broadcasting | [Read](01-tensor-foundations/README.md) | [Open](01-tensor-foundations/experiment.ipynb) |
| 02 | Digital audio sampling, frequency, and sine waves | [Read](02-digital-audio-foundations/README.md) | In progress |
| 03 | STFT and mel features | [Read](03-stft-and-mel-features/README.md) | [Open](03-stft-and-mel-features/experiment.ipynb) |

## Planned dependency order

1. Tensor foundations
2. Digital-audio foundations
3. STFT and mel features
4. Text processing and batching
5. Text encoder and prior distribution
6. Posterior encoder
7. Normalizing flow
8. Monotonic Alignment Search
9. Duration prediction
10. Waveform decoder
11. Discriminators and losses
12. Full training and inference paths

## File convention

Each numbered chapter uses:

- `README.md` for concepts, equations, and conclusions.
- `experiment.ipynb` for code written during practice.
- `tests.py` for focused checks when a notebook is no longer sufficient.

Temporary experiments belong in `scratch/` and should move into a numbered chapter once their purpose is clear.
