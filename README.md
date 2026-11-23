# VITS2 From-Scratch Learning Path

This directory contains focused experiments and notes for rebuilding VITS2 while understanding each dependency before integrating the full model.

## Current chapters

| Chapter | Topic | Notes | Experiment |
|---:|---|---|---|
| 01 | Tensor shapes, masks, and broadcasting | [Read](01-tensor-foundations/README.md) | [Open](01-tensor-foundations/experiment.ipynb) |
| 02 | Digital audio sampling, frequency, and sine waves | [Read](02-digital-audio-foundations/README.md) |  [Open](02-digital-audio-foundations/experiment.ipynb)|
| 03 | STFT and mel features | [Read](03-stft-and-mel-features/README.md) | [Open](03-stft-and-mel-features/experiment.ipynb) |
| 04 | Arabic text processing and batching | [Read](04-text-processing-and-batching/README.md) | [Open](04-text-processing-and-batching/experiment.ipynb) |
| 05 | Text encoder and prior distribution | [Read](05-text-encoder-and-prior/README.md) | [Open](05-text-encoder-and-prior/experiment.ipynb) |
| 06 | Posterior encoder | [Read](06-posterior-encoder/README.md) | [Open](06-posterior-encoder/experiment.ipynb) |
| 07 | Normalizing flow | [Read](07-normalizing-flow/README.md) | [Open](07-normalizing-flow/experiment.ipynb) |
| 08 | Monotonic Alignment Search | [Read](08-monotonic-alignment-search/README.md) | [Open](08-monotonic-alignment-search/experiment.ipynb) |
| 09 | Duration prediction | [Read](09-duration-prediction/README.md) | [Open](09-duration-prediction/experiment.ipynb) |
| 10 | Waveform decoder | [Read](10-waveform-decoder/README.md) | [Open](10-waveform-decoder/experiment.ipynb) |
| 11 | Discriminators and losses | [Read](11-discriminators-and-losses/README.md) | [Open](11-discriminators-and-losses/experiment.ipynb) |
| 12 | Full training and inference paths | [Read](12-full-training-and-inference/README.md) | [Open](12-full-training-and-inference/experiment.ipynb) |

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
