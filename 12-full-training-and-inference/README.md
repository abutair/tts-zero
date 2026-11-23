# Full Training and Inference Paths for VITS2

This chapter assembles the data, text, acoustic, alignment, duration, flow, decoder, discriminator, optimization, checkpoint, and synthesis paths into one verified system.

Primary references: `training/models/vits2.py` and `training/trainers/trainer.py`.

## Complete training pathway

```text
text + waveform + lengths + speaker
  -> speaker embedding
  -> mel extraction
  -> text encoder: x, m_p, logs_p, x_mask
  -> posterior encoder: z, m_q, logs_q, y_mask
  -> flow forward: z -> z_p
  -> Gaussian scores + MAS path
  -> durations and duration loss
  -> expand prior from text time to mel time
  -> random z segment -> decoder -> wav_hat
  -> matching target waveform segment
  -> discriminator update
  -> generator update with all losses
```

## Complete inference pathway

```text
text + text length + speaker
  -> text encoder
  -> duration predictor reverse/sample
  -> exp, length_scale, ceil
  -> output length, y_mask, monotonic path
  -> expand m_p and logs_p over acoustic time
  -> z_p = m_p + noise * exp(logs_p) * noise_scale
  -> flow reverse: z_p -> z
  -> waveform decoder
  -> waveform
```

Training observes target audio and can use posterior encoding and MAS. Inference has only text and speaker, so it must use predicted durations, prior sampling, reverse flow, and full decoding.

## Speaker conditioning

The reference speaker table maps:

```text
speaker indices [B] -> speaker vectors [B,256]
```

The same vector conditions the text encoder, posterior encoder, flow, duration predictor, and decoder. Shape and speaker-map consistency are system-wide contracts.

## Three time axes

Never mix:

```text
T_text    token positions
T_mel     acoustic/latent frames
T_samples waveform samples
```

The expected relationship is approximately:

```text
T_samples ~= T_mel * hop_length
```

But actual mel lengths depend on STFT centering and padding. The reference calculates training `mel_length = wav_length // hop_length`; verify that this matches the extractor's actual frame count for your configuration.

## Expanding the text prior

MAS attention has shape `[B,T_text,T_mel]`. Expand token-level prior parameters with:

```text
m_p_mel    = einsum("bci,bij->bcj", m_p_text, attention)
logs_p_mel = einsum("bci,bij->bcj", logs_p_text, attention)
```

Each mel frame receives the parameters of its aligned token.

## Random segment training

The decoder trains on fixed latent segments rather than entire utterances. If segment size is 32 mel frames and hop length is 256:

```text
waveform segment = 32 * 256 = 8192 samples
```

Latent start index `i` corresponds to waveform sample start `i*hop_length`. The target slice must use the same start.

## Two optimization phases

The trainer runs a discriminator step and a generator step with separate optimizers and AMP scalers.

Discriminator:

1. run generator;
2. detach generated waveform;
3. compute real/fake discriminator scores;
4. backpropagate D loss;
5. unscale, clip D gradients, step D.

Generator:

1. run generator again with graph;
2. calculate mel, adversarial, feature, KL, and duration losses;
3. sum generator objective;
4. backpropagate;
5. unscale, clip G gradients, step G.

## Mixed precision and clipping order

With gradient scaling:

```text
scaled_loss.backward()
scaler.unscale_(optimizer)
clip_grad_norm_(parameters)
scaler.step(optimizer)
scaler.update()
```

Clipping scaled gradients would use meaningless magnitudes, so unscale first.

## Inference controls

- `noise_scale`: latent acoustic variation around the prior.
- `noise_scale_w`: stochastic duration variation.
- `length_scale`: multiplies durations and controls speaking rate.

Change one at a time when studying behavior.

## Checkpoint compatibility

A reproducible checkpoint depends on more than weights:

- generator and discriminator states;
- optimizer, scheduler, and scaler states as applicable;
- epoch and global step;
- model/audio configuration;
- vocabulary order and symbol count;
- speaker ID map;
- sample rate, hop, FFT, mel settings;
- code version and random seeds.

Changing vocabulary order or speaker mapping silently changes embedding meaning.

## Evaluation

Use fixed sentences, fixed speakers, controlled seeds/noise settings, consistent text cleaning, and the same audio configuration. Save audio plus configuration and checkpoint identity. Listening alone is not enough; also track loss curves, alignment validity, durations, waveform range, and nonfinite values.

# Experiment roadmap

## Task 1: Draw the complete shape graph

Label every training tensor from raw batch to each loss. Distinguish all three time axes.

## Task 2: Assemble speaker conditioning

Trace one speaker index through every conditioned component.

## Task 3: Verify waveform-to-mel lengths

Compare predicted frame lengths with actual extractor output under the configured centering behavior.

## Task 4: Run text and posterior encoders

Check masks, means, log-scales, and time dimensions.

## Task 5: Run flow and MAS

Transform `z` to `z_p`, score alignments, recover the path, and derive durations.

## Task 6: Expand prior parameters

Use einsum and verify each mel frame receives its aligned token parameters.

## Task 7: Slice latent and waveform segments

Choose valid starts and prove latent-frame/waveform-sample correspondence.

## Task 8: Assemble the discriminator step

Detach `wav_hat`, compute D loss, backward, unscale, clip, and update only D.

## Task 9: Assemble the generator step

Recompute outputs and combine adversarial, feature, mel, KL, and duration losses.

## Task 10: Add AMP and gradient clipping

Verify operation order and finite gradient norms.

## Task 11: Build checkpoint save/resume

Round-trip all required state and confirm epoch/global-step continuation.

## Task 12: Build deterministic evaluation

Use fixed sentences, speakers, seeds, and synthesis settings.

## Task 13: Trace inference durations

Predict `logw`, apply controls, ceil durations, and create `y_mask` plus path.

## Task 14: Trace inference latent synthesis

Expand prior, sample `z_p`, reverse flow, decode, and verify waveform length.

## Task 15: Test control parameters

Change `noise_scale`, `noise_scale_w`, and `length_scale` independently.

## Task 16: Create end-to-end smoke tests

Use a tiny configuration to catch NaNs, shape errors, invalid paths, broken gradients, and incorrect optimizer ownership.

## Task 17: Final system checklist

Document data, configuration, vocabulary, speaker-map, and checkpoint assumptions required for a reproducible run.

# Debugging order

1. Find the first boundary with the wrong shape.
2. Distinguish text, mel, and sample time.
3. Verify every length conversion and mask sum.
4. Check MAS path and duration invariants.
5. Verify latent/waveform slice correspondence.
6. Check detach and optimizer ownership.
7. Inspect each loss and gradient norm for finiteness.
8. Test checkpoint resume before long training.
9. Run a fixed inference case after every major change.

# Questions you must answer

1. Which modules are training-only?
2. Why must token-level priors be expanded?
3. How do latent and waveform slices correspond?
4. Why are D and G updated separately?
5. What does each inference control change?
6. Which metadata must remain checkpoint-compatible?
7. How would you locate the first source of NaNs?

# Completion criteria

Chapter 12 is complete when you can trace full training and inference graphs, run finite smoke-test losses and gradients, round-trip checkpoints, and synthesize a correctly sized waveform from text-only inputs.

After Chapter 12, return to the production model and replace each trusted reference component only when your independent chapter implementation passes its tests.
