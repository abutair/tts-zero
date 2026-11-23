# Waveform Decoder for VITS2

This chapter converts latent acoustic frames into waveform samples using the HiFi-GAN generator.

Primary reference: `HiFiGANGenerator`, `ResBlock1`, and `ResBlock2` in `training/models/modules/decoder.py`.

## End-to-end pathway

```text
latent z [B,192,T_frames]
    -> Conv1d kernel 7, channels 512
    -> upsample x8, channels 256 + MRF
    -> upsample x8, channels 128 + MRF
    -> upsample x2, channels 64  + MRF
    -> upsample x2, channels 32  + MRF
    -> Conv1d to 1 channel + tanh
waveform [B,1,T_frames*256]
```

## Frame-to-sample conversion

The configured upsampling rates are:

```text
[8, 8, 2, 2]
product = 256
```

This equals the STFT hop length. Therefore one latent frame corresponds to 256 waveform samples. A training segment of 32 latent frames produces `32*256=8192` samples.

## Transposed-convolution arithmetic

For dilation one:

```text
L_out = (L_in-1)*stride - 2*padding + kernel + output_padding
```

The reference uses `padding=(kernel-stride)//2`. Calculate every stage rather than assuming exact multiplication.

## Channel schedule

The pre-convolution maps latent channels to 512. Every upsampling stage halves channels:

```text
512 -> 256 -> 128 -> 64 -> 32 -> 1
```

Speaker conditioning is projected to 512 channels and added after the pre-convolution, broadcasting across latent time.

## Multi-receptive-field fusion

After each upsampling layer, three parallel residual blocks use kernels `3`, `7`, and `11`. Their outputs are added and divided by the number of kernels. They are averaged, not concatenated.

Different kernels let the decoder combine fine and broad waveform patterns.

## ResBlock1

For every dilation in `[1,3,5]`:

```text
leaky_relu -> dilated Conv1d
leaky_relu -> ordinary Conv1d
residual addition
```

Padding is selected to preserve time. Residual connections retain a direct information path while convolutions learn refinements.

## Weight normalization

Upsampling and residual convolutions use weight normalization during training. After training, `remove_weight_norm()` removes the reparameterization for simpler inference.

## Waveform output

The final path is:

```text
leaky_relu -> Conv1d(C,1,kernel=7) -> tanh
```

`tanh` bounds the output approximately to `[-1,1]`. It does not guarantee good audio or prevent clipping caused by later rescaling.

# Experiment roadmap

## Task 1: Calculate production upsampling

Compute cumulative factors and sample counts for 1, 32, and 87 latent frames.

## Task 2: Verify ConvTranspose1d length formula

Predict every stage length from stride, kernel, and padding; compare with PyTorch.

## Task 3: Build the pre-convolution

Map latent channels to initial decoder channels while preserving time.

## Task 4: Add speaker conditioning

Project `[B,C_speaker]` and broadcast `[B,C_initial,1]` over latent time.

## Task 5: Implement one upsampling stage

Apply leaky ReLU and ConvTranspose1d; predict channels and length first.

## Task 6: Build ResBlock1

Implement dilated/ordinary convolution pairs with residual additions.

## Task 7: Compare receptive fields

Calculate context for kernels `3,7,11` with dilations `1,3,5`.

## Task 8: Build MRF averaging

Run parallel residual blocks and average their same-shaped outputs.

## Task 9: Assemble every stage

Track channel halving and cumulative time expansion in a table.

## Task 10: Produce waveform output

Apply the post-convolution and tanh. Verify shape and numeric range.

## Task 11: Test segment correspondence

Prove 32 frames create exactly 8192 samples with production settings.

## Task 12: Compare with the reference generator

Use compatible small settings or production settings and compare shapes and outputs under identical weights.

## Task 13: Final assertions

Collect length, channel, range, residual, MRF, and reference checks.

# Debugging order

1. Calculate every transposed-convolution output length.
2. Confirm rate/kernel list lengths match.
3. Track channels separately from time.
4. Index residual blocks by stage and kernel.
5. Average rather than concatenate MRF outputs.
6. Verify total stride equals hop length.
7. Keep the waveform channel dimension until the intended squeeze.

# Questions you must answer

1. Why must total upsampling match hop length?
2. What does a transposed convolution learn?
3. Why use several residual kernels?
4. How does dilation widen context?
5. Why average MRF outputs?
6. What does final tanh guarantee and not guarantee?

# Completion criteria

Chapter 10 is complete when you can predict every decoder shape, derive output lengths, explain MRF processing, and generate correctly sized bounded waveforms.

Chapter 11 trains waveform realism using discriminators and multiple complementary losses.
