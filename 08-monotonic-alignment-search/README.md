# Monotonic Alignment Search for VITS2

This chapter scores every text-token/acoustic-frame pairing and uses dynamic programming to find the highest-scoring monotonic path.

Primary reference: score construction and `_mas` in `training/models/vits2.py`.

## End-to-end pathway

```text
text prior m_p, logs_p [B, C, T_text]
flowed acoustic z_p    [B, C, T_mel]
    -> Gaussian scores [B, T_text, T_mel]
    -> valid-pair mask [B, T_text, T_mel]
    -> dynamic programming: stay or advance
    -> backtrack best binary path
attention [B, 1, T_text, T_mel]
    -> durations [B, 1, T_text]
```

## Meaning of an alignment score

`score[b, i, j]` is the diagonal-Gaussian log probability of acoustic latent frame `j` under the prior predicted for text token `i`, summed across latent channels:

```text
log p(z_j | token_i)
= sum_c[-0.5*log(2*pi) - logs_i
        -0.5*(z_j-m_i)^2*exp(-2*logs_i)]
```

The reference expands this quadratic into four einsum-friendly terms. Your notebook must compare the optimized expansion with the direct equation.

## Pair mask

```text
x_mask [B, 1, T_text]
y_mask [B, 1, T_mel]
pair_mask = x_mask.transpose(1,2) * y_mask
          [B, T_text, T_mel]
```

Invalid scores receive a large negative value so the best path cannot enter padding.

## Legal path moves

At every next mel frame the path may:

- stay on the current text token; or
- advance exactly one text token.

It may not move backward, skip a text token, or skip mel time. Every valid mel frame belongs to exactly one text token.

## Dynamic programming recurrence

For current token `i` and mel frame `j`:

```text
stay    = value[i, j-1]
advance = value[i-1, j-1]
value[i, j] = score[i, j] + max(stay, advance)
```

Initialize only text position zero as reachable at mel frame zero. Save which predecessor won in a direction table.

## Backtracking

Begin at the last valid text token and last valid mel frame. Move backward through time; when the direction table says the forward path advanced, decrement the text index.

The output must satisfy:

```text
one active token per valid mel frame
token indices never decrease forward in time
first frame uses first token
last valid frame uses last valid token
path is zero outside the pair mask
```

## Durations

Sum the path over mel time:

```text
w = attention.sum(dim=-1)  # [B,1,T_text]
```

For each sample, valid token durations must sum to the valid mel length.

## Diagonal training prior

The reference can add a Gaussian-shaped bonus near proportional diagonal progress early in training. Its strength decays linearly to zero by a configured final epoch. This is a temporary guide, not the final alignment rule.

# Experiment roadmap

## Task 1: Draw legal paths by hand

For `T_text=3` and `T_mel=6`, draw several legal and illegal binary paths.

## Task 2: Build pair masks

Combine unequal text and mel lengths. Inspect the valid rectangle for each sample.

## Task 3: Compute direct Gaussian scores

Use tiny `m`, `logs`, and `z` tensors and calculate every pairing explicitly.

## Task 4: Reproduce the optimized score expansion

Implement the reference's four terms and prove equality with the direct formula.

## Task 5: Penalize invalid states

Apply the pair mask and verify invalid positions can never win.

## Task 6: Implement one DP step

Calculate stay/advance candidates and store the winning direction.

## Task 7: Run the forward DP

Iterate over mel frames while retaining direction choices for every state.

## Task 8: Backtrack the path

Use per-sample valid endpoints and reconstruct a binary path.

## Task 9: Validate path invariants

Check frame coverage, monotonic indices, endpoints, binary values, and mask compliance.

## Task 10: Extract durations

Sum attention over mel time and prove duration totals equal mel lengths.

## Task 11: Add and decay the diagonal prior

Visualize or print its bonus matrix and calculate strength at several epochs.

## Task 12: Compare with reference MAS

Run identical scores and masks through both implementations and compare paths.

## Task 13: Final assertions

Collect score, mask, DP, path, duration, and reference checks.

# Debugging order

1. Label text and mel axes explicitly.
2. Initialize unreachable states to negative infinity.
3. Disallow advance into text position zero.
4. Use each sample's valid text and mel endpoint.
5. Backtrack only active mel frames.
6. Verify one path value per valid frame.
7. Verify durations sum to mel length.

# Questions you must answer

1. What does `score[i,j]` mean?
2. Why are only stay and advance legal?
3. Why must directions be stored?
4. How do masks determine endpoints?
5. Which invariants make the result monotonic?
6. How does the path become duration targets?

# Completion criteria

Chapter 08 is complete when direct and optimized scores match, best paths obey every invariant, duration sums are correct, and your MAS matches the reference.

Chapter 09 learns and samples token durations so inference can align text without target audio.
