"""Build the completed Chapter 05 text-encoder notebook."""

from __future__ import annotations

import json
from pathlib import Path


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
# Chapter 05 — Text Encoder and Prior Distribution

This notebook builds the text side of VITS2 from padded token IDs to a contextual representation and token-level Gaussian prior.

```text
text IDs [B, T] + lengths [B]
  → embeddings [B, T, C]
  → masked Transformer [B, C, T]
  → prior mean and log-scale [B, C_latent, T]
```

Small dimensions make every intermediate tensor inspectable. The final checkpoint copies identical weights into the project reference class and proves numerical equivalence.
"""
    ),
    markdown(
        """
## Setup

The notebook uses deterministic seeds and zero dropout for equivalence tests. Production training uses dropout `0.1`.
"""
    ),
    code(
        """
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(17)
torch.set_printoptions(precision=4, sci_mode=False)

print("PyTorch version:", torch.__version__)
"""
    ),
    markdown(
        """
## Task 1 — Recreate the input contract

We use three sequences with different valid lengths. IDs after each length are padding. The mask must be `[B, 1, T]` so it broadcasts across `[B, C, T]` features.
"""
    ),
    code(
        """
def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


text = torch.tensor([
    [1, 4, 5, 6, 2, 0, 0],
    [1, 7, 8, 2, 0, 0, 0],
    [1, 9, 10, 11, 12, 13, 2],
], dtype=torch.long)
text_lengths = torch.tensor([5, 4, 7], dtype=torch.long)

batch_size, max_text_length = text.shape
n_vocab = 20
hidden_channels = 8
filter_channels = 16
n_heads = 2
head_dim = hidden_channels // n_heads
n_layers = 2
kernel_size = 3
out_channels = 6
speaker_cond_dim = 5

x_mask_bool = sequence_mask(text_lengths, max_text_length).unsqueeze(1)
x_mask = x_mask_bool.to(torch.float32)

print("Text shape:", text.shape)
print("Lengths:", text_lengths.tolist())
print("Mask shape:", x_mask.shape)
print("Mask:", x_mask_bool)

assert text.shape == (3, 7)
assert text.dtype == torch.long
assert x_mask.shape == (3, 1, 7)
assert torch.equal(x_mask_bool.sum(dim=(1, 2)), text_lengths)
assert int(text.max().item()) < n_vocab
assert hidden_channels % n_heads == 0
"""
    ),
    markdown(
        """
## Task 2 — Embedding initialization and scaling

The reference initializes embedding weights with standard deviation `C**-0.5`, then multiplies looked-up vectors by `sqrt(C)`. Their expected combined scale is approximately one.
"""
    ),
    code(
        """
embedding = nn.Embedding(n_vocab, hidden_channels)
expected_weight_std = hidden_channels ** -0.5
nn.init.normal_(embedding.weight, mean=0.0, std=expected_weight_std)

raw_embeddings = embedding(text)
scaled_embeddings = raw_embeddings * math.sqrt(hidden_channels)

print("Embedding table shape:", embedding.weight.shape)
print("Raw embedding shape:", raw_embeddings.shape)
print("Requested weight std:", expected_weight_std)
print("Observed table std:", embedding.weight.std().item())
print("Observed looked-up std before scaling:", raw_embeddings.std().item())
print("Observed looked-up std after scaling:", scaled_embeddings.std().item())

assert embedding.weight.shape == (n_vocab, hidden_channels)
assert raw_embeddings.shape == (batch_size, max_text_length, hidden_channels)
assert scaled_embeddings.shape == raw_embeddings.shape
assert torch.allclose(
    scaled_embeddings,
    raw_embeddings * math.sqrt(hidden_channels),
)
"""
    ),
    markdown(
        """
## Task 3 — Speaker conditioning

One speaker vector per sample is projected to `C_hidden`, unsqueezed to `[B, 1, C]`, and added identically at every text position.
"""
    ),
    code(
        """
speaker_embeddings = torch.randn(batch_size, speaker_cond_dim)
speaker_projection = nn.Linear(speaker_cond_dim, hidden_channels)

projected_speaker = speaker_projection(speaker_embeddings)
speaker_over_time = projected_speaker.unsqueeze(1)
conditioned_embeddings = scaled_embeddings + speaker_over_time
speaker_delta = conditioned_embeddings - scaled_embeddings

print("Speaker input shape:", speaker_embeddings.shape)
print("Projected speaker shape:", projected_speaker.shape)
print("Broadcast condition shape:", speaker_over_time.shape)
print("Conditioned text shape:", conditioned_embeddings.shape)

assert projected_speaker.shape == (batch_size, hidden_channels)
assert speaker_over_time.shape == (batch_size, 1, hidden_channels)
assert conditioned_embeddings.shape == scaled_embeddings.shape
for time_index in range(max_text_length):
    assert torch.allclose(speaker_delta[:, time_index, :], projected_speaker)
"""
    ),
    markdown(
        """
## Task 4 — Layouts and LayerNorm

`LayerNorm(C)` expects channels last, so normalization happens in `[B, T, C]`. `Conv1d` expects `[B, C, T]`, requiring a transpose afterward.
"""
    ),
    code(
        """
layer_norm = nn.LayerNorm(hidden_channels)
normalized_tokens = layer_norm(conditioned_embeddings)
channels_first = conditioned_embeddings.transpose(1, 2)
normalized_channels_first = normalized_tokens.transpose(1, 2)

token_means = normalized_tokens.mean(dim=-1)
token_variances = normalized_tokens.var(dim=-1, unbiased=False)

print("Tokens-last shape:", conditioned_embeddings.shape)
print("Channels-first shape:", channels_first.shape)
print("Normalized token means:")
print(token_means)
print("Normalized token variances:")
print(token_variances)

assert channels_first.shape == (batch_size, hidden_channels, max_text_length)
assert normalized_channels_first.shape == channels_first.shape
assert torch.allclose(token_means, torch.zeros_like(token_means), atol=1e-5)
assert torch.allclose(token_variances, torch.ones_like(token_variances), atol=2e-4)
"""
    ),
    markdown(
        """
## Task 5 — Q/K/V projections and head splitting

`1x1` convolutions mix channels independently at each time position. Splitting heads reorganizes `C = H × D` without changing the number of elements.
"""
    ),
    code(
        """
q_projection = nn.Conv1d(hidden_channels, hidden_channels, 1)
k_projection = nn.Conv1d(hidden_channels, hidden_channels, 1)
v_projection = nn.Conv1d(hidden_channels, hidden_channels, 1)

q_full = q_projection(normalized_channels_first)
k_full = k_projection(normalized_channels_first)
v_full = v_projection(normalized_channels_first)

q_heads = q_full.view(batch_size, n_heads, head_dim, max_text_length)
k_heads = k_full.view(batch_size, n_heads, head_dim, max_text_length)
v_heads = v_full.view(batch_size, n_heads, head_dim, max_text_length)

print("Q full/head shapes:", q_full.shape, q_heads.shape)
print("K full/head shapes:", k_full.shape, k_heads.shape)
print("V full/head shapes:", v_full.shape, v_heads.shape)

assert hidden_channels == n_heads * head_dim
assert q_heads.shape == (batch_size, n_heads, head_dim, max_text_length)
assert k_heads.shape == q_heads.shape
assert v_heads.shape == q_heads.shape
assert q_heads.numel() == q_full.numel()
"""
    ),
    markdown(
        """
## Task 6 — Scaled scores and key masking

Scores compare every query token with every key token. The key mask broadcasts across heads and query positions. Softmax operates over key positions.
"""
    ),
    code(
        """
unscaled_scores = torch.matmul(q_heads.transpose(2, 3), k_heads)
scaled_scores = unscaled_scores / math.sqrt(head_dim)

attention_key_mask = x_mask.unsqueeze(1)
masked_scores = scaled_scores.masked_fill(attention_key_mask == 0, -1e4)
attention_weights = F.softmax(masked_scores, dim=-1)

invalid_key_mask = (~x_mask_bool).unsqueeze(1).expand_as(attention_weights)
invalid_key_probabilities = attention_weights.masked_select(invalid_key_mask)
attention_sums = attention_weights.sum(dim=-1)

print("Scores shape:", scaled_scores.shape)
print("Mean absolute unscaled score:", unscaled_scores.abs().mean().item())
print("Mean absolute scaled score:", scaled_scores.abs().mean().item())
print("Largest padded-key probability:", invalid_key_probabilities.max().item())
print("Attention row-sum range:", attention_sums.min().item(), attention_sums.max().item())

assert scaled_scores.shape == (batch_size, n_heads, max_text_length, max_text_length)
assert torch.allclose(scaled_scores, unscaled_scores / math.sqrt(head_dim))
assert invalid_key_probabilities.max() < 1e-7
assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)
"""
    ),
    markdown(
        """
## Task 7 — Weighted values and head merging

Attention-weighted values produce `[B, H, T, D]`. Reordering and merging `H × D` recovers `[B, C, T]`, followed by a final channel-mixing projection.
"""
    ),
    code(
        """
weighted_values = torch.matmul(attention_weights, v_heads.transpose(2, 3))
merged_heads = weighted_values.transpose(2, 3).contiguous().view(
    batch_size, hidden_channels, max_text_length
)
attention_output_projection = nn.Conv1d(hidden_channels, hidden_channels, 1)
manual_attention_output = attention_output_projection(merged_heads)

print("Weighted values shape:", weighted_values.shape)
print("Merged heads shape:", merged_heads.shape)
print("Attention output shape:", manual_attention_output.shape)

assert weighted_values.shape == (batch_size, n_heads, max_text_length, head_dim)
assert merged_heads.shape == (batch_size, hidden_channels, max_text_length)
assert manual_attention_output.shape == merged_heads.shape
"""
    ),
    markdown(
        """
## Task 8 — Convolutional FFN

The FFN expands channels, applies GELU and dropout, then contracts back to the hidden size. Symmetric padding with an odd kernel preserves time length. Masks are applied before and after convolutions.
"""
    ),
    code(
        """
class FFNScratch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(filter_channels, in_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x * mask)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x * mask)
        return x * mask


ffn = FFNScratch(hidden_channels, filter_channels, kernel_size, p_dropout=0.0)
ffn_input = channels_first * x_mask
ffn_hidden = ffn.conv1(ffn_input)
ffn_output = ffn(ffn_input, x_mask)

print("FFN input shape:", ffn_input.shape)
print("Expanded hidden shape:", ffn_hidden.shape)
print("FFN output shape:", ffn_output.shape)

assert ffn_hidden.shape == (batch_size, filter_channels, max_text_length)
assert ffn_output.shape == (batch_size, hidden_channels, max_text_length)
assert torch.count_nonzero(ffn_output.masked_select((~x_mask_bool).expand_as(ffn_output))) == 0
"""
    ),
    markdown(
        """
## Task 9 — One encoder layer

This implementation matches the reference: pre-norm attention, residual addition, pre-norm FFN, residual addition, then final time masking.
"""
    ),
    code(
        """
class MultiHeadAttentionScratch(nn.Module):
    def __init__(self, channels: int, n_heads: int, p_dropout: float) -> None:
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError("channels must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.q_proj = nn.Conv1d(channels, channels, 1)
        self.k_proj = nn.Conv1d(channels, channels, 1)
        self.v_proj = nn.Conv1d(channels, channels, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, channels, time = q.shape
        q_heads = self.q_proj(q).view(batch, self.n_heads, self.head_dim, time)
        k_heads = self.k_proj(kv).view(batch, self.n_heads, self.head_dim, time)
        v_heads = self.v_proj(kv).view(batch, self.n_heads, self.head_dim, time)

        scores = torch.matmul(q_heads.transpose(2, 3), k_heads) / math.sqrt(self.head_dim)
        key_mask = mask.unsqueeze(1)
        scores = scores.masked_fill(key_mask == 0, -1e4)
        weights = self.dropout(F.softmax(scores, dim=-1))

        output = torch.matmul(weights, v_heads.transpose(2, 3))
        output = output.transpose(2, 3).contiguous().view(batch, channels, time)
        return self.out_proj(output)


class TransformerEncoderLayerScratch(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttentionScratch(hidden_channels, n_heads, p_dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.ffn = FFNScratch(hidden_channels, filter_channels, kernel_size, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        normalized = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = residual + self.dropout(self.attention(normalized, normalized, mask))

        residual = x
        normalized = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = residual + self.dropout(self.ffn(normalized, mask))
        return x * mask


encoder_layer = TransformerEncoderLayerScratch(
    hidden_channels, filter_channels, n_heads, kernel_size, p_dropout=0.0
)
layer_input = channels_first * x_mask
layer_output = encoder_layer(layer_input, x_mask)

print("Layer input shape:", layer_input.shape)
print("Layer output shape:", layer_output.shape)
print("Mean absolute residual update:", (layer_output - layer_input).abs().mean().item())

assert layer_output.shape == layer_input.shape
assert torch.count_nonzero(layer_output.masked_select((~x_mask_bool).expand_as(layer_output))) == 0
"""
    ),
    markdown(
        """
## Task 10 — Stack encoder layers

Each layer has independent parameters and preserves `[B, C_hidden, T]`. Depth changes the values, not the external shape.
"""
    ),
    code(
        """
class TransformerEncoderScratch(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerScratch(
                hidden_channels, filter_channels, n_heads, kernel_size, p_dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


transformer = TransformerEncoderScratch(
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout=0.0,
)
encoded_text = transformer(layer_input, x_mask)

print("Number of layers:", len(transformer.layers))
print("Stack input/output shapes:", layer_input.shape, encoded_text.shape)
print("Mean absolute total update:", (encoded_text - layer_input).abs().mean().item())

assert len(transformer.layers) == n_layers
assert encoded_text.shape == layer_input.shape
assert transformer.layers[0] is not transformer.layers[1]
assert torch.count_nonzero(encoded_text.masked_select((~x_mask_bool).expand_as(encoded_text))) == 0
"""
    ),
    markdown(
        """
## Task 11 — Prior projection and split

A `1x1` convolution produces `2 × C_latent` channels. Splitting channels yields the prior mean and log standard deviation.
"""
    ),
    code(
        """
prior_projection = nn.Conv1d(hidden_channels, out_channels * 2, 1)
prior_stats = prior_projection(encoded_text) * x_mask
m_p_manual, logs_p_manual = prior_stats.split(out_channels, dim=1)

print("Prior stats shape:", prior_stats.shape)
print("Prior mean shape:", m_p_manual.shape)
print("Prior log-scale shape:", logs_p_manual.shape)

assert prior_stats.shape == (batch_size, out_channels * 2, max_text_length)
assert m_p_manual.shape == (batch_size, out_channels, max_text_length)
assert logs_p_manual.shape == m_p_manual.shape
assert torch.count_nonzero(m_p_manual.masked_select((~x_mask_bool).expand_as(m_p_manual))) == 0
assert torch.count_nonzero(logs_p_manual.masked_select((~x_mask_bool).expand_as(logs_p_manual))) == 0
"""
    ),
    markdown(
        """
## Task 12 — Interpret and sample the Gaussian

`logs_p` is log standard deviation: `std = exp(logs_p)` and `variance = exp(2*logs_p)`. Sampling uses `m_p + epsilon*std`. The sample is masked afterward because zero log-scale corresponds to standard deviation one, not zero.
"""
    ),
    code(
        """
prior_std = torch.exp(logs_p_manual)
prior_variance = torch.exp(2.0 * logs_p_manual)
epsilon = torch.randn_like(m_p_manual)
latent_mask = x_mask.expand_as(m_p_manual)
prior_sample = (m_p_manual + epsilon * prior_std) * latent_mask

print("Minimum standard deviation:", prior_std.min().item())
print("Maximum standard deviation:", prior_std.max().item())
print("Sample shape:", prior_sample.shape)
print("Padded log-scale is zero, so padded std is:", prior_std[0, 0, -1].item())

assert torch.all(prior_std > 0)
assert torch.allclose(prior_variance, prior_std.square())
assert prior_sample.shape == m_p_manual.shape
assert torch.count_nonzero(prior_sample.masked_select((~x_mask_bool).expand_as(prior_sample))) == 0
"""
    ),
    markdown(
        """
## Task 13 — Assemble the full text encoder

The module below matches the reference architecture and parameter names. This allows direct state-dictionary transfer for the final comparison.
"""
    ),
    code(
        """
class TextEncoderScratch(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        speaker_cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if speaker_cond_dim > 0:
            self.speaker_proj = nn.Linear(speaker_cond_dim, hidden_channels)
        else:
            self.speaker_proj = None

        self.encoder = TransformerEncoderScratch(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(token_ids) * math.sqrt(self.hidden_channels)

        if self.speaker_proj is not None and speaker_emb is not None:
            x = x + self.speaker_proj(speaker_emb).unsqueeze(1)

        x = x.transpose(1, 2)
        mask = sequence_mask(lengths, x.shape[2]).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * mask, mask)

        stats = self.proj(x) * mask
        mean, log_scale = stats.split(self.out_channels, dim=1)
        return x, mean, log_scale, mask


scratch_encoder = TextEncoderScratch(
    n_vocab=n_vocab,
    out_channels=out_channels,
    hidden_channels=hidden_channels,
    filter_channels=filter_channels,
    n_heads=n_heads,
    n_layers=n_layers,
    kernel_size=kernel_size,
    p_dropout=0.0,
    speaker_cond_dim=speaker_cond_dim,
)
scratch_encoder.eval()

with torch.no_grad():
    x_full, m_p, logs_p, mask_full = scratch_encoder(
        text, text_lengths, speaker_embeddings
    )

print("Encoded text shape:", x_full.shape)
print("Prior mean shape:", m_p.shape)
print("Prior log-scale shape:", logs_p.shape)
print("Mask shape:", mask_full.shape)

assert x_full.shape == (batch_size, hidden_channels, max_text_length)
assert m_p.shape == (batch_size, out_channels, max_text_length)
assert logs_p.shape == m_p.shape
assert mask_full.shape == (batch_size, 1, max_text_length)
"""
    ),
    markdown(
        """
## Task 14 — Prove padding invariance

Changing only padded token IDs must not alter any valid output. This is stronger than merely checking that padded outputs are zero.
"""
    ),
    code(
        """
altered_padded_text = text.clone()
replacement_ids = [15, 16, 17, 18, 19]
replacement_index = 0
for batch_index, length in enumerate(text_lengths.tolist()):
    for time_index in range(length, max_text_length):
        altered_padded_text[batch_index, time_index] = replacement_ids[
            replacement_index % len(replacement_ids)
        ]
        replacement_index += 1

with torch.no_grad():
    x_altered, m_altered, logs_altered, altered_mask = scratch_encoder(
        altered_padded_text, text_lengths, speaker_embeddings
    )

valid_x_mask = mask_full.bool().expand_as(x_full)
valid_prior_mask = mask_full.bool().expand_as(m_p)

print("Original text:", text)
print("Altered padded text:", altered_padded_text)
print("Max valid x difference:", (x_full - x_altered).masked_select(valid_x_mask).abs().max().item())
print("Max valid mean difference:", (m_p - m_altered).masked_select(valid_prior_mask).abs().max().item())
print("Max valid log-scale difference:", (logs_p - logs_altered).masked_select(valid_prior_mask).abs().max().item())

assert torch.equal(mask_full, altered_mask)
assert torch.allclose(x_full.masked_select(valid_x_mask), x_altered.masked_select(valid_x_mask))
assert torch.allclose(m_p.masked_select(valid_prior_mask), m_altered.masked_select(valid_prior_mask))
assert torch.allclose(logs_p.masked_select(valid_prior_mask), logs_altered.masked_select(valid_prior_mask))
assert torch.count_nonzero(x_altered.masked_select(~valid_x_mask)) == 0
assert torch.count_nonzero(m_altered.masked_select(~valid_prior_mask)) == 0
assert torch.count_nonzero(logs_altered.masked_select(~valid_prior_mask)) == 0
"""
    ),
    markdown(
        """
## Task 15 — Compare with the project reference

Only the reference `commons.py` and `encoder.py` files are loaded. Identical weights and inputs must produce identical outputs.
"""
    ),
    code(
        """
def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


current_path = Path.cwd().resolve()
project_root = next(
    (
        candidate
        for candidate in [current_path, *current_path.parents]
        if (candidate / "training" / "models" / "modules" / "encoder.py").is_file()
    ),
    None,
)
if project_root is None:
    raise RuntimeError("Could not locate the project root")

modules_directory = project_root / "training" / "models" / "modules"
reference_package_name = "_chapter05_reference_modules"
reference_package = types.ModuleType(reference_package_name)
reference_package.__path__ = [str(modules_directory)]
sys.modules[reference_package_name] = reference_package

load_module(
    f"{reference_package_name}.commons",
    modules_directory / "commons.py",
)
reference_encoder_module = load_module(
    f"{reference_package_name}.encoder",
    modules_directory / "encoder.py",
)

ReferenceTextEncoder = reference_encoder_module.TextEncoder
reference_encoder = ReferenceTextEncoder(
    n_vocab=n_vocab,
    out_channels=out_channels,
    hidden_channels=hidden_channels,
    filter_channels=filter_channels,
    n_heads=n_heads,
    n_layers=n_layers,
    kernel_size=kernel_size,
    p_dropout=0.0,
    speaker_cond_dim=speaker_cond_dim,
)
reference_encoder.load_state_dict(scratch_encoder.state_dict(), strict=True)
reference_encoder.eval()

with torch.no_grad():
    x_reference, m_reference, logs_reference, mask_reference = reference_encoder(
        text, text_lengths, speaker_embeddings
    )

print("Maximum x difference:", (x_full - x_reference).abs().max().item())
print("Maximum mean difference:", (m_p - m_reference).abs().max().item())
print("Maximum log-scale difference:", (logs_p - logs_reference).abs().max().item())
print("Masks equal:", torch.equal(mask_full, mask_reference))

assert torch.allclose(x_full, x_reference, atol=1e-7, rtol=1e-6)
assert torch.allclose(m_p, m_reference, atol=1e-7, rtol=1e-6)
assert torch.allclose(logs_p, logs_reference, atol=1e-7, rtol=1e-6)
assert torch.equal(mask_full, mask_reference)
"""
    ),
    markdown(
        """
## Task 16 — Final chapter assertions

The final checkpoint collects the structural and semantic guarantees, including the real production configuration arithmetic.
"""
    ),
    code(
        """
# Core input and mask contract
assert text.dtype == torch.long
assert text_lengths.shape == (batch_size,)
assert mask_full.shape == (batch_size, 1, max_text_length)
assert torch.equal(mask_full.sum(dim=(1, 2)).long(), text_lengths)

# Attention and layer contract
assert hidden_channels % n_heads == 0
assert head_dim == hidden_channels // n_heads
assert attention_weights.shape == (batch_size, n_heads, max_text_length, max_text_length)
assert invalid_key_probabilities.max() < 1e-7
assert encoded_text.shape == (batch_size, hidden_channels, max_text_length)

# Prior semantics
assert m_p.shape == (batch_size, out_channels, max_text_length)
assert logs_p.shape == m_p.shape
assert torch.all(torch.exp(logs_p) > 0)
assert torch.allclose(torch.exp(2 * logs_p), torch.exp(logs_p).square())

# Padding safety
assert torch.count_nonzero(x_full.masked_select(~valid_x_mask)) == 0
assert torch.count_nonzero(m_p.masked_select(~valid_prior_mask)) == 0
assert torch.count_nonzero(logs_p.masked_select(~valid_prior_mask)) == 0

# Reference equivalence
assert torch.allclose(x_full, x_reference, atol=1e-7, rtol=1e-6)
assert torch.allclose(m_p, m_reference, atol=1e-7, rtol=1e-6)
assert torch.allclose(logs_p, logs_reference, atol=1e-7, rtol=1e-6)

# Production configuration arithmetic
production_hidden = 192
production_heads = 2
production_head_dim = production_hidden // production_heads
production_filter = 768
production_latent = 192
assert production_head_dim == 96
assert production_filter == 4 * production_hidden
assert 2 * production_latent == 384

print("All Chapter 05 assertions passed.")
print("Final small-model shape path:")
print("  token IDs:       ", tuple(text.shape))
print("  text mask:       ", tuple(mask_full.shape))
print("  encoded text:    ", tuple(x_full.shape))
print("  prior mean:      ", tuple(m_p.shape))
print("  prior log-scale: ", tuple(logs_p.shape))
print("Production head dimension:", production_head_dim)
"""
    ),
    markdown(
        """
# Chapter reflection

1. **Why scale embeddings?** The initialization shrinks with hidden size; multiplication by `sqrt(C)` restores an order-one input scale.
2. **Why alternate layouts?** `LayerNorm(C)` expects channels last, while `Conv1d` expects channels second.
3. **Why scale attention logits?** Dot-product magnitude grows with head dimension; scaling stabilizes softmax.
4. **What does the attention mask do?** It prevents padded keys from contributing. Final layer masking separately zeros padded query positions.
5. **Why two prior halves?** One half is the mean and the other is log standard deviation for a diagonal Gaussian.
6. **Why is `logs_p` not log-variance?** Sampling uses `exp(logs_p)` directly as standard deviation; variance is `exp(2*logs_p)`.
7. **What did the perturbation test prove?** Padded token identity cannot affect valid text or prior outputs.
8. **What comes next?** Chapter 06 builds the posterior encoder from observed mel features and samples acoustic latents.
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
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("experiment.ipynb")
output_path.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {output_path} with {len(cells)} cells")
