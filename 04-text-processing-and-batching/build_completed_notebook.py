"""Build the completed Chapter 04 learning notebook."""

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
# Chapter 04 — Arabic Text Processing and Batching

This notebook rebuilds the complete path from Arabic Unicode text to the tensors consumed by the VITS2 text encoder.

```text
raw text → clean symbols → token IDs [T]
         → padded IDs [B, T] + lengths [B]
         → mask [B, 1, T]
         → embeddings [B, T, C]
         → encoder layout [B, C, T]
```

The first twelve tasks are independent implementations. Task 13 compares them with the finished project reference. This ordering keeps the reference from hiding the mechanics we want to understand.
"""
    ),
    markdown(
        """
## Setup

Only general Python and PyTorch tools are imported here. The project tokenizer is deliberately not imported until Task 13.
"""
    ),
    code(
        """
import re
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(7)
print("PyTorch version:", torch.__version__)
"""
    ),
    markdown(
        """
## Task 1 — Tiny vocabulary on paper

For the ordered vocabulary below, the list position is the token ID.

Predictions:

```text
<pad>=0, a=1, b=2, c=3, h=4, i=5, t=6
cat → [3, 1, 6]
hi  → [4, 5]
padded batch shape → [2, 3]
lengths → [3, 2]
mask shape → [2, 1, 3]
```
"""
    ),
    code(
        """
tiny_symbols = ["<pad>", "a", "b", "c", "h", "i", "t"]
tiny_symbol_to_id = {symbol: index for index, symbol in enumerate(tiny_symbols)}

cat_ids = [tiny_symbol_to_id[character] for character in "cat"]
hi_ids = [tiny_symbol_to_id[character] for character in "hi"]

tiny_lengths = torch.tensor([len(cat_ids), len(hi_ids)], dtype=torch.long)
tiny_padded = torch.tensor([cat_ids, hi_ids + [tiny_symbol_to_id["<pad>"]]], dtype=torch.long)
tiny_positions = torch.arange(tiny_padded.shape[1]).unsqueeze(0)
tiny_mask = (tiny_positions < tiny_lengths.unsqueeze(1)).unsqueeze(1)

print("Symbol IDs:", tiny_symbol_to_id)
print("cat IDs:", cat_ids)
print("hi IDs:", hi_ids)
print("Padded batch:\\n", tiny_padded)
print("Lengths:", tiny_lengths)
print("Mask:\\n", tiny_mask)
print("Shapes:", tiny_padded.shape, tiny_lengths.shape, tiny_mask.shape)

assert cat_ids == [3, 1, 6]
assert hi_ids == [4, 5]
assert tiny_padded.shape == (2, 3)
assert tiny_lengths.tolist() == [3, 2]
assert tiny_mask.shape == (2, 1, 3)
"""
    ),
    markdown(
        """
## Task 2 — Construct reversible lookup tables

Both dictionaries should be derived from one ordered source of truth. Manually maintaining two independent tables could let them disagree.
"""
    ),
    code(
        """
tiny_id_to_symbol = {index: symbol for index, symbol in enumerate(tiny_symbols)}

assert len(tiny_symbols) == len(set(tiny_symbols))
assert len(tiny_symbol_to_id) == len(tiny_symbols)
assert len(tiny_id_to_symbol) == len(tiny_symbols)

for symbol in tiny_symbols:
    token_id = tiny_symbol_to_id[symbol]
    assert tiny_id_to_symbol[token_id] == symbol

print("Forward table:", tiny_symbol_to_id)
print("Inverse table:", tiny_id_to_symbol)
print("All tiny-vocabulary round trips passed.")
"""
    ),
    markdown(
        """
## Task 3 — Inspect Arabic as Unicode data

Arabic is displayed right-to-left, but Python iterates it in logical Unicode order. Tashkeel are separate code points, so they add positions to a character-level sequence.
"""
    ),
    code(
        """
unicode_examples = {
    "plain": "سلام",
    "with_tashkeel": "سَلَام",
    "punctuation_and_whitespace": "  مرحباً،\\tيا عالم!  ",
}

for label, text in unicode_examples.items():
    print(f"\\n{label}")
    print("repr:", repr(text))
    print("logical symbols:", list(text))
    print("length:", len(text))
    for character in text:
        code_point = f"U+{ord(character):04X}"
        name = unicodedata.name(character, "UNNAMED")
        print(f"  {repr(character):5} {code_point} {name}")

assert list("سلام") == ["س", "ل", "ا", "م"]
assert len("سَلَام") > len("سلام")
"""
    ),
    markdown(
        """
## Task 4 — Recreate the project vocabulary

The ordering is a checkpoint contract: changing it changes which embedding row each ID selects. PAD, BOS, and EOS occupy the first three rows.
"""
    ),
    code(
        """
PAD_SYMBOL = "_"
BOS_SYMBOL = "^"
EOS_SYMBOL = "$"

ARABIC_CHARS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإؤئةىـ"
TASHKEEL = "ًٌٍَُِّْ"
PUNCTUATION = "،؟؛.!?,-: "

SPECIAL_SYMBOLS = [PAD_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
TEXT_SYMBOLS = ARABIC_CHARS + TASHKEEL + PUNCTUATION
SYMBOLS = SPECIAL_SYMBOLS + list(TEXT_SYMBOLS)

SYM_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
ID_TO_SYM = {index: symbol for index, symbol in enumerate(SYMBOLS)}

PAD_ID = SYM_TO_ID[PAD_SYMBOL]
BOS_ID = SYM_TO_ID[BOS_SYMBOL]
EOS_ID = SYM_TO_ID[EOS_SYMBOL]
N_SYMBOLS = len(SYMBOLS)
TEXT_SYMBOL_SET = set(TEXT_SYMBOLS)

print("Number of symbols:", N_SYMBOLS)
print("PAD/BOS/EOS IDs:", PAD_ID, BOS_ID, EOS_ID)
print("First 12 symbols:", SYMBOLS[:12])

assert len(SYMBOLS) == len(set(SYMBOLS)), "Vocabulary contains duplicates"
assert len(SYM_TO_ID) == len(SYMBOLS)
assert len(ID_TO_SYM) == len(SYMBOLS)
assert (PAD_ID, BOS_ID, EOS_ID) == (0, 1, 2)
assert not set(SPECIAL_SYMBOLS) & TEXT_SYMBOL_SET
assert all(ID_TO_SYM[SYM_TO_ID[symbol]] == symbol for symbol in SYMBOLS)
"""
    ),
    markdown(
        """
## Task 5 — Whitespace normalization

Whitespace normalization and vocabulary filtering are kept separate so each transformation can be inspected independently.
"""
    ),
    code(
        """
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", text).strip()


whitespace_example = "  مرحباً،\\t\\n  يا   عالم!  "
normalized_example = normalize_whitespace(whitespace_example)

print("Raw:", repr(whitespace_example))
print("Normalized:", repr(normalized_example))

assert normalized_example == "مرحباً، يا عالم!"
assert normalize_whitespace("  أ  ب\\tج\\nد  ") == "أ ب ج د"
assert not normalized_example.startswith(" ")
assert not normalized_example.endswith(" ")
assert "  " not in normalized_example
"""
    ),
    markdown(
        """
## Task 6 — Vocabulary filtering and unknown audit

Unsupported characters should not disappear invisibly. This implementation reports them and uses a text-only whitelist, so raw `_`, `^`, and `$` cannot become structural IDs.
"""
    ),
    code(
        """
def audit_clean_text(text: str) -> dict[str, object]:
    normalized = normalize_whitespace(text)
    removed = [character for character in normalized if character not in TEXT_SYMBOL_SET]
    cleaned = "".join(character for character in normalized if character in TEXT_SYMBOL_SET)
    return {
        "raw": text,
        "normalized": normalized,
        "cleaned": cleaned,
        "removed": removed,
        "removed_count": len(removed),
    }


def clean_arabic(text: str) -> str:
    return str(audit_clean_text(text)["cleaned"])


audit_example = "  السَّلَام\\tعليكم🙂 _^$  "
audit = audit_clean_text(audit_example)

for key, value in audit.items():
    print(f"{key:14}: {value!r}")

assert "🙂" in audit["removed"]
assert "_" in audit["removed"]
assert "^" in audit["removed"]
assert "$" in audit["removed"]
assert all(character not in audit["cleaned"] for character in SPECIAL_SYMBOLS)
"""
    ),
    markdown(
        """
## Task 7 — Encode, add boundaries, and decode

Encoding cleans first, converts each text symbol to its categorical ID, then optionally inserts BOS/EOS. Decoding removes structural IDs and reverses the lookup.
"""
    ),
    code(
        """
def text_to_ids(text: str, add_bos_eos: bool = True) -> list[int]:
    cleaned = clean_arabic(text)
    token_ids = [SYM_TO_ID[character] for character in cleaned]
    if add_bos_eos:
        token_ids = [BOS_ID] + token_ids + [EOS_ID]
    return token_ids


def ids_to_text(token_ids: list[int]) -> str:
    structural_ids = {PAD_ID, BOS_ID, EOS_ID}
    return "".join(
        ID_TO_SYM[token_id]
        for token_id in token_ids
        if token_id not in structural_ids
    )


raw_example = "  مرحباً،   يا عالم!  "
cleaned_example = clean_arabic(raw_example)
ids_without_boundaries = text_to_ids(raw_example, add_bos_eos=False)
ids_with_boundaries = text_to_ids(raw_example, add_bos_eos=True)
decoded_example = ids_to_text(ids_with_boundaries)

print("Raw:", repr(raw_example))
print("Cleaned:", repr(cleaned_example))
print("IDs without BOS/EOS:", ids_without_boundaries)
print("IDs with BOS/EOS:", ids_with_boundaries)
print("Decoded:", repr(decoded_example))

assert decoded_example == cleaned_example
assert ids_with_boundaries[0] == BOS_ID
assert ids_with_boundaries[-1] == EOS_ID
assert len(ids_with_boundaries) == len(ids_without_boundaries) + 2
"""
    ),
    markdown(
        """
## Task 8 — Create one dataset-style item

One dataset item has no batch dimension. Collation creates the batch later. Token IDs use `torch.long` because they index embedding rows.
"""
    ),
    code(
        """
sample_raw_text = "سلام"
sample_ids = text_to_ids(sample_raw_text)
sample_item = {
    "text": torch.tensor(sample_ids, dtype=torch.long),
    "text_length": len(sample_ids),
    "raw_text": sample_raw_text,
}

print(sample_item)
print("Text shape:", sample_item["text"].shape)
print("Text dtype:", sample_item["text"].dtype)

assert sample_item["text"].ndim == 1
assert sample_item["text"].dtype == torch.long
assert sample_item["text_length"] == sample_item["text"].shape[0]
"""
    ),
    markdown(
        """
## Task 9 — Pad a batch manually

Each row preserves its original sequence in the valid prefix. Only positions after its stored length receive `PAD_ID`.
"""
    ),
    code(
        """
batch_raw_texts = ["سلام", "مرحبا", "أنا"]
batch_token_tensors = [
    torch.tensor(text_to_ids(text), dtype=torch.long)
    for text in batch_raw_texts
]

text_lengths = torch.tensor(
    [tokens.shape[0] for tokens in batch_token_tensors],
    dtype=torch.long,
)
batch_size = len(batch_token_tensors)
max_text_length = int(text_lengths.max().item())

padded_text = torch.full(
    (batch_size, max_text_length),
    fill_value=PAD_ID,
    dtype=torch.long,
)

for row, tokens in enumerate(batch_token_tensors):
    padded_text[row, : tokens.shape[0]] = tokens

padding_per_row = max_text_length - text_lengths
total_padding = int(padding_per_row.sum().item())

print("Raw texts:", batch_raw_texts)
print("Lengths:", text_lengths.tolist())
print("Padded text:\\n", padded_text)
print("Shape:", padded_text.shape)
print("Padding per row:", padding_per_row.tolist())
print("Total padding:", total_padding)

for row, tokens in enumerate(batch_token_tensors):
    length = tokens.shape[0]
    assert torch.equal(padded_text[row, :length], tokens)
    assert torch.all(padded_text[row, length:] == PAD_ID)

assert padded_text.shape == (batch_size, max_text_length)
assert total_padding == batch_size * max_text_length - int(text_lengths.sum().item())
"""
    ),
    markdown(
        """
## Task 10 — Convert lengths into a text mask

The initial comparison creates `[B, T]`; inserting the singleton channel dimension creates `[B, 1, T]`, which broadcasts over `[B, C, T]` features.
"""
    ),
    code(
        """
def make_sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    mask_2d = positions < lengths.unsqueeze(1)
    return mask_2d.unsqueeze(1)


text_mask = make_sequence_mask(text_lengths, max_text_length)
fake_channels = 3
fake_features = torch.arange(
    batch_size * fake_channels * max_text_length,
    dtype=torch.float32,
).reshape(batch_size, fake_channels, max_text_length) + 1
masked_fake_features = fake_features * text_mask

print("Text mask:\\n", text_mask)
print("Mask shape:", text_mask.shape)
print("Valid positions per row:", text_mask.sum(dim=(1, 2)).tolist())
print("Masked fake features:\\n", masked_fake_features)

assert text_mask.shape == (batch_size, 1, max_text_length)
assert torch.equal(text_mask.sum(dim=(1, 2)), text_lengths)

for row, length in enumerate(text_lengths.tolist()):
    assert torch.count_nonzero(masked_fake_features[row, :, length:]) == 0
"""
    ),
    markdown(
        """
## Task 11 — Reproduce padding with PyTorch

The library helper should reproduce the manual result exactly. It is now a convenience rather than a mystery.
"""
    ),
    code(
        """
library_padded_text = pad_sequence(
    batch_token_tensors,
    batch_first=True,
    padding_value=PAD_ID,
)

print("Manual batch:\\n", padded_text)
print("Library batch:\\n", library_padded_text)
print("Exactly equal:", torch.equal(padded_text, library_padded_text))

assert torch.equal(padded_text, library_padded_text)
"""
    ),
    markdown(
        """
## Task 12 — Cross the embedding boundary

Token IDs `[B, T]` select rows from an embedding table and become floating-point features `[B, T, C]`. The encoder transposes these to `[B, C, T]`.

Because this embedding does not set `padding_idx`, padded positions are generally nonzero before masking.
"""
    ),
    code(
        """
hidden_channels = 4
embedding = nn.Embedding(N_SYMBOLS, hidden_channels)

embedded_text = embedding(padded_text)
encoder_layout = embedded_text.transpose(1, 2)
masked_encoder_features = encoder_layout * text_mask.to(encoder_layout.dtype)

invalid_positions = (~text_mask).expand_as(encoder_layout)
padded_values_before_mask = encoder_layout.masked_select(invalid_positions)
padded_values_after_mask = masked_encoder_features.masked_select(invalid_positions)

print("Token IDs shape:", padded_text.shape)
print("Embedded shape:", embedded_text.shape)
print("Encoder layout shape:", encoder_layout.shape)
print("Mask shape:", text_mask.shape)
print("Absolute padded sum before masking:", padded_values_before_mask.abs().sum().item())
print("Absolute padded sum after masking:", padded_values_after_mask.abs().sum().item())

assert embedded_text.shape == (batch_size, max_text_length, hidden_channels)
assert encoder_layout.shape == (batch_size, hidden_channels, max_text_length)
assert padded_values_before_mask.abs().sum() > 0
assert torch.count_nonzero(padded_values_after_mask) == 0
"""
    ),
    markdown(
        """
## Task 13 — Compare with the reference implementation

The independent implementation should match the reference for ordinary supported text. A deliberate structural-character input exposes a reference edge case: its cleaner treats `_`, `^`, and `$` as allowed raw characters because they are present in the full model vocabulary.
"""
    ),
    code(
        """
current_path = Path.cwd().resolve()
project_root = next(
    (candidate for candidate in [current_path, *current_path.parents]
     if (candidate / "training").is_dir()),
    None,
)
if project_root is None:
    raise RuntimeError("Could not find the project root containing training/")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib.util
import types


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import only the reference files needed by this chapter. Importing the
# training.datasets package would also import audio code and require torchaudio.
text_directory = project_root / "training" / "datasets" / "text"
reference_package_name = "_chapter04_reference_text"
reference_package = types.ModuleType(reference_package_name)
reference_package.__path__ = [str(text_directory)]
sys.modules[reference_package_name] = reference_package

reference_symbols = load_module(
    f"{reference_package_name}.symbols",
    text_directory / "symbols.py",
)
reference_arabic = load_module(
    f"{reference_package_name}.arabic",
    text_directory / "arabic.py",
)
reference_collate = load_module(
    "_chapter04_reference_collate",
    project_root / "training" / "datasets" / "collate.py",
)

reference_clean_arabic = reference_arabic.clean_arabic
reference_ids_to_text = reference_arabic.ids_to_text
reference_text_to_ids = reference_arabic.text_to_ids
TTSCollate = reference_collate.TTSCollate

REFERENCE_BOS_ID = reference_symbols.BOS_ID
REFERENCE_EOS_ID = reference_symbols.EOS_ID
REFERENCE_PAD_ID = reference_symbols.PAD_ID
REFERENCE_SYMBOLS = reference_symbols.SYMBOLS

ordinary_examples = ["سلام", "  مرحباً،   يا عالم!  ", "أنا"]
for text in ordinary_examples:
    assert clean_arabic(text) == reference_clean_arabic(text)
    assert text_to_ids(text) == reference_text_to_ids(text)
    assert ids_to_text(text_to_ids(text)) == reference_ids_to_text(reference_text_to_ids(text))

assert SYMBOLS == REFERENCE_SYMBOLS
assert (PAD_ID, BOS_ID, EOS_ID) == (
    REFERENCE_PAD_ID,
    REFERENCE_BOS_ID,
    REFERENCE_EOS_ID,
)

reference_items = []
for index, text in enumerate(batch_raw_texts):
    token_tensor = torch.tensor(reference_text_to_ids(text), dtype=torch.long)
    waveform = torch.arange(index + 2, dtype=torch.float32)
    reference_items.append({
        "text": token_tensor,
        "text_length": token_tensor.shape[0],
        "wav": waveform,
        "wav_length": waveform.shape[0],
        "speaker_idx": index,
    })

reference_batch = TTSCollate()(reference_items)
assert torch.equal(reference_batch["text"], padded_text)
assert torch.equal(reference_batch["text_length"], text_lengths)

edge_text = "_^$سلام"
independent_edge_clean = clean_arabic(edge_text)
reference_edge_clean = reference_clean_arabic(edge_text)
reference_edge_ids = reference_text_to_ids(edge_text)

print("Ordinary examples match the reference.")
print("Reference collator matches the manual batch.")
print("Structural edge input:", repr(edge_text))
print("Independent cleaner:", repr(independent_edge_clean))
print("Reference cleaner:", repr(reference_edge_clean))
print("Reference edge IDs:", reference_edge_ids)

assert independent_edge_clean == "سلام"
assert reference_edge_clean == edge_text
assert independent_edge_clean != reference_edge_clean
assert reference_edge_ids[1:4] == [REFERENCE_PAD_ID, REFERENCE_BOS_ID, REFERENCE_EOS_ID]
"""
    ),
    markdown(
        """
## Task 14 — Final chapter assertions

These checks summarize the complete contract. They deliberately test shapes, dtypes, values, reversibility, padding behavior, and the separation between transcript symbols and structural symbols.
"""
    ),
    code(
        """
# Vocabulary and structural-token invariants
assert len(SYMBOLS) == len(set(SYMBOLS))
assert len({PAD_ID, BOS_ID, EOS_ID}) == 3
assert not set(SPECIAL_SYMBOLS) & TEXT_SYMBOL_SET
assert all(ID_TO_SYM[SYM_TO_ID[symbol]] == symbol for symbol in SYMBOLS)

# Cleaning and round-trip invariants
assert clean_arabic(raw_example) == clean_arabic(raw_example)
assert ids_to_text(text_to_ids(raw_example)) == clean_arabic(raw_example)
assert all(
    token_id not in {PAD_ID, BOS_ID, EOS_ID}
    for character in TEXT_SYMBOLS
    for token_id in [SYM_TO_ID[character]]
)

# Dataset-item and batching invariants
assert sample_item["text"].ndim == 1
assert sample_item["text"].dtype == torch.long
assert padded_text.ndim == 2
assert padded_text.dtype == torch.long
assert text_lengths.shape == (batch_size,)
assert text_mask.shape == (batch_size, 1, max_text_length)
assert torch.equal(text_mask.sum(dim=(1, 2)), text_lengths)
assert torch.equal(padded_text, library_padded_text)

# Embedding and masking invariants
assert embedded_text.shape == (batch_size, max_text_length, hidden_channels)
assert encoder_layout.shape == (batch_size, hidden_channels, max_text_length)
assert torch.count_nonzero(
    masked_encoder_features.masked_select((~text_mask).expand_as(masked_encoder_features))
) == 0

print("All Chapter 04 assertions passed.")
print("Final shape path:")
print("  padded token IDs:", tuple(padded_text.shape))
print("  text lengths:    ", tuple(text_lengths.shape))
print("  text mask:       ", tuple(text_mask.shape))
print("  embeddings:      ", tuple(embedded_text.shape))
print("  encoder layout:  ", tuple(encoder_layout.shape))
"""
    ),
    markdown(
        """
# Chapter reflection

1. **What information is lost during cleaning?** Unsupported characters and the original whitespace pattern are lost. This is why removed characters must be audited.
2. **Why must vocabulary order remain stable?** An ID selects a specific embedding row. Reordering symbols changes the meaning of trained rows.
3. **Why preserve lengths after padding?** The rectangular tensor does not by itself carry a safe, persistent boundary between real tokens and padding.
4. **Why is the mask `[B, 1, T]`?** The singleton channel dimension broadcasts the same valid-time decision over every feature channel in `[B, C, T]`.
5. **What changes at the embedding boundary?** Categorical integer IDs `[B, T]` become learned floating-point vectors `[B, T, C]`.
6. **What should change in the reference tokenizer?** Raw-text filtering should use a text-only symbol set so transcript characters cannot accidentally produce PAD/BOS/EOS IDs. Unknown removals should also be measured rather than remaining invisible.

Chapter 05 begins with `encoder_layout` and `text_mask`, then builds the text encoder and its prior mean/log-scale outputs.
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
