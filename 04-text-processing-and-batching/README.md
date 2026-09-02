# Text Processing and Batching for Arabic VITS2

This chapter converts Arabic text into the padded integer tensors consumed by the VITS2 text encoder.

The goal is not merely to call a tokenizer. The goal is to understand every decision between a Unicode string and an encoder input: which symbols exist, how symbols receive stable IDs, what normalization is safe, how special tokens work, how variable-length sequences form a batch, and how lengths become masks.

The project reference uses an **Arabic character-level representation**. It does not currently use a phonemizer, subword tokenizer, or learned tokenizer. We will reproduce and test that simpler design first. Later, we can compare it with phoneme-based input without confusing two separate problems.

## Learning objectives

By the end of this chapter, you should be able to explain and verify:

- The difference between raw text, normalized text, symbols, token IDs, and embeddings.
- Why a neural network receives integer IDs rather than Python characters.
- Why vocabulary order is part of the saved model's meaning.
- Why padding, beginning-of-sequence, and end-of-sequence need reserved IDs.
- Why the padding ID must not also represent a real text symbol.
- How Arabic Unicode text differs from its visual right-to-left display.
- Why tashkeel count as their own symbols and token positions.
- What the current cleaner keeps, changes, and removes.
- Why silently removing unsupported symbols can hide dataset problems.
- How unequal token sequences become `[B, T_text]`.
- Why the original lengths `[B]` must survive padding.
- How text lengths become a mask shaped `[B, 1, T_text]`.
- Why token tensors use `torch.long`.
- How an embedding changes `[B, T_text]` into `[B, T_text, C]`.
- Why the VITS2 encoder transposes that result to `[B, C, T_text]`.

## Prerequisite

Complete [Chapter 01](../01-tensor-foundations/README.md) first. This chapter reuses padding, original lengths, broadcasting, masks shaped `[B, 1, T]`, and the `[B, C, T]` convention used by `Conv1d`.

Chapters 02 and 03 built the acoustic side of a training example. Chapter 04 builds the matching text side.

## 1. The complete text pathway

```text
raw transcript
    -> conservative normalization and filtering
clean text
    -> split into symbols in logical Unicode order
symbol sequence
    -> symbol-to-ID lookup
token IDs [T_text]
    -> optional BOS/EOS insertion
token IDs with boundaries [T_text]
    -> batch padding
padded token IDs [B, T_text_max] + lengths [B]
    -> sequence mask
text mask [B, 1, T_text_max]
    -> embedding lookup
embedded text [B, T_text_max, C]
    -> transpose for Conv1d-based processing
encoder input [B, C, T_text_max]
```

Here:

```text
B          = batch size
T_text     = number of tokens in one sequence
T_text_max = longest token sequence in the batch
C          = embedding or hidden-channel size
```

These are different representations of the same text. A string is useful to a human; token IDs are useful for lookup; embeddings are useful for learned computation.

## 2. Vocabulary: the contract between text and model

A vocabulary is an ordered collection of every symbol the tokenizer may emit. The reference project groups symbols in this order:

```text
special symbols
Arabic characters
tashkeel
punctuation and space
```

Its special symbols are:

```text
_  padding
^  beginning of sequence (BOS)
$  end of sequence (EOS)
```

Because padding is first, its ID is `0`.

An ID is simply a symbol's position in the vocabulary:

```text
symbols     = ["<pad>", "a", "b", "c"]
positions   = [      0,   1,   2,   3]

symbol_to_id["c"] == 3
id_to_symbol[3]   == "c"
```

The number itself has no linguistic meaning. ID `3` is not “more” than ID `2`. IDs are addresses used by an embedding table.

### Why vocabulary order must remain stable

Suppose a trained checkpoint learned:

```text
ID 10 -> embedding for ب
```

If the vocabulary is reordered and ID `10` later means `ت`, the checkpoint will still retrieve row `10`. It will silently use the learned representation of `ب` for `ت`.

Therefore the vocabulary is model configuration, not an incidental Python list. Once training begins, changing symbol order invalidates the meaning of existing embedding rows unless the checkpoint is migrated deliberately.

### Vocabulary invariants

Your experiment must eventually prove:

```text
len(symbols) == len(set(symbols))
len(symbol_to_id) == len(symbols)
len(id_to_symbol) == len(symbols)
padding_id == 0
id_to_symbol[symbol_to_id[s]] == s
```

The first check catches duplicate symbols. A duplicate in the list could create an embedding row that is never reachable through the dictionary.

## 3. Character tokens versus phoneme tokens

The current reference uses Unicode characters as tokens:

```text
"سلام" -> ["س", "ل", "ا", "م"] -> four character IDs
```

Advantages:

- Small and understandable implementation.
- No external grapheme-to-phoneme system is required.
- The original spelling is mostly recoverable from the IDs.
- It is a good representation for learning the complete data path.

Limitations:

- Arabic spelling does not always specify every vowel.
- The same written form can have different pronunciations.
- Diacritics may be inconsistent or absent.
- The model must learn more pronunciation behavior from data.

A phoneme representation moves some pronunciation decisions into text preprocessing. That may help speech quality, but it also introduces a phonemizer, its symbol set, language rules, and new failure cases. We will not hide the basic batching pipeline behind that complexity in this chapter.

## 4. Arabic text and Unicode

Arabic is displayed right-to-left, but Python stores and iterates the string in **logical order**. Do not reverse the string before tokenization.

```text
Displayed Arabic: سلام
Logical symbols:  س, ل, ا, م
```

The renderer chooses contextual letter shapes. Initial, medial, final, and isolated visual forms should normally not become separate vocabulary symbols.

### A displayed mark may be a separate token

Tashkeel are Unicode characters. A base letter and its fatha are two code points and therefore two character tokens in this project:

```text
بَ -> [ب, َ]
```

Consequently, `len(text)` counts Unicode code points, not visible glyphs, spoken phonemes, bytes, or audio frames.

### Unicode questions to investigate

Two strings that look similar can contain different code points. Your experiments should inspect:

- Arabic versus Persian forms of visually similar letters.
- Arabic and Latin punctuation.
- ordinary spaces versus unusual whitespace.
- letters with tashkeel.
- tatweel (`ـ`).
- precomposed and combining forms where applicable.

Use `repr(text)`, `list(text)`, `len(text)`, and `ord(character)` when visual display is ambiguous.

## 5. Text cleaning is a linguistic decision

The reference cleaner currently performs two operations:

1. Replace every run of whitespace with one ordinary space and remove spaces at the ends.
2. Keep only characters present in the vocabulary.

Conceptually:

```text
raw text
    -> collapse whitespace
    -> trim outer whitespace
    -> vocabulary whitelist
clean text
```

This is intentionally minimal, but “minimal” does not mean harmless.

### Filtering versus normalization

Filtering asks whether an exact character is allowed. Normalization asks whether a character should be transformed into a chosen canonical form.

Deleting an unsupported character is not the same as normalizing it. Deciding whether two Arabic letter forms should remain distinct requires linguistic and dataset reasoning, not just code.

### The danger of silent deletion

If an unsupported character disappears silently, training continues, but the transcript no longer matches the intended sentence. That can be harder to discover than a loud error.

Your chapter experiment should produce an unknown-character audit:

```text
original characters
cleaned characters
removed characters
removed count
```

Possible policies include:

- raise an error;
- emit an explicit `<unk>` token;
- remove the character and record it;
- normalize it to a supported character when linguistically justified.

The reference currently removes unsupported characters. Reproduce that behavior first, then evaluate the trade-off rather than treating it as automatically correct.

### Reference edge case: structural symbols in raw text

The reference cleaner checks membership in the complete symbol table, and that table includes `_`, `^`, and `$`. As a result, those literal characters can survive cleaning and be encoded as PAD, BOS, or EOS even when they came from the transcript rather than from the tokenizer.

That conflicts with the intended separation between text symbols and structural symbols. Your experiment must expose this behavior with a deliberately chosen input. Do not silently “fix” it before reproducing it; first observe the failure, explain why it is dangerous, and then compare policies such as:

- reject structural characters found in raw text;
- filter against a text-only symbol set;
- represent special tokens with values that cannot collide with transcript characters.

## 6. Special tokens

### Padding (`PAD`)

Padding fills unused positions so several sequences can share one rectangular tensor.

```text
sequence A: [11, 12, 13]
sequence B: [21, 22]

padded batch:
[
  [11, 12, 13],
  [21, 22,  0],
]
```

If `0` were also a real symbol, the model could not determine from the ID alone whether the last position contained data or padding. Reserve it exclusively for padding.

This is the intended design contract. The current reference cleaner does not fully enforce it for a literal underscore in raw input, which is why the edge-case test above matters.

### Beginning and end (`BOS`, `EOS`)

The reference adds BOS and EOS by default:

```text
clean symbols:       [س, ل, ا, م]
with boundaries: [BOS, س, ل, ا, م, EOS]
```

Therefore the stored text length includes both boundary tokens:

```text
text_length = number_of_clean_symbols + 2
```

for a nonempty sequence when both are enabled.

BOS/EOS are real valid positions and must be marked `True` in the text mask. PAD positions are invalid and must be `False`.

### Empty-text edge case

If cleaning removes every input character and BOS/EOS are enabled, the encoded sequence may still contain two tokens. That does **not** mean the transcript is acceptable. Dataset validation should separately reject text that becomes linguistically empty.

## 7. Encoding and decoding

Encoding performs a symbol-to-integer lookup. Decoding performs the inverse lookup.

For a valid supported sentence, the useful round-trip property is:

```text
decode(encode(raw_text)) == clean(raw_text)
```

The decoder normally omits PAD, BOS, and EOS because they are structural markers rather than transcript characters.

This round trip tests mapping consistency. It does not prove that the cleaner made the correct linguistic decision.

## 8. Why token tensors use `torch.long`

The token tensor contains row indices for an embedding table. PyTorch embedding lookup expects integer index tensors, normally `torch.long` (`int64`).

```text
token IDs: [B, T_text], dtype = torch.long
```

Floating-point IDs would suggest interpolation or measurement, but token IDs are discrete addresses. Converting IDs to `float32` makes them invalid as embedding indices.

## 9. Batching variable-length sequences

Assume the encoded sequences have lengths `6`, `4`, and `5`. The batch maximum is `6`.

After padding:

```text
text.shape         == [3, 6]
text_lengths.shape == [3]
text_lengths       == [6, 4, 5]
```

The padded tensor alone cannot reveal the original lengths safely. We retain them because later modules need to know which positions are data.

For sample `b`:

```text
padding_count[b] = T_text_max - text_lengths[b]
```

Total padding waste is:

```text
B * T_text_max - sum(text_lengths)
```

This motivates length-based bucketing for large datasets: batching examples of similar sizes reduces wasted memory and computation. Bucketing changes efficiency, not the meaning of a sample.

## 10. Lengths become masks

For lengths `[3, 5]` and maximum length `5`:

```text
positions = [0, 1, 2, 3, 4]

position < 3 -> [True, True, True, False, False]
position < 5 -> [True, True, True, True,  True]
```

The initial comparison produces `[B, T_text]`. VITS2 then inserts the channel dimension:

```text
text_mask.shape == [B, 1, T_text]
```

This shape broadcasts over encoder channels:

```text
features [B, C, T_text]
mask     [B, 1, T_text]
result   [B, C, T_text]
```

`True` means a valid token position; `False` means padding. The mask length is based on encoded tokens, including BOS/EOS—not raw characters before cleaning.

## 11. From token IDs to embeddings

An embedding table has conceptual shape:

```text
[number_of_symbols, hidden_channels]
```

Each token ID selects one row:

```text
token IDs           [B, T_text]
embedding lookup -> [B, T_text, C]
transpose         -> [B, C, T_text]
```

The embedding output is floating point because its values are learned features. The token input remains integer-valued.

In the reference text encoder:

```text
x = embedding(token_ids)          # [B, T_text, C]
x = transpose time and channels   # [B, C, T_text]
x_mask                            # [B, 1, T_text]
```

The encoder masks features before and after relevant operations so padded positions do not contaminate valid computation.

### A subtle padding point

An input ID equal to `PAD_ID` does not guarantee that its embedding vector is zero. That only happens automatically when the embedding is configured with the appropriate padding behavior, or when the output is explicitly masked.

The reference embedding does not pass a `padding_idx`; it relies on `x_mask` to remove padded positions in the encoder path. This is a design choice worth testing rather than assuming.

## 12. The reference-project contract

The final reference pathway is distributed across these components:

```text
symbols.py
    defines ordered symbols and both lookup tables

arabic.py
    cleans text, encodes it, and decodes IDs

tts_dataset.py
    converts one transcript to a 1-D LongTensor
    stores its pre-padding token length

collate.py
    pads multiple text tensors to [B, T_text_max]
    preserves lengths as [B]

encoder.py
    embeds IDs, transposes to [B, C, T_text], and builds [B, 1, T_text] mask
```

This chapter will reproduce that contract in a small independent notebook before relying on the finished training implementation.

# Experiment roadmap

Build one canonical notebook named `experiment.ipynb`. Complete the tasks in order. Each code section should be preceded by a Markdown cell containing:

- the question being tested;
- your predicted result;
- the relevant shapes;
- what would count as failure.

Do not begin by importing the finished project tokenizer. Rebuild the core ideas using small values, then compare with the reference near the end.

## Task 1: Tiny vocabulary on paper

Use this deliberately small vocabulary:

```text
["<pad>", "a", "b", "c", "h", "i", "t"]
```

Before coding, predict:

1. Every symbol's ID.
2. The encoded IDs for `"cat"`.
3. The encoded IDs for `"hi"`.
4. The padded batch shape.
5. The padded values.
6. The original lengths.
7. The boolean mask.

Expected structural result:

```text
batch shape:   [2, 3]
lengths shape: [2]
mask shape:    [2, 1, 3]
```

## Task 2: Construct reversible lookup tables

Build symbol-to-ID and ID-to-symbol mappings from the tiny ordered vocabulary.

Test that every ID and symbol is unique, both maps have the vocabulary's size, and looking up a symbol and reversing the result returns the original symbol.

Do not manually type both dictionaries. Derive both from the one ordered source of truth.

## Task 3: Inspect Arabic as Unicode data

Choose at least three short strings:

1. Arabic letters without tashkeel.
2. Arabic letters with at least one diacritic.
3. Arabic text containing punctuation and repeated whitespace.

For each, inspect:

```text
repr(text)
list(text)
len(text)
Unicode code point of each symbol
```

Write down why visual right-to-left rendering does not change tensor order.

## Task 4: Recreate the project vocabulary

Define the groups separately:

```text
special symbols
Arabic characters
tashkeel
punctuation
```

Combine them once into the final ordered vocabulary. Then derive:

```text
symbol_to_id
id_to_symbol
padding_id
bos_id
eos_id
number_of_symbols
```

Assertions must catch duplicate symbols and verify the three special IDs.

Also keep a separate set of symbols allowed to originate in transcripts. Structural symbols belong to the model vocabulary but should not automatically belong to the raw-text whitelist.

## Task 5: Whitespace normalization

Start only with whitespace behavior. Test ordinary spaces, tabs, newlines, leading whitespace, trailing whitespace, and repeated whitespace.

Predict the normalized string before running the cell. The invariant is:

```text
no outer whitespace
no run of internal whitespace longer than one ordinary space
```

## Task 6: Vocabulary filtering and unknown audit

Add the vocabulary whitelist as a separate conceptual step. Use a sentence containing supported Arabic text plus deliberately unsupported characters.

Record:

```text
raw text
whitespace-normalized text
clean text
removed symbols
removed count
```

Do not merely print the cleaned sentence. The purpose is to make information loss observable.

Then answer: when would raising an error be safer than silently removing a character?

Finally, test raw strings containing `_`, `^`, and `$`. Record what the reference does, why PAD/BOS/EOS appearing from raw text is ambiguous, and what rule your independent tokenizer should enforce.

## Task 7: Encode, add boundaries, and decode

Implement encoding in stages:

```text
raw text
clean text
character IDs
BOS + character IDs + EOS
```

Then decode while omitting structural tokens.

Prove:

```text
decode(encode(raw_text)) == clean(raw_text)
```

Also test encoding with boundary tokens disabled and explain how the length changes.

## Task 8: Create one dataset-style item

Convert one encoded ID list into a tensor and attach the metadata required before batching:

```text
text:        torch.LongTensor with shape [T_text]
text_length: scalar Python integer or scalar integer value
raw_text:    original transcript
```

Verify:

```text
text.ndim == 1
text.dtype == torch.long
text_length == text.shape[0]
```

Explain why this item is not shaped `[1, T_text]` yet. The batch dimension belongs to collation, not to an individual sample.

## Task 9: Pad a batch manually

Encode at least three Arabic strings with different token lengths. Without using the final collator, create:

```text
padded_text [B, T_text_max]
text_lengths [B]
```

Before constructing the batch, predict `B`, every encoded length, `T_text_max`, the final shape, padding count per row, and total padding count.

Check every row individually: valid positions must match its original sequence, and remaining positions must equal `PAD_ID`.

## Task 10: Reuse length-to-mask reasoning

Turn `text_lengths` into:

```text
text_mask [B, 1, T_text_max]
```

Verify for every row:

```text
text_mask[b].sum() == text_lengths[b]
```

Then multiply a fake feature tensor shaped `[B, C, T_text_max]` by the mask. Confirm that all padded time positions become zero across every channel.

## Task 11: Use the standard padding utility

After the manual version is understood, reproduce the same batch with PyTorch's sequence-padding utility.

Compare the manual and library results exactly:

```text
manual_padded_text == library_padded_text
```

Passing this checkpoint means the helper is replacing understood mechanics, not hiding them.

## Task 12: Cross the embedding boundary

Create a small embedding table using the vocabulary size and a small hidden-channel value such as `4` or `8`.

Predict and verify:

```text
token IDs:           [B, T_text]
embedded tokens:     [B, T_text, C]
transposed features: [B, C, T_text]
mask:                [B, 1, T_text]
```

Inspect padded embedding positions before masking and after masking. Explain what you observe.

Do not build the Transformer here. That belongs to Chapter 05.

## Task 13: Compare with the reference implementation

Only after your independent version passes, compare it against the project's reference text functions and collator.

Use the same input strings and verify:

- cleaned strings match;
- encoded ID lists match;
- decoded strings match;
- padded batches match;
- lengths match;
- all dtypes and shapes match.

If they differ, identify the first stage where the values diverge. Do not debug the whole pipeline at once.

## Task 14: Final chapter assertions

The finished notebook should prove:

```text
vocabulary contains no duplicates
PAD, BOS, and EOS have distinct IDs
raw transcript characters cannot produce PAD, BOS, or EOS IDs
mapping inversion works for every symbol
cleaning is deterministic
round-trip decoding returns cleaned supported text
token tensors use torch.long
individual token tensors are [T_text]
padded batches are [B, T_text_max]
length tensors are [B]
masks are [B, 1, T_text_max]
mask sums equal original token lengths
masked padded features are zero
embedding output is [B, T_text_max, C]
encoder-layout output is [B, C, T_text_max]
```

# Debugging strategy

When a test fails, locate the first broken boundary:

1. Print `repr(raw_text)` rather than trusting visual appearance.
2. Inspect `list(raw_text)` and Unicode code points.
3. Compare raw and whitespace-normalized text.
4. List removed or unknown symbols explicitly.
5. Confirm the ordered vocabulary has no duplicates.
6. Inspect each symbol-to-ID lookup before adding BOS/EOS.
7. Confirm length is measured after cleaning and boundary insertion.
8. Confirm one sample is `[T_text]`, not `[1, T_text]`.
9. Confirm IDs have dtype `torch.long`.
10. Confirm batch padding uses `PAD_ID`.
11. Compare each padded row with its original unpadded tensor.
12. Regenerate the mask from stored lengths.
13. Label every tensor dimension before attempting broadcasting.
14. Inspect embedding output before blaming the encoder.

## Common reasoning mistakes

### “Arabic is right-to-left, so I should reverse the tokens”

Right-to-left is a display rule. Tokenization follows logical string order. Reversing the Python string corrupts the sequence.

### “`len(text)` is the number of spoken sounds”

It counts Python string elements/code points in this character-level pipeline. A diacritic can add a token, while one written character may not correspond cleanly to one phoneme.

### “Cleaning only makes text prettier”

Cleaning changes training data. Removing or merging characters can change pronunciation and alignment.

### “The padded tensor tells us its own true length”

Only if padding can never occur as real data and you deliberately scan for it. Stored lengths are explicit, efficient, and remain correct after later layers turn padded features nonzero.

### “Token IDs are numerical features”

They are categorical addresses. Their ordering and numerical distance carry no intended linguistic meaning.

### “Padding IDs automatically become zero embeddings”

That depends on embedding configuration. Always inspect the actual behavior and preserve the mask.

### “Unsupported characters can be dropped because the model cannot use them”

Silent removal can create transcript/audio mismatch. Unknown-symbol frequency is a dataset-quality signal and should be audited.

### “Anything in the vocabulary is valid raw text”

The model vocabulary also contains structural tokens. A raw-text whitelist should distinguish ordinary transcript symbols from PAD/BOS/EOS.

# Questions you must be able to answer

1. Why must vocabulary order remain unchanged after training begins?
2. Why is `PAD_ID` reserved rather than shared with a real symbol?
3. Why does `text_length` include BOS and EOS in this project?
4. What is the difference between filtering and normalization?
5. Why can two visually similar Arabic strings produce different IDs?
6. Why should Arabic text not be reversed before tokenization?
7. Why are token IDs `torch.long` while embeddings are floating point?
8. Why is a dataset sample `[T_text]` but a batch `[B, T_text]`?
9. Why retain lengths after padding?
10. Why is the encoder mask `[B, 1, T_text]`?
11. What happens to the shape during embedding and transposition?
12. Why might a padded position contain a nonzero embedding?
13. How would you detect unsupported symbols before they damage training data?
14. What does a successful encode/decode round trip prove—and what does it not prove?
15. Why should the raw-text whitelist differ from the complete model vocabulary?

# Chapter completion criteria

Chapter 04 is complete when you can explain every transition in:

```text
Arabic string
    -> symbols
    -> token IDs [T_text]
    -> padded IDs [B, T_text_max]
    -> embeddings [B, T_text_max, C]
    -> encoder layout [B, C, T_text_max]
```

and your notebook independently demonstrates that:

1. The vocabulary and inverse mappings are stable and reversible.
2. Arabic Unicode behavior is inspected rather than guessed from display.
3. Cleaning decisions and removed characters are observable.
4. BOS/EOS and padding semantics are correct.
5. Unequal sequences are padded without losing their original lengths.
6. Lengths create correct broadcastable masks.
7. Token IDs cross the embedding boundary with the expected dtype and shapes.
8. Results match the reference pipeline only after the independent implementation works.

The next chapter will use this output to build the **text encoder and prior distribution**. It will begin at token embeddings and study attention, feed-forward layers, masking inside the encoder, and the predicted prior mean and log-scale.
