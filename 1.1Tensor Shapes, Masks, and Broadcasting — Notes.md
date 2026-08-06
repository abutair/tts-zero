# Tensor Shapes, Masks, and Broadcasting — Notes

## 1. Tensors

A tensor is simply a container of numbers.

### 0D Tensor (Scalar)

```text
5
```

Shape:

```text
[]
```

---

### 1D Tensor (Vector)

```text
5 8 2 9
```

Shape:

```text
[4]
```

Meaning:

* One dimension
* Four elements

---

### 2D Tensor (Matrix)

```text
1 2 3
4 5 6
```

Shape:

```text
[2,3]
```

Meaning:

* 2 rows
* 3 columns

---

### 3D Tensor

Think of it as **a stack of matrices (pages)**.

```text
Page 0

1 2 3
4 5 6


Page 1

7 8 9
10 11 12
```

Shape:

```text
[2,2,3]
```

Meaning:

* 2 pages
* 2 rows per page
* 3 columns

---

# 2. Numerical Shape vs Symbolic Shape

Example:

```text
[
 [11 12 13 0 0]
 [21 22 23 24 25]
]
```

Numerical shape:

```text
[2,5]
```

In NLP we usually write it as

```text
[B,T]
```

where

* B = Batch size (number of sentences)
* T = Sequence length (number of positions/tokens)

Example:

```text
[B,T]

↓

[2,5]
```

The symbolic shape only gives names to the dimensions. It does **not** change the tensor.

---

# 3. unsqueeze()

`unsqueeze(dim)` inserts a new dimension of size **1**.

Example:

```python
x = torch.tensor([7,8,9])
```

Shape:

```text
[3]
```

---

## unsqueeze(0)

```python
x.unsqueeze(0)
```

Result:

```text
[[7,8,9]]
```

Shape:

```text
[1,3]
```

Meaning:

* 1 row
* 3 columns

---

## unsqueeze(1)

```python
x.unsqueeze(1)
```

Result:

```text
[
 [7]
 [8]
 [9]
]
```

Shape:

```text
[3,1]
```

Meaning:

* 3 rows
* 1 column

---

# 4. Creating the Mask

Given:

```python
lengths = torch.tensor([3,5])
```

This means:

* Sentence 1 has length 3
* Sentence 2 has length 5

---

Create positions:

```python
positions = torch.arange(5)
```

Result:

```text
0 1 2 3 4
```

Shape:

```text
[5]
```

---

Then:

```python
positions = positions.unsqueeze(0)
```

Result:

```text
[[0 1 2 3 4]]
```

Shape:

```text
[1,5]
```

---

Next:

```python
lengths = lengths.unsqueeze(1)
```

Result:

```text
[
 [3]
 [5]
]
```

Shape:

```text
[2,1]
```

---

Compare:

```python
mask = positions < lengths
```

Result:

```text
[
 [True True True False False]
 [True True True True  True ]
]
```

Shape:

```text
[B,T]
```

The mask is created by comparing:

```text
Position < Length
```

For sentence 1:

```text
0 < 3 ✓
1 < 3 ✓
2 < 3 ✓
3 < 3 ✗
4 < 3 ✗
```

For sentence 2:

```text
0 < 5 ✓
1 < 5 ✓
2 < 5 ✓
3 < 5 ✓
4 < 5 ✓
```

Notice:

The mask depends only on:

* positions
* lengths

It does **not** depend on the token values.

---

# 5. Broadcasting

Broadcasting lets tensors with a dimension of size **1** stretch to match another tensor.

Example 1

```text
A

1 2 3
4 5 6
```

Shape:

```text
[2,3]
```

```text
B

10 20 30
```

Shape:

```text
[1,3]
```

Broadcasts to:

```text
10 20 30
10 20 30
```

Result:

```text
11 22 33
14 25 36
```

---

Example 2

```text
A

1 2 3
4 5 6
```

Shape:

```text
[2,3]
```

```text
B

10
20
```

Shape:

```text
[2,1]
```

Broadcasts to:

```text
10 10 10
20 20 20
```

Result:

```text
11 12 13
24 25 26
```

Important:

Broadcasting **does not copy data**.

It behaves **as if** the data were repeated.

---

# 6. Why is the Mask [B,1,T]?

After building the mask we have

```text
[B,T]
```

Example:

```text
[
 [1 1 1 0 0]
 [1 1 1 1 1]
]
```

The code then does

```python
mask = mask.unsqueeze(1)
```

Result:

```text
[B,1,T]
```

Example:

```text
[
 [
  [1 1 1 0 0]
 ],

 [
  [1 1 1 1 1]
 ]
]
```

---

# 7. Why Add the Extra Dimension?

Initially, it looks unnecessary.

For tensors shaped `[B,T]`, the mask could stay `[B,T]`.

However, later in a neural network, each token is converted into a **feature vector** (an embedding).

Instead of:

```text
11
12
13
```

we might have:

```text
11 -> [0.2, 1.1, -0.4]
12 -> [0.7, 0.5,  2.3]
13 -> [1.4,-0.9,  0.8]
```

Now the tensor shape becomes

```text
[B,C,T]
```

where

* B = Batch
* C = Feature channels (embedding dimension)
* T = Sequence length

The padding positions are identical across all feature channels.

Therefore we only need **one mask**:

```text
[B,1,T]
```

The middle dimension has size **1**, so broadcasting stretches it across every feature channel.

Conceptually:

```text
Mask

1 1 1 0 0
```

becomes

```text
Feature 0

1 1 1 0 0

Feature 1

1 1 1 0 0

Feature 2

1 1 1 0 0
```

without storing multiple copies.

---

# Key Takeaways

* A tensor is just a container of numbers.
* Shapes describe the structure of a tensor.
* Symbolic shapes (e.g. `[B,T]`) simply give names to dimensions.
* `unsqueeze()` inserts a new dimension of size 1.
* The mask is created from **positions** and **lengths**, not token values.
* Broadcasting stretches dimensions of size 1 to match larger dimensions.
* The mask shape `[B,1,T]` exists so the same mask can be applied to every feature channel in tensors of shape `[B,C,T]`.
* The dimension of size **1** is not storing extra information; it exists to enable broadcasting.
