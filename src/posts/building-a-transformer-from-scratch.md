---
title: Building a Transformer from scratch
date: 2026-07-15
summary: A deep, line-by-line walkthrough of reimplementing "Attention Is All You Need" in PyTorch — the attention math, every reshape, masking, positional encodings, teacher forcing, mixed-precision training, and a length-penalized beam search.
tags: [NLP, Transformer, PyTorch]
---

I wanted to *really* understand the Transformer, so I did the thing you're
supposed to do: I reimplemented the 2017 [*Attention Is All You
Need*](https://arxiv.org/abs/1706.03762) paper from scratch in PyTorch and
trained it on English→German translation. Reading the paper gives you the
shape of the architecture. Writing it gives you all the little decisions the
paper glosses over — the masks, the tensor gymnastics, the numerics, the
difference between a training loss that drops and a model that actually
translates.

This is the long version. I'm going to walk through nearly every component,
in the order data flows through it, and explain not just *what* each line
does but *why* it has to be that way. The full code is on
[GitHub](https://github.com/sadra-etaei/mini-transformer).

**Roadmap:** the big picture → scaled dot-product attention → multi-head
attention and every reshape → the feed-forward network → positional
encodings → residuals and LayerNorm → encoder and decoder layers → masking →
assembling the full model → tokenization and batching → teacher forcing and
the training loop → the loss → fitting it on one GPU → beam search →
evaluation. It's a lot, but each piece is small.

## The big picture

Before 2017, sequence-to-sequence models were built on recurrent networks
(RNNs, LSTMs): you fed tokens in one at a time, carrying a hidden state
forward. That's inherently sequential — you can't compute step 50 until
you've computed step 49 — so it's slow to train and it struggles to connect
words that are far apart, because information has to survive being passed
through dozens of intermediate states.

The Transformer throws recurrence out entirely. Its central idea: let every
token look directly at every other token in a single operation, and weight
how much attention it pays to each. There's no distance penalty — the first
and last word of a sentence are one step apart — and the whole sequence is
processed in parallel.

The architecture is an **encoder–decoder** pair. The encoder reads the
English sentence and produces a stack of context-aware vectors, one per
token. The decoder generates the German sentence one token at a time, each
step attending both to what it has produced so far (self-attention) and to
the encoder's output (cross-attention). Both halves are built from the same
two Lego bricks — **attention** and a **feed-forward network** — stacked six
deep. Let's build those bricks.

## Scaled dot-product attention: the one equation

Everything rests on this:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The names come from a database analogy that genuinely helps. Imagine a
lookup table. A **query** ($Q$) is what you're searching for. Each entry has
a **key** ($K$) you match against, and a **value** ($V$) you retrieve. A
normal dictionary does a hard match — one key wins. Attention does a *soft*
match: it compares the query against every key, turns those comparisons into
weights that sum to 1, and returns a blended average of all the values.

Concretely, for one query vector:

1. **Compare.** Take the dot product of the query with every key. A dot
   product is large when two vectors point the same way, so it's a similarity
   score. That's $QK^\top$.
2. **Scale.** Divide by $\sqrt{d_k}$ (more on why in a second).
3. **Normalize.** Apply softmax across the keys, turning the raw scores into
   a probability distribution — the attention weights.
4. **Retrieve.** Multiply those weights by the value vectors and sum. Keys
   the query matched strongly contribute most.

In code it's almost a direct transcription:

```python
class ScaledDotProductAttn(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, V), attn
```

### Why divide by √dₖ?

This is the "easy to skip, costly to omit" detail. Suppose the components of
$Q$ and $K$ are independent, each with mean 0 and variance 1. Their dot
product is a sum of $d_k$ products; each product has variance 1, so the sum
has **variance $d_k$** and standard deviation $\sqrt{d_k}$. With
$d_k = 64$ that's a spread of ±8 or more.

Feed numbers that large into softmax and it saturates: one weight rounds to
almost 1, the rest to almost 0. That's a problem because the gradient of
softmax in its saturated region is nearly zero — the model can barely learn.
Dividing by $\sqrt{d_k}$ rescales the variance back to 1, keeping the scores
in the range where softmax is smooth and gradients flow. One division is the
difference between a network that trains and one that stalls.

### The masking value

Note `masked_fill(mask == 0, float('-inf'))`: we set masked positions to
negative infinity *before* the softmax, so $e^{-\infty} = 0$ and those keys
get exactly zero weight. It has to be `-inf` (or a very large negative like
`-1e9`), not a small number — a gentle nudge like `-1e-9` would leave the
forbidden positions almost fully visible, which is a genuinely nasty bug
because the model still trains, just slightly wrong.

## Multi-head attention: einsum and every reshape, in detail

This is the section I most wish someone had written for me. Multi-head
attention isn't conceptually hard, but the *bookkeeping* — which axis is
which, when to reshape, when memory is contiguous — is where nearly all my
bugs lived. Let me go slowly through every shape.

### What "multi-head" actually means

Instead of computing attention once over the full `d_model`-dimensional
vectors, we split those vectors into `n_heads` smaller chunks of size
`d_k = d_model / n_heads`, run attention independently in each chunk, then
glue the results back together. Each head gets to specialize — one might
track subject–verb agreement, another local word order, another
long-distance references — and they don't interfere because attention is
computed separately per head. It's the difference between asking one person
for their overall impression and asking eight specialists and combining their
answers.

So for a model with `d_model = 512` and `n_heads = 8`, each head works in a
`d_k = 64`-dimensional subspace. The whole job of the reshaping is to
*carve out* those 8 independent subspaces, do the math, and *stitch* them
back into one `512`-dimensional vector. The module holds four linear layers —
`W_q`, `W_k`, `W_v` to build the queries/keys/values, and `W_o` to mix the
concatenated heads back together at the end:

```python
class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

The `assert` matters: if `d_model` isn't divisible by `n_heads`, the split
doesn't tile evenly and you'll get a shape error three steps later that's
much harder to trace back to its cause.

### A quick primer on einsum

`torch.einsum` (Einstein summation) lets you describe a tensor operation by
*naming every axis with a letter* and writing what you want the output to
look like. The rule is delightfully simple:

- Each input tensor gets a group of letters, one per axis, in order.
- Any letter that appears in the inputs but **not** in the output is
  **summed over** (contracted).
- Any letter that appears in both inputs is **matched up** (multiplied
  element-wise along that axis before summing).
- Letters that survive to the output become its axes, in the order you
  write them.

That's the entire mental model. `torch.matmul` forces you to remember that
it batches over leading dimensions and contracts the last axis of the first
tensor with the second-to-last of the other — rules you have to hold in your
head. With einsum the rule is written *in the string*, right next to the
operation, so the code is its own shape documentation.

### Step 1 — projecting Q, K, V

We start with input of shape `[batch, seq_len, d_model]`. The three linear
layers keep that shape (they map `d_model → d_model`):

```python
Q = self.W_q(Q)   # [Batch, Q_Len, D_Model]
K = self.W_k(K)   # [Batch, K_Len, D_Model]
V = self.W_v(V)   # [Batch, K_Len, D_Model]
```

Note `Q_Len` and `K_Len` can differ — in cross-attention the decoder's
queries attend to the encoder's keys/values, which have a different length.
Keeping them as separate letters (`q` vs `k`) later is what makes that case
just work with no special handling.

### Step 2 — splitting into heads with `view`

Now the key move. That last `d_model` axis is really `n_heads × d_k` values
laid out contiguously in memory. `view` lets us *reinterpret* those same
numbers as two axes without moving any data:

```python
# [Batch, Seq_Len, D_Model] -> [Batch, Seq_Len, Heads, D_k]
Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
K = K.view(batch_size, -1, self.n_heads, self.d_k)
V = V.view(batch_size, -1, self.n_heads, self.d_k)
```

The `-1` says "infer this axis" — it's the sequence length, which we let
PyTorch compute. Because the last dimension was already `n_heads * d_k`,
splitting it into `[n_heads, d_k]` is free: element `[b, s, h, d]` is exactly
the number that used to sit at position `h * d_k + d` in the old last axis.
No copy happens, which is why this is fast — `view` only rewrites the
tensor's *metadata* (its shape and strides), not its bytes.

A subtle point: many tutorials immediately `.transpose(1, 2)` here to get
`[Batch, Heads, Seq_Len, D_k]`, because they want the head axis next to the
batch axis so `matmul` will treat `(batch, heads)` as batch dimensions. That
transpose is exactly the fiddly, error-prone step I wanted to avoid — so I
*don't* transpose, and let einsum handle the axis ordering instead.

### Step 3 — the scores: QKᵀ per head

Here's where einsum earns its keep. I want, for every batch and every head,
the `Q_Len × K_Len` matrix of dot products between each query and each key:

```python
scores = torch.einsum('b q h d, b k h d -> b h q k', Q, K) / math.sqrt(self.d_k)
```

Read the string carefully:

- First tensor `Q` has axes `b q h d` (batch, query pos, head, d_k).
- Second tensor `K` has axes `b k h d` (batch, key pos, head, d_k).
- The letter **`d` is missing from the output** — so we sum over it. That
  summation *is* the dot product: for a fixed query and key we multiply
  their `d_k` components and add them up.
- `b` and `h` appear in both inputs and the output — they're just carried
  along, batched. Attention in head 3 never mixes with head 5.
- `q` and `k` each appear in only one input and both survive to the output,
  so they become the two axes of the resulting score matrix.

Result: `[Batch, Heads, Q_Len, K_Len]`. To write this with `matmul` I'd have
had to transpose `K`'s last two axes *and* have already moved the head axis
forward. Here I just declared the shape I wanted. The `/ √d_k` is the same
scaling derived above, keeping the softmax gradients healthy.

### Step 4 — softmax and the weighted sum of values

Softmax runs over the `k` (key) axis — each query produces a probability
distribution over all keys, answering "how much of my attention budget goes
to each of them?":

```python
attention = F.softmax(scores, dim=-1)   # [Batch, Heads, Q_Len, K_Len]
```

Then the second einsum uses those weights to blend the value vectors:

```python
out = torch.einsum('b h q k, b k h d -> b q h d', attention, V)
```

Again, walk the letters:

- `attention` is `b h q k`, `V` is `b k h d`.
- **`k` is missing from the output** → we sum over the keys. For each query
  `q`, that's exactly "weighted sum of all value vectors, weighted by the
  attention probabilities."
- `b` and `h` are carried through (per batch, per head).
- Notice I asked for the output as `b q h d`, deliberately putting the head
  axis back *after* the sequence axis — because that's the layout I need for
  the final merge.

Result: `[Batch, Q_Len, Heads, D_k]`.

### Step 5 — merging the heads back together

The heads have done their independent work; now I concatenate each head's
`d_k`-vector back into a single `d_model`-vector and mix them with `W_o`:

```python
out = out.contiguous().view(batch_size, -1, self.n_heads * self.d_k)
return self.W_o(out)   # final linear projection, [Batch, Q_Len, D_Model]
```

This `view` is the exact inverse of Step 2: collapse `[Heads, D_k]` back into
one `d_model` axis. The final `W_o` is not decorative — without it the heads
would just sit side by side, never allowed to combine their findings. `W_o`
is what lets "head 2 found the subject and head 5 found the verb" turn into a
single coherent representation.

And here's the one gotcha that *will* bite you — **`.contiguous()`**.

`view` can only reinterpret memory that's laid out in the standard row-major
order. But the einsum in Step 4 produced a tensor whose *logical* axis order
(`b q h d`) doesn't match its *physical* memory order — internally the head
axis is still stored where it was, so the bytes for `[b, q, h, d]` aren't
sitting one after another the way `view` needs. Calling `.contiguous()`
forces PyTorch to actually copy the data into that clean row-major layout,
after which `view` is legal again. Skip it and you get the classic:

```text
RuntimeError: view size is not compatible with input tensor's size and
stride ... use .reshape(...) instead
```

(You *could* use `.reshape()`, which silently does the copy for you when
needed — but `.contiguous().view()` makes the one unavoidable copy explicit,
which I prefer when I'm learning where the costs are.)

### The whole pipeline at a glance

Here's every shape flowing through one multi-head attention block, from the
input embeddings to the output. `B` = batch, `L` = sequence length,
`H` = heads, `dₖ` = per-head dimension (`d_model / H`). In self-attention the
query and key lengths are equal; I write `Lq`/`Lk` to show where they're
allowed to differ (cross-attention):

<svg viewBox="0 0 720 580" role="img" aria-label="Tensor shapes flowing through a multi-head attention block" style="width:100%;height:auto;max-width:640px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>

  <!-- Box 1: Input -->
  <rect x="40" y="14" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="32" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">input embeddings</text>
  <text x="205" y="49" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, L, d_model]</text>

  <!-- arrow 1 -->
  <line x1="205" y1="60" x2="205" y2="96" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="82" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">W_q · W_k · W_v  (linear)</text>

  <!-- Box 2: Q,K,V -->
  <rect x="40" y="98" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="116" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">Q, K, V</text>
  <text x="205" y="133" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, L, d_model]</text>

  <!-- arrow 2 -->
  <line x1="205" y1="144" x2="205" y2="180" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="166" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">view — split into H heads</text>

  <!-- Box 3: split heads -->
  <rect x="40" y="182" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="200" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">per-head Q, K, V</text>
  <text x="205" y="217" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, L, H, dₖ]</text>

  <!-- arrow 3 -->
  <line x1="205" y1="228" x2="205" y2="264" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="245" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">einsum bqhd,bkhd→bhqk</text>
  <text x="224" y="259" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">÷√dₖ, then softmax over keys</text>

  <!-- Box 4: attn weights -->
  <rect x="40" y="266" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="284" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">attention weights</text>
  <text x="205" y="301" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, H, Lq, Lk]</text>

  <!-- arrow 4 -->
  <line x1="205" y1="312" x2="205" y2="348" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="334" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">einsum bhqk,bkhd→bqhd  (× V)</text>

  <!-- Box 5: context -->
  <rect x="40" y="350" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="368" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">per-head context</text>
  <text x="205" y="385" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, Lq, H, dₖ]</text>

  <!-- arrow 5 -->
  <line x1="205" y1="396" x2="205" y2="432" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="418" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">contiguous + view — merge heads</text>

  <!-- Box 6: merged -->
  <rect x="40" y="434" width="330" height="44" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="205" y="452" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">merged heads</text>
  <text x="205" y="469" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, Lq, d_model]</text>

  <!-- arrow 6 -->
  <line x1="205" y1="480" x2="205" y2="516" style="stroke:var(--text-faint)" marker-end="url(#arrow)"/>
  <text x="224" y="502" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">W_o (output projection)</text>

  <!-- Box 7: output -->
  <rect x="40" y="518" width="330" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="205" y="536" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">output</text>
  <text x="205" y="553" text-anchor="middle" style="font-size:14px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">[B, Lq, d_model]</text>
</svg>

The shape leaves as `[B, L, d_model]` and comes back as `[B, Lq, d_model]` —
same rank, ready to drop straight into the residual connection. Everything in
between is the temporary detour into head-space and back. Once I switched to
einsum, an entire category of "why is my tensor `[8, 32, ...]` instead of
`[32, 8, ...]`" bugs simply evaporated: the string `b q h d, b k h d ->
b h q k` *is* the documentation, the shape assertion, and the implementation
all at once.

## The feed-forward network

Attention moves information *between* positions, but it's mostly a weighted
average — a fundamentally linear mixing operation. The model needs a source
of real non-linear processing too, and that's the position-wise
feed-forward network (FFN): a little two-layer MLP applied to each token
independently.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.Linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.Linear2(self.dropout(F.relu(self.Linear1(x))))
```

It expands each token's vector up to a wider hidden dimension `d_ff` (in the
paper, 2048 for a 512-dim model — a 4× expansion; I used `d_ff = 1024` for my
256-dim model, keeping the same ratio), applies a ReLU non-linearity, then
projects back down to `d_model`. "Position-wise" means the *same* MLP is
applied to every token separately — there's no mixing across positions here;
that already happened in attention. This is where a surprisingly large
fraction of the model's parameters (and its capacity to transform features)
actually lives. A useful mental model: attention decides *what to look at*,
the FFN decides *what to think about it*.

## Positional encoding

Here's a consequence of throwing out recurrence that's easy to miss:
attention is **permutation-invariant**. If you shuffle the input tokens, the
set of attention scores is the same set — the model has no built-in notion of
order. "Dog bites man" and "man bites dog" would look identical. We have to
*inject* position information manually.

The paper's trick is to add a fixed pattern of sines and cosines to the
embeddings, with a different frequency for each dimension:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        exp_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * exp_term)   # even dims
        pe[:, 1::2] = torch.cos(position * exp_term)   # odd dims
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

A few details worth unpacking:

- **The `exp`/`log` reformulation.** Computing $10000^{2i/d}$ directly for
  large exponents is numerically shaky. The identity
  $10000^{-2i/d} = e^{-2i \cdot \ln(10000) / d}$ turns a risky power into a
  stable exponential — that's exactly what `exp_term` is.
- **Why sinusoids?** Two reasons. First, for any fixed offset $k$,
  $PE_{pos+k}$ is a linear function of $PE_{pos}$ — so the model can learn to
  attend by *relative* position just from these features. Second, because
  the pattern is defined by a formula rather than learned, it extrapolates to
  sequences longer than any seen in training.
- **`register_buffer`.** This tells PyTorch that `pe` is persistent state
  that is *not* a trainable parameter. It won't get gradients or be updated
  by the optimizer, but it *will* move to the GPU with `model.to(device)` and
  be saved in the checkpoint. Exactly right for a fixed lookup table.
- **The slice in `forward`.** `self.pe` is precomputed out to `max_len`, but
  a given batch is only `x.size(1)` tokens long, so we add just that prefix.

## Residual connections and LayerNorm

Every sublayer in the Transformer — each attention block, each FFN — is
wrapped in the same two-part envelope. First a **residual connection**: add
the sublayer's input back to its output.

The reason is gradient flow. Stack six layers (or in bigger models, ninety-
six) and gradients have to propagate all the way back through every one; they
tend to shrink or explode en route. A residual `x + sublayer(x)` gives the
gradient a direct "skip" path back to earlier layers — it can always flow
through the `+` untouched, so deep stacks stay trainable. It also means each
sublayer only has to learn a *correction* to its input, not reconstruct the
whole representation from scratch.

Second, **layer normalization**: for each token, normalize its `d_model`
features to zero mean and unit variance, then apply a learned scale and
shift. This keeps the magnitude of activations stable as they pass through
many layers, which makes training far less finicky. Unlike batch norm, it
normalizes across features *within one token*, so it doesn't depend on the
other examples in the batch — important when sequences have varying lengths.

This code uses **post-norm** — `norm(x + sublayer(x))` — which is the
original paper's arrangement. (Modern implementations often move to
**pre-norm**, `x + sublayer(norm(x))`, which is a bit more stable for very
deep models and usually needs less learning-rate warmup. Worth knowing the
distinction; both appear everywhere.)

## Encoder and decoder layers

With attention, the FFN, residuals, and norm in hand, a layer is just those
pieces in sequence. The encoder layer is self-attention then feed-forward,
each wrapped in dropout → residual → norm:

```python
def forward(self, x, mask):
    attn_output = self.attn(x, x, x, mask)
    x = self.norm1(x + self.dropout(attn_output))   # residual + norm
    ff_out = self.ff(x)
    x = self.norm2(x + self.dropout(ff_out))
    return x
```

Notice `self.attn(x, x, x, mask)` — query, key, and value are all the same
tensor. That's what makes it *self*-attention: every token attends to every
other token in the same sequence.

The decoder layer has **three** sublayers instead of two. Its self-attention
uses a causal mask (so it can't peek ahead — more on that next), and then a
new middle sublayer, *cross-attention*, is inserted:

```python
def forward(self, x, encoder_output, src_mask, tgt_mask):
    attn_output = self.attn(x, x, x, tgt_mask)                 # masked self-attn
    x = self.norm1(x + self.dropout(attn_output))
    cross = self.cross_attn(x, encoder_output, encoder_output, src_mask)  # cross-attn
    x = self.norm2(x + self.dropout(cross))
    ff_out = self.ff(x)
    x = self.norm3(x + self.dropout(ff_out))
    return x
```

The cross-attention line is the bridge between the two languages. Look at its
arguments: the **query** comes from the decoder (`x` — what I'm trying to
generate), but the **key and value** come from `encoder_output` (the encoded
source sentence). So each German position I'm producing gets to ask the whole
English sentence "which of you is relevant to me right now?" This is exactly
where the model learns alignment — that German *Hund* should attend to
English *dog*.

## Masking: where the bugs live

This was the part the paper made look trivial and the implementation made
humbling. There are two different masks doing two different jobs, and both
are about controlling *what a query is allowed to see*.

The **source mask** hides padding. Because sentences in a batch have
different lengths, we pad the short ones with `[PAD]` tokens up to the batch
maximum — but those are meaningless filler, and attention must never spend
weight on them:

```python
def create_src_mask(self, src, pad_id):
    return (src != pad_id).unsqueeze(1).unsqueeze(2)
```

`(src != pad_id)` is a boolean tensor of shape `[Batch, Seq_Len]` — `True`
for real tokens, `False` for padding. The two `unsqueeze` calls insert axes
to make it `[Batch, 1, 1, Seq_Len]`, which **broadcasts** cleanly against the
`[Batch, Heads, Q_Len, K_Len]` score tensor: the same key-padding mask
applies across every head and every query. Broadcasting is why we can get
away with such a small mask.

The **target mask** has to do all of that *and* prevent each position from
peeking at future tokens — otherwise the model cheats during training by
looking at the answer it's supposed to predict. So it's a padding mask AND a
lower-triangular causal mask, combined with a logical `and`:

```python
def create_tgt_mask(self, target, pad_id):
    batch_size, seq_len = target.size()
    target_pad_mask = (target != pad_id).unsqueeze(1).unsqueeze(2)   # [B,1,1,L]
    target_sub_mask = torch.tril(torch.ones((seq_len, seq_len),
                                 device=target.device)).bool()       # [L,L]
    return target_pad_mask & target_sub_mask                         # [B,1,L,L]
```

`torch.tril` builds a lower-triangular matrix of ones — row *i* has ones in
columns 0…*i* and zeros after. Read as a mask, that says position *i* may
attend to positions ≤ *i* and nothing later. That single triangular matrix is
the entire trick behind autoregressive generation. Combined with the pad mask
by `&`, it broadcasts to `[Batch, 1, Seq_Len, Seq_Len]` and then across all
heads. Get this even slightly wrong — an off-by-one, a transposed triangle —
and your training loss looks *amazing* while your model has learned nothing
but how to read ahead, then produces garbage the moment it has to generate
for real. It's the highest ratio of "tiny code, huge consequence" in the
whole project.

## Assembling the full Transformer

The top-level module wires everything together: two embedding tables (source
and target), the shared positional encoding, a stack of encoder layers, a
stack of decoder layers, and a final linear layer that projects back to
vocabulary size.

```python
def forward(self, src, tgt, src_pad_id=0, tgt_pad_id=0):
    src_mask = self.create_src_mask(src, src_pad_id)
    tgt_mask = self.create_tgt_mask(tgt, tgt_pad_id)

    enc = self.dropout(self.pos_encode(self.src_embeds(src) * math.sqrt(self.d_model)))
    for layer in self.encoder_layers:
        enc = layer(enc, src_mask)

    dec = self.dropout(self.pos_encode(self.tgt_embeds(tgt) * math.sqrt(self.d_model)))
    for layer in self.decoder_layers:
        dec = layer(dec, enc, src_mask, tgt_mask)

    logits = self.fc_out(dec)
    return F.softmax(logits, -1)
```

Reading it top to bottom: build both masks; embed the source, scale, add
positions, and run it through the encoder stack to get `enc`; embed the
target the same way and run it through the decoder stack, feeding in `enc` and
both masks; finally project each decoder output to a distribution over the
target vocabulary.

One line deserves a spotlight: `self.src_embeds(src) * math.sqrt(d_model)`.
Why scale the embeddings up? The embedding vectors start out small (roughly
unit-ish per component), while the positional encodings are sines and cosines
with amplitude around 1. Multiplying the embeddings by $\sqrt{d_{model}}$
puts them on a comparable scale to the positional signal we're about to add,
so neither drowns out the other. The paper mentions this almost in passing;
it's easy to drop and mildly harmful to omit.

(I'll flag one thing to revisit later: this `forward` returns `softmax`
probabilities rather than raw logits. That choice ripples into the loss and
the decoder, and it's the main thing I'd change — see the end.)

## The data pipeline: tokenization and batching

Models don't consume text, they consume integer IDs. I trained a **byte-pair
encoding (BPE)** tokenizer with HuggingFace `tokenizers`, one for each
language:

```python
def build_tokenizer(dataset, lang, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
                         vocab_size=vocab_size)
    tokenizer.train_from_iterator((item["translation"][lang] for item in dataset),
                                  trainer=trainer)
    return tokenizer
```

BPE starts from individual characters and repeatedly merges the most frequent
adjacent pair into a new token, until it reaches the target vocabulary size.
The payoff is that it never hits a truly unknown word: common words become
single tokens, while a rare word gracefully falls back to a handful of
subword pieces. Four IDs are reserved for special tokens — `[PAD]` for
padding, `[UNK]` for anything unrepresentable, and `[BOS]`/`[EOS]` to mark
the beginning and end of a sequence.

Batching then turns raw text pairs into padded integer tensors, wrapping each
target in the begin/end markers:

```python
def process_batch(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_ids = src_tokenizer.encode(item["translation"]["en"]).ids
        tgt_ids = [BOS_IDX] + tgt_tokenizer.encode(item["translation"]["de"]).ids + [EOS_IDX]
        src_batch.append(torch.tensor(src_ids))
        tgt_batch.append(torch.tensor(tgt_ids))
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded
```

`pad_sequence` stacks variable-length sequences into a rectangular tensor by
padding the short ones — and it pads with `PAD_IDX`, precisely the token our
masks are built to ignore. The whole pipeline is internally consistent: pad
with `[PAD]`, then mask out `[PAD]`. I trained on a 30,000-sentence subset of
WMT'14 English–German with a batch size of 16 and a 16,000-token vocabulary —
small enough to iterate on a single GPU.

## Teacher forcing and the training loop

The decoder is autoregressive: token *t* depends on tokens 1…*t*−1. At
inference we truly generate left to right, but training that way would be
painfully slow. Instead we use **teacher forcing**: feed the decoder the
entire ground-truth target sequence at once, and rely on the causal mask to
stop each position from seeing its own answer. Every position is trained in
parallel, yet none can cheat.

The mechanism is a one-token shift:

```python
tgt_input = tgt[:, :-1]   # everything but the last token  → decoder input
tgt_label = tgt[:, 1:]    # everything but the first token → what to predict
```

If the target is `[BOS] Der Hund [EOS]`, then `tgt_input` is
`[BOS] Der Hund` and `tgt_label` is `Der Hund [EOS]`. Position by position,
the model sees `[BOS]` and must predict `Der`, sees `[BOS] Der` and must
predict `Hund`, and so on. The causal mask guarantees that when predicting
`Hund` it can't peek at the real `Hund` sitting one step ahead.

The loop itself:

```python
for i, (src, tgt) in enumerate(dataloader):
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
    tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

    with torch.autocast('cuda'):
        probs = model(src, tgt_input, PAD_IDX, PAD_IDX)
        log_probs = torch.log(torch.clamp(probs, 1e-9))
        loss = criterion(log_probs.reshape(-1, log_probs.size(-1)),
                         tgt_label.reshape(-1))
        loss = loss / GRAD_ACC_STEPS

    scaler.scale(loss).backward()
    if (i + 1) % GRAD_ACC_STEPS == 0 or (i + 1) == len(dataloader):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

There's a lot packed in here — the next three sections pull it apart.

## The loss, and a log-softmax subtlety

The criterion is `nn.NLLLoss(ignore_index=PAD_IDX)`. Negative log-likelihood
loss expects **log-probabilities** as input: for each position it looks up
the log-probability the model assigned to the correct token and averages the
negatives. The `ignore_index=PAD_IDX` is important — it drops padding
positions from the loss entirely, so the model isn't graded on predicting
filler.

Because my `forward` returns plain `softmax` probabilities, I have to take
their log before the loss:

```python
log_probs = torch.log(torch.clamp(probs, 1e-9))
```

The `clamp` floors the probabilities at `1e-9` so that a value that rounded
to exactly 0 doesn't become `log(0) = -inf` and blow up the loss. Then
`reshape(-1, vocab_size)` flattens `[Batch, Seq_Len, Vocab]` down to
`[Batch·Seq_Len, Vocab]`, and the labels flatten to a matching 1-D vector, so
the loss is computed over every (position, correct-token) pair at once.

This works, but it's the numerically clumsy path — doing `softmax` then `log`
separately loses precision. The clean version is to return raw logits and use
`nn.CrossEntropyLoss`, which fuses log-softmax and NLL into one numerically
stable operation and needs no `clamp` at all. (I list this in "what I'd
change.")

## Fitting it on one GPU

The paper trains on 8 GPUs; I had one. Three techniques make a
paper-scale-ish model fit and train on modest hardware.

**Mixed precision** (`torch.autocast` + `GradScaler`). Autocast runs the
heavy matmuls in 16-bit floating point, which is roughly twice as fast and
uses about half the memory, while keeping precision-sensitive ops in 32-bit.
The catch: 16-bit gradients can *underflow* to zero. The `GradScaler` fixes
this by multiplying the loss by a large factor before `backward()` (scaling
the gradients up into representable range), then `unscale_`-ing them before
the optimizer step. It also silently skips any step where the gradients
overflowed to inf/NaN, adjusting the scale factor automatically.

**Gradient accumulation.** Only 16 sequences fit in memory at once, but tiny
batches make for noisy gradient estimates. So instead of stepping every
batch, I accumulate gradients over `GRAD_ACC_STEPS = 16` batches and step
once — an *effective* batch size of 16 × 16 = 256. Two details make this
correct: the loss is divided by `GRAD_ACC_STEPS` (so accumulated gradients
average rather than sum), and `optimizer.zero_grad()` is called only *after*
a step, so gradients from consecutive batches pile up in between.

**Gradient clipping.** `clip_grad_norm_(..., max_norm=1.0)` rescales the
gradient vector whenever its norm exceeds 1.0. Early Transformer training is
prone to sudden loss spikes; clipping caps how far any single step can move
the weights, which keeps those spikes from derailing the run. Note the
ordering — it comes *after* `scaler.unscale_`, because you must clip the real
gradients, not the artificially scaled-up ones.

The optimizer is Adam with `lr = 1e-4`, `betas = (0.9, 0.98)`, and
`eps = 1e-9` — the momentum settings are the paper's, chosen to play well
with the way Transformer gradients behave.

## Decoding: beam search with a length penalty

Training teaches the model to predict the next token. Actually *translating*
means searching for a high-probability whole sequence — and that's a
different problem. **Greedy decoding**, always taking the single most likely
next token, is fast but shortsighted: a locally attractive token can lead
into a globally mediocre sentence, with no way to back out.

**Beam search** keeps the `k` best partial sequences ("beams") alive at every
step instead of just one. Each step, it expands every unfinished beam by its
top-`k` next tokens, scores all the resulting candidates, and keeps only the
best `k` overall. It's a middle ground between greedy (`k = 1`) and an
infeasible exhaustive search over all sequences.

```python
for _ in range(max_len):
    candidates = []
    for log_prob, tgt, finished in beams:
        if finished:
            candidates.append((log_prob, tgt, True))   # carry completed beams unchanged
            continue
        output = model(src, tgt, pad_idx, pad_idx)
        logits = output[:, -1, :]                       # distribution over next token
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
        topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size)
        for next_prob, next_id in zip(topk_log_probs, topk_ids):
            new_tgt = torch.cat([tgt, next_id.view(1, 1)], dim=1)
            candidates.append((log_prob + next_prob.item(),
                               new_tgt, next_id.item() == end_idx))
    candidates.sort(key=get_penalized_score, reverse=True)
    beams = candidates[:beam_size]
```

Sequence scores are sums of log-probabilities, and every extra token adds a
*negative* number — so longer sequences score worse purely for being longer.
Unchecked, beam search develops a bias toward short, truncated translations
that stop early. The standard fix is the Google NMT **length penalty**:

```python
def get_penalized_score(candidate):
    c_log_prob, c_tgt, _ = candidate
    length = c_tgt.size(1) - 1               # exclude the <sos> token
    penalty = ((5 + length) / 6) ** alpha
    return c_log_prob / penalty
```

Dividing the log-probability by a growing function of length rewards longer
sequences just enough to cancel the built-in bias, so a fluent 20-word
sentence can beat a clipped 5-word one on merit. The `alpha` knob controls
the strength: `alpha = 0` disables it, and I used `alpha = 0.6` (the paper's
value), which stops the model from giving up early. Two more details in the
loop: beams that have emitted `[EOS]` are marked `finished` and carried
forward without expansion, and the whole search stops once every beam has
finished or we hit `max_len`.

## Evaluating with BLEU

Loss going down is reassuring but not the same as *good translations*. The
standard metric for machine translation is **BLEU**, which measures n-gram
overlap between the model's output and a human reference (with a penalty for
being too short). I generate translations for the test set with the same beam
search, then score them with `sacrebleu`:

```python
bleu = sacrebleu.corpus_bleu(hypotheses, [references])
```

I used `sacrebleu` specifically because it handles tokenization and
normalization in a standardized way, which is what makes BLEU scores
comparable across papers — a raw, hand-rolled BLEU is almost impossible to
compare against anyone else's number.

## The hyperparameters, in one place

| Setting | Value | Note |
| --- | --- | --- |
| `d_model` | 256 | token/embedding dimension |
| `n_heads` | 8 | so `d_k = 32` per head |
| `n_layers` | 6 | encoder and decoder each |
| `d_ff` | 1024 | feed-forward hidden size (4× `d_model`) |
| `dropout` | 0.1 | applied after each sublayer |
| vocab size | 16,000 | per language, BPE |
| batch size | 16 | × 16 accumulation = 256 effective |
| optimizer | Adam | `lr 1e-4`, `betas (0.9, 0.98)`, `eps 1e-9` |
| grad clip | 1.0 | max gradient norm |
| epochs | 5 | on a 30k-sentence subset |
| beam size / `alpha` | 4 / 0.6 | decoding |

## What I'd change

A few things I did the "learning" way rather than the production way:

- **Return logits, not probabilities.** The single change I'd make first.
  My `forward` ends in `softmax`, which forces the training loop to take a
  `log` (with a `clamp` guard against `log(0)`) and, more awkwardly, makes
  the beam search apply `log_softmax` *on top of* already-softmaxed values —
  a redundant, numerically lossy double-normalization. Returning raw logits
  and switching to `CrossEntropyLoss` fixes all of it at once and is strictly
  more stable.
- **Add learning-rate warmup.** The paper's Noam schedule (warm up, then
  decay) matters more than it looks; a flat `1e-4` leaves quality on the
  table.
- **Label smoothing.** The paper uses 0.1; it consistently improves BLEU by
  discouraging the model from becoming overconfident.
- **Pre-norm instead of post-norm.** A small rearrangement that makes deep
  stacks train more smoothly with less warmup sensitivity.
- **Learned positional embeddings** would be a one-line swap worth comparing
  against the fixed sinusoids.

## The point of all this

None of this is state of the art — that wasn't the goal. The goal was to turn
"I've read about attention" into "I've debugged an off-by-one in a causal
mask at midnight," and those are very different kinds of understanding.
Reading the paper, the Transformer looks like a tidy stack of boxes. Building
it, you learn that the boxes are the easy part and the *connective tissue* —
the masks, the reshapes, the scaling factors, the shift between input and
label — is where the real understanding lives. If you've only ever *used*
Transformers, I can't recommend reimplementing one highly enough. The masks
alone will teach you more than any diagram.
