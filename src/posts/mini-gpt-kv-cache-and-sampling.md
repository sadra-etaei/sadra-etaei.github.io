---
title: "GPT from scratch"
date: 2026-08-12
summary: A decoder-only GPT built from first principles in PyTorch and trained on TinyStories — but this time the interesting part isn't the architecture, it's making it generate fast (KV caching, O(n) instead of O(n²)) and generate well (temperature, top-k, top-p, beam search).
tags: [NLP, GPT, PyTorch]
---

I've now implemented the Transformer architecture in three different ways. First the original
[encoder–decoder](/#/posts/building-a-transformer-from-scratch) for
translation. Then [BERT](/#/posts/bert-from-scratch), the encoder-only,
bidirectional one. This project is the third : a **decoder-only GPT** — the causal, left-to-right architecture that
every modern LLM descends from — built from first principles in
PyTorch and trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

But I've written about attention and residual streams at length already, so
this post spends its depth somewhere new. The architecture is not that *different*. 
The new thing is **generation**:
how to make an autoregressive model produce text *fast* (KV caching, which
turns quadratic generation into linear) and produce it *well* (a full sampling
toolkit — temperature, top-k, top-p, beam search). Those are the parts that make a deployed LLM
usable. Code is on [GitHub](https://github.com/sadra-etaei/mini-gpt); it
started as an [ARENA](https://arena.education/) exercise.

## The architecture, briefly

A decoder-only GPT is a stack of pre-norm transformer blocks over learned
token and position embeddings, ending in a LayerNorm and an unembedding to
vocabulary logits. If you want the attention internals spelled out reshape by
reshape, that's [the first post](/#/posts/building-a-transformer-from-scratch);
here's the whole block in five lines:

```python
def forward(self, resid_pre):
    resid_mid  = self.attn(self.ln1(resid_pre)) + resid_pre   # attention sublayer
    resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid    # MLP sublayer
    return resid_post
```

Two deliberate choices distinguish it from the models in the earlier posts.
It's **pre-norm** — LayerNorm is applied *inside* each residual branch,
`x + sublayer(norm(x))`, rather than BERT's post-norm `norm(x + sublayer(x))`.
Pre-norm keeps a clean, un-normalized residual "highway" running the full
depth of the network, which is what lets GPT-scale models train stably; it's
the modern default. And attention keeps each head's weights as their own axis
— `W_Q` has shape `(n_heads, d_model, d_head)` :

```python
q = einops.einsum(x, self.W_Q,
    "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head") + self.b_Q
attn_scores = einops.einsum(q, k,
    "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K")
```
Two things I got wrong myself :
The causal  mask must be built from a *fresh ones matrix*, not by
thresholding the score tensor. `triu` on the scores keeps their values, so any
future-position score that happened to land at exactly `0.0` would read as
"not masked" and leak through. And masked positions are filled with `-1e5`,
not `-inf`, so a fully-masked row can never become `0/0 = NaN` in the softmax:

```python
mask = t.triu(t.ones(q_posn, k_posn, device=...), diagonal=1).bool()
return attn_scores.masked_fill(mask, self.IGNORE)   # IGNORE = -1e5
```


## Why generation is secretly quadratic

Here's the problem that KV caching exists to solve. To generate token *n+1*, a
naive autoregressive loop feeds the *entire* sequence so far back through the
model and keeps only the last position's prediction. But almost all of that
work is redundant: the keys and values for tokens 1…*n* are identical to what
they were on the previous step — the model recomputes them from scratch every
single time.

Generating a sequence of length *n* that way costs *O(n²)* work (each step
reprocesses a growing prefix), and the attention score matrix is rebuilt in
full at every step. For a chatbot streaming hundreds of tokens, that's unacceptable.

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 320" role="img" aria-label="Generating without a cache recomputes the whole attention triangle each step; with a KV cache only the new query row is computed" style="width:100%;height:auto;max-width:660px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <!-- WITHOUT CACHE -->
  <text x="160" y="28" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">without a cache</text>
  <text x="160" y="46" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">recompute the whole prefix, every step</text>
  <g transform="translate(100,64)">
    <g style="fill:var(--accent-soft);stroke:var(--accent)">
      <rect x="0" y="0" width="24" height="24"/>
      <rect x="0" y="24" width="24" height="24"/><rect x="24" y="24" width="24" height="24"/>
      <rect x="0" y="48" width="24" height="24"/><rect x="24" y="48" width="24" height="24"/><rect x="48" y="48" width="24" height="24"/>
      <rect x="0" y="72" width="24" height="24"/><rect x="24" y="72" width="24" height="24"/><rect x="48" y="72" width="24" height="24"/><rect x="72" y="72" width="24" height="24"/>
      <rect x="0" y="96" width="24" height="24"/><rect x="24" y="96" width="24" height="24"/><rect x="48" y="96" width="24" height="24"/><rect x="72" y="96" width="24" height="24"/><rect x="96" y="96" width="24" height="24"/>
    </g>
  </g>
  <text x="160" y="220" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">O(n²) per token</text>
  <text x="160" y="238" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">all cells recomputed</text>
  <!-- WITH CACHE -->
  <text x="520" y="28" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">with a KV cache</text>
  <text x="520" y="46" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">reuse stored keys &amp; values</text>
  <g transform="translate(460,64)">
    <!-- cached rows (reused) -->
    <g style="fill:var(--code-bg);stroke:var(--border)">
      <rect x="0" y="0" width="24" height="24"/>
      <rect x="0" y="24" width="24" height="24"/><rect x="24" y="24" width="24" height="24"/>
      <rect x="0" y="48" width="24" height="24"/><rect x="24" y="48" width="24" height="24"/><rect x="48" y="48" width="24" height="24"/>
      <rect x="0" y="72" width="24" height="24"/><rect x="24" y="72" width="24" height="24"/><rect x="48" y="72" width="24" height="24"/><rect x="72" y="72" width="24" height="24"/>
    </g>
    <!-- new query row (computed) -->
    <g style="fill:var(--accent-soft);stroke:var(--accent)">
      <rect x="0" y="96" width="24" height="24"/><rect x="24" y="96" width="24" height="24"/><rect x="48" y="96" width="24" height="24"/><rect x="72" y="96" width="24" height="24"/><rect x="96" y="96" width="24" height="24"/>
    </g>
  </g>
  <text x="600" y="176" style="font-size:10px;fill:var(--text-faint)">← new query</text>
  <text x="520" y="220" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">O(n) per token</text>
  <text x="520" y="238" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">one new row · rest reused</text>
  <text x="350" y="290" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">attention score matrix at generation step n · rows = queries, columns = keys</text>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">The whole idea of the KV cache: the shaded cells are what actually gets computed at generation step n. Without a cache the model rebuilds the entire triangle; with one, it computes a single new query row against stored keys and values.</figcaption>
</figure>

The fix is to *store* the keys and values once computed, and only ever compute
them for the genuinely new token. That's the **KV cache**.

## The KV cache

The store itself is deliberately boring — a preallocated block of memory per
layer, not a list that grows:

```python
# layout per tensor: (n_layers, batch, max_len, n_heads, d_head)
self.k = t.zeros(shape, device=device, dtype=dtype)
self.v = t.zeros(shape, device=device, dtype=dtype)
self.length = 0                      # how many positions are currently valid
```

Preallocation matters more than it looks. The easy implementation grows the
cache with `torch.cat` each step — but concatenation reallocates and copies the
*entire* cache on every token, which quietly reintroduces the exact *O(n²)*
memory traffic the cache was supposed to remove. Allocating `max_len` up front
and writing into a slice keeps each step genuinely constant-cost. The layout is
chosen so the `max_len` axis sits exactly where `posn_K` sits in the attention
einsum, so a stored slice drops straight in with no permute.

Writing to it is a slice assignment; the model advances the length once after
*all* layers have written at the same offset:

```python
def append(self, layer, k_new, v_new):
    end = self.length + k_new.shape[1]
    self.k[layer, :, self.length:end] = k_new
    self.v[layer, :, self.length:end] = v_new
    return self.k[layer, :, :end], self.v[layer, :, :end]   # the full valid span
```

Then attention does the one asymmetric thing that makes the whole scheme work:
the query stays short (just the new token), while keys and values are extended
to cover the entire prefix from the cache.

```python
past_len = cache.length
k, v = cache.append(layer, k, v)     # k, v now span the whole prefix; q does not
```

At decode time that makes the score matrix `(posn_Q = 1, posn_K = n)` — a
single row instead of a full triangle. That's the linear-cost generation in one
shape change.

### The two bugs I encountered

KV caching is famous for two subtle bugs, and I left comments on both because I
walked into them:

**Positional embeddings must index from `past_len`, not 0.** With a cache, the
tokens you pass in are just the *new* ones — so their absolute positions start
at `past_len`, not at zero. Get this wrong and every generated token gets
position 0; the model degenerates into nonsense without ever raising an error.

```python
def forward(self, tokens, past_len=0):
    end = past_len + tokens.shape[1]
    return self.W_pos[past_len:end].unsqueeze(0).expand(batch, -1, -1)
```

**The causal mask diagonal shifts by `past_len`.** The score matrix is no
longer square, so a query at absolute position `past_len + i` may legitimately
see keys `0 … past_len + i`. The mask's diagonal moves accordingly. And at
single-token decode (`q_posn == 1`) the new token is allowed to see *every*
earlier token, so the mask hides nothing — which means you can skip building it
entirely:

```python
if q_posn == 1 and k_posn == past_len + 1:
    return attn_scores            # decode masks nothing — don't waste 3 kernel launches
mask = t.triu(t.ones(q_posn, k_posn), diagonal=1 + past_len).bool()
```

That early return is a small but real optimization: decoding one token is
*kernel-launch-bound*, not compute-bound, so skipping the `ones`/`triu`/`masked_fill`
trio per layer per token actually shows up in the wall clock.

### Prefill and decode

Put together, generation has two phases. **Prefill**: one forward pass over the
whole prompt to populate the cache. **Decode**: one forward pass per new token,
each over a single position. The reference `model.py` and the cached
`model_kv.py` share parameter names and shapes, so — with `cache=None` the cached model
reproduces the plain one *bit for bit*. The cache is a pure speed optimization
that changes no outputs, which makes it trivial to A/B:

```python
logits = self(tokens, cache)                 # prefill (or a full pass, if uncached)
for _ in range(max_new_tokens):
    next_token = logits[:, -1].argmax(-1, keepdim=True)
    tokens = t.cat([tokens, next_token], dim=-1)
    logits = self(next_token, cache) if use_cache else self(tokens)   # the only difference
```

## Sampling: turning logits into text

Fast generation is half the story; the other half is *what* to generate. Greedy
decoding — always take the argmax — is repetitive and lifeless. The sampler
implements the standard toolkit, each knob shaping the probability distribution
differently before a token is drawn.

- **Temperature** divides the logits before softmax. Below 1 sharpens the
  distribution (safer, more repetitive); above 1 flattens it (wilder, more
  errors); 0 collapses to greedy.
- **Frequency penalty** subtracts a multiple of each token's running count from
  its logit, directly discouraging repetition.
- **Top-k** keeps only the *k* highest-probability tokens and renormalizes.
- **Top-p (nucleus)** keeps the smallest set of tokens whose cumulative
  probability reaches *p* — an *adaptive* cutoff that keeps many tokens when the
  model is unsure and few when it's confident.

The difference between the two truncation strategies is worth a picture:

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 300" role="img" aria-label="Top-k keeps a fixed number of tokens; top-p keeps a cumulative-probability mass" style="width:100%;height:auto;max-width:640px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <!-- baseline -->
  <line x1="56" y1="230" x2="660" y2="230" style="stroke:var(--border)"/>
  <!-- bars: descending probabilities -->
  <g>
    <rect x="60"  y="110" width="44" height="120" style="fill:var(--accent);stroke:var(--accent)"/>
    <rect x="112" y="135" width="44" height="95"  style="fill:var(--accent);stroke:var(--accent)"/>
    <rect x="164" y="158" width="44" height="72"  style="fill:var(--accent);stroke:var(--accent)"/>
    <rect x="216" y="176" width="44" height="54"  style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <rect x="268" y="190" width="44" height="40"  style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <rect x="320" y="200" width="44" height="30"  style="fill:var(--code-bg);stroke:var(--border)"/>
    <rect x="372" y="208" width="44" height="22"  style="fill:var(--code-bg);stroke:var(--border)"/>
    <rect x="424" y="214" width="44" height="16"  style="fill:var(--code-bg);stroke:var(--border)"/>
    <rect x="476" y="219" width="44" height="11"  style="fill:var(--code-bg);stroke:var(--border)"/>
    <rect x="528" y="223" width="44" height="7"   style="fill:var(--code-bg);stroke:var(--border)"/>
  </g>
  <text x="360" y="252" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">vocabulary, sorted by probability →</text>
  <!-- top-k cut -->
  <line x1="210" y1="70" x2="210" y2="234" style="stroke:var(--text-muted);stroke-dasharray:5 3"/>
  <text x="210" y="60" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">top-k (k = 3)</text>
  <text x="210" y="46" text-anchor="middle" style="font-size:10px;fill:var(--text-faint)">fixed count</text>
  <!-- top-p cut -->
  <line x1="314" y1="90" x2="314" y2="234" style="stroke:var(--accent);stroke-dasharray:5 3"/>
  <text x="314" y="82" text-anchor="start" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">top-p (p = 0.9)</text>
  <text x="314" y="98" text-anchor="start" style="font-size:10px;fill:var(--text-faint)">cumulative mass</text>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Top-k always keeps the same number of tokens; top-p keeps however many are needed to reach p of the probability mass — wider when the model is uncertain, narrower when it's confident.</figcaption>
</figure>

Top-p  — sort, take the cumulative sum, and find
the cutoff with a binary search:

```python
logits_sorted, indices = logits.sort(descending=True, stable=True)
cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
n_keep = torch.searchsorted(cumul_probs, top_p, side="left").item() + 1
keep_idx = indices[:max(n_keep, min_tokens_to_keep)]
```

For a more deliberate search there's also **beam search**, which keeps the *k*
highest-probability running sequences instead of committing to one token at a
time, with a `no_repeat_ngram_size` option that forbids any n-gram the beam has
already emitted — a cheap, effective cure for the loops beam search is prone to.

## Training on TinyStories

I trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
— a synthetic corpus of simple children's stories written with a small
vocabulary. It's a lovely tiny testbed: a 124M-parameter model can actually
learn to produce *coherent* text on it, so you get real "once upon a time…"
completions rather than the word-salad a model this size produces on web-scale
data. Training is standard next-token prediction, teacher-forced, with the loss
read off the shifted logits:

```python
log_probs = logits[:, :-1].log_softmax(dim=-1)                       # predict token t+1
log_probs_for_tokens = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
```

The training loop:

- **bf16 autocast, and notably *no* `GradScaler`.** The translation project
  needed a gradient scaler because fp16 gradients underflow to zero; bfloat16
  has the *same exponent range as fp32*, so gradients stay representable and the
  scaler is simply unnecessary. The one thing autocast still promotes back to
  fp32 is the `log_softmax`, keeping the loss numerically stable while the
  matmuls run in bf16.
- **Not naming the logits.** `loss = -get_log_probs(self.model(tokens), tokens).mean()`
  deliberately never binds the logits to a variable — doing so would pin a
  `(batch, seq, d_vocab)` tensor alive for the whole backward pass. Autograd
  doesn't need it, so letting it go free reclaims ~0.38 GiB at batch 8.
- **Slicing before the softmax.** `log_softmax` allocates a tensor the size of
  its input — the single biggest allocation in the step — so slicing off the
  last position *first* shrinks it.

One hardware note : above ~batch 10 on this
GPU, the NVIDIA Windows driver *silently pages* the VRAM overflow into system
RAM instead of raising `OutOfMemoryError`. Training still "works" — about 14×
slower. If your steps suddenly take seconds instead of milliseconds, that's why.

## The hyperparameters, in one place

| Setting | Value | Note |
| --- | --- | --- |
| parameters | 124M | GPT-2 small shape |
| `d_model` / `n_layers` / `n_heads` | 768 / 12 / 12 | `d_head` = 64 |
| `d_mlp` | 3072 | 4× `d_model`, GELU |
| context length | 1024 | learned position embeddings |
| norm placement | pre-norm | `x + sublayer(norm(x))` |
| vocabulary | 50,257 | GPT-2 BPE tokenizer |
| dataset | TinyStories | `seq_len` 256 |
| optimizer | AdamW | `lr 1e-3`, wd 1e-2 |
| precision | bf16 autocast | no GradScaler needed |
| generation | KV cache | prefill + single-token decode |

## What I'd change

- **Weight tying.** The token embedding `W_E` and the unembedding `W_U` are
  separate matrices here; GPT-2 ties them (the same reason it helped in
  [BERT](/#/posts/bert-from-scratch)). Tying would save ~39M parameters — a
  third of the model.
- **RoPE instead of learned position embeddings.** Rotary embeddings generalize
  past the training context length and are the modern standard; learned
  embeddings hard-cap the model at `n_ctx`.
- **FlashAttention.** The attention here materializes the full score matrix;
  a fused kernel would cut memory and time, especially at longer context.
- **Grouped-query attention.** The KV cache is the memory bottleneck at long
  context; sharing K/V across query heads (GQA/MQA) shrinks it directly, and
  would be a natural next extension of the cache I already built.

## Personal Notes

This was the first time I used einops instead of torch.einsum or torch reshape 
and it was genuinely easier.
I wasn't able to train the 124M param model , I trained it with a smaller Config at around 25m params which was 
enough for the TinyStories dataset 
