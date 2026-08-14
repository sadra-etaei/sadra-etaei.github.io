---
title: "BERT from scratch"
date: 2026-07-22
summary: Reimplementing BERT in PyTorch — the bidirectional encoder, the three summed embeddings, masked language modeling with the 80/10/10 trick, next-sentence prediction, weight tying, and pretraining it to fill in the blanks.
tags: [NLP, BERT, PyTorch]
---

When I [built the Transformer](/#/posts/building-a-transformer-from-scratch),
I only built its *decoder* side — a left-to-right model that generates one token at
a time, forbidden by a causal mask from looking ahead. BERT is different: 
take the Transformer's **encoder**, throw away the causal mask
entirely, and train it to understand text rather than generate it. This post
is my explanation of my from-scratch PyTorch reimplementation of
[BERT](https://arxiv.org/abs/1810.04805) — the architecture, the two
pretraining objectives, and the small details that make masked language
modeling actually work. Code is on
[GitHub](https://github.com/sadra-etaei/BERT).

## The one idea: bidirectionality

A generative language model reads left to right because it *has* to — it's
predicting the future, and seeing the future would be cheating. But most tasks
we care about (classification, question answering, named-entity recognition)
aren't generation; they're *understanding*, and understanding a word benefits
from seeing everything around it, on both sides.

So BERT drops the causal mask. Every token attends to every other token, in
both directions. The word "bank" can look at both "river" three words to its
left and "flooded" two words to its right before deciding what it means.

<svg viewBox="0 0 640 260" role="img" aria-label="Bidirectional attention versus causal attention" style="width:100%;height:auto;max-width:560px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <!-- BERT full attention -->
  <g>
    <text x="130" y="30" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">BERT (encoder)</text>
    <text x="130" y="48" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">every token sees every token</text>
    <g transform="translate(70,64)">
      <!-- 5x5 grid all filled -->
      <g>
        <rect x="0" y="0" width="120" height="120" style="fill:none;stroke:var(--border)"/>
        <g style="fill:var(--accent-soft);stroke:var(--accent)">
          <rect x="0" y="0" width="24" height="24"/><rect x="24" y="0" width="24" height="24"/><rect x="48" y="0" width="24" height="24"/><rect x="72" y="0" width="24" height="24"/><rect x="96" y="0" width="24" height="24"/>
          <rect x="0" y="24" width="24" height="24"/><rect x="24" y="24" width="24" height="24"/><rect x="48" y="24" width="24" height="24"/><rect x="72" y="24" width="24" height="24"/><rect x="96" y="24" width="24" height="24"/>
          <rect x="0" y="48" width="24" height="24"/><rect x="24" y="48" width="24" height="24"/><rect x="48" y="48" width="24" height="24"/><rect x="72" y="48" width="24" height="24"/><rect x="96" y="48" width="24" height="24"/>
          <rect x="0" y="72" width="24" height="24"/><rect x="24" y="72" width="24" height="24"/><rect x="48" y="72" width="24" height="24"/><rect x="72" y="72" width="24" height="24"/><rect x="96" y="72" width="24" height="24"/>
          <rect x="0" y="96" width="24" height="24"/><rect x="24" y="96" width="24" height="24"/><rect x="48" y="96" width="24" height="24"/><rect x="72" y="96" width="24" height="24"/><rect x="96" y="96" width="24" height="24"/>
        </g>
      </g>
    </g>
  </g>
  <!-- Decoder causal -->
  <g>
    <text x="470" y="30" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">Decoder (GPT-style)</text>
    <text x="470" y="48" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">only the past is visible</text>
    <g transform="translate(410,64)">
      <rect x="0" y="0" width="120" height="120" style="fill:none;stroke:var(--border)"/>
      <g style="fill:var(--accent-soft);stroke:var(--accent)">
        <rect x="0" y="0" width="24" height="24"/>
        <rect x="0" y="24" width="24" height="24"/><rect x="24" y="24" width="24" height="24"/>
        <rect x="0" y="48" width="24" height="24"/><rect x="24" y="48" width="24" height="24"/><rect x="48" y="48" width="24" height="24"/>
        <rect x="0" y="72" width="24" height="24"/><rect x="24" y="72" width="24" height="24"/><rect x="48" y="72" width="24" height="24"/><rect x="72" y="72" width="24" height="24"/>
        <rect x="0" y="96" width="24" height="24"/><rect x="24" y="96" width="24" height="24"/><rect x="48" y="96" width="24" height="24"/><rect x="72" y="96" width="24" height="24"/><rect x="96" y="96" width="24" height="24"/>
      </g>
    </g>
  </g>
  <text x="320" y="230" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">rows = queries · columns = keys · shaded = attended</text>
</svg>

That single change — a full attention matrix instead of a triangular one — is
the whole architectural difference. But it forces a new question: if a token
can see the entire sentence, how do you train it to *predict* anything without
the answer leaking in? That's what the pretraining objectives are for. First,
the architecture.

## The input: three embeddings, summed

Every input token is represented as the sum of **three** learned embeddings,
which is how BERT packs word identity, sentence structure, and position into
one vector:

<svg viewBox="0 0 660 300" role="img" aria-label="BERT input as the sum of token, segment, and position embeddings" style="width:100%;height:auto;max-width:640px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="da" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <g style="font-family:'JetBrains Mono',monospace;font-size:11px;text-anchor:middle">
    <!-- tokens row -->
    <text x="40" y="40" text-anchor="end" style="fill:var(--text-muted)">tokens</text>
    <g style="fill:var(--accent-soft);stroke:var(--accent)">
      <rect x="70" y="24" width="80" height="26" rx="4"/><rect x="160" y="24" width="80" height="26" rx="4"/><rect x="250" y="24" width="80" height="26" rx="4"/><rect x="340" y="24" width="80" height="26" rx="4"/><rect x="430" y="24" width="80" height="26" rx="4"/><rect x="520" y="24" width="80" height="26" rx="4"/>
    </g>
    <g style="fill:var(--accent)">
      <text x="110" y="41">[CLS]</text><text x="200" y="41">my</text><text x="290" y="41">dog</text><text x="380" y="41">[MASK]</text><text x="470" y="41">cute</text><text x="560" y="41">[SEP]</text>
    </g>
    <!-- plus -->
    <text x="35" y="80" style="fill:var(--text-faint);font-size:16px">+</text>
    <!-- segment row -->
    <text x="40" y="96" text-anchor="end" style="fill:var(--text-muted)">segment</text>
    <g style="fill:var(--code-bg);stroke:var(--border)">
      <rect x="70" y="80" width="80" height="26" rx="4"/><rect x="160" y="80" width="80" height="26" rx="4"/><rect x="250" y="80" width="80" height="26" rx="4"/><rect x="340" y="80" width="80" height="26" rx="4"/><rect x="430" y="80" width="80" height="26" rx="4"/><rect x="520" y="80" width="80" height="26" rx="4"/>
    </g>
    <g style="fill:var(--text-muted)">
      <text x="110" y="97">A</text><text x="200" y="97">A</text><text x="290" y="97">A</text><text x="380" y="97">A</text><text x="470" y="97">A</text><text x="560" y="97">A</text>
    </g>
    <text x="35" y="136" style="fill:var(--text-faint);font-size:16px">+</text>
    <!-- position row -->
    <text x="40" y="152" text-anchor="end" style="fill:var(--text-muted)">position</text>
    <g style="fill:var(--code-bg);stroke:var(--border)">
      <rect x="70" y="136" width="80" height="26" rx="4"/><rect x="160" y="136" width="80" height="26" rx="4"/><rect x="250" y="136" width="80" height="26" rx="4"/><rect x="340" y="136" width="80" height="26" rx="4"/><rect x="430" y="136" width="80" height="26" rx="4"/><rect x="520" y="136" width="80" height="26" rx="4"/>
    </g>
    <g style="fill:var(--text-muted)">
      <text x="110" y="153">0</text><text x="200" y="153">1</text><text x="290" y="153">2</text><text x="380" y="153">3</text><text x="470" y="153">4</text><text x="560" y="153">5</text>
    </g>
  </g>
  <!-- arrow down to sum -->
  <line x1="335" y1="170" x2="335" y2="200" style="stroke:var(--text-faint)" marker-end="url(#da)"/>
  <text x="360" y="190" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">sum → LayerNorm → dropout</text>
  <rect x="70" y="206" width="530" height="30" rx="5" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="335" y="226" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">input embeddings   [B, T, hidden_size]</text>
  <text x="335" y="262" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">token identity + which sentence + where in the sequence</text>
</svg>

In code it's an addition of three `nn.Embedding` lookups:

```python
embeddings = (
    self.word_embeddings(input_ids) +
    self.position_embeddings(position_ids) +
    self.sentence_embeddings(token_type_ids)
)
embeddings = self.dropout(self.LayerNorm(embeddings))
```

Two things that are different from the Transformer. First, the
**position embeddings are learned** (`nn.Embedding(max_position, hidden)`),
not the fixed sinusoids of the 2017 paper — BERT just gives each of the 512
positions its own trainable vector. Second, there's a third stream,
`sentence_embeddings` (the "segment" or `token_type` embedding), with a
vocabulary of exactly two: it tags each token as belonging to sentence A or
sentence B. That's the hook the next-sentence objective hangs on. Note also
`padding_idx=0` on the word embeddings, which pins the `[PAD]` token's vector
to zero.

## Tokenization: WordPiece

BERT tokenizes with **WordPiece**,I train it with HuggingFace `tokenizers`, using BERT's own
normalizer (lowercasing, accent stripping) and pre-tokenizer:

```python
tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tok.normalizer = normalizers.BertNormalizer(lowercase=True)
tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
trainer = trainers.WordPieceTrainer(vocab_size=vocab, special_tokens=specials)
```


## The encoder block

Each of BERT's identical layers is self-attention followed by a feed-forward
network, each wrapped in residual-and-LayerNorm ,The self-attention 
is basically the same 

```python
scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
if attention_mask is not None:
    scores += attention_mask        # additive mask: 0 for real, -inf for padding
```

The mask is **additive**, not multiplicative. Padding positions get a huge
negative number added to their scores so softmax drives them to zero. That
number is built once, up front, from the 0/1 attention mask:

```python
def _extended_mask(self, attention_mask):
    ext = attention_mask[:, None, None, :].to(dtype=torch.float32)
    return (1.0 - ext) * torch.finfo(torch.float32).min
```

The `[:, None, None, :]` reshapes `[B, T]` into `[B, 1, 1, T]` so it
broadcasts across heads and query positions — a real token contributes
`(1 - 1) * min = 0`, a pad contributes `(1 - 0) * min = -inf`. Crucially,
there is no *causal* mask here; the only thing being hidden is padding.

A couple of details differ from the Transformer and are worth
naming:

- **GELU, not ReLU.** The feed-forward network uses the smoother GELU
  activation, which is BERT's choice and tends to help slightly:
  `F.gelu(self.dense1(x))`.
- **Post-norm.** Normalization is applied *after* the residual add,
  `LayerNorm(input + sublayer(input))` — the original arrangement. You can see
  it in the attention block:

```python
def forward(self, input, attn_mask=None):
    hidden_states = self.selfattn(input, attn_mask)
    hidden_states = self.dropout(self.dense(hidden_states))
    return self.LayerNorm(input + hidden_states)     # residual, then norm
```

How does BERT learn language with no labels? 

## Objective 1: Masked Language Modeling

Here's the elegant solution to the "bidirectional models can't predict without
cheating" problem: **corrupt the input**. Randomly hide 15% of the tokens and
ask the model to reconstruct them from the surrounding context on both sides.
"the ___ sat on the mat" — in order to fill the blank the model
must build a genuine understanding of the whole sentence.

But there's a problem the paper handles with a  **80/10/10 rule**.
If masked tokens were always replaced with a literal `[MASK]` token, the model
would only ever learn to reason about `[MASK]` symbols — and `[MASK]` never
appears when you later fine-tune on real text, creating a train/test mismatch.
So of the 15% of tokens chosen for prediction:

<svg viewBox="0 0 660 220" role="img" aria-label="The 80/10/10 masking scheme" style="width:100%;height:auto;max-width:600px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="ma" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <rect x="40" y="90" width="150" height="40" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="115" y="108" text-anchor="middle" style="font-size:12px;fill:var(--accent)">15% chosen</text>
  <text x="115" y="123" text-anchor="middle" style="font-size:10px;fill:var(--text-muted)">for prediction</text>
  <g style="stroke:var(--text-faint)">
    <line x1="190" y1="105" x2="250" y2="45" marker-end="url(#ma)"/>
    <line x1="190" y1="110" x2="250" y2="110" marker-end="url(#ma)"/>
    <line x1="190" y1="115" x2="250" y2="175" marker-end="url(#ma)"/>
  </g>
  <g style="font-family:'JetBrains Mono',monospace">
    <rect x="255" y="26" width="365" height="40" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="270" y="50" style="font-size:12px;fill:var(--text)">80% → replace with [MASK]</text>
    <rect x="255" y="90" width="365" height="40" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="270" y="114" style="font-size:12px;fill:var(--text)">10% → replace with a random token</text>
    <rect x="255" y="154" width="365" height="40" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="270" y="178" style="font-size:12px;fill:var(--text)">10% → leave unchanged</text>
  </g>
  <text x="437" y="212" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">…but the model is scored on predicting all three</text>
</svg>

`mask_tokens` implements exactly this, using `-100` as the label for every
*non*-selected position (PyTorch's `CrossEntropyLoss` ignores that index, so
only the chosen 15% contribute to the loss):

```python
masked = torch.bernoulli(prob_matrix).bool()   # ~15% selected
labels[~masked] = -100                          # ignore everything else

replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
input_ids[replace_mask] = mask_token_id         # 80% -> [MASK]

random_mask = (torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
               & masked & ~replace_mask)         # 50% of the remaining 20% -> 10%
input_ids[random_mask] = random_tokens[random_mask]
# the last 10% are left as the original token
```

Forcing the model to sometimes predict a token that looks unchanged — or that
was swapped for something random — means it can never fully trust the surface
form. It has to keep a rich contextual representation of *every* position, just
in case that's the one being scored.

## Objective 2: Next Sentence Prediction

The second objective teaches BERT about relationships *between* sentences.
Feed it two segments packed as `[CLS] A [SEP] B [SEP]`, and ask a binary
question: is B the real sentence that followed A, or a random impostor?

The prediction is read off the special `[CLS]` token — position 0 — which acts
as a sequence-level summary. A "pooler" (a dense layer with tanh) turns its
final vector into the sentence-pair representation, and a small head classifies
it:

```python
pooled_output = F.tanh(self.pooler(sequence_output[:, 0]))   # the [CLS] vector
nsp_logits = self.nsp_head(pooled_output)                     # 2-way: isNext / notNext
```

Both objectives are trained *together*, and the total loss is simply their
sum:

```python
loss = mlm_loss + nsp_loss
```
This part was apparently unnecessary and only added complications, since they dropped it later when introducing RoBerta
## Building the training examples

The dataset builder does the packing. For each anchor sentence it flips a coin:
half the time it pairs it with the true following sentence (label 0), half the
time with a random one (label 1) — a balanced NSP task:

```python
if random.random() < 0.5:
    b, nsp = self.enc[i + 1], 0        # true next sentence
else:
    b, nsp = random.choice(self.enc), 1  # random impostor

ids   = [CLS] + a + [SEP] + b + [SEP]
types = [0] * (len(a) + 2) + [1] * (len(b) + 1)   # segment A vs B
```

Then it right-pads everything to `MAX_LEN` and builds a parallel attention
mask of 1s (real) and 0s (padding) — the very mask that
`_extended_mask` later turns into `-inf`. When two sentences overflow the
budget, it trims tokens off whichever is longer until they fit.

## Weight tying and initialization

Two things worth noting in the model constructor. First, all weights
are initialized from a tight normal distribution (`std=0.02`, BERT's value),
with LayerNorm gains at 1 and biases at 0. Second — and this is the thing that suprised me  —
the MLM output projection **shares its weight matrix** with the input word
embeddings:

```python
self.mlm_decoder.weight = self.embeddings.word_embeddings.weight
```

This is *weight tying*: the matrix that maps a token id to a vector on the way
in is the same one (transposed) that maps a vector back to vocabulary logits on
the way out. It apparently saves a huge number of parameters (the vocab projection is
often the single biggest matrix in the model) and reflects a real symmetry
between "what does this word mean" and "which word is this."

## The training loop

Pretraining ties it together with a few production-minded touches:

- **Dynamic masking.** The corruption is regenerated *fresh every step* on CPU,
  so the model rarely sees the exact same masked version of a sentence twice —
  free data augmentation, and strictly better than masking once up front.
- **A warmup-then-linear-decay schedule.** The learning rate ramps up over the
  first 1,000 steps and decays linearly afterward, the standard recipe for
  keeping early Transformer training stable:

```python
def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)                     # linear warmup
    return max(0.0, (TOTAL_STEPS - step) / (TOTAL_STEPS - WARMUP))  # linear decay
```

- **Mixed precision + gradient clipping**, exactly as before — `autocast`, a
  `GradScaler`, unscale, clip to norm 1.0, step.
- **An efficiency trick** The MLM loss only scores the ~15%
  masked positions, so projecting *every* token to the 8,000-word vocabulary is
  mostly wasted compute. Instead I gather only the masked hidden states first,
  then run the decoder on those:

```python
seq_out, pooled = model.encode(corrupt, attn, types)
sel = mlm_labels != -100          # (B, T) boolean: which positions are masked
hid = seq_out[sel]                # gather ONLY masked positions -> (num_masked, H)
hid = model.mlm_ln(F.gelu(model.mlm_transform(hid)))
mlm_logits = model.mlm_decoder(hid)   # decode just those — several times faster
```

Since the loss ignores everything else anyway, this is exact, not an
approximation — just a much cheaper way to compute the same number.

## Does it actually learn?

The training script ends with a fill-mask sanity check: take real sentences,
hide a word in the middle, and see whether the model recovers it in its top-5
guesses.

```python
x = full.copy(); x[pos] = MASK
logits, _, _ = model(torch.tensor([x], device=DEVICE))
topk = logits[0, pos].topk(5).indices.tolist()
```

It's a small model trained on a small corpus — nowhere near the real BERT's
understanding — but the mechanism is genuinely the same one that powered a
generation of NLP.

## The hyperparameters, in one place

| Setting | Value | Note |
| --- | --- | --- |
| hidden size | 256 | (reference bert-base: 768) |
| layers | 6 | (bert-base: 12) |
| attention heads | 8 | head dim = 32 |
| FFN inner size | 1024 | 4× hidden |
| max sequence length | 64 | packed sentence pairs |
| vocab size | 8,000 | WordPiece |
| activation | GELU | — |
| norm placement | post-norm | `LayerNorm(x + sublayer)` |
| MLM masking | 15% @ 80/10/10 | dynamic, per step |
| optimizer | AdamW | `lr 5e-4`, wd 0.01 |
| schedule | warmup 1k → linear decay | 20,000 total steps |

## What I'd change

- **Whole-word masking.** Masking individual WordPiece fragments lets the model
  cheat by completing a word from its own visible pieces; masking all fragments
  of a word together is a well-known improvement.
- **Drop NSP.** Later work (RoBERTa) found next-sentence prediction adds little
  and sometimes hurts; replacing it with more MLM on longer spans is stronger.
- **A real corpus.** Shakespeare is a fun, tiny testbed, but BERT's magic comes
  from scale — Wikipedia-sized data and far more steps.
- **Downstream fine-tuning.** The whole point of pretraining is transfer;
  wiring up a classification head and fine-tuning on a real task would close
  the loop.


