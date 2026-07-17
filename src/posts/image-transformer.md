---
title: "The Image Transformer: generating images one pixel at a time"
date: 2026-07-17
summary: A detailed walkthrough of reimplementing the Image Transformer paper in PyTorch — images as sequences, 2D local attention with query and memory blocks, the causal mask in two dimensions, and sampling CIFAR-10 images pixel by pixel.
tags: [Image Generation, Transformer, PyTorch]
---

After reimplementing the original Transformer for
[translation](/#/posts/building-a-transformer-from-scratch), I wanted to see
how far the "everything is a sequence" idea stretches. The answer, it turns
out, is *very* far: the [Image Transformer](https://arxiv.org/abs/1802.05751)
(Parmar et al., 2018) treats an image as nothing more than a long sequence of
pixel intensities and trains a decoder to predict the next one — exactly like
a language model, except the "words" are numbers from 0 to 255.

This post walks through my PyTorch reimplementation: how an image becomes a
sequence, why full attention breaks immediately, and the paper's central
contribution — **2D local attention** — which was by far the trickiest thing
I've built so far. The code is on
[GitHub](https://github.com/sadra-etaei/img-transformer).

## An image is a sequence, if you squint

A 32×32 RGB image is 32 × 32 × 3 = **3,072 numbers**, each an integer from 0
to 255. Flatten them in *raster order* — left to right, top to bottom, the
three channel values of each pixel in a row — and you have a sequence of
3,072 tokens drawn from a vocabulary of size 256.

<svg viewBox="0 0 700 300" role="img" aria-label="Raster-scan flattening of an image into a sequence" style="width:100%;height:auto;max-width:620px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="rarrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <!-- pixel grid 6x4 -->
  <g>
    <rect x="40" y="30" width="240" height="160" rx="6" style="fill:none;stroke:var(--border)"/>
    <!-- cells -->
    <g style="stroke:var(--border)">
      <line x1="80" y1="30" x2="80" y2="190"/><line x1="120" y1="30" x2="120" y2="190"/>
      <line x1="160" y1="30" x2="160" y2="190"/><line x1="200" y1="30" x2="200" y2="190"/>
      <line x1="240" y1="30" x2="240" y2="190"/>
      <line x1="40" y1="70" x2="280" y2="70"/><line x1="40" y1="110" x2="280" y2="110"/>
      <line x1="40" y1="150" x2="280" y2="150"/>
    </g>
    <!-- scan arrows -->
    <g style="stroke:var(--accent);fill:none" stroke-width="1.6">
      <line x1="52" y1="50" x2="266" y2="50" marker-end="url(#rarrow)"/>
      <path d="M 266 50 C 292 50, 292 90, 266 90 L 54 90" marker-end="url(#rarrow)"/>
      <path d="M 54 90 C 28 90, 28 130, 54 130 L 266 130" marker-end="url(#rarrow)"/>
    </g>
    <text x="160" y="215" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">raster order: row by row, left to right</text>
  </g>
  <!-- arrow to sequence -->
  <line x1="300" y1="110" x2="352" y2="110" style="stroke:var(--text-faint)" marker-end="url(#rarrow)"/>
  <text x="326" y="98" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">flatten</text>
  <!-- sequence strip -->
  <g>
    <g style="font-size:11px;font-family:'JetBrains Mono',monospace">
      <rect x="370" y="90" width="40" height="40" rx="5" style="fill:var(--accent-soft);stroke:var(--accent)"/>
      <text x="390" y="114" text-anchor="middle" style="fill:var(--accent)">R₀</text>
      <rect x="414" y="90" width="40" height="40" rx="5" style="fill:var(--code-bg);stroke:var(--border)"/>
      <text x="434" y="114" text-anchor="middle" style="fill:var(--text-muted)">G₀</text>
      <rect x="458" y="90" width="40" height="40" rx="5" style="fill:var(--code-bg);stroke:var(--border)"/>
      <text x="478" y="114" text-anchor="middle" style="fill:var(--text-muted)">B₀</text>
      <rect x="502" y="90" width="40" height="40" rx="5" style="fill:var(--code-bg);stroke:var(--border)"/>
      <text x="522" y="114" text-anchor="middle" style="fill:var(--text-muted)">R₁</text>
      <rect x="546" y="90" width="40" height="40" rx="5" style="fill:var(--code-bg);stroke:var(--border)"/>
      <text x="566" y="114" text-anchor="middle" style="fill:var(--text-muted)">G₁</text>
      <text x="612" y="114" style="fill:var(--text-faint)">…</text>
    </g>
    <text x="495" y="155" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">3,072 tokens, each ∈ {0 … 255}</text>
  </g>
</svg>

Then the modeling task is identical to language modeling: predict token $t$
given tokens $1 \dots t-1$, with a 256-way softmax instead of a
vocabulary-sized one. In my implementation each intensity gets an embedding,
and one extra embedding id acts as the start token — the "before the first
pixel" marker:

```python
self.embeds = nn.Embedding(self.NUM_INTENSITIES + 1, d_model)  # 256 + start
self.fc_out = nn.Linear(d_model, self.NUM_INTENSITIES)         # 256-way softmax
```

The loss is plain cross-entropy over intensities, and quality is reported in
**bits per dimension** — the average number of bits the model needs to encode
each subpixel, which is just the negative log-likelihood converted from nats
to bits:

```python
def bits_per_dim(nll_nats):
    return nll_nats / math.log(2)
```

Lower is better; a model that assigned uniform probability to all 256 values
would sit at exactly 8 bits/dim.

## The quadratic wall

Here's why you can't just reuse the text Transformer unchanged. Attention
compares every position with every other position, so its cost grows with the
*square* of sequence length. A 3,072-token image means

$$
3072^2 \approx 9.4 \text{ million}
$$

attention scores *per head, per layer* — and with 8 heads and 12 layers
that's roughly 900 million scores for a single 32×32 image. Double the image
resolution and it multiplies by 16. Full attention over raw pixels simply
doesn't scale.

The Image Transformer's answer is **local attention**: each position attends
only to a neighborhood around it rather than the whole image. The insight
that makes this reasonable is that images are spatially coherent — a pixel's
value is overwhelmingly determined by nearby pixels. And with 12 stacked
layers, information still propagates across the whole image, one
neighborhood-hop per layer — the effective receptive field grows with depth
even though each layer is local.

## 2D local attention: query blocks and memory blocks

This is the heart of the paper and of the implementation. The image is tiled
into non-overlapping **query blocks** (I use the paper's CIFAR-10 shape of
8 rows × 32 columns). Every pixel in a query block attends to the same
**memory block**: the query block itself, plus a halo of `memory_pad` extra
rows above and columns to either side.

<svg viewBox="0 0 700 380" role="img" aria-label="2D local attention query block and memory block layout" style="width:100%;height:auto;max-width:620px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <!-- legend -->
  <g style="font-size:12px">
    <rect x="60" y="20" width="14" height="14" rx="3" style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <text x="82" y="32" style="fill:var(--text-muted)">query block</text>
    <rect x="200" y="20" width="14" height="14" rx="3" style="fill:var(--code-bg);stroke:var(--text-faint)"/>
    <text x="222" y="32" style="fill:var(--text-muted)">memory halo (visible context)</text>
    <rect x="440" y="20" width="14" height="14" rx="3" style="fill:none;stroke:var(--border);stroke-dasharray:3 3"/>
    <text x="462" y="32" style="fill:var(--text-muted)">rest of the image</text>
  </g>
  <!-- image frame -->
  <rect x="60" y="60" width="580" height="280" rx="8" style="fill:none;stroke:var(--border);stroke-dasharray:4 4"/>
  <text x="70" y="330" style="font-size:11px;fill:var(--text-faint)">image (padded to tile exactly)</text>
  <!-- memory halo: rows above -->
  <rect x="180" y="120" width="360" height="50" style="fill:var(--code-bg);stroke:var(--text-faint)"/>
  <text x="360" y="150" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">h_m rows above</text>
  <!-- memory halo: left -->
  <rect x="180" y="170" width="60" height="70" style="fill:var(--code-bg);stroke:var(--text-faint)"/>
  <text x="210" y="209" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">w_m</text>
  <!-- memory halo: right -->
  <rect x="480" y="170" width="60" height="70" style="fill:var(--code-bg);stroke:var(--text-faint)"/>
  <text x="510" y="209" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">w_m</text>
  <!-- query block -->
  <rect x="240" y="170" width="240" height="70" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
  <text x="360" y="200" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">query block  h_q × w_q</text>
  <text x="360" y="222" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">every pixel here attends to the whole shaded region</text>
  <!-- memory block outline -->
  <rect x="180" y="120" width="360" height="120" style="fill:none;stroke:var(--text-muted);stroke-width:1.4"/>
  <text x="360" y="262" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">memory block  (h_q + h_m) × (w_q + 2·w_m)</text>
</svg>

Two things make this design efficient:

1. **Attention cost becomes linear in image size.** Each query attends to a
   fixed-size memory (`(8+8) × (32+16) = 768` positions in my config) rather
   than all 3,072. The number of blocks grows with the image; the per-block
   cost doesn't.
2. **All queries in a block share one memory**, so the computation batches
   into dense tensor ops — no per-pixel gathering. The block dimension just
   becomes one more batch axis.

### Carving the image into blocks

The input arrives as a flat sequence `[B, H·W, D]`, so the first job is to
get it back into 2D and tile it. Padding makes the tiling exact:

```python
x = x.view(B, height, width, D)
pad_b = (h_q - height % h_q) % h_q     # pad bottom so rows tile evenly
pad_r = (w_q - width % w_q) % w_q      # pad right so cols tile evenly
x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
```

Extracting the query blocks is the same split-axes trick as multi-head
attention, one dimension up. `view` splits each spatial axis into
(block index, offset inside block), and a `permute` groups the two block
indices together:

```python
Q = (self.W_q(x)
     .view(B, n_h, h_q, n_w, w_q, D)      # split H -> (n_h, h_q), W -> (n_w, w_q)
     .permute(0, 1, 3, 2, 4, 5)           # [B, n_h, n_w, h_q, w_q, D]
     .reshape(B, n_blocks, l_q, D))       # flatten to blocks of l_q pixels
```

### `F.unfold`: the sliding-window workhorse

The memory blocks are harder because they **overlap** — each one extends into
its neighbors' territory. Extracting overlapping patches by hand means messy
index arithmetic, but PyTorch has an operation built for exactly this:
`F.unfold`, the same im2col primitive that underlies convolution. Give it a
kernel size and a stride and it extracts every sliding window as a column:

```python
def to_memory(t):
    t = t.permute(0, 3, 1, 2)                       # [B, D, H, W] — unfold wants channels-first
    t = F.pad(t, (w_m, w_m, h_m, 0))                # halo: left, right, top (never below)
    t = F.unfold(t, (k_h, k_w), stride=(h_q, w_q))  # [B, D·l_m, n_blocks]
    return t.view(B, D, l_m, n_blocks).permute(0, 3, 2, 1)

K = to_memory(self.W_k(x))
V = to_memory(self.W_v(x))
```

The stride equals the *query* block size while the kernel is the larger
*memory* size — that's what makes consecutive windows overlap by exactly the
halo. And note the padding is asymmetric: `h_m` rows above but **zero below**,
because in generation order the pixels below a block don't exist yet; there
is nothing to attend to down there.

### Attention with one extra letter

After splitting heads, the einsum is the text-Transformer one with a block
axis `n` along for the ride:

```python
scores = torch.einsum('bnqhd,bnkhd->bnhqk', Q, K) / math.sqrt(self.d_k)
attn = self.dropout(F.softmax(scores, dim=-1))
out = torch.einsum('bnhqk,bnkhd->bnqhd', attn, V)
```

Same mental model as before — `d` contracts (dot product), `k` contracts
(weighted sum), everything else is batched — except now `b` *and* `n` are
carried along, so each of the `n_blocks` neighborhoods computes its attention
independently and in parallel. This is the payoff of the blocking: what
looks like an exotic sparse attention pattern is just dense attention with
one more batch dimension.

Afterwards, the inverse permute/reshape puts pixels back in image order and
slices off the bottom/right padding.

## The causal mask, now in two dimensions

In the text Transformer the causal mask was one `torch.tril`. Here "the past"
is a more complicated shape, because generation proceeds block by block, and
within each block in raster order. From the viewpoint of one query pixel,
a memory position is visible if it falls in any of three regions:

<svg viewBox="0 0 700 330" role="img" aria-label="Causal mask regions inside a memory block" style="width:100%;height:auto;max-width:560px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <!-- memory block frame -->
  <rect x="80" y="40" width="480" height="240" style="fill:none;stroke:var(--text-muted);stroke-width:1.4"/>
  <!-- rows above -->
  <rect x="80" y="40" width="480" height="80" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="320" y="85" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">① rows above the query block — always visible</text>
  <!-- left -->
  <rect x="80" y="120" width="80" height="160" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="120" y="196" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">②</text>
  <text x="120" y="214" text-anchor="middle" style="font-size:10px;fill:var(--text-muted)">left halo</text>
  <!-- query block area -->
  <g>
    <!-- visible raster part -->
    <path d="M 160 120 L 480 120 L 480 170 L 330 170 L 330 200 L 160 200 Z" style="fill:var(--accent-soft);stroke:none"/>
    <text x="285" y="150" text-anchor="middle" style="font-size:12px;fill:var(--accent)">③ earlier in raster order</text>
    <!-- current pixel -->
    <rect x="330" y="170" width="24" height="24" style="fill:var(--accent);stroke:none"/>
    <text x="368" y="187" style="font-size:11px;fill:var(--text)">← current pixel</text>
    <!-- future -->
    <text x="320" y="245" text-anchor="middle" style="font-size:12px;fill:var(--text-faint)">masked: later in raster order + right halo below</text>
    <!-- query block outline -->
    <rect x="160" y="120" width="320" height="160" style="fill:none;stroke:var(--accent);stroke-width:1.5"/>
  </g>
  <text x="320" y="310" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">visible = above  |  left  |  (in-block ∧ raster ≤ self)</text>
</svg>

The implementation computes each region from coordinates measured relative to
the query block's top-left corner:

```python
above = mr < 0                                    # ① earlier block rows
left = (mr >= 0) & (mc < 0)                       # ② halo to the left
in_block = ((mr >= 0) & (mc >= 0) & (mc < w_q)
            & (mr * w_q + mc <= qr * w_q + qc))   # ③ raster order inside the block
causal = above | left | in_block                  # [l_q, l_m]
```

The comparison `mr * w_q + mc <= qr * w_q + qc` is the 2D generalization of
`torch.tril`: it linearizes both positions into raster indices and keeps
memory positions at or before the query. One mask matrix of shape
`[l_q, l_m]` is shared by every block, because the geometry is identical
everywhere — that's another gift of the uniform blocking.

Two extra subtleties stack on top:

- **Image-validity masking.** The bottom/right padding pixels (and halo
  positions hanging off the image edge) are not real; a second mask, built by
  running `F.unfold` over a 0/1 "is this a real pixel" map with the *same*
  kernel and stride, knocks them out per block.
- **The self-slot rule.** A padded query position could end up with *every*
  memory slot masked — and a softmax over all `-inf` produces NaN, which then
  poisons the whole backward pass. The fix: every query is always allowed to
  see itself. Those padded rows get discarded after attention anyway, so this
  changes nothing real — it just keeps the softmax finite. This single line
  cost me the most debugging time of the entire project; NaN losses point at
  everything except their actual cause.

## 2D positional encoding

A pixel's identity is its (row, column) pair, so the paper splits the
embedding dimension in half: the first `d_model/2` dimensions carry a
sinusoidal encoding of the **row**, the other half of the **column** (in the
channel-expanded coordinate system, since each pixel spans three sequence
positions):

```python
rows = pos // (width * channels)
cols = pos % (width * channels)
pe = torch.cat([self.row_pe[rows], self.col_pe[cols]], dim=-1)
return x + pe.unsqueeze(0)
```

Same sinusoid formula as the text Transformer (same `exp`/`log` stability
trick too), just applied twice and concatenated. The model gets genuine 2D
coordinates instead of a single flat position — attending "one row up" is as
learnable as attending "one token back."

## Making it autoregressive: the shifted input

Training uses teacher forcing, exactly like the translation model: feed the
ground-truth image in, predict every position in parallel, let the mask
prevent cheating. But there's a wrinkle. Generation doesn't proceed in plain
raster order over the image — it proceeds **block by block**, and in raster
order *within* each block. The "previous token" for the first pixel of block
7 is the *last* pixel of block 6, which is not its raster-order neighbor.

So the shift-right has to happen in *generation* order. The code builds a
permutation `perm` mapping generation step → raster index, applies it, shifts
by one with the start token at the front, and permutes back:

```python
perm = self._generation_order(H, W * C, images.device)   # step -> raster index
inv = torch.empty_like(perm)
inv[perm] = torch.arange(perm.numel(), device=perm.device)  # raster index -> step

seq_gen = seq.gather(1, perm.unsqueeze(0).expand(B, -1))    # reorder to generation order
inp_gen = torch.cat([start, seq_gen[:, :-1]], dim=1)        # shift right + start token
inp = inp_gen.gather(1, inv.unsqueeze(0).expand(B, -1))     # back to raster order
```

Building the inverse permutation with `inv[perm] = arange(...)` is a tidy
idiom worth stealing: if `perm` sends step `s` to position `p`, then `inv`
sends position `p` back to step `s`, and both directions are one `gather`
away. The result: every position's input is the intensity generated *just
before it* in block order, while the tensor stays in raster layout so the
positional encoding and attention blocking still line up.

## Training on CIFAR-10

The training script follows the paper's CIFAR-10 configuration: 12 decoder
layers, `d_model = 512`, `d_ff = 2048`, 8 heads, dropout 0.3 (heavier than
the text model — with 50,000 training images, a ~50M-parameter model overfits
enthusiastically). Images are kept as raw `uint8` intensities:

```python
to_hwc = transforms.Compose([
    transforms.PILToTensor(),      # keeps 0-255 integers, no float scaling
    transforms.Lambda(lambda t: t.permute(1, 2, 0).long()),
])
```

That `PILToTensor` (rather than the usual `ToTensor`) matters: the standard
transform would rescale to `[0, 1]` floats, but here the intensities *are*
the class labels — they have to stay integers.

This time I also implemented the paper's **Noam learning-rate schedule**
rather than a flat rate — warm up linearly, then decay with the inverse
square root of the step:

```python
def noam_lr(step, d_model, warmup):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
```

(That's why the optimizer is created with the odd-looking `lr=1.0` — the
schedule multiplies it every step, so the "real" learning rate lives entirely
in `noam_lr`.) Early Transformer training is unstable while LayerNorm
statistics settle; the warmup keeps the first few thousand steps small, and
this schedule was one of my listed regrets from the translation project, so
it felt good to close the loop. Checkpointing saves `last.pt` every epoch and
`best.pt` whenever test bits/dim improves, with the config dict stored inside
the checkpoint so `sample.py` can rebuild the exact architecture.

## Sampling: the slow, honest way

Generation is where autoregressive image models pay their bill. To sample an
image you run the *entire model once per subpixel*, in generation order:

```python
for s in range(flat.size(1)):                    # 3,072 iterations for 32×32×3
    f = perm[s]
    logits = self.forward(images)[:, f] / temperature
    flat[:, f] = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
```

Each step: full forward pass, take the logits at the position being
generated, divide by temperature, sample from the resulting 256-way
distribution, write the value into the image, repeat. `temperature < 1`
sharpens the distribution toward the most likely intensities (cleaner but
blander samples); `temperature > 1` flattens it (more diverse, more noise).

Yes — that's 3,072 full forward passes per image, and no, it isn't fast. A
production implementation would cache the per-layer activations for already-
generated pixels instead of recomputing them (KV caching), which is at the
top of my improvements list below.

## The hyperparameters, in one place

| Setting | Value | Note |
| --- | --- | --- |
| dataset | CIFAR-10 | 32×32 RGB, intensities kept as raw 0–255 ints |
| sequence length | 3,072 | 32 × 32 × 3 subpixels |
| `d_model` / `d_ff` | 512 / 2048 | paper's CIFAR-10 config |
| `n_layers` / `n_heads` | 12 / 8 | decoder-only |
| `query_shape` | 8 × 32 | 256 queries per block |
| `memory_pad` | (8, 8) | memory block 16 × 48 = 768 positions |
| dropout | 0.3 | heavy — small dataset, big model |
| LR schedule | Noam | warmup 4,000 steps, `lr=1.0` base |
| grad clip | 1.0 | max norm |
| metric | bits/dim | uniform baseline = 8.0 |

## What I'd change

- **KV caching for sampling.** The 3,072-forward-pass loop recomputes
  everything for every subpixel; caching would make sampling orders of
  magnitude faster and is the single highest-value improvement.
- **The categorical treatment of intensities** ignores that 200 and 201 are
  nearly the same color — the model has to *learn* that intensity space is
  ordered. A discretized logistic mixture (as in PixelCNN++) bakes it in.
- **Mixed precision.** The translation project used it; this one doesn't
  yet, and a 12-layer model on 3,072-long sequences would benefit even more.
- **Attention visualizations.** The per-block attention weights are sitting
  right there in the forward pass; rendering which memory pixels each query
  attends to would make a great follow-up post.

## What building it taught me

The translation Transformer taught me attention's *mechanics*; this project
taught me its *geometry*. Almost every hard bug lived in the seams between
coordinate systems — raster order vs. block order vs. generation order,
image coordinates vs. block-relative coordinates, real pixels vs. padding.
The model itself is the same residual rhythm as before; what's new is the
choreography of reshapes around it, and the discipline of keeping five
different orderings straight in your head at once. If the first project's
lesson was "the masks will teach you," this one's is: *in two dimensions,
everything is a mask.*
