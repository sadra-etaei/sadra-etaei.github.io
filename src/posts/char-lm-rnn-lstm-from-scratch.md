---
title: "Teaching Python to write Shakespeare: RNNs and LSTMs from scratch"
date: 2026-07-19
summary: A character-level language model built with nothing but Python and NumPy — hand-written backprop through time, why vanilla RNNs fail to vanishing gradients, and how the LSTM's cell-state highway finally makes the loss go down.
tags: [NLP, LSTM, NumPy]
---

For my other from-scratch projects I let PyTorch handle the calculus. This
time I wanted to understand the gradients, so I built character-level language
models with **only Python and NumPy** — every forward pass, every
backpropagation-through-time step, every optimizer update written by hand.
It's a homage to Andrej Karpathy's famous
[char-rnn](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), trained
on the same tiny-Shakespeare dataset, but with no autograd to hide behind.

The project is really a story in three acts, and the plot has a genuine
villain. The [vanilla RNN](https://github.com/sadra-etaei/char-lm) couldn't
learn. Stacking it into a deep RNN didn't help. Only after I implemented a
multi-layer **LSTM** did the model start producing something that looks like
English. This post is about *why* — what breaks, and what the LSTM's gates
actually fix. Code's on [GitHub](https://github.com/sadra-etaei/char-lm).

## Character language modeling in one screen

The task: given the characters so far, predict the next one. No words, no
tokenizer — just raw characters. The vocabulary is every distinct character
in the text (~65 for Shakespeare), and each is represented as a **one-hot**
column vector:

```python
chars = list(set(text))
vocab_size = len(chars)
char2id = {ch: i for i, ch in enumerate(chars)}
x = np.zeros((vocab_size, 1))
x[char_id] = 1        # one-hot encoding of a single character
```

The model reads a sequence, and at each step emits a score vector over the
vocabulary. Softmax turns scores into probabilities, and the loss is
cross-entropy — the negative log-probability of the *correct* next character:

```python
p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))   # softmax
loss += -np.log(p[target_id, 0] + 1e-15)                    # cross-entropy
```

The `y - np.max(y)` shift is the standard trick to keep `exp` from
overflowing — it doesn't change the result, since softmax is invariant to a
constant offset. That's the whole objective. Everything interesting is in how
the score vector `y` gets computed, and how the gradient of that loss flows
back through time.

## Act I — the Vanilla RNN

A recurrent network keeps a hidden state $h_t$ that it updates at every step,
mixing the new input with the memory of everything before it:

$$
h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h), \qquad
y_t = W_{hy}\, h_t + b_y
$$

The forward pass is a direct transcription — and note it stores every hidden
state, because backprop will need them all:

```python
def forward(self, inputs, h_prev):
    self.h_states = {-1: h_prev}
    outputs = []
    for t, x in enumerate(inputs):
        h = np.tanh(np.dot(self.W_hh, self.h_states[t-1]) + np.dot(self.W_xh, x)) + self.b_h
        self.h_states[t] = h
        outputs.append(np.dot(self.W_hy, h) + self.b_y)
    return outputs, self.h_states[len(inputs)-1]
```

### Backprop through time, by hand

This is the part autograd normally does for you. The loss at every timestep
depends on $h_t$, but $h_t$ also feeds $h_{t+1}$ — so the gradient at step $t$
has two sources: the output at $t$, and everything downstream flowing back
through the recurrence. We walk *backwards* through the sequence, carrying a
running `dh_next`:

```python
dh_next = np.zeros((self.hidden_size, 1))
for t in reversed(range(len(self.inputs))):
    dh_t = np.dot(self.W_hy.T, d_outputs[t]) + dh_next   # output grad + future grad
    dtanh = (1 - h_t**2) * dh_t                          # through tanh: tanh'(x) = 1 - tanh²(x)
    dW_xh += np.dot(dtanh, x_t.T)
    dW_hh += np.dot(dtanh, h_prev.T)
    dh_next = np.dot(self.W_hh.T, dtanh)                 # gradient handed to the previous step
```

That last line is the crux of the whole story. To send the gradient one step
further back in time, we multiply by $W_{hh}^\top$. Do that over a sequence of
length 25 and the gradient reaching the earliest step has been multiplied by
$W_{hh}^\top$ twenty-five times over.

The optimizer is **AdaGrad**, which gives each parameter its own learning
rate by dividing by the accumulated sum of squared gradients:

```python
mem += dparam * dparam                          # accumulate squared gradients forever
param -= (lr / np.sqrt(mem + eps)) * dparam
```

## Why it failed: the vanishing gradient

The vanilla RNN trained, but its loss plateaued and its samples never
progressed past noise. The culprit is that repeated multiplication by
$W_{hh}^\top$. Chaining the recurrence backward gives a product of Jacobians:

$$
\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i} \operatorname{diag}\!\big(1 - h_i^2\big)\, W_{hh}^\top
$$

Two forces squeeze this toward zero. First, $\tanh'(x) = 1 - \tanh^2(x)$ is at
most 1 and usually much less, so every factor shrinks the signal. Second, if
the largest singular value of $W_{hh}$ is below 1, repeated multiplication
decays geometrically. The result: gradients from far-back timesteps arrive
essentially at zero, so the network **cannot learn long-range dependencies** —
it never connects a closing quote to the opening one twenty characters
earlier. (The mirror-image failure, exploding gradients when that singular
value exceeds 1, is the other half of the same problem.)

<svg viewBox="0 0 700 260" role="img" aria-label="Unrolled RNN showing gradient vanishing backward through time" style="width:100%;height:auto;max-width:640px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="fwd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <!-- hidden state chain -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:13px">
    <!-- nodes -->
    <circle cx="110" cy="130" r="26" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="110" y="135" text-anchor="middle" style="fill:var(--text-muted)">h₁</text>
    <circle cx="270" cy="130" r="26" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="270" y="135" text-anchor="middle" style="fill:var(--text-muted)">h₂</text>
    <circle cx="430" cy="130" r="26" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="430" y="135" text-anchor="middle" style="fill:var(--text-muted)">h₃</text>
    <circle cx="590" cy="130" r="26" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="590" y="135" text-anchor="middle" style="fill:var(--text-muted)">h₄</text>
  </g>
  <!-- forward arrows -->
  <g style="stroke:var(--text-faint)">
    <line x1="136" y1="130" x2="244" y2="130" marker-end="url(#fwd)"/>
    <line x1="296" y1="130" x2="404" y2="130" marker-end="url(#fwd)"/>
    <line x1="456" y1="130" x2="564" y2="130" marker-end="url(#fwd)"/>
  </g>
  <!-- W_hh labels on forward -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:10px;fill:var(--text-faint)">
    <text x="190" y="120" text-anchor="middle">W_hh</text>
    <text x="350" y="120" text-anchor="middle">W_hh</text>
    <text x="510" y="120" text-anchor="middle">W_hh</text>
  </g>
  <!-- inputs / outputs -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:12px;fill:var(--text-faint)">
    <text x="110" y="196" text-anchor="middle">x₁</text><line x1="110" y1="182" x2="110" y2="160" style="stroke:var(--border)" marker-end="url(#fwd)"/>
    <text x="270" y="196" text-anchor="middle">x₂</text><line x1="270" y1="182" x2="270" y2="160" style="stroke:var(--border)" marker-end="url(#fwd)"/>
    <text x="430" y="196" text-anchor="middle">x₃</text><line x1="430" y1="182" x2="430" y2="160" style="stroke:var(--border)" marker-end="url(#fwd)"/>
    <text x="590" y="196" text-anchor="middle">x₄</text><line x1="590" y1="182" x2="590" y2="160" style="stroke:var(--border)" marker-end="url(#fwd)"/>
  </g>
  <!-- backward gradient arrows, shrinking -->
  <g style="stroke:var(--accent);fill:none">
    <line x1="564" y1="104" x2="456" y2="104" stroke-width="4" marker-end="url(#fwd)"/>
    <line x1="404" y1="104" x2="296" y2="104" stroke-width="2.4" marker-end="url(#fwd)" style="opacity:0.75"/>
    <line x1="244" y1="104" x2="136" y2="104" stroke-width="1.1" marker-end="url(#fwd)" style="opacity:0.45"/>
  </g>
  <text x="590" y="46" text-anchor="middle" style="font-size:11px;fill:var(--accent)">strong gradient</text>
  <text x="110" y="46" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">≈ vanished</text>
  <text x="350" y="30" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">each backward step multiplies by Wₕₕᵀ · tanh′ → the signal decays</text>
</svg>

## Act II — stacking didn't save it

My next hypothesis was that the model just wasn't expressive enough, so I
built an $N$-layer RNN: the hidden state of layer $l$ becomes the input to
layer $l+1$, with a separate weight set per layer.

```python
for l in range(self.num_layers):
    h = np.tanh(np.dot(self.W_hh[l], self.hs[t-1][l]) +
                np.dot(self.W_xh[l], layer_input) + self.b_h[l])
    self.hs[t].append(h)
    layer_input = h        # this layer's output feeds the next layer
```

Now the backward pass has gradient flowing in *two* directions — backward
through time (`dh_next[l]`) and downward through the stack (`dh_down`) — which
was excellent practice at keeping the bookkeeping straight:

```python
for l in reversed(range(self.num_layers)):
    dh = dh_down + dh_next[l]                  # combine depth grad + time grad
    dtanh = (1 - self.hs[t][l]**2) * dh
    ...
    dh_next[l] = np.dot(self.W_hh[l].T, dtanh)   # -> back in time
    np.clip(dh_next[l], -5, 5, out=dh_next[l])   # tame exploding gradients
    dh_down = np.dot(self.W_xh[l].T, dtanh)      # -> down to the layer below
```

I added two upgrades here: **gradient clipping** (clamp every gradient to
`[-5, 5]` so a spike can't blow up the weights) and **RMSProp** instead of
AdaGrad. The difference matters — AdaGrad accumulates squared gradients
*forever*, so its effective learning rate marches monotonically to zero and
training stalls; RMSProp uses an exponentially *decaying* average, so it keeps
adapting:

```python
mem *= decay          # forget old gradient magnitudes (decay = 0.99)
mem += (1 - decay) * (dparam * dparam)
param -= (lr / np.sqrt(mem + eps)) * dparam
```

It still didn't learn. Depth adds representational power, but it does nothing
about the fundamental problem: the gradient *through time* still passes
through the same saturating tanh recurrence at every layer. More layers, same
vanishing. The architecture itself had to change.

## Act III — the LSTM, and the highway that fixes everything

The Long Short-Term Memory cell solves vanishing gradients with one
structural idea: alongside the hidden state, carry a separate **cell state**
$c_t$ that is updated almost entirely by *addition*, and let a set of learned
**gates** decide what to write to it, erase from it, and read out of it.

### The four gates, in one matrix multiply

Each layer has four gate computations, and rather than four separate weight
matrices, I stack the previous hidden state and the input into one vector `z`
and use a single big matrix — one `dot` instead of four:

```python
z = np.vstack((self.h_states[t-1][l], layer_input))   # concatenate [h_prev; x]
gates = np.dot(self.W[l], z) + self.b[l]               # one matmul -> all 4 gates
f = sigmoid(gates[0:H])            # forget gate  — what to erase from the cell
i = sigmoid(gates[H:2*H])          # input gate   — how much new info to admit
g = np.tanh(gates[2*H:3*H])        # candidate    — the new info itself
o = sigmoid(gates[3*H:4*H])        # output gate  — what to read out as h
```

The three sigmoid gates output values in $(0, 1)$ — soft switches. The cell
update and the new hidden state then read:

$$
c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)
$$

```python
c_t = f * self.c_states[t-1][l] + i * g
h_t = o * np.tanh(c_t)
```

<svg viewBox="0 0 700 300" role="img" aria-label="LSTM cell showing the additive cell-state highway and the four gates" style="width:100%;height:auto;max-width:660px;display:block;margin:1.5rem auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="lstma" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <!-- cell state highway -->
  <line x1="40" y1="60" x2="660" y2="60" style="stroke:var(--accent);stroke-width:3" marker-end="url(#lstma)"/>
  <text x="70" y="48" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">c₍ₜ₋₁₎</text>
  <text x="640" y="48" text-anchor="end" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">cₜ</text>
  <!-- forget multiply -->
  <circle cx="250" cy="60" r="15" style="fill:var(--bg);stroke:var(--accent)"/>
  <text x="250" y="65" text-anchor="middle" style="font-size:14px;fill:var(--accent)">×</text>
  <!-- input add -->
  <circle cx="410" cy="60" r="15" style="fill:var(--bg);stroke:var(--accent)"/>
  <text x="410" y="66" text-anchor="middle" style="font-size:16px;fill:var(--accent)">+</text>
  <!-- gate boxes -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:12px">
    <rect x="215" y="150" width="70" height="34" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="250" y="172" text-anchor="middle" style="fill:var(--text-muted)">f  σ</text>
    <rect x="330" y="150" width="70" height="34" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="365" y="172" text-anchor="middle" style="fill:var(--text-muted)">i  σ</text>
    <rect x="415" y="150" width="70" height="34" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="450" y="172" text-anchor="middle" style="fill:var(--text-muted)">g  tanh</text>
    <rect x="545" y="150" width="70" height="34" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="580" y="172" text-anchor="middle" style="fill:var(--text-muted)">o  σ</text>
  </g>
  <!-- gate to op wires -->
  <g style="stroke:var(--text-faint)">
    <line x1="250" y1="150" x2="250" y2="77" marker-end="url(#lstma)"/>
    <line x1="380" y1="150" x2="405" y2="77" marker-end="url(#lstma)"/>
    <line x1="450" y1="150" x2="418" y2="77" marker-end="url(#lstma)"/>
  </g>
  <!-- input concat -->
  <rect x="300" y="235" width="200" height="34" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="400" y="257" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">z = [ h₍ₜ₋₁₎ ; xₜ ]</text>
  <g style="stroke:var(--border)">
    <line x1="340" y1="235" x2="250" y2="186" marker-end="url(#lstma)"/>
    <line x1="380" y1="235" x2="365" y2="186" marker-end="url(#lstma)"/>
    <line x1="430" y1="235" x2="450" y2="186" marker-end="url(#lstma)"/>
    <line x1="470" y1="235" x2="580" y2="186" marker-end="url(#lstma)"/>
  </g>
  <!-- output read -->
  <circle cx="580" cy="60" r="15" style="fill:var(--bg);stroke:var(--text-muted)"/>
  <text x="580" y="65" text-anchor="middle" style="font-size:13px;fill:var(--text-muted)">×</text>
  <line x1="580" y1="150" x2="580" y2="77" style="stroke:var(--text-faint)" marker-end="url(#lstma)"/>
  <line x1="580" y1="75" x2="580" y2="140" style="stroke:var(--text-muted)"/>
  <text x="600" y="110" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">hₜ</text>
  <line x1="595" y1="60" x2="595" y2="110" style="stroke:var(--text-muted)"/>
</svg>

### Why this fixes the gradient

Look at how the gradient travels backward along the cell state. In the vanilla
RNN, one step back meant multiplying by $W_{hh}^\top$. In the LSTM it's just:

```python
dc_next[l] = dc * f     # gradient to c_{t-1}: elementwise multiply by the forget gate
```

No weight matrix, no tanh derivative — the gradient flows back through the
cell state by an **elementwise multiply by the forget gate**. When the model
learns to remember something, $f \approx 1$, and the gradient passes backward
essentially untouched across hundreds of steps. This additive, un-squashed
path is the "constant error carousel," and it's the entire reason LSTMs learn
long-range structure where RNNs can't.

I gave it a nudge in the right direction by initializing the **forget-gate
bias to 1**, so the cell defaults to *remembering* before it has learned when
to forget:

```python
self.b[l][0:hidden_size, 0] = 1.0   # forget gate starts "open"
```

### Backprop through the gates

There's no autograd here either, so every gate's derivative is written out by
hand — the payoff for understanding exactly how error splits between "back in
time" and "down in depth":

```python
do = dh * np.tanh(c_t) * (o * (1 - o))       # sigmoid'
dc = dh * o * (1 - np.tanh(c_t)**2) + dc_next[l]
df = dc * c_prev * (f * (1 - f))
di = dc * g * (i * (1 - i))
dg = dc * i * (1 - g**2)                      # tanh'
d_gates = np.vstack((df, di, dg, do))
dW[l] += np.dot(d_gates, z.T)                 # one matmul, mirrors the forward
dz = np.dot(self.W[l].T, d_gates)
dh_next[l] = dz[:self.hidden_size]           # top half -> previous timestep
dh_down    = dz[self.hidden_size:]           # bottom half -> layer below
```

Splitting `dz` back into its two halves — the top going to the previous
hidden state, the bottom to the layer below — is the exact inverse of the
`vstack` in the forward pass. Getting that split right is the whole game.

## Training tricks that made it work

- **Truncated BPTT.** The full text is far too long to backprop through, so
  training processes it in short chunks (`seq_length` characters) while
  *carrying the hidden and cell states across chunks*. The model keeps its
  memory; only the gradient is truncated at the chunk boundary.
- **Smoothed loss.** Per-chunk loss is noisy, so I track an exponential
  moving average for a readable training curve:
  `smooth_loss = 0.999 * smooth_loss + 0.001 * loss`.
- **Temperature sampling.** At generation time, dividing the logits by a
  temperature before softmax controls the risk the model takes — low
  temperature (I used 0.7) yields more confident, coherent text; high
  temperature is wilder and more error-prone:

```python
y = y / temperature
p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
id = np.random.choice(range(vocab_size), p=p.ravel())
```

## The payoff: watching it learn

Same model, same code — only the number of training iterations changes. At the
very start it emits pure noise:

```text
iteration 0:
kTEbCbw-K-ksKddbeRK'FRcdxdrxKKdx'xxxdxxxxxxxxxxxxxxTxxxxxxuuanFFnFF...
```

After tens of thousands of iterations, the two-layer LSTM has picked up
Shakespeare's *shape* — character names in caps, line breaks, punctuation,
and mostly-real English words:

```text
iteration 50000:
Second Citizen:
My morrow, my honour means is of my boon,
Gentlemen on, then hove the world, my house,
And you m...

MARIANA:
O crave you to you: not many mourning him
```

It's not perfect — "hove the world" is not exactly the Bard — but for a model
that is *pure NumPy* and started from random noise, watching structured
English emerge from a hand-written backward pass was genuinely thrilling.

## The hyperparameters, in one place

| Setting | Vanilla RNN | N-layer RNN | LSTM |
| --- | --- | --- | --- |
| hidden size | 512 | 512 | 512 |
| layers | 1 | 3 | 2 |
| optimizer | AdaGrad | RMSProp | RMSProp |
| grad clip | — | ±5 | ±1 |
| seq length | 25 | 100 | 50 |
| learning rate | 0.01 | 0.01 | 0.001 |
| forget bias init | — | — | 1.0 |
| **learned Shakespeare?** | **no** | **no** | **yes** |

## What I'd change

- **Minibatching.** Everything runs one sequence at a time with explicit
  Python loops over timesteps — correct, but slow. Vectorizing across a batch
  of sequences would be a large speedup (and is what makes the GPU worth it).
- **Adam over RMSProp.** Adding the momentum term usually converges faster and
  more stably.
- **Peephole connections or an LSTM variant** (or a GRU) would be a natural
  next experiment now that the plumbing exists.
- **A proper train/validation split** to measure overfitting instead of eyeing
  the samples.

## What building it taught me

Frameworks make it far too easy to treat backprop as magic. Writing the
backward pass of an LSTM by hand — splitting the gradient between time and
depth, deriving each gate's local derivative, watching a single misplaced
`np.clip` or transpose quietly poison the loss — turned "I know the LSTM
equations" into "I know why every term is there." And the central lesson is
one no diagram had ever really landed for me until I watched my own RNN fail
and my own LSTM succeed on the same data: **architecture is how you shape the
flow of gradients.** The LSTM didn't win because it was bigger. It won because
it gave the gradient a road home.
