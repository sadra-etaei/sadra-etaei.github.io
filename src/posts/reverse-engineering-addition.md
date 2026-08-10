---
title: "Reverse-engineering addition: the circuit inside a 0.5B model"
date: 2026-08-10
summary: I take a model I didn't build — Qwen2.5-0.5B — and pull one skill apart to see the gears: which attention heads fetch the digits, which MLPs do the math, how the carry works, and the experiments that came back null.
tags: [Interpretability, LLMs, TransformerLens]
---

Every model I've written about so far, I built. The
[Transformer](/#/posts/building-a-transformer-from-scratch),
[BERT](/#/posts/bert-from-scratch), the
[Image Transformer](/#/posts/image-transformer), a
[char-level LSTM](/#/posts/char-lm-rnn-lstm-from-scratch) — the fun was in
wiring the thing together from a blank file and watching the loss come down.
This post is the opposite exercise. I take a model I *didn't* build —
Qwen2.5-0.5B, an off-the-shelf open-weight LLM — and try to pull one specific
skill apart to see the gears. The skill is the one every kid learns first:
adding two numbers. The question is not *can* it add (it can), but **how** —
which attention heads, which MLPs, in which layers, doing what. Code is on
[GitHub](https://github.com/sadra-etaei/arithmetic_circuits).

There's a small, active literature on this. Stolfo et al. showed with causal mediation
that in GPT-2 and Pythia, attention shuttles the operands to the final position and MLPs
do the actual math. Nikankin et al. argued LLMs don't run a clean algorithm at all but a
"bag of heuristics" — a pile of neurons that each fire on a narrow input pattern.
Kantamneni & Tegmark found bigger models represent numbers on a helix and add by
literally rotating it (the "Clock" algorithm). I wanted to see, on the smallest model
that can actually do the task, which of these pictures shows up — and whether the *carry*,
the one genuinely hard part of multi-digit addition, has a mechanism I could point to.

Some of what follows is a clean, causal circuit I'm fairly confident in. Some of it is a
striking pattern I *can't* yet nail down — including one experiment that came back a flat
null and changed how I think about the problem. I've kept the negatives in, because in
interpretability the negatives are where you learn what you were fooling yourself about.

## The one twist that shapes everything: MSB-first

Before any experiment, one observation about the task itself sets the whole frame. When
you prompt the model with `457+386=`, the very next token it emits is the **most
significant** digit of the answer — the `8` in `843`. It writes the answer left to right,
big end first.

That's a strong constraint. To commit to the leading digit, the model has to have already
resolved the *entire* carry chain — the units carry into the tens, the tens carry into the
hundreds — because the leading digit depends on all of it. There's no "compute the units
first and carry upward" over multiple steps; it all has to happen in a single forward pass,
at a single token position: the `=`. So the `=` token is the natural focal point of the
whole investigation. Whatever "adding" means mechanically, it culminates there.

## Does it even add? (capability first)

You can't reverse-engineer a computation the model can't do — you'd just be studying its
failures. So the first job is picking a model that genuinely adds, and confirming it.

GPT-2-small, the classic interpretability target, is hopeless here: ~1% on 2-digit
addition, 0% on 3-digit. Nothing to interpret. I settled on **Qwen2.5-0.5B** for three
reasons: it's small enough to dissect completely (24 layers, `d_model` 896, 14 heads), it
tokenizes numbers **one digit per token** — `123+456=579` becomes
`['1','2','3','+','4','5','6','=','5','7','9']`, which keeps digit positions aligned across
examples and makes everything downstream cleaner — and, crucially, it can actually do the
task:

| Width    | Accuracy (8-shot) | Across carry counts     |
|----------|-------------------|-------------------------|
| 2-digit  | **98.7%**         | robust (0/1/2 carries)  |
| 3-digit  | **94.7%**         | robust (0–3 carries)    |

The evaluation is deliberately tokenizer-agnostic: greedy-generate a few tokens after
`=`, parse the first integer out, compare to the true sum. And from the start I tagged
every problem with its *carry structure*, because carries are the whole reason multi-digit
addition is interesting:

```python
def carry_bits(a, b):
    c1 = int((a % 10) + (b % 10) >= 10)                    # units -> tens
    c2 = int((a // 10 % 10) + (b // 10 % 10) + c1 >= 10)   # tens  -> hundreds
    c3 = int((a // 100 % 10) + (b // 100 % 10) + c2 >= 10) # hundreds -> thousands
    return c1, c2, c3
```

Notice the model's accuracy barely moves with the number of carries. Whatever it's doing,
it handles carry propagation gracefully — which only makes the "how" more interesting.

## Where does the answer appear? The logit lens

The cheapest first probe of depth is the **logit lens**: take the residual stream at the
`=` position after each layer, pretend it's the final layer, and decode it straight through
the unembedding. If the correct leading digit is already "readable" at layer L, the model
has effectively computed it by then.

```python
resid = cache["resid_post", L][:, -1, :]     # residual at '=' after layer L
logits = model.unembed(model.ln_final(resid))  # decode as if final
```

<figure style="margin:1.75rem 0">
  <img src="/figures/logit_lens.png" alt="Logit lens: the leading answer digit is undecodable until the last few layers, then resolves sharply around layers 20-22." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Logit lens: the leading answer digit is undecodable until the last few layers, then resolves sharply around layers 20–22.</figcaption>
</figure>

The result is almost comically sharp. For the first ~19 of 24 layers the correct digit is
*random* — mean rank ~150,000 out of a 152k vocabulary. Then across **layers 20–22** it
snaps into place: rank collapses to ~0, probability jumps past 0.9. The answer is assembled
in the last handful of layers.

One honest caveat, and it matters for everything after: the logit lens is
**correlational**. It tells you where the answer becomes *readable through the unembedding*,
not where the model actually computes it. The real work — moving operands around, resolving
carries — could happen earlier and only get rotated into the readable subspace late. To
find out where the computation *causally* lives, you have to intervene.

## Following the operands: activation patching

Activation patching is the workhorse. The setup is a **denoising** patch: run a *corrupted*
prompt (I swap operand `a` for a different 3-digit number, keeping `b` fixed), then splice
the *clean* residual stream back in at a single (layer, position) and measure how much of
the correct answer is restored. Restoration near 1 means "the operand information this
location holds is sufficient to produce the right sum"; near 0 means "irrelevant." The
metric is the logit difference between the clean and corrupted leading digits:

```python
def metric(logits):                       # at the '=' position
    last = logits[:, -1, :]
    return last[idx, clean_id] - last[idx, corr_id]

def hook(act, hook, P=P, clean_act=clean_act):
    act[:, P, :] = clean_act[:, P, :]     # patch clean into corrupt at position P
    return act
```

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 280" role="img" aria-label="Activation patching: splice a clean activation into a corrupted run and measure how much of the answer returns" style="width:100%;height:auto;max-width:640px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="patchA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--accent)"/>
    </marker>
    <marker id="patchB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <!-- clean row -->
  <text x="52" y="52" text-anchor="end" style="font-size:12px;fill:var(--text-muted)">clean</text>
  <g style="font-family:'JetBrains Mono',monospace;font-size:13px;fill:var(--text-muted)">
    <g style="fill:var(--code-bg);stroke:var(--border)"><rect x="74" y="32" width="30" height="30" rx="4"/><rect x="114" y="32" width="30" height="30" rx="4"/><rect x="154" y="32" width="30" height="30" rx="4"/><rect x="234" y="32" width="30" height="30" rx="4"/><rect x="274" y="32" width="30" height="30" rx="4"/><rect x="314" y="32" width="30" height="30" rx="4"/><rect x="374" y="32" width="30" height="30" rx="4"/></g>
    <g style="fill:var(--text-muted)"><text x="89" y="52" text-anchor="middle">4</text><text x="129" y="52" text-anchor="middle">5</text><text x="169" y="52" text-anchor="middle">7</text><text x="209" y="52" text-anchor="middle">+</text><text x="249" y="52" text-anchor="middle">3</text><text x="289" y="52" text-anchor="middle">8</text><text x="329" y="52" text-anchor="middle">6</text><text x="389" y="52" text-anchor="middle">=</text></g>
  </g>
  <text x="430" y="52" style="font-size:11px;fill:var(--text-faint)">→ cache every activation</text>
  <!-- corrupted row -->
  <text x="52" y="192" text-anchor="end" style="font-size:12px;fill:var(--text-muted)">corrupted</text>
  <g style="font-family:'JetBrains Mono',monospace;font-size:13px">
    <g style="fill:var(--code-bg);stroke:var(--text-faint)"><rect x="114" y="172" width="30" height="30" rx="4"/><rect x="154" y="172" width="30" height="30" rx="4"/></g>
    <g style="fill:var(--code-bg);stroke:var(--border)"><rect x="234" y="172" width="30" height="30" rx="4"/><rect x="274" y="172" width="30" height="30" rx="4"/><rect x="314" y="172" width="30" height="30" rx="4"/><rect x="374" y="172" width="30" height="30" rx="4"/></g>
    <!-- spliced position -->
    <rect x="74" y="172" width="30" height="30" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.6;stroke-dasharray:3 2"/>
    <g style="fill:var(--text-muted)"><text x="129" y="192" text-anchor="middle">1</text><text x="169" y="192" text-anchor="middle">9</text><text x="209" y="192" text-anchor="middle">+</text><text x="249" y="192" text-anchor="middle">3</text><text x="289" y="192" text-anchor="middle">8</text><text x="329" y="192" text-anchor="middle">6</text><text x="389" y="192" text-anchor="middle">=</text></g>
    <text x="89" y="192" text-anchor="middle" style="fill:var(--accent)">4</text>
  </g>
  <!-- splice arrow -->
  <line x1="89" y1="64" x2="89" y2="170" style="stroke:var(--accent);stroke-dasharray:4 3" marker-end="url(#patchA)"/>
  <text x="102" y="112" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--accent)">splice clean activation</text>
  <text x="102" y="128" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">at one (layer, position)</text>
  <!-- metric -->
  <line x1="405" y1="187" x2="452" y2="187" style="stroke:var(--text-faint)" marker-end="url(#patchB)"/>
  <rect x="456" y="170" width="216" height="34" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="564" y="191" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">restoration = logit diff at '='</text>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Activation patching (denoising): run the corrupted prompt, splice one clean activation back in at a single (layer, position), and measure how much of the correct answer returns at '='. Sweeping every cell maps where the operand information causally lives.</figcaption>
</figure>

Sweeping every (layer × position) gives a causal map. I keep `b` fixed between the clean and
corrupted prompts so that the only thing changing is `a` — which, as I'll get to below, turns
out to be less of a clean control than I first assumed.

<figure style="margin:1.75rem 0">
  <img src="/figures/patching_resid.png" alt="Residual-stream patching. Operand information waits at its own token positions through layer ~16, then hands off to the '=' position from layer 17 on." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Residual-stream patching. Operand information waits at its own token positions through layer ~16, then hands off to the '=' position from layer 17 on.</figcaption>
</figure>

The map has a clean **L-shape**, and it tells a precise story:

- **Layers 0–16 — the operand waits at its own position.** Patching clean `a` at its
  hundreds-digit position restores ~0.72–0.84; the tens digit ~0.15; the units peaks around
  0.03 in the middle layers and is ~0 from layer 9 on. (The hundreds digit dominates because
  we're scoring the *leading* answer digit — it's the most sensitive input.) The information
  just sits there.
- **A sharp handoff between layer 16 and 17.** At the `=` position, restoration jumps
  **0.08 → 0.99** — and *simultaneously* the operand columns collapse to ~0. After layer 17
  the `=` position **alone** determines the answer; the original operand tokens become
  causally irrelevant.

Here's the inference I like most: information can only move *between* token positions via
**attention** (MLPs act position-by-position). So the operand fetch — reading the digits off
their positions and writing them onto `=` — has to be attention, and it has to happen in
**layer 16**, because `=` is insufficient at the input to layer 16 (0.08) and sufficient at
the input to layer 17 (0.99). That's a falsifiable, specific claim, and the next experiment
tests it.

What about `b`? I originally set this up thinking `b`'s positions were a free control — `b`
is identical in the clean and corrupted prompts, so patching there should do nothing. That
reasoning is wrong, and the plot shows why. Attention is causal and `b` comes *after* `a`, so
`b`'s residual quietly absorbs information about `a`. Patching there doesn't inject nothing;
it injects whatever `a`-information has leaked into `b`'s positions. And that's exactly what
you see: `b`'s units position sits at 0 through layer 1, climbs to **0.064 by layer 13**, then
collapses back to ~0 at layer 15 — right when layer 16 goes and re-reads the operands from
their own positions. It's a small effect, but it's a measurement, not a control. The real
control here is the *shape* of the map: the restoration is concentrated in `a`'s digits and
then in `=`, in that order, which is what a routing story predicts and what a diffuse
"everything matters a bit" story does not.

## Naming the components

Patching whole residual streams tells you *where*; to get *which components*, you patch the
attention output and the MLP output separately, layer by layer, at `=`.

<figure style="margin:1.75rem 0">
  <img src="/figures/m4_components.png" alt="Left: attention contributes only at layer 16 (the fetch); MLPs do the computation across layers 16-21, peaking at 20. Right: per-head patching localizes the fetch to a few layer-16 heads." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Left: attention contributes only at layer 16 (the fetch); MLPs do the computation across layers 16–21, peaking at 20. Right: per-head patching localizes the fetch to a few layer-16 heads.</figcaption>
</figure>

The division of labour is astonishingly clean:

- **Attention's only causal job is a single fetch at layer 16.** Patching attention output
  restores 0.99 at layer 16 and ~0 at *every other layer*.
- **The MLPs do all the computing**, spread across layers 16–21 and peaking at layer 20
  (+0.70). **Attention routes; MLPs compute.** This is exactly the picture Stolfo et al.
  reported for GPT-2 and Pythia — now localized, layer by layer, on a 0.5B model.

And when you go one level finer, to individual heads, the fetch turns out to be done by a
small set of layer-16 heads that are **specialized by digit place** — you can see it in
where each one attends from the `=` position:

| Head       | Attends to           | Role            | Restoration |
|------------|----------------------|-----------------|------------:|
| **L16.H11**| operand `a` hundreds (0.83) | hundreds fetch | **+0.49** |
| L16.H0     | `a` hundreds (0.77)  | hundreds fetch  | +0.04 |
| L16.H8     | `b` hundreds (0.77)  | hundreds fetch  | +0.05 |
| L16.H6     | `a` tens (0.79)      | tens fetch      | +0.11 |
| L16.H10    | `b` tens (0.55)      | tens fetch      | +0.09 |
| L14.H13    | `b` units (0.78)     | units fetch     | +0.03 |

Different heads copy the hundreds, tens, and units digits, each onto the `=` position, where
the MLP stack adds them up. The hundreds-fetchers dominate the restoration score — which
makes sense, because we're scoring the *leading* answer digit, and that's what the hundreds
feed. So far, so tidy. Then I went after the carry, and it got humbling.

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 400" role="img" aria-label="The discovered addition circuit: layer-16 heads fetch each digit place onto the equals position, where the MLP stack computes the answer" style="width:100%;height:auto;max-width:600px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="circA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--accent)"/>
    </marker>
    <marker id="circB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <!-- output -->
  <text x="400" y="30" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">output token</text>
  <rect x="360" y="40" width="80" height="42" rx="8" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="400" y="69" text-anchor="middle" style="font-size:22px;font-weight:600;font-family:'JetBrains Mono',monospace;fill:var(--accent)">8</text>
  <text x="452" y="66" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">→ 843</text>
  <!-- mlp -> output -->
  <line x1="400" y1="120" x2="400" y2="84" style="stroke:var(--text-faint)" marker-end="url(#circB)"/>
  <!-- MLP stack -->
  <rect x="362" y="122" width="76" height="86" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
  <g style="stroke:var(--border)"><line x1="362" y1="150" x2="438" y2="150"/><line x1="362" y1="178" x2="438" y2="178"/></g>
  <text x="400" y="142" text-anchor="middle" style="font-size:12px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">MLPs</text>
  <text x="400" y="169" text-anchor="middle" style="font-size:10px;fill:var(--text-faint)">layers</text>
  <text x="400" y="197" text-anchor="middle" style="font-size:10px;fill:var(--text-faint)">16–21</text>
  <text x="452" y="160" style="font-size:12px;fill:var(--text-muted)">compute</text>
  <text x="452" y="177" style="font-size:12px;fill:var(--text-faint)">the sum</text>
  <!-- band -> mlp -->
  <line x1="400" y1="232" x2="400" y2="208" style="stroke:var(--text-faint)" marker-end="url(#circB)"/>
  <!-- attention band -->
  <rect x="55" y="234" width="430" height="44" rx="8" style="fill:var(--accent-soft);stroke:var(--accent)"/>
  <text x="250" y="255" text-anchor="middle" style="font-size:12px;fill:var(--accent)">layer 16 · attention</text>
  <text x="250" y="270" text-anchor="middle" style="font-size:10px;fill:var(--text-muted)">digit-place fetch heads (L16.H11, H0, H6 …)</text>
  <!-- fetch arrows from digits up to '=' column -->
  <g style="stroke:var(--accent);opacity:0.7">
    <line x1="90" y1="320" x2="392" y2="280" marker-end="url(#circA)"/>
    <line x1="130" y1="320" x2="394" y2="280" marker-end="url(#circA)"/>
    <line x1="170" y1="320" x2="396" y2="280" marker-end="url(#circA)"/>
    <line x1="250" y1="320" x2="404" y2="280" marker-end="url(#circA)"/>
    <line x1="290" y1="320" x2="406" y2="280" marker-end="url(#circA)"/>
    <line x1="330" y1="320" x2="408" y2="280" marker-end="url(#circA)"/>
  </g>
  <line x1="400" y1="322" x2="400" y2="280" style="stroke:var(--accent)" marker-end="url(#circA)"/>
  <!-- token row -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:14px">
    <g style="fill:var(--code-bg);stroke:var(--border)"><rect x="75" y="322" width="30" height="30" rx="4"/><rect x="115" y="322" width="30" height="30" rx="4"/><rect x="155" y="322" width="30" height="30" rx="4"/><rect x="235" y="322" width="30" height="30" rx="4"/><rect x="275" y="322" width="30" height="30" rx="4"/><rect x="315" y="322" width="30" height="30" rx="4"/></g>
    <rect x="385" y="322" width="30" height="30" rx="4" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <g style="fill:var(--text-muted)"><text x="90" y="343" text-anchor="middle">4</text><text x="130" y="343" text-anchor="middle">5</text><text x="170" y="343" text-anchor="middle">7</text><text x="209" y="343" text-anchor="middle" style="fill:var(--text-faint)">+</text><text x="250" y="343" text-anchor="middle">3</text><text x="290" y="343" text-anchor="middle">8</text><text x="330" y="343" text-anchor="middle">6</text></g>
    <text x="400" y="343" text-anchor="middle" style="fill:var(--accent)">=</text>
  </g>
  <g style="font-size:11px;fill:var(--text-faint)">
    <text x="130" y="372" text-anchor="middle">operand a</text>
    <text x="290" y="372" text-anchor="middle">operand b</text>
  </g>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">The circuit, assembled: layer-16 attention heads — each specialized by digit place — fetch the operand digits onto the '=' position, where the MLP stack (layers 16–21) adds them and writes the leading digit. Attention routes, MLPs compute.</figcaption>
</figure>

## The carry: where it gets interesting (and humbling)

The carry is the only genuinely hard part. `9+7` in the units column has to reach up and
add one to the tens column. Where does that happen? My plan was simple: train linear probes
to read each carry bit off the residual stream at `=`, layer by layer, and watch where each
one "turns on."

<figure style="margin:1.75rem 0">
  <img src="/figures/m5_carry_probes.png" alt="Carry probes at '='. All carries are decodable early (a confound), but after the layer-16 fetch the leading-digit carries c2/c3 sharpen while c1 (not needed yet) decays." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Carry probes at '='. All carries are decodable early (a confound), but after the layer-16 fetch the leading-digit carries c2/c3 sharpen while c1 (not needed yet) decays.</figcaption>
</figure>

The naive reading of this plot is a trap, and falling into it taught me something. **All
three carries are linearly decodable from layer ~1**, at ~90%+. If you stopped here you'd
announce "the model computes carries immediately!" — and you'd be wrong. It's a confound:
layer-0/1 attention smears the operand digits onto *every* position including `=`, and a
carry bit is just a threshold on a digit sum (`a_units + b_units ≥ 10`), so it's trivially
readable *wherever the digits happen to be present*. **A probe shows information is present,
not that it's used.** This reconciles perfectly with the patching result: that early `=`
copy is causally inert, because the layer-16 fetch re-reads the operand positions and
overwrites `=`.

The *real* signal is not the height of the curves but what happens at the fetch. After
layer 16:

- **c2** (tens→hundreds) — the carry the *leading* digit needs — sharpens to **0.97**.
- **c1** (units→tens) — needed only for the *tens* digit, which is emitted later, at a
  different position — **decays to ~0.75**.

(I also probed **c3**, the hundreds→thousands carry, and it climbs to 0.99 — but I've stopped
quoting it as evidence, because it's very nearly circular. In this dataset operands run
100–999, so `c3 = 1` is the same event as "the sum is four digits," which is the same event as
"the leading digit is a `1`." A probe reading c3 off the late layers is reading the model's
own output decision, not an independent carry. c2 is the honest evidence here.)

So at `=` the model computes exactly the carry relevant to the digit it's about to emit, and
lets the irrelevant one fade. That's a signature of **position-local, most-significant-first
carry computation**: it doesn't resolve the whole sum's carry structure at `=`, only what the
leading digit requires.

I tried to confirm the position-locality directly by teacher-forcing the full answer
(`457+386=843`) and probing each carry at each *output* position. The prediction: c1 should
become sharp at the position that emits the *tens* digit, where it's finally needed. It does
— c1 climbs back to ~1.0 there — but I have to be honest that once the answer digits are in
context they leak the carries, so those answer-position probes are confounded too. The clean
part of the signal remains the c1-vs-c2 divergence *at* `=`.

## Then the causal test — and the null

Probes are correlational. To claim the model *uses* a carry representation, you have to
intervene on it. So I took the c2 "direction" the probe had found (its readout weight vector,
mapped back into residual space), and **ablated** it from the residual at `=` across layers
16–23 — projecting it out, mean-centering the component. If that direction is the causal
carry signal, removing it should selectively break the cases that *have* a carry (c2=1),
pushing the leading digit off by one (the model "forgets to carry").

<figure style="margin:1.75rem 0">
  <img src="/figures/m6_c2_ablation.png" alt="Ablating the c2 direction does not selectively break carry cases: both groups drop about equally." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Ablating the c2 direction does not selectively break carry cases: both groups drop about equally.</figcaption>
</figure>

It didn't work. Ablating the c2 direction barely moved anything, and it hit the **no-carry
control slightly harder** than the carry cases (−0.040 vs −0.035), with only 8% of carry cases
going off-by-one. A flat **null**: whatever that direction is, the model's leading digit does
not depend on it.

My first instinct was to write this up as "the carry is distributed and redundant, so no
single direction is load-bearing." That's *a* reading, and it's the one that flatters the
result — but thinking about the setup, the experiment can't actually establish it. The
ablation happens at `=`, **after** layer 16 has already deposited all six operand digits
there. The quantity that *determines* c2 — the tens digits of both operands — is still sitting
in the residual stream, untouched. So the MLPs downstream are free to just recompute the carry
from its inputs. Deleting a readout while leaving its inputs intact is a test that fails
whether or not the feature is real.

So the honest conclusion is narrower than I wanted: **this ablation cannot isolate the carry.**
"The carry is smeared across many directions" and "the carry is recomputed downstream of where
I ablated" both predict exactly what I measured, and this experiment doesn't separate them.
What it does rule out is the tidy story I went in with — a single linear carry feature that the
leading digit reads off and depends on. The right next test is a *carry-controlled* patch:
minimal pairs whose tens digits differ only in whether they trigger a carry, holding the
hundreds digits fixed, so the carry's causal effect can be isolated from the operand values
that generate it.

Either way, the contrast with the routing result stands. The fetch was easy to intervene on
and behaved exactly as predicted; the carry resisted the same style of intervention. That
asymmetry is real even if my explanation for it isn't settled.

## Is it real, or an artifact of one prompt?

A fair objection to everything above: it all used one fixed 6-shot prompt format. Maybe the
"layer-16 fetch" is an accident of that exact prompt. So I re-ran the residual patch under
different formats — 3-shot, the canonical 6-shot, a 6-shot with completely different
exemplars, and 10-shot — and looked at the `=` restoration curve.

<figure style="margin:1.75rem 0">
  <img src="/figures/m6_robustness.png" alt="The layer-16 fetch is format-invariant: all four prompt formats' '=' curves overlap and jump at the same place." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">The layer-16 fetch is format-invariant: all four prompt formats' '=' curves overlap and jump at the same place.</figcaption>
</figure>

The curves lie almost exactly on top of each other, and the fetch appears at **layer 17 in
every case** (i.e., layer-16 attention writes it; it's readable at the input to 17). The
routing circuit is a stable property of the model, not of the prompt.

## Does it use a Clock?

Back to the question I opened with. Bigger models — GPT-J, Pythia-6.9B, Llama-8B — are
reported to represent numbers on a *helix*: a magnitude line wound with circles at periods
2, 5, 10, and 100, added by literally rotating it (the "Clock" algorithm). Does a 0.5B model
do anything so tidy?

There's a wrinkle first. Those results lean on models with a *single token per number*, whose
embedding row you can Fourier-analyze directly. Qwen tokenizes one digit per token, so there's
no embedding for "42." So I look in two places — the raw `0`–`9` digit embeddings, and the
residual stream at the last digit of a number (which has attended to all its digits) as a
stand-in for the value — and for each I measure how much variance a `[cos, sin]` pair at each
period explains:

```python
def fourier_power(X, N, periods):        # X: reps, N: their integer values
    Xc = X - X.mean(0, keepdims=True)
    out = {}
    for T in periods:
        ang = 2 * np.pi * N / T
        B = np.stack([np.cos(ang), np.sin(ang)], 1)   # a circle at period T
        out[T] = proj_energy(Xc, B)                    # variance explained
    return out
```

<figure style="margin:1.75rem 0">
  <img src="/figures/m7_fourier_spectrum.png" alt="Number reps carry Fourier features at T=2,5,10 — but only in the embedding; they wash out by layer 2 for context-free numbers." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Number reps carry Fourier features at T=2,5,10 — but only in the embedding; they wash out by layer 2 for context-free numbers.</figcaption>
</figure>

Good news for the Clock camp, and it's stronger than I expected. In the embedding (layers 0–1),
the number representations light up at exactly **T = 2, 5, and 10** — 0.30, 0.18 and 0.08 of
the variance, against ≤0.014 at every other period I tried. Those are precisely the periods the
big-model papers report. And the raw digit embeddings are even more periodic: **T = 10 explains
0.39 of their variance, more than the linear magnitude component (0.24).** Qwen's number
embeddings really are built from circular ingredients.

(One caveat on those numbers, in fairness: a period is fit with a `[cos, sin]` pair — two free
parameters — while "linear" gets one. The periodic terms are flattered by the extra degree of
freedom, so read "T10 beats linear" as suggestive, not as a clean win.)

Bad news: the structure doesn't survive. By layer 2 the periodic power — and the linear
magnitude component with it — collapses to almost nothing for context-free numbers, with only
a faint echo in the last few layers. And the ten digit embeddings, projected onto their top two
components, trace a *magnitude-ordered arc* (0 at one end, 9 at the other) rather than closing
into a ring:

<figure style="margin:1.75rem 0">
  <img src="/figures/m7_digit_ring.png" alt="The ten digit embeddings sit on a magnitude-ordered curve rather than closing into a ring." style="width:100%;height:auto;display:block;border-radius:8px;background:#fff;padding:0.6rem;border:1px solid var(--border)" />
  <figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">The ten digit embeddings sit on a magnitude-ordered curve rather than closing into a ring.</figcaption>
</figure>

Those last two observations pull in opposite directions and I want to be straight about that:
T10 carries the most variance of any single component, yet the top-2 PCA doesn't close into a
circle. Both can hold at once — 39% is plenty to be real and still leave the first two
principal components dominated by magnitude plus a partial arc — but it means the picture is
"strongly periodic ingredients, not an obvious clock face," not "no periodicity."

So my honest read: **the Clock's ingredients are clearly present in the embeddings, but I found
no evidence of a multi-digit Clock being maintained through the computation** in this small
model. That fits everything else — a carry I couldn't isolate, MLPs that smear the work across
six layers — and leans toward the "bag of heuristics" picture rather than the crisp trigonometry
the 6–8B models show.

The load-bearing caveat, though, is the same trap as before: reading *bare, out-of-context*
numbers at their last token is a weak probe — that residual is mostly busy predicting the next
token, not preserving a value. The features could be alive and used *during addition* and
invisible to this measurement. Settling it needs an in-context probe (read the operands while
the model is actually adding) and a causal Fourier-ablation. That's the honest boundary of
what this shows: the periodic structure exists at the input; whether Qwen-0.5B *computes* with
it is still open.

## How this fits the literature

Three findings line up with, and one pushes against, prior work:

- **"Attention routes, MLPs compute"** replicates Stolfo et al.'s causal-mediation picture
  from GPT-2/Pythia — here pinned down component-by-component on a much smaller model.
- **Digit-place-specialized fetch heads** are a concrete, model-specific version of the
  "digit-by-digit" view that's been showing up in recent work; I can name the individual
  layer-16 heads and the digit each one grabs.
- The **MSB-first, position-local carry** behavior is, as far as I know, a less-reported
  angle, and it falls naturally out of the fact that the model emits the answer big-end-first.
- The **carry-ablation null** and the **absent multi-digit Clock** both lean toward Nikankin et
  al.'s "bag of heuristics" rather than a crisp algorithm — with the caveat that neither is a
  positive demonstration of distributed computation, only a failure to find the tidy
  alternative. The periodic features that power the "Clock/Fourier algorithm" in Kantamneni &
  Tegmark and Zhou et al.'s 6–8B models are unmistakably present in Qwen's *embeddings*; I just
  can't show they survive into its computation. Whether that gap is about model size, or about
  my probes being too weak to see an in-context Clock, is the open question I'd most want to
  close.

## The setup, in one place

| Setting                 | Value                                             |
|-------------------------|---------------------------------------------------|
| model                   | Qwen2.5-0.5B (base), 24 layers, `d_model` 896     |
| heads / head dim / MLP  | 14 / 64 / 4864                                     |
| tokenization            | one digit per token; no BOS                        |
| framework               | TransformerLens 3.7, fp32, single RTX 3070 (8 GB)  |
| task                    | 3-digit + 3-digit addition, fixed few-shot prefix  |
| focal position          | the `=` token (predicts the leading digit)         |
| patching metric         | logit diff (clean vs corrupted leading digit)      |
| capability              | 98.7% (2-digit) / 94.7% (3-digit), 8-shot          |

## What I'd change

- **A real causal handle on the carry.** The linear-direction ablation was the obvious test and
  it was the wrong one — it left the carry's inputs sitting in the residual stream for the MLPs
  to recompute from. The carry-controlled minimal pairs described above are the fix, ideally
  with SAE/transcoder features standing in for the raw probe direction.
- **Error bars.** Every number here is a single run with a single seed. Nothing in the story
  hinges on a marginal effect, but the probe accuracies in particular deserve multiple splits.
- **A second model.** Everything here is one model. Does the same layer-fraction fetch +
  digit-place heads appear in Pythia or Llama-3.2-1B? Universality is the interesting question.
- **Zero-shot and non-digit-aligned tokenizers.** The clean single-digit tokenization made
  this tractable; models that chunk digits (Llama-3) would test how much the story depends on it.
- **A proper Clock test.** I found the Fourier features sitting in the embeddings but couldn't
  confirm the model *uses* them — the context-free probe was too weak. The real test is an
  in-context operand probe (read the numbers while it's adding) plus a causal Fourier-ablation.

## What it taught me

Building models teaches you what the pieces *are*. Taking one apart teaches you what they're
*for* — and how easily you can lie to yourself about it. The satisfying part of this project
was the routing circuit: a single attention layer, a few heads each grabbing one digit place,
an MLP stack that adds. It's a clean little machine and the interventions confirm it. The
humbling part was the carry, twice over. First I had a probe that looked like it was reading a
carry and was mostly reading operands that happened to be lying around — the probe-vs-causal
gap the field keeps warning about, walked straight into. Then I designed an ablation to settle
it, got a clean null, and started writing it up as "the carry is distributed" before noticing
that my experiment couldn't distinguish that from "the model just recomputes the carry from
inputs I left untouched." The null was real; my first explanation of it wasn't earned.

What I'm left with is an asymmetry rather than a complete story. The routing half of this
circuit is legible and behaves under intervention exactly as predicted. The carry half defeated
two attempts to pin it down, and the most useful thing I can say is precisely *how* they
failed, so the next test can be better designed. That's less satisfying than a clean answer,
but writing down the version I could actually defend — instead of the version that made a
better story — is the part of this project I'd want to keep.
