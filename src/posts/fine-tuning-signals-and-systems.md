---
title: "Fine-tuning a 1.5B param model on signals and systems"
date: 2026-08-18
summary: A LoRA fine-tune of Qwen2.5-Math-1.5B on undergraduate signals and systems problems 
tags: [LoRA, fine-tuning, evaluation]
---

In this project I wanted to see if I can improve a small open-weight model`s
performance on a complicated subject like undergraduate level Signals & Systems 
so I finetuned Qwen2.5-Math-1.5B on a small synthetic dataset and then evaluated it
on problems from Oppenheim and Willsky (our classes reference)

Code is on
[GitHub](https://github.com/sadra-etaei/S-S_finetuning).

## The setup

The plan was standard rejection-sampling fine-tuning (RFT, sometimes STaR) :

1. take problems with checkable answers,
2. sample k solutions from the model at temperature,
3. keep only the traces that reach the correct answer,
4. LoRA on the survivors.

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 170" role="img" aria-label="Rejection-sampling fine-tuning pipeline: sample k solutions, keep only the verifier-approved ones, LoRA on the survivors" style="width:100%;height:auto;max-width:660px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <defs>
    <marker id="rft" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" style="fill:var(--text-faint)"/>
    </marker>
  </defs>
  <g style="font-family:'Inter',sans-serif">
    <!-- box 1 -->
    <rect x="10" y="48" width="150" height="60" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="85" y="72" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">problems with</text>
    <text x="85" y="90" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">checkable answers</text>
    <!-- box 2 -->
    <rect x="185" y="48" width="150" height="60" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="260" y="72" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">sample k solutions</text>
    <text x="260" y="90" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">k = 4 · T = 0.8</text>
    <!-- box 3 (verifier) -->
    <rect x="360" y="48" width="150" height="60" rx="8" style="fill:var(--accent-soft);stroke:var(--accent);stroke-width:1.5"/>
    <text x="435" y="72" text-anchor="middle" style="font-size:12px;fill:var(--accent)">verifier</text>
    <text x="435" y="90" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">keep only correct</text>
    <!-- box 4 -->
    <rect x="535" y="48" width="150" height="60" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="610" y="72" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">LoRA on</text>
    <text x="610" y="90" text-anchor="middle" style="font-size:12px;fill:var(--text-muted)">the survivors</text>
    <!-- arrows -->
    <g style="stroke:var(--text-faint)">
      <line x1="160" y1="78" x2="184" y2="78" marker-end="url(#rft)"/>
      <line x1="335" y1="78" x2="359" y2="78" marker-end="url(#rft)"/>
      <line x1="510" y1="78" x2="534" y2="78" marker-end="url(#rft)"/>
    </g>
    <!-- step numbers -->
    <g style="font-size:10px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">
      <text x="85" y="38" text-anchor="middle">1</text>
      <text x="260" y="38" text-anchor="middle">2</text>
      <text x="435" y="38" text-anchor="middle">3</text>
      <text x="610" y="38" text-anchor="middle">4</text>
    </g>
    <text x="435" y="132" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">a sample becomes training data if a function says it's correct</text>
  </g>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Rejection-sampling fine-tuning: the model learns only from its own successful attempts, so the entire method rests on the verifier in the middle.</figcaption>
</figure>

The model never sees a reasoning trace written by a human. It learns from its own
successful attempts. The appeal is that you never author a single derivation —
but it puts the entire weight of the method on one component : **the verifier**.
An answer becomes training data if and only if a function says it's correct.
There's no human in the loop, no reward model, nothing else.

## Why I generated the problems instead of using the textbook

I had the textbook and its solutions manual as PDFs, and my first instinct was to
extract problems from them. Both turned out to be unusable, in ways worth knowing
before you try the same thing :

**The solutions manual** is a 116-page scan. One image per page, four book pages
per scan, skewed, no text layer at all. It's also partial cover for a 987-page
book, and its solutions are mostly derivations and "show that" proofs rather than
final answers a program could check.

**The textbook** has an OCR text layer, but the maths is mangled. `e-5t` for
$e^{-3t}$. `J` for the integral sign. `{3` for β. Extracted problems would need
per-problem repair by a model, and would still be a commercial work reproduced
into a training set.

So I wrote a generator instead : 31 problem families across 8 chapters, ~5,300
problems, where the answer is computed symbolically with sympy rather than
written by hand. Laplace and z-transforms with regions of convergence,
convolution, Fourier series and transforms, sampling and aliasing, LCCDEs, system
property classification. The PDFs got used for exactly one thing — reading the
chapter structure to check my coverage against the real syllabus.

## The verifier is the load-bearing part

Comparing two answers in this domain is not string matching. `\frac{1}{s+3}` and
`(s+3)**(-1)` are the same. `Re{s} > -3` and `\text{Re}(s) > -3` are the same.
`e^{-2t}` and `e^{-2t}u(t)` are **not** — they differ for t < 0, and that
difference is the entire concept of causality.

Two domain traps a generic maths verifier gets wrong :

**`j` is the imaginary unit**, not a free symbol. Every signals text writes
`1/(2+jw)`. A parser that treats `j` as a variable silently accepts nonsense.

**`u[0]` must equal 1.** sympy's `Heaviside(0)` defaults to `1/2`. In continuous
time that's a measure-zero set and nobody cares. In discrete time n=0 is a real
sample point, so every DT comparison quietly breaks :

```python
def ud(x):
    """Discrete unit step; u[0] = 1, not 1/2."""
    return sp.Heaviside(x, 1)
```

The comparison itself is **numeric first, symbolic second**, which is the
opposite of what I expected going in. `simplify()` on expressions full of
`Heaviside` and nested exponentials is slow and, worse, *inconclusive* — it
returns a non-zero difference for expressions that are equal, which silently
throws away correct training data. Sampling both expressions at ~14 random points
(straddling every step discontinuity, skipping poles) has a far lower
false-negative rate. Symbolic simplification only runs as a fallback when
sampling says "different".

Impulses get handled separately, because `DiracDelta` can't be sampled at all.
Terms containing it are split off, reduced with the sifting property
f(t)·δ(t−a) = f(a)·δ(t−a), and matched shift by shift. A nice consequence :
`t*DiracDelta(t)` correctly compares equal to `0`.

### The gate that caught five bugs before I trained anything

The single most useful thing I built was three lines long. Before writing any
dataset, feed every problem's **own ground truth** back through the verifier,
rendered exactly the way the prompt asks a model to render it :

```python
def roundtrip_ok(prob):
    rendered = prob.answer.pretty()
    v = check(prob.answer, f"\\boxed{{{rendered}}}")
    return v.ok, v.reason
```

If the verifier rejects the correct answer, the problem is dropped and reported.
On the first run this caught five real bugs :

- infinite ROC bounds never compared equal, because `oo - oo` is `nan`, so every
  half-plane region of convergence failed;
- a symbol-assumption mismatch — `Symbol('t')` and `Symbol('t', real=True)` are
  different objects, so substitution silently no-oped and *every* expression
  evaluated to something non-numeric;
- three notation contracts my own parser couldn't read back.

Without that gate all five would have shown up later as a mysteriously low eval
score, and I'd have blamed the model.

The mirror-image test matters just as much. Round-tripping only proves the
verifier accepts correct answers — a verifier hardcoded to `return True` passes
it. So the test suite also **corrupts** every family's ground truth (sign flip,
off-by-one, dropped factor) and asserts rejection.

## Measurement bug #1 : 12.5 points of "capability" that was a token budget

First baseline : **39.0%**. Squarely in the "green" band I'd pre-registered, so I
almost moved straight on to training.

Then I looked at the failures. 21% of completions never emitted `\boxed{}` at
all, and their tails looked like this :

```
...\frac{-8 \pm \sqrt{-196}}{2} = \frac{-8 \pm 14i
```

Cut off mid-expression. My `--max-new-tokens` default was 768, and the model was
running out of budget partway through long derivations. Splitting the run :

| | correct |
|---|---|
| completions that finished | 49.2% |
| completions that ran out of tokens | 1.2% |

Truncation wasn't spread evenly either — it hit `lti` (32.6%) and `ztransform`
(29.8%) hardest, the two topics with the longest derivations. Those were exactly
the topics that looked worst, and they looked worst largely because the model
never got to finish.

Re-running at 2048 tokens : **51.5%**, with truncation at 0.0%.

Twelve and a half points of apparent model weakness, and none of it was the
model. If you report an eval number without checking your truncation rate, this
bug is probably in your numbers too.

**The fix, generalized :** truncation, unparseable, and wrong are three different
findings and should never share a bucket. One is a generation-config bug, one is
a prompt-contract bug, and only the third tells you anything about the model. My
eval script now counts them separately.

## Round one : a result that looked good

With a clean baseline I ran the actual pipeline. 1,200 problems × k=4 samples,
filter through the verifier, 1,116 traces kept, LoRA rank 16 for 2 epochs.

Evaluated on 161 held-out problems — written by me in textbook prose, produced by
no generator family, so a gain there means something transferred rather than the
model memorizing my templates :

**68.9% → 75.2%, +6.2%.** McNemar p = 0.110.

Not significant, but the effect was *larger* on the prose set than on the
synthetic one (+3.0%), which is the opposite of the memorization signature. I
figured I just needed more eval problems for the power to resolve it. So I wrote
111 more, doubling the held-out set to 272.

## Measurement bug #2 :

The expanded run came back at **+2.9%, p = 0.291.** The effect had shrunk.

Worse — the *original 161 problems* now read **+3.7%** instead of +6.2%. Same
problems. Same two models. Greedy decoding, `do_sample=False`. Nothing about the
models had changed.

The cause is a cache bug I'd written : when the eval set grew, the cache-miss
path regenerated all 272 completions instead of just the 111 new ones. And
regenerating changed the answers. I checked directly :

```
same model, same prompts, greedy decoding, 200 problems
  identical completions : 77/200 = 38.5%
  DIFFERENT             : 123
```

**61.5% of greedy completions changed** because the batch size and padding
changed. Batched matmuls aren't associative — different batch composition shifts
the numerics enough to flip token choices, and a flipped token early in a
derivation changes everything after it.

The practical consequence is that my noise floor was about **2.5 points**, which
was *larger than the entire effect I'd been trying to measure* in round one.



## Why round one actually failed

The replication failure told me the +6.2% wasn't solid. It didn't tell me why the
method underperformed. For that I went back to the raw samples — 
and measured what rejection sampling actually had to work with.

First, the pass@k curve :

```
pass@1 = 43.8%        headroom = pass@4 - pass@1 = +18.6 points
pass@2 = 53.2%
pass@3 = 59.3%
pass@4 = 62.4%
```

So headroom wasn't the problem. There were 18.6 points of capability sitting
between what the model does reliably and what it can do occasionally, and
distilling that gap into pass@1 is exactly what RFT is for.

Then I split the problems by how often they were solved :

| | share | |
|---|---|---|
| solved 4/4 | 27.5% | already reliable — **nothing to teach** |
| solved 0/4 | 37.6% | unreachable — **no trace can ever be harvested** |
| solved 1–3/4 | 34.9% | **the only band RFT can act on** |

<figure style="margin:1.75rem 0">
<svg viewBox="0 0 700 250" role="img" aria-label="Of the k=4 outcomes per problem, only the 1-3 correct band is useful for rejection-sampling fine-tuning; solved-0 is unreachable and solved-4 has nothing to teach" style="width:100%;height:auto;max-width:640px;display:block;margin:0 auto;font-family:'Inter',sans-serif">
  <text x="336" y="24" text-anchor="middle" style="font-size:11px;fill:var(--text-faint)">how many of the k = 4 samples were correct →</text>
  <!-- buckets -->
  <g style="font-family:'JetBrains Mono',monospace;font-size:14px;text-anchor:middle">
    <rect x="90"  y="40" width="84" height="60" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="132" y="76" style="fill:var(--text-faint)">0/4</text>
    <rect x="192" y="40" width="84" height="60" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <text x="234" y="76" style="fill:var(--accent)">1/4</text>
    <rect x="294" y="40" width="84" height="60" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <text x="336" y="76" style="fill:var(--accent)">2/4</text>
    <rect x="396" y="40" width="84" height="60" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)"/>
    <text x="438" y="76" style="fill:var(--accent)">3/4</text>
    <rect x="498" y="40" width="84" height="60" rx="6" style="fill:var(--code-bg);stroke:var(--border)"/>
    <text x="540" y="76" style="fill:var(--text-faint)">4/4</text>
  </g>
  <!-- region brackets -->
  <g style="stroke:var(--border)">
    <line x1="90" y1="112" x2="174" y2="112"/>
    <line x1="498" y1="112" x2="582" y2="112"/>
  </g>
  <line x1="192" y1="112" x2="480" y2="112" style="stroke:var(--accent)"/>
  <text x="132" y="130" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">unreachable</text>
  <text x="132" y="146" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">37.6%</text>
  <text x="336" y="130" text-anchor="middle" style="font-size:11px;fill:var(--accent)">the useful band — RFT's only leverage</text>
  <text x="336" y="146" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-muted)">34.9%</text>
  <text x="540" y="130" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">nothing to teach</text>
  <text x="540" y="146" text-anchor="middle" style="font-size:11px;font-family:'JetBrains Mono',monospace;fill:var(--text-faint)">27.5%</text>
  <!-- punchline -->
  <rect x="90" y="176" width="492" height="52" rx="8" style="fill:var(--code-bg);stroke:var(--border)"/>
  <text x="336" y="198" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">Round 1 harvested <tspan style="fill:var(--accent);font-weight:600">45.5%</tspan> of its traces from the 4/4 dead zone —</text>
  <text x="336" y="216" text-anchor="middle" style="font-size:11px;fill:var(--text-muted)">teaching the model what it already did perfectly.</text>
</svg>
<figcaption style="font-family:var(--font-sans);font-size:0.82rem;color:var(--text-faint);text-align:center;margin-top:0.6rem;line-height:1.5">Rejection sampling can only convert <em>unreliable</em> into <em>reliable</em>: a problem solved in every sample has nothing left to teach, and one solved in none yields no trace at all. Only the 1–3 band is actionable.</figcaption>
</figure>

And then the number that explained everything. Of my 1,116 training traces,
**45.5% came from problems the model already solved 4 times out of 4.**

Almost half my training data was teaching the model to do things it already did
perfectly.


## The fix

Keep only problems in the 1..k−1 band, and up-weight the scarce hard ones :

```python
# rejection sampling converts *unreliable* into *reliable*. A problem solved in
# every sample has nothing left to teach; a problem solved in none yields nothing
# regardless. The useful band is 1..k-1.
if not (args.band_lo <= n_ok <= len(comps) - 1):
    reasons[f"dropped: outside band ({n_ok}/{len(comps)})"] += 1
    continue
```

Re-filtering the **same raw samples** — no new sampling, — changed the mix
completely :

| source problem | round 1 | round 2 |
|---|---:|---:|
| solved 4/4 (useless) | 45.5% | **0%** |
| solved 3/4 | 19.7% | 26.5% |
| solved 2/4 | 22.8% | 41.0% |
| solved 1/4 (hardest) | 12.0% | **32.5%** |

I also dropped to 1 epoch. Round one's final train loss was 0.085, which on
self-generated text is an overfitting signal rather than a learning one — the
model already assigns high probability to its own outputs, so there isn't much
gradient to begin with.

## Round two

Same base model, same raw samples, same eval, same everything except the filter
and the epoch count :

| run | base | tuned | delta | McNemar p |
|---|---|---|---|---|
| v1 — all correct traces, 2 epochs | 65.8% | 68.8% | +2.9% | 0.291 |
| **v2 — difficulty-banded, 1 epoch** | 65.8% | **72.8%** | **+7.0%** | **0.0043** |

The base completions are byte-identical between those two rows  — so the
two tuned models are measured against exactly the same reference.

Per topic :

| topic | base | v2 | delta | n |
|---|---|---|---|---|
| ztransform | 50.0% | 75.0% | **+25.0%** | 44 |
| ctfs | 60.0% | 70.0% | +10.0% | 20 |
| laplace | 68.1% | 76.6% | +8.5% | 47 |
| sampling | 66.7% | 74.1% | +7.4% | 27 |
| signals | 91.2% | 97.1% | +5.9% | 34 |
| ctft | 85.2% | 85.2% | 0.0% | 27 |
| dtft | 50.0% | 50.0% | 0.0% | 24 |
| lti | 59.2% | 55.1% | **−4.1%** | 49 |

The z-transform result carries it. That's satisfying, because a discrete-time
weakness was visible in the very first baseline — the model scored 100% on
`dy/dt + ay = bx → h(t) = be^{-at}u(t)` and 0% on the discrete twin
`y[n] − ay[n−1] = bx[n] → h[n] = b·aⁿu[n]`. Same concept, and it only broke in
discrete time.

## What actually changed

All ten below are from the held-out prose set — problems written by hand, in
textbook language, produced by no generator family. Eight the tuning fixed, two
it broke.

**1. Convergence is not the same as being defined.**

> **Determine the Laplace transform of $x(t) = \sin(5t)u(t)$, together with its region of convergence.**
>
> base : $\boxed{\frac{5}{s^2+25}; \text{Re}\{s\} > -\infty}$ ❌
>
> tuned : $\boxed{\frac{5}{s^2+25}; \Re(s) > 0}$ ✅

Both get the transform right. The base model then reasons that `s²+25` is
non-zero for all real s and concludes the region of convergence is the whole
plane. It's confusing *the algebraic expression is defined* with *the defining
integral converges*. This exact mistake appeared in six of the fixed problems.

<details><summary>chain of thought — base ❌</summary>

```text
To determine the Laplace transform of \( x(t) = \sin(5t)u(t) \), we will follow these steps:

1. **Identify the Laplace transform formula for \(\sin(\omega t)u(t)\):**
   The Laplace transform of \(\sin(\omega t)u(t)\) is given by:
   \[ \mathcal{L}\{\sin(\omega t)u(t)\} = \frac{\omega}{s^2 + \omega^2} \]

2. **Substitute \(\omega = 5\) into the formula:**
   \[ \mathcal{L}\{\sin(5t)u(t)\} = \frac{5}{s^2 + 5^2} = \frac{5}{s^2 + 25} \]

3. **Determine the region of convergence:**
   The Laplace transform \(\frac{5}{s^2 + 25}\) is valid for all \(s\) such that the
   denominator is non-zero. The denominator \(s^2 + 25\) is always positive for all
   real \(s\) because \(s^2 \geq 0\) and \(25 > 0\). Therefore, the Laplace transform
   is valid for all \(s\) in the complex plane. In terms of the real part of \(s\):
   \[ \text{Re}\{s\} \in (-\infty, \infty) \]

…(truncated — the base concludes Re{s} > -∞)
```

</details>

<details><summary>chain of thought — tuned ✅</summary>

```text
To determine the Laplace transform of \( x(t) = \sin(5t)u(t) \), we will follow these steps:

1. **Identify the Laplace transform of the function \(\sin(5t)u(t)\):**
   The Laplace transform of \(\sin(\omega t)u(t)\) is given by:
   \[ \mathcal{L}\{\sin(\omega t)u(t)\} = \frac{\omega}{s^2 + \omega^2} \]
   For \(\omega = 5\):
   \[ \mathcal{L}\{\sin(5t)u(t)\} = \frac{5}{s^2 + 25} \]

2. **Determine the region of convergence:**
   The denominator \(s^2 + 25\) is zero when \(s = \pm 5i\), which are complex numbers.
   Since the Laplace transform is defined for all \(s\) except these points, the region
   of convergence is:
   \[ \text{FOC: } \Re(s) > 0 \]

3. **Combine the results:**
   \[ \boxed{\frac{5}{s^2 + 25}; \Re(s) > 0} \]
```

</details>

**2. The unit step is part of the answer.**

> **Determine the step response of the system with impulse response $h(t) = e^{-5t}u(t)$.**
>
> base : $\boxed{\frac{1}{5}(1 - e^{-5t})}$ ❌
>
> tuned : $\boxed{\frac{1}{5}(1 - e^{-5t}) u(t)}$ ✅

The derivations are near-identical; the base just drops `u(t)`, asserting a step
response that is non-zero before the step arrives. A causality error, not a
formatting quibble — and the reason the verifier has to sample at negative t.

**3. Eigenfunctions, instead of convolving.**

> **An LTI system has $h(t) = 2e^{-3t}u(t)$. The input is the everlasting exponential $x(t) = e^{-t}$, present for all $t$. Determine $y(t)$.**
>
> base : $\boxed{e^{-t} - e^{-3t}}$ ❌
>
> tuned : $\boxed{e^{-t}}$ ✅

The base convolves as if the input switched on at t=0 and picks up a transient
that doesn't exist. The tuned model applies the eigenfunction property : an
everlasting exponential gives $H(s_0)e^{s_0 t}$, with $H(-1) = 2/(-1+3) = 1$.

**4. A cascade collapses to a single impulse.**

> **A system with impulse response $h_1[n] = (1/2)^n u[n]$ is cascaded with a second whose impulse response is $h_2[n] = \delta[n] - \tfrac{1}{2}\delta[n-1]$. Determine the overall impulse response.**
>
> base : $\left(\frac{1}{2}\right)^n u[n] - 1$ ❌
>
> tuned : $\left(\frac{1}{2}\right)^n \delta[n]$ ✅

Cascaded systems convolve, and here the second system is the inverse of the first, so the
whole chain collapses to $h_1 * h_2 = \delta[n]$. The base grinds through the convolution
and botches the boundary term; the tuned model reaches the clean cancellation — and
$(1/2)^n\delta[n] = \delta[n]$, since the impulse is non-zero only at $n=0$.

<details><summary>chain of thought — base ❌</summary>

```text
To determine the overall impulse response we convolve h_1[n] and h_2[n]:

\[ h[n] = h_1[n] * h_2[n] = \sum_{k=-\infty}^{\infty} h_1[k] h_2[n-k] \]

with h_1[n] = (1/2)^n u[n] and h_2[n] = δ[n] - (1/2)δ[n-1]. Splitting into two sums and
using the sifting property of δ[n-k]:

\[ h[n] = (1/2)^n u[n] - (1/2)(1/2)^{n-1} u[n-1] \]

For n = 0, u[-1] = 0 so the second term vanishes; for n < 0 both terms are zero...

…(truncated — the base ends at (1/2)^n u[n] - 1)
```

</details>

<details><summary>chain of thought — tuned ✅</summary>

```text
The overall impulse response is the convolution h[n] = h_1[n] * h_2[n]:

\[ h[n] = \sum_{k} (1/2)^k u[k] \left( δ[n-k] - (1/2) δ[n-k-1] \right) \]

Using the sifting property δ[n-k] -> f[n]:

\[ h[n] = (1/2)^n u[n] - (1/2)(1/2)^{n-1} u[n-1] \]

Simplifying the second term, (1/2)(1/2)^{n-1} = (1/2)^n, so the two terms cancel down to
a single impulse at n = 0...

…(truncated)
```

</details>

**5. Two-sided signals need an intersection, not a half-plane.**

> **Determine the region of convergence of the z-transform of $x[n] = (1/2)^n u[n] - 3^n u[-n-1]$.**
>
> base : $\boxed{|z| > \frac{1}{2}}$ ❌
>
> tuned : $\boxed{\frac{1}{2} < |z| < 3}$ ✅

The base handles the right-sided term and stops. The ROC of a sum is the
*intersection* of the terms' ROCs, and the left-sided term contributes |z| < 3 —
so the answer is an annulus. Forgetting the second constraint is the standard
z-transform exam mistake, and it's the topic that moved most (+25%).

**6. A remembered transform pair, with a spurious factor.**

> **The Fourier transform of a signal is $X(j\omega) = e^{-j\omega}/(3 + j\omega)$. Determine $x(t)$.**
>
> base : $\frac{1}{3} e^{-3(t-1)} u(t-1)$ ❌
>
> tuned : $e^{-3(t-1)} u(t-1)$ ✅

Both recognise the $e^{-j\omega}$ as a one-second time shift and both invert
$1/(3+j\omega)$. But the base misremembers the pair as $\tfrac{1}{3}e^{-3t}u(t)$, inserting
a $1/a$ factor that isn't there — the correct pair is $e^{-at}u(t)\leftrightarrow 1/(a+j\omega)$,
with no leading fraction.

<details><summary>chain of thought — base ❌</summary>

```text
To find the inverse Fourier transform of X(jω) = e^{-jω}/(3 + jω):

\[ X(j\omega) = \frac{1}{3 + j\omega} \cdot e^{-j\omega} \]

The inverse transform of 1/(3 + jω) is (1/3) e^{-3t} u(t) [sic], and e^{-jω} is a phase
shift corresponding to a one-unit time shift, δ(t - 1). Therefore:

\[ x(t) = \frac{1}{3} e^{-3(t-1)} u(t-1) \]

…(truncated)
```

</details>

<details><summary>chain of thought — tuned ✅</summary>

```text
To find the inverse Fourier transform of X(jω) = e^{-jω}/(3 + jω):

\[ X(j\omega) = \frac{1}{3 + j\omega} \cdot e^{-j\omega} \]

We recognise 1/(3 + jω) as the transform of e^{-3t} u(t):

\[ \mathcal{F}\{e^{-3t} u(t)\} = \frac{1}{3 + j\omega} \]

The factor e^{-jω} = e^{-jω·1} is a time shift by τ = 1, so:

\[ x(t) = e^{-3(t-1)} u(t-1) \]

\[ \boxed{e^{-3(t-1)} u(t-1)} \]
```

</details>

**7. rad/s is not Hz.**

> **A signal has no spectral content above 750 rad/s. Determine the largest sampling interval T, in seconds, permitting exact reconstruction.**
>
> base : $\boxed{\frac{1}{1500}}$ ❌
>
> tuned : $\boxed{\frac{\pi}{750}}$ ✅

The base treats 750 as though it were in hertz and writes 1/(2f). In rad/s the
answer is $T = \pi/\omega_m$. A factor of $2\pi$, and the kind of error that is invisible
unless you check units.

<details><summary>chain of thought — base ❌</summary>

```text
To determine the largest sampling interval T for a signal with no spectral content
above 750 rad/s, we use the sampling theorem: the sampling rate must be at least twice
the highest frequency component.

\[ f_s \geq 2 \times 750 = 1500 \]

The sampling interval is the inverse of the sampling rate:

\[ T = \frac{1}{f_s} = \frac{1}{1500} \text{ seconds} \]

\[ \boxed{\frac{1}{1500}} \]
```

</details>

<details><summary>chain of thought — tuned ✅</summary>

```text
...the highest frequency component is 750 rad/s, so the sampling rate must be at least
2 × 750 = 1500 rad/s. The sampling rate is 2π/T, so we need 2π/T ≥ 1500. Therefore the
largest sampling interval is:

\[ T = \frac{2\pi}{1500} = \frac{\pi}{750} \]

\[ \boxed{\frac{\pi}{750}} \]
```

</details>

**8. Reading a Fourier coefficient off the exponential form.**

> **For $x(t) = 2 + 3\cos(t)$ with fundamental frequency 1 rad/s, determine the Fourier series coefficient $a_{-1}$.**
>
> base : $3\pi$ ❌
>
> tuned : $\frac{3}{2}$ ✅

The clean route is to write $x(t) = 2 + \tfrac{3}{2}e^{jt} + \tfrac{3}{2}e^{-jt}$ and read
$a_{-1} = \tfrac{3}{2}$ straight off the $e^{-jt}$ term. The base instead sets up the
analysis integral and mishandles it, ending with a stray $3\pi$; the tuned model uses the
exponential decomposition and gets it in one line.

<details><summary>chain of thought — base ❌</summary>

```text
The Fourier coefficients are a_k = (1/T) ∫_0^T x(t) e^{-j k ω_0 t} dt, with ω_0 = 1 and
T = 2π. Splitting x(t) = 2 + 3cos(t):

\[ a_k = \frac{1}{2\pi}\int_0^{2\pi} 2 e^{-jkt} dt + \frac{1}{2\pi}\int_0^{2\pi} 3\cos(t) e^{-jkt} dt \]

The first integral is 0. Using cos(t) = (e^{jt}+e^{-jt})/2 in the second...

…(truncated — the base's integral evaluation drifts to 3π)
```

</details>

<details><summary>chain of thought — tuned ✅</summary>

```text
For a periodic signal, x(t) = Σ a_k e^{j k ω_0 t}. We need a_{-1}, the coefficient of
e^{-jt}. Writing x(t) = 2 + 3cos(t) in complex-exponential form:

\[ x(t) = 2 + \frac{3}{2} e^{jt} + \frac{3}{2} e^{-jt} \]

The coefficient of e^{-jt} is 3/2, so:

\[ a_{-1} = \frac{3}{2} \]

\[ \boxed{\frac{3}{2}} \]
```

</details>

### And what it broke

**9. Impulse response confused with step response.**

> **A causal system obeys $dy(t)/dt + 6y(t) = 2x(t)$. Determine $h(t)$.**
>
> base : $\boxed{2e^{-6t}u(t)}$ ✅
>
> tuned : $\boxed{\frac{1}{3}(1 - e^{-6t})u(t)}$ ❌

The tuned model produced the *step* response instead. Step responses were heavily
represented in the training traces, and here it reached for the wrong template.

**10. Over-applying a learned rule.**

> **A signal is bandlimited so that $X(jw) = 0$ for $|w| > 2000$ rad/s. State its Nyquist rate in rad/s.**
>
> base : $\boxed{4000}$ ✅
>
> tuned : $\boxed{8000}$ ❌

The answer is 2ω_m = 4000. The tuned model doubled it again — plausibly
importing the "squaring doubles the bandwidth" rule from the x²(t) problems in
its training set into a problem with no squaring in it.

<details><summary>chain of thought — base ✅</summary>

```text
The Nyquist rate is twice the highest frequency component in the signal. The signal is
bandlimited to |w| > 2000 rad/s, so the highest frequency is 2000 rad/s:

\[ f_s = 2 \times 2000 = 4000 \text{ rad/s} \]

\[ \boxed{4000} \]
```

</details>

<details><summary>chain of thought — tuned ❌</summary>

```text
The signal occupies -2000 to 2000 rad/s, so its bandwidth is 2000 - (-2000) = 4000 rad/s.
The Nyquist rate is twice the bandwidth of the signal:

\[ 2 \times 4000 = 8000 \text{ rad/s} \]

\[ \boxed{8000} \]
```

</details>

Both regressions are the same failure mode : the model learned a pattern and
fired it where it doesn't apply. That's worth knowing, because it's the specific
risk of training on a narrow generated distribution, and it argues for keeping
some general maths in the mix.

## Honest caveats

This is the second configuration I tried. v1 failed, I diagnosed it, three things
changed at once, and v2 worked. That makes the p-value slightly optimistic and
this is **not an ablation** — I can't attribute the gain to the band filter
alone. What keeps it from being a fishing expedition is that the change was
*predicted from a measured defect* (45.5% wasted traces) before the retrain, not
found by sweeping hyperparameters.

Splitting the held-out set by when I wrote it, since the extension was authored
after seeing round-1 results :

| split | delta | p |
|---|---|---|
| original 161 (written before) | +8.7% | 0.0066 |
| extension 111 (written after) | +4.5% | 0.332 |

Both point the same way, but the extension isn't individually significant. The
true effect is probably nearer +4–5% than +8.7%.

And **37.6% of problems remain unreachable.** No trace can be harvested for a
problem the model never solves, at any k. Self-sampling cannot break that ceiling
by construction.

## The numbers, in one place

| SETTING | VALUE | NOTE |
|---|---|---|
| base model | Qwen2.5-Math-1.5B-Instruct | 4-bit QLoRA |
| LoRA | r=16, α=32, dropout 0.05 | attention + MLP |
| training | 1 epoch, lr 1e-4, cosine | completion-only loss |
| SFT set | 1,238 traces / 371 problems | difficulty band 1–3 of 4 |
| sampling | k=4 at T=0.8, 1,200 problems | ~7.5 h on a laptop 3070 |
| eval (synthetic) | 795 held out from generators | measures task learning |
| eval (prose) | 272 hand-written | measures transfer |
| peak VRAM | 2.93 GB | 8 GB card, batch 1 × accum 16 |
| result | 65.8% → 72.8% | +7.0%, McNemar p = 0.0043 |

## What I'd change

**A teacher for the never-solved band.** 37.6% of problems produce no traces at
any k, and that's the ceiling self-sampling can't touch. Generating traces for
those with a larger model, filtered through the same verifier, is the one lever
that breaks it instead of working inside it.

**Ablate the v2 changes.** Band filter, hard-weighting, and epoch count moved
together. I don't actually know the split.

**Fix the `lti` regression.** It contributed the most traces (366) and was the
only topic that went backwards. That's a thread worth pulling.

**vLLM.** Sampling took 7.5 hours with HuggingFace `generate`. Continuous
batching would cut that by an order of magnitude, and at batch 24 I was already
getting 6.8× over batch 2 just from batching properly — autoregressive decoding
is memory-bandwidth-bound, so a small batch wastes almost the whole GPU.

**A bigger prose eval.** At n=272 the 95% interval is still ±5.9 points.


