# Teaching a 27-cell mushroom-body model to read, and proving what it cannot read

**FlyBrain — research report**

This is the narrative write-up. Every number in it comes from a script in this
repository, and the read-out scripts hash their own weights before and after they
run, so the claims are checkable rather than asserted. `README.md` is the short
version, `docs/methods.md` is the modelling detail, `MODEL_CARD.md` is the
limitations. This document is the argument.

**v0.2.0 adds a second experiment and one revised negative claim.** The `PN → KC`
wiring can now be loaded from the measured hemibrain v1.2 connectome, and it was
trained against two controls (§3.5, §7.1). Section 7.1 formally retracts the
v0.1.0 statement that no connectome was present. The null result there is
reported as a null result.

---

## 1. Summary

A 512-neuron spiking circuit with 13,824 plastic synapses learned a 27-symbol
alphabet (A–Z plus a blank) from hand-drawn 5×7 bitmaps, trained by a three-factor
reward-modulated STDP rule. It reaches **100.0%** holdout accuracy at four looks
per character, converging in **102.5 seconds** on one CPU thread.

It was then shown ten digits, and it got **0 of 300 right**.

The second result is the point of the paper. The model has no digit output cell,
so being wrong about a digit is a structural guarantee rather than a measurement
error — which makes it the one experiment in the project that cannot be faked by
a buggy evaluation loop, a leaked test set, or an over-optimistic threshold. It is
a negative control with a proof of impossibility attached, and it is what licenses
the headline accuracy rather than the other way round.

The same read-out was then run over a whole book: 132 pages, 209,939 characters.
**168,985 letters, none wrong.** All 402 errors in the entire document were
spaces. Every one of the 1,050 digits was wrong, as it must be.

Finally, the fixed `PN → KC` expansion was replaced with the **measured**
hemibrain v1.2 connectome — 157 projection neurons, 1,802 Kenyon cells, 185,994
traced synapses — and the same task was trained on it three times. It reached
**100.0%, 243/243**, matching the published result. A degree-matched shuffle that
preserves every KC's exact partner count and weight multiset while randomising
*which* partners **also reached 100.0%, 243/243 — a paired difference of exactly
zero on every seed.** The honest conclusion is that at this task the specific
detected partner identities contribute nothing measurable over equal-degree
random partners. That is a negative result about this model, and §7.1 explains
what it does and does not license.

---

## 2. What was built

A classifier whose layers are named after the parts of an insect mushroom body.
The default configuration is entirely synthetic. The `wiring="connectome"`
configuration replaces the `PN → KC` matrix with measured anatomy — but only that
matrix, so the anatomy is still borrowed *vocabulary* for every other stage. See
§7.1.

| Stage | Units (default · connectome) | Update rule | Plastic |
| --- | --- | --- | --- |
| Receptor sheet | 35 · 35 | 5×7 bitmap, greyscale | no |
| Projection neurons | 128 · 157 | graded logistic | no |
| Kenyon cells | 512 (k=60) · 1802 (k=211) | sparse top-*k*, RMS-normalised, Poisson | no |
| MBONs | 27 · 27 | single-compartment LIF, β=0.90, bias 0.12 | **yes** |
| Dopamine | — | phasic burst, τ=3 steps, 1-step delay | — |

**Plastic synapses, KC→MBON only:** 13,824 (512 × 27) in the default
configuration, 48,654 (1,802 × 27) in the connectome configuration. Everything
upstream is fixed in both. Chance is 3.7%.

The `PN → KC` matrix is the one place the two configurations differ:

| Configuration | `PN → KC` |
| --- | --- |
| `wiring="random"` (default) | invented. Each KC samples 16 PNs uniformly. |
| `wiring="connectome"` | **measured.** 12,426 traced hemibrain connections, 185,994 synapses. |
| `wiring="connectome-shuffled"` | the same, with each KC's partner list permuted in place — exact degree and weight multiset preserved. |

Two design choices are worth naming because they were arrived at by being wrong:

- **The projection neurons are graded, not spiking.** A LIF with these parameters
  saturates outside a narrow input band, which destroys the glyph pattern at the
  first stage. The graded logistic preserves relative intensity, which is what a
  5×7 bitmap is mostly made of.
- **The KC code is RMS-normalised, not peak-normalised.** Peak normalisation
  throws away total activity, and total activity is the only signal that
  distinguishes the blank glyph from a dim letter.
- **A measured `PN → KC` matrix is re-scaled to unit L2 row norm.** This was a
  deliberate control, not a convenience: the connectome arm must not win because
  its rows happen to be louder. Two of the three modelled choices — √ synapse
  weighting and all-excitatory sign — are applied identically to every arm. The
  third, the row norm, is exact for both connectome arms and only *nominal* for
  the random arm, which draws weights with variance 1/`fan_in`: its rows average
  0.96 with a spread of 0.20–2.15. That asymmetry is disclosed rather than
  smoothed over, and it does not touch the comparison that carries the result,
  since the connectome and shuffled arms share all three choices exactly.

Three-factor reward-modulated STDP on KC→MBON:

```
dw = lr · D(t) · e(t)
```

`e(t)` is an eligibility trace (decay 0.97) that bridges the reward delay; `D(t)`
is the phasic dopamine burst. With `credit_mode="pattern"` the credited row's
update is `direction · pattern / norm`, so the burst contributes only its **sign**
and the magnitude comes from the centred KC pattern. That is a covariance rule,
which is why the rows converge to class centroids rather than drifting with reward
magnitude.

There is exactly one weight-mutating function in the project,
`MushroomBody.apply_dopamine()`. Everything that reads the circuit out is a pure
forward pass. That property is what makes §5 mechanical instead of rhetorical.

---

## 3. Results

### 3.1 The alphabet

| Measurement | Result |
| --- | --- |
| Classes | 27 (`A`–`Z` + space) |
| Chance | 3.7% |
| Clean-glyph holdout, 4 looks | **100.0%** |
| First epoch within 2 points of final | **59** |
| Train loop, 400 epochs, one thread | **102.5 s** |
| Trainable synapses | 13,824 |

Convergence at epoch 59 means roughly 85% of the 400-epoch budget does no
measurable work; the budget is set past convergence rather than at it.

**Honest caveat on the 100.0%.** The trainer's own `evaluate()` runs 27 classes ×
3 repeats = 81 trials, and every trial re-encodes the glyph from a Poisson
generator. Re-running the same evaluation against the *same checkpoint* can land
on 80/81 and report **98.8%**. Both numbers are honest; the difference is the
sampling of one trial. The large-*n* figure that is not vulnerable to this is
§3.3 — 168,985 letters, none wrong.

### 3.2 The negative control: it cannot count

Ten digits were drawn on the receptor sheet 30 times each. The model has no
output cell for any of them.

| Digit | Ink (lit px) | Usually says | Consistency | Mean margin |
| --- | --- | --- | --- | --- |
| `0` | 19 | `G` | 15/30 (or `O`) | 0.034 |
| `1` | 10 | `I` | 30/30 | 0.311 |
| `2` | 14 | `Q` | 23/30 | 0.081 |
| `3` | 14 | `Z` | 30/30 | 0.238 |
| `4` | 14 | *(blank)* | 30/30 | 0.213 |
| `5` | 17 | `D` | 29/30 | 0.085 |
| `6` | 15 | `B` | 14/30 (or `G`) | 0.037 |
| `7` | 11 | `Z` | 30/30 | 0.504 |
| `8` | 17 | `B` | 19/30 | 0.040 |
| `9` | 15 | `O` | 23/30 | 0.051 |

Pooled over 300 presentations:

- **digits recognised: 0/300**
- **answers inside the vocabulary: 300/300** — no symbol is ever invented
- alphabet control in the same process: **810/810**
- letters: mean winner-minus-runner-up margin **0.359**
- digits: mean margin **0.159**, with **92/300** below 0.05

The margin collapse is real, and the temptation is to sell it as an abstention
signal. It is not one. **208 of the 300 digit answers are not near-ties**, so a
margin threshold would miss most of the confusion. It is reported because it was
measured, not because it works.

What the control buys is a proof of mechanism: the model is wrong about digits for
exactly one reason — there is no output cell — and not because of a softmax
artefact, a threshold, or a decoding bug. When a model fails *for a stated,
structural reason* and fails 300 times out of 300, the rest of its accuracy is
being produced by something real.

### 3.3 Reading an actual book

One PDF, 132 pages, 669,627 bytes. The model read it one character at a time,
four looks each, and answered with its own 27 symbols.

| | |
| --- | --- |
| Document | *Industrial Society and Its Future* (132 pages) |
| Characters extracted | 209,939 |
| Characters presented | 204,556 |
| Stepped over (no glyph) | 5,383 |
| Wall clock | 1,300.2 s, one thread, **157.3 char/s** |
| **Letters** | **168,985 / 168,985 — 100.0%** (margin 0.3135) |
| **Spaces** | **34,119 / 34,521 — 98.8%** (margin 0.3290) |
| **Digits** | **0 / 1,050 — 0.0%** (margin 0.1801) |
| All presented | 203,104 / 204,556 — 99.29% |
| **Answerable (letters + spaces)** | **203,104 / 203,506 — 99.8%** |
| Chance | 3.7% |

**All 402 errors are spaces.** The blank glyph is the hardest class in the
alphabet — zero lit pixels, so its code is nothing but baseline Poisson noise —
and it is the sole source of error in the entire run. The first error is at
character 331.

The digits, counted separately and never folded into the headline:

| Digit | Shown | What it said |
| --- | --- | --- |
| `0` | 96 | `G` ×56, `O` ×40 |
| `1` | 253 | `I` ×253 |
| `2` | 162 | `Q` ×139, `D` ×23 |
| `3` | 91 | `Z` ×90, `D` ×1 |
| `4` | 78 | *(blank)* ×78 |
| `5` | 76 | `D` ×72, `G` ×2, `U` ×2 |
| `6` | 71 | `B` ×35, `G` ×30, `C` ×6 |
| `7` | 74 | `Z` ×74 |
| `8` | 67 | `B` ×55, `O` ×11, `C` ×1 |
| `9` | 82 | `O` ×66, `S` ×16 |

The confusions reproduce the synthetic control — `1`→`I`, `7`→`Z`, `3`→`Z` — but
now over hundreds of natural instances of each glyph instead of 30 drawn ones.
That is the strongest evidence that the control measured the model rather than
the test rig.

### 3.4 Two denominators that disagree, and why

The extractor classifies characters with `str.isalpha()`. The reader presents
characters based on membership of the 5×7 font's glyph table (`is_input`). These
are not the same predicate, and a real book shows the gap:

| Account | Count |
| --- | --- |
| Characters `str.isalpha()` calls letters | 169,527 |
| **PDF typographic ligatures with no glyph** (`ﬁ` 245, `ﬀ` 169, `ﬃ` 73, `ﬂ` 54, `ﬄ` 1) | **542** |
| Letters actually presentable to the model | 168,985 |
| Characters the extractor marks unreadable (punctuation) | 4,841 |
| Ligatures stepped over as well | 542 |
| Total stepped over | **5,383** |
| 168,985 + 34,521 + 1,050 + 5,383 | **209,939** ✓ |

Nothing was quietly dropped to make a denominator prettier. `str.isalpha()` is not
a test for "the model can see this", and using it as one — here, and repeatedly
during development — is the same bug class. Use the glyph table.

### 3.5 The measured connectome, against two controls

The synthetic `PN → KC` expansion was replaced with the measured hemibrain v1.2
wiring and the same task was trained on it. `connectome_compare.py` holds
everything fixed except that one matrix, and runs three arms:

| Arm | `PN → KC` | What it isolates |
| --- | --- | --- |
| `connectome` | the measured wiring | — |
| `connectome-shuffled` | each KC's partner list permuted in place, exact degree and weight multiset preserved | partner **identity** |
| `random` | the synthetic path at the connectome's dimensions, uniform fan-in 6 | the synthetic baseline at matched scale |

Three seeds, 400 epochs, 81-glyph holdout, 4 looks per answer:

| Arm | seed 7 | seed 11 | seed 13 | pooled | conv. epoch | train s |
| --- | --- | --- | --- | --- | --- | --- |
| `connectome` | 81/81 | 81/81 | 81/81 | **243/243 = 100.0%** | 54 / 98 / 74 | 287 |
| `connectome-shuffled` | 81/81 | 81/81 | 81/81 | **243/243 = 100.0%** | 53 / 88 / 72 | 257 |
| `random` | 77/81 | 74/81 | 81/81 | 232/243 = 95.5% | 98 / 76 / 69 | 254 |

Paired `connectome` − `connectome-shuffled`: **+0 items, on every seed.**

The measured wiring reproduces the published result at its real dimensions:
1,802 Kenyon cells, 185,994 measured synapses, 48,654 plastic synapses, 100.0%
holdout, ~290 s on one thread. It runs, it converges, and it works.

**The shuffle matches it exactly.** Permuting each Kenyon cell's partner list in
place — preserving that cell's exact partner count *and* its exact multiset of
synapse weights, changing only *which* partners they are — does not move the
number at all. 243/243 against 243/243, three seeds, zero difference.

The `random` arm is where the only gap appears, and it is a gap that must not be
attributed to the connectome:

| Arm | fan-in per KC |
| --- | --- |
| `connectome` | the real **distribution**: min 1, median 6, mean 6.90, max 21 |
| `connectome-shuffled` | the real **distribution**, exactly |
| `random` | uniform **6** for every KC |

The two connectome arms share the real fan-in *distribution*; the random arm
matches only its *mean*. So that comparison bundles partner identity with fan-in
distribution and cannot separate them. Since the shuffle already eliminates
identity as an explanation, the residual ~4.5-point difference is attributable to
**the shape of the fan-in distribution and the larger circuit** (1,802 KCs and
48,654 plastic synapses are both 3.5× the default), not to which specific
partners the electron microscope detected.

**Limitations that cut against this conclusion.** 27 clean hand-drawn glyphs is
an easy task. The connectome and shuffled arms both sit at 81/81, so this is a
comparison at ceiling: a 100.0% vs 100.0% tie demonstrates that the test cannot
separate the arms, not that they are equivalent. The holdout is 81 items (one
flake = 1.23 points) and there are three seeds — enough to see a consistent
ceiling, not enough for an interval worth quoting. And the layer that actually
learns, `KC → MBON` (48,654 plastic synapses), remains entirely invented, so any
genuine anatomical advantage would have to survive that bottleneck to be visible.

This is a **negative result about this model**, not a claim about *Drosophila*.
It is reported because it is what the experiment returned.

---

## 4. Reproducibility, and two different kinds of hash

Three independent 400-epoch training runs, same seed, same hyperparameters. The
first two were run here; the third was run in a clean copy of the repository as
part of the clone-and-run verification (§10).

- **weights**: identical across all three. Weight digest
  `626bece30a76927e8fd124eef1c3fdc37e31c87b07c4d20636656141ee85d802`, `mbon.w`
  maximum absolute difference **0.0**, `torch.equal` → `True`.
- **`.pt` files**: different in each. `1e56a34a…`, `d231c8bc…`, `56891cdf…`.

So: one weight digest, three distinct files, and the clean copy reproduced the
digest without being told what it should be.

The second row is not a failure. `torch.save` writes a zip archive containing a
timestamp, so saving **one unchanged in-memory brain twice** produces two
different file hashes — this was tested directly, and it does. The conclusion:

- a **weight digest** is the artifact you compare between runs to establish
  reproducibility;
- a **checkpoint file hash** identifies a particular file on disk, which is
  exactly what the before/after guard needs it for, since that guard compares the
  *same path* across a read-only run.

Conflating the two would have produced a false reproducibility scare. Stating the
distinction is cheaper than defending a wrong claim later.

---

## 5. The negative control is mechanical, not rhetorical

`flybrain/proof.py` hashes all three weight matrices and the checkpoint file
before and after a read-out run. Both read-out scripts import it — there is one
implementation, not three copies.

Around the 204,556-presentation book read:

```
weights sha256     before 626bece30a76927e8fd124eef1c3fdc3...
                   after  626bece30a76927e8fd124eef1c3fdc3...   unchanged
checkpoint sha256  before 1e56a34a4400ad4dc928e5989540d849...
                   after  1e56a34a4400ad4dc928e5989540d849...   unchanged
94,954 bytes on disk, epoch 400 - byte-identical, not rewritten
```

If a weight moves during a read-only run the script does not warn. It crashes.
That is the design: a read-out that quietly learns would invalidate its own
result, so the guard has to be fatal.

The same guard runs around the 300 digit presentations, with the same result.

The JSON artifacts carry `answers_sha256` as well — for the book read,
`d895f5d2429627cc45714b3dc83904091c2f41637676f8faa4c5d890588599af`, a hash of the
model's 204,556-character answer stream. Two runs agreeing on every answer share
that hash, so a reader can verify the run without needing a copy of the book.

---

## 6. Three bugs worth publishing

### 6.1 The reward rule that starved itself

The `Config` defaults are `reward_mode="operant"`, `lr_schedule="const"`,
`lr=0.004`. That recipe **plateaus at 60.5%** and does not move. It was re-run to
confirm: 400 epochs, 60.5%, exactly the ceiling the old docstring quoted.

Under `operant`, the reward depends on the model's own answer, so an early wrong
answer suppresses the sugar that would have taught the right one. The rule starves
itself. The fix is `reward_mode="pavlovian"` — the true letter always precedes the
sugar, so sugar is delivered on every trial and *baseline adaptation* turns it
into a reward-prediction error. Combined with `lr_schedule="inv"`
(`lr_half_life=300`) so the step anneals instead of random-walking:

```python
Config(lr=0.020, credit_mode="pattern", reward_mode="pavlovian",
       lr_schedule="inv", lr_half_life=300, epochs=400, seed=7)
```

`make_checkpoint.OLD-60pct.py` is kept in the repository because it reproduces the
plateau on demand.

### 6.2 The four identical looks

`encode_from_config(cfg, ch)` with no generator builds a **fresh** `Generator`
seeded `cfg.seed + 5` on every call. So this:

```python
counts = brain.mb.present(encode_from_config(cfg, ch))
for _ in range(3):
    counts = counts + brain.mb.present(encode_from_config(cfg, ch))   # same spikes
```

sums the *identical* spike train four times. The averaging reduces no variance at
all. It surfaced as the blank glyph reading 59% in the browser while `trial()` —
which does pass the generator — measured 99.5% on the same code. Passing
`gen=brain.gen` makes each look an independent draw.

### 6.3 The 7× overshoot

Calling `apply_dopamine()` without naming a credited row falls back to
`delta = gain · eligibility`, where the gain is the raw burst sum (~5.6) against a
weight scale of ~0.053. That is a ~7× overshoot, and it ran a trained brain down
to chance (7%) on a **single click** in the UI. The `credit=`/`punish=` arguments
are not optional in practice.

---

## 7. What this is not

### 7.1 One layer is a real connectome, and it was tested

**This section replaces an earlier claim that there is no connectome here. That
claim was true in v0.1.0 and is false in v0.2.0.** It is left visible, rather
than deleted, because the reversal is itself part of the record: the original
version of this report stated, correctly at the time, that *"No FlyWire, no
hemibrain, no neuPrint, no Codex, no electron-microscopy reconstruction of any
kind is loaded, downloaded, or referenced at runtime. There is no wiring diagram
in this repository."* As of v0.2.0 there is.

The `PN → KC` expansion can now be loaded from the **hemibrain v1.2** connectome
(Scheffer et al. 2020, eLife 9:e57443, CC BY 4.0) instead of sampled from a
generator. What that changes, and what it does not, is the whole point:

> **A connectome replaces the *wiring*, not the *task*.**

| Element | Status | Detail |
| --- | --- | --- |
| Which PNs feed which KCs | **Measured** | 12,426 traced connections, 185,994 synapses |
| KC fan-in | **Measured** | min 1, median 6, mean 6.90, max 21 |
| KC population | **Measured** | 1,802 KCs receiving PN input |
| Synapse magnitude | Modelled | √(synapse count), applied identically to every arm |
| Row scale | Modelled | unit L2 norm per KC — exact for both connectome arms, nominal only for the random arm (rows average 0.96, spread 0.20–2.15) |
| Synapse sign | Modelled | all excitatory (PN→KC is cholinergic) |
| Receptor sheet, PNs, KCs, MBONs, `KC → MBON`, dopamine, the task | **Invented** | unchanged from v0.1.0 |

The extraction required one non-obvious correction, and it is the part worth
citing. The natural rule — a neuron is a projection neuron if its cell type
contains `PN` — yielded 428 PNs, of which **271 had exactly zero synapses onto
any Kenyon cell**. Verified three ways: those 271 have 39–780 outgoing edges
each (median 126) and exactly zero onto KCs; their names are legitimate
(`DA1_vPN`, `M_lvPNm24`, `WEDPN8C`); and the curated tables resolve it, with 140
of 166 `mPN` having no KC output against 118 of 181 `uPN`. Those are
lateral-horn-projecting cells, which is correct anatomy. Membership is therefore
decided by **connectivity, not nomenclature**. Result: **157 PNs, 1,802 KCs**,
unmatched edges 271 → 2.

**The comparison.** `connectome_compare.py` trains three arms on the same task,
same seed, same hyperparameters, with the `PN → KC` matrix as the only variable.
Three seeds, 400 epochs, 81-glyph holdout:

| Arm | seed 7 | seed 11 | seed 13 | pooled |
| --- | --- | --- | --- | --- |
| `connectome` | 81/81 | 81/81 | 81/81 | **243/243 = 100.0%** |
| `connectome-shuffled` | 81/81 | 81/81 | 81/81 | **243/243 = 100.0%** |
| `random` (uniform fan-in 6, same dims) | 77/81 | 74/81 | 81/81 | 232/243 = 95.5% |

Paired `connectome` − `connectome-shuffled`: **+0 items on every seed.**

**The finding is a negative one, and it is reported as such.** Destroying partner
identity while preserving each KC's exact degree and weight multiset changes
nothing measurable. The real connectome and its shuffled control are
indistinguishable at 243/243. The `random` arm differs by ~4.5 points, but it
differs along *two* axes at once — partner identity **and** fan-in distribution,
since the connectome arms carry the real fan-in spread (min 1, median 6, mean
6.90, max 21) while the random arm uses a uniform 6. Because the shuffle
eliminates identity as the explanation, the residual gap is attributable to the
**fan-in distribution and circuit size**, not to which specific partners the
electron microscope detected.

**And the test is underpowered in the direction that flatters this conclusion,
which must be said out loud.** 27 clean glyphs is an easy task; once an arm
reaches 81/81 there is no headroom, so a 100.0% vs 100.0% tie is evidence that
the test cannot separate the arms, **not** evidence that they are equivalent.
The holdout is 81 items (one flake = 1.23 points). Three seeds is enough to see a
consistent ceiling, not enough for a confidence interval. And the layer that
actually learns — `KC → MBON`, 48,654 plastic synapses — remains fully invented,
so any real anatomical advantage would still have to survive that bottleneck.

What the experiment legitimately establishes is narrower and still worth
publishing: the real `PN → KC` wiring is correctly extracted, correctly scaled,
and trains to the published accuracy at its real dimensions; and the specific
detected partner identities are load-bearing for nothing at this task. **The
second claim is a negative result about this model. It is not a claim about
*Drosophila*.**

### 7.2 A fruit fly does not read

Nothing here should be cited as a claim about what an insect brain does. This
borrows anatomical vocabulary to make an architecture concrete. It is a toy —
an unusually well-instrumented toy.

### 7.3 It is not OCR

One character at a time, from a rendered bitmap. No word model, no language model,
no layout analysis, no segmentation, and it has never seen a photograph of text.
Every result is a statement about the hand-drawn 5×7 glyphs in
`flybrain/encoding.py`. Change the font and every number changes. There is no
MNIST, no EMNIST, and no scraped dataset.

### 7.4 One destroyed long run

Two `serve.py` processes ended up bound to the same port — one on the project
virtualenv, one on the system Python that happened to be on `PATH`. Requests
alternated between them, so `/api/state` reported `epoch 2021` while a measurement
script reported `epoch 61`: two different brains answering the same question. The
rogue process's `POST /api/save` overwrote `runs/flybrain.pt` with a 60-epoch
bootstrap.

The long run is gone. It was not recoverable. It is documented here because a
methods section that omits its own accidents is not a methods section. Two guards
came out of it: `FlyBrain.save()` refuses to overwrite a longer run with a shorter
one (copying the longer file to `<path>.bak` first), and `serve.main()` refuses to
start on an occupied port.

The forensic evidence is still in the repository:
`runs/flybrain.CLOBBERED-epoch61.pt`. The recipe was recovered from it — the
clobbered checkpoint still carried its training config, so diffing
`torch.load(ckpt)["config"]` against `Config()` named the three fields that
mattered.

And the reason this report does not claim 2000 epochs: the extra 1,900 epochs were
never doing any work. Rebuilding from scratch takes 102.5 seconds and converges by
epoch 59.

---

## 8. Threats to validity

| Threat | Mitigation present | Residual risk |
| --- | --- | --- |
| Evaluation leak or buggy test loop | Digits cannot be named, so the negative control cannot pass by accident | The alphabet figure still depends on the split; the split is in the code |
| Overfitting to a tiny test set | 168,985 letters read in sequence; also 1,050 digits counted separately | The test set is the same 27 glyphs as training, augmented |
| Measurement changes the model | Weights and checkpoint hashed before/after; mismatch is fatal | None for read-out; training runs are not guarded mid-run |
| Cherry-picked runs | Reproducibility claim is a weight digest over three independent runs | Different seeds were not swept |
| Denominator games | All 209,939 characters accounted for; digits and unreadable counted | Accuracy is reported in three buckets so the framing is visible |
| Structural error rate | Blank glyph is the sole error source and is reported as such | 98.8% on spaces could be improved by more looks |
| **Task too easy to discriminate the connectome arms** | Stated, not hidden: §3.5 reports that 81/81 is a ceiling, so the tie cannot be read as equivalence | A harder task would be needed to test the claim properly; not attempted here |
| **Fan-in distribution not isolated from partner identity** | The shuffle isolates identity; the fan-in asymmetry is spelled out in §3.5 | The `random` arm should be re-run at the real fan-in distribution to separate the two |
| **Connectome arm claims only one measured layer** | §7.1 enumerates measured vs modelled vs invented line by line | `KC → MBON`, the plastic layer, remains invented and unvalidated |

The one thing this report does **not** establish is that any of this transfers.
Everything measured here is measured on a 5×7 font this project drew itself. The
connectome result additionally establishes nothing about *Drosophila*: it says
which cells connect to which, which is measured, and nothing about what they do
with it, which is not.

---

## 9. Artifacts

| Path | Contents |
| --- | --- |
| `runs/flybrain.pt` | Trained circuit, default wiring, epoch 400, 94,954 bytes |
| `runs/digit_proof.json` | 300 digit presentations, per-digit table, margins, hashes |
| `runs/manifesto_read.json` | 204,556 presentations, per-bucket counts, digit answers, first errors, `answers_sha256`, proof block |
| `runs/connectome_compare.json` | seed 7 comparison run: metadata, extraction rule, calyx filter rationale, per-arm results and accuracy curve |
| `runs/compare_seeds1113.json` | seeds 11 and 13 comparison run, same schema |
| `runs/compare_seed7.log`, `runs/compare_seeds1113.log` | Console logs of both comparison runs |
| `runs/wiring_<arm>_seed<n>.pt` | Per-arm checkpoints. **Not tracked** (~229 KB × 9 = 2 MB); regenerable in ~40 min from the tracked `data/hemibrain_mb.npz` plus the committed seed. Every number they would prove is already in `runs/connectome_compare.json`, including the full accuracy curve. |
| `data/hemibrain_mb.npz` | Derived hemibrain v1.2 subgraph: 157 PN → 1802 KC, 68 MBON, 322 DAN. 173.9 KB |
| `flybrain/connectome.py` | Loader. The only module that knows a connectome exists |
| `tools/build_connectome.py` | Extract the above from the public hemibrain bucket |
| `connectome_compare.py` | The three-arm experiment |
| `runs/flybrain.CLOBBERED-epoch61.pt` | Evidence for §7.4 |
| `runs/schedule.log`, `runs/sweep.log` | Output of the long sweeps |

## 10. How to reproduce

```powershell
python -m venv .venv-flybrain
.\.venv-flybrain\Scripts\python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS=1

.\.venv-flybrain\Scripts\python.exe make_checkpoint.py --quiet
.\.venv-flybrain\Scripts\python.exe digit_proof.py --repeats 30 --json runs/digit_proof.json
.\.venv-flybrain\Scripts\python.exe serve.py                        # http://127.0.0.1:8000/
.\.venv-flybrain\Scripts\python.exe read_document.py <your-book.pdf> --json runs/book_read.json
```
The connectome experiment needs two more steps. `build_connectome.py` needs no
credentials — the hemibrain bucket is public (CC BY 4.0) — and downloads ~46 MB.
The derived subgraph it writes is 174 KB and is committed, so the download is
only required if you want to rebuild the extract yourself.

```powershell
.\\.venv-flybrain\Scripts\python.exe tools/build_connectome.py    # ~46 MB download, -> data/hemibrain_mb.npz
.\\.venv-flybrain\Scripts\python.exe connectome_compare.py --seeds 7,11,13 --epochs 400
```

That is nine trainings at ~260–310 s each, so budget roughly 40 minutes on one
thread. `--quick --epochs 8` will smoke-test the plumbing in about 20 seconds
without producing a number worth quoting.
| Claim | How to check it | Measured |
| --- | --- | --- |
| Train loop from scratch | time `make_checkpoint.py` | 102.5 s, 400 epochs, one thread |
| Converges long before 400 | the `first within 2pt` line it prints | epoch 59 |
| Reproducible weights | retrain, compare weight digests | `626bece3…`, identical |
| Digits cannot be named | `digit_proof.py --repeats 30` | 0/300, 300/300 in vocabulary |
| Reading changes no weights | before/after hashes in both scripts | unchanged, both |
| A book can be read | `read_document.py` on a 132-page PDF | 99.8% answerable |
| The real connectome trains | `connectome_compare.py`, `connectome` arm | 100.0% holdout, 3 seeds |
| Partner identity is not the ingredient | the `connectome-shuffled` arm, paired per seed | +0 items, every seed |
| Fan-in distribution is the open question | compare `random` at uniform 6 vs the connectome's spread | not yet isolated — §3.5 |

### 10.1 Verified from a clean copy

The open-source claim — clone it and it runs — is a claim, so it was tested as
one. The repository was copied into an empty directory with the virtualenv and
caches excluded, and the entire sequence above was executed there with no
inherited state:

| Step | Result |
| --- | --- |
| `python -m venv .venv-flybrain` | Python 3.10.11, `include-system-site-packages = false` |
| `pip install -r requirements.txt` | resolved cleanly, torch 2.14.0 + pypdf 6.19.0 |
| `pytest -q` | **14 passed** at v0.1.0; **33** as of v0.2.0 (19 connectome tests added) |
| `make_checkpoint.py --quiet` | **100.0%**, first within 2 pt at **epoch 59**, 116.7 s for the whole command (102.5 s of it the train loop) |
| `digit_proof.py --repeats 30` | **0/300** digits, **300/300** in vocabulary, **810/810** alphabet control, margins 0.359 / 0.159 |
| `serve.py` | `GET /` → 200, 15,069 bytes at v0.1.0 / **15,071** as of v0.2.0; `GET /api/state` → 200 with the canonical config |
| `read_document.py` on the 132-page PDF | 100.0% on letters, 0/1,050-shape behaviour on digits |

The run also produced the third file hash of §4 — `56891cdf…` in the clean copy
against `1e56a34a…` here — with the identical weight digest. The reproducibility
distinction was not just argued; it showed up unprompted in the verification.

Two things to expect before they surprise you:

1. `make_checkpoint.py` overwrites `runs/flybrain.pt`, and your checkpoint hash
   **will not match the one printed above** even though the weights will. That is
   `torch.save` timestamping its archive, not a different model — see §4.
2. `read_document.py` does **not** write a transcript by default. A full
   204,000-character transcript of a book is a copy of the book. `--save-text`
   will write one for your own reading; don't commit it.

## 11. Citing

See `CITATION.cff`. MIT licensed.
