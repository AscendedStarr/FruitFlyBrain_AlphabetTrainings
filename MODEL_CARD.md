# Model card — FlyBrain

A short, honest description of what this thing is, what it was measured on, and
where it will let you down. Written to be read *before* the README's claims are
quoted anywhere.

## What it is

A spiking classifier whose layers are named after the parts of an insect
mushroom body, trained by a three-factor reward-modulated Hebbian rule. It reads
individual characters from a 35-pixel bitmap and answers with one of **27**
symbols: `A`–`Z` and a blank.

The default configuration is entirely synthetic. As of v0.2.0 the fixed
`PN → KC` expansion can instead be loaded from the measured **hemibrain v1.2**
connectome — 157 projection neurons, 1,802 Kenyon cells, 185,994 traced synapses
— which makes exactly one layer of the network anatomical and leaves every other
layer invented. See [The connectome arm](#the-connectome-arm).

## What it is not

- **Not a simulation of a fly.** As of v0.2.0 the `PN -> KC` wiring can be loaded
  from the measured **hemibrain v1.2** connectome, so one layer of this network is
  no longer invented. Everything else still is. A connectome replaces the
  *wiring*, not the *task*: the 5x7 glyph encoding, the 27 output labels, the
  mapping of 68 real MBONs onto 27 letters, the learning rule, the dopamine model
  and the Poisson spiking are all choices this project made. No fly reads
  letters, and this network is not evidence about what a fly brain does. See
  [The connectome arm](#the-connectome-arm) for exactly which parts are measured
  and which are modelled. The default `wiring="random"` path is unchanged from
  v0.1.0 and loads no connectome at all.
- **Not an OCR system.** It reads one character at a time from a rendered
  bitmap. It has no word model, no language model, no layout analysis, and no
  ability to segment a page. It has never seen a photograph of text.
- **Not evidence about biology.** A fruit fly does not read. Nothing here should
  be cited as a claim about what an insect brain does. It is a toy that borrows
  anatomical vocabulary to make the architecture concrete — and now, in one
  layer, the actual anatomy.

## Intended use

- Teaching, demonstration, and a worked example of R-STDP.
- A testbed for negative controls: the digit experiment is the interesting part
  of this repository, not the 100% accuracy.

## Not intended for

- Any decision affecting a person.
- Any task requiring text extraction. Use a real OCR engine.
- Any claim that a machine "understands" the characters it classifies.

## Data

- **Input vocabulary:** 27 hand-drawn 5×7 glyphs (`FONT_5X7`) plus 10 hand-drawn
  digits (`FONT_DIGITS`). Both are literal string art in `flybrain/encoding.py`.
  There is no external font, no MNIST, no EMNIST, and no scraped dataset.
- **Training/evaluation:** the same 27 glyphs, split into a train fold and a
  holdout fold. Augmentation during training only (3% pixel flip, 50% chance of
  a ±1 pixel jitter).
- **Document reading:** text extracted from a PDF, EPUB, or plain-text file via
  `pypdf` (PDF only). Scanned PDFs have no text layer and are refused.

## Metrics

| Measurement | Value | Conditions |
| --- | --- | --- |
| Chance | 3.7% | 1 of 27 classes |
| Clean-glyph holdout | 100.0% | `epochs=400`, 4 looks/answer |
| `evaluate()` holdout | 98.8–100% | varies with where in the session it is called; the receptor sheet is Poisson-sampled from a running generator, so repeated evaluations are independent draws |
| Algorithms recognised | 0/300 | 30 sweeps × 10 digits, epoch 400, 4 looks |
| Answers inside vocabulary | 300/300 | every digit answered with a letter |
| Alphabet control, same run | 810/810 | 27 clean glyphs × 30 sweeps, 4 looks |
| Plastic synapses | 13,824 | `wiring="random"` (the default): KC→MBON only, 512 × 27 |
| Plastic synapses | 48,654 | `wiring="connectome"`: KC→MBON only, 1802 × 27 |
| Train to 100% | 102.5 s | 400 epochs, CPU, `OMP_NUM_THREADS=1`; first within 2 points at epoch 59 (default config) |

The two figures that matter together: **0/300 on digits** and **300/300 answers
inside the vocabulary**. The fly is never right about a digit and never invents
a symbol outside its 27 cells.

### The connectome arm

`wiring="connectome"` swaps the `PN → KC` matrix for the measured hemibrain v1.2
anatomy (157 PNs, 1,802 KCs, 185,994 traced PN→KC synapses) and trains the same
task with the same hyperparameters. `connectome_compare.py` runs it against two
controls — a degree-matched shuffle that keeps every KC's exact partner *count*
but randomises *which* PNs, and the v0.1.0 random-expansion path at the
connectome's dimensions with the connectome's median fan-in.

See `runs/connectome_compare.json` for the raw numbers and the README's
[The connectome arm](#the-connectome-arm) section for what they do and do not
show. The short version, stated without hedging: the measured wiring reaches the
same accuracy as the published random wiring, and the partner-identity shuffle
matches it exactly — so at this task, on this model, the specific detected
partner identities add nothing measurable over equal-degree random partners. What
the connectome changes is **scale and fan-in distribution**, not behaviour.

**Read that as a negative result about this model, not about the fly.** The only
measured layer is `PN → KC`. `KC → MBON` — the plastic layer, the one that
actually learns — is still random, dense, and the largest gap in the model.

### Reading a real document

One 132-page PDF (209,939 characters) read with `read_document.py` at epoch 400,
4 looks, with the checkpoint hashed before and after. Letters were defined as the
168,985 characters that have a glyph, not the 169,527 that `str.isalpha()`
accepts — 542 PDF ligatures (`ﬁ`, `ﬀ`, `ﬃ`, `ﬂ`, `ﬄ`) have no glyph and are
stepped over with the punctuation.

| Bucket | Shown | Right | Accuracy | Mean margin |
| --- | --- | --- | --- | --- |
| letter | 168,985 | 168,985 | 100.0% | 0.313 |
| space | 34,521 | 34,119 | 98.8% | 0.329 |
| digit | 1,050 | 0 | 0.0% | 0.180 |
| **all** | **204,556** | **203,104** | **99.3%** | — |
| answerable (letters + spaces) | 203,506 | 203,104 | 99.8% | — |

Chance is 3.7%, so 99.8% on answerable characters and 0% on 1,050 digits is the
intended shape: the missing capability is exactly the missing output cells. All
402 errors are spaces. Throughput was 157 characters/s; the run took 1,300 s.

## Out-of-vocabulary behaviour, stated plainly

A digit can be drawn on the receptor sheet but has no output cell. When shown
one, the fly returns its best-matching letter. The mean winner-minus-runner-up
margin collapses from **0.359** on letters to **0.159** on digits, and 92 of 300
digit answers are near-ties (margin below 0.05) — but **208 of 300 are not**.

**The margin is not a working abstention signal.** Do not use it as an
"I don't know" detector. It is reported because it is measured, not because it
is reliable.

## Known limitations

1. **27 symbols.** No digits, no punctuation, no lowercase, no accents. Text is
   uppercased before presentation.
2. **Hand-drawn font dependency.** Every result is a statement about the glyphs
   in `encoding.py`, not about typography. Change the font and every number
   changes.
3. **Single-look evaluation understates it.** `evaluate()` presents each glyph
   once by default; the 4-look readout is what the UI and the published numbers
   use.
4. **The blank class is the hard one.** It has 0 lit pixels, so its code is
   nothing but baseline Poisson noise. It needs the look-averaging to be read
   correctly, and in the document read it was the **sole** source of error: all
   402 misses out of 204,556 were spaces, and every one of the 168,985 letters
   was right.
5. **PDF text layer only.** Scanned pages fail with an explicit message rather
   than silently returning nothing.
6. **One documented training accident.** A long run was destroyed by a second
   server process writing over the checkpoint. See the README's post-mortem.
   The mitigation is a downgrade guard in `FlyBrain.save()` and a port check in
   `serve.main()`.
7. **The connectome arm is one layer, and its result is a null.** `PN → KC` is
   measured; `KC → MBON`, the learning rule, the dopamine model and the encoding
   are not. The measured wiring reaches 100.0% on an 81-glyph holdout, but so
   does a degree-matched shuffle, on all three seeds — so the specific detected
   partner identities add nothing measurable at this task. Do **not** cite this as
   evidence that connectomes do or don't matter, or as a claim about *Drosophila*.
   The task is at ceiling and the plastic layer is still invented; both caveats
   are in the README's [The connectome arm](#the-connectome-arm) section.
8. **The connectome arm is slower and larger.** 1,802 KCs and 48,654 plastic
   synapses — 3.5× the default — for ~287 s per training run against 102.5 s. It
   buys no measured accuracy at this task.
