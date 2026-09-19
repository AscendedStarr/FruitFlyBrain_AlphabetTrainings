# Model card — FlyBrain

A short, honest description of what this thing is, what it was measured on, and
where it will let you down. Written to be read *before* the README's claims are
quoted anywhere.

## What it is

A spiking classifier whose layers are named after the parts of an insect
mushroom body, trained by a three-factor reward-modulated Hebbian rule. It reads
individual characters from a 35-pixel bitmap and answers with one of **27**
symbols: `A`–`Z` and a blank.

## What it is not

- **Not a connectome.** No FlyWire, hemibrain, neuPrint, or any other
  electron-microscopy reconstruction is loaded, downloaded, or referenced at
  runtime. Nothing here is an emulation of a specific animal's wiring diagram.
  The layer sizes are *inspired by* published cell counts; the connectivity is
  random and dense where a real mushroom body is sparse and stereotyped.
- **Not an OCR system.** It reads one character at a time from a rendered
  bitmap. It has no word model, no language model, no layout analysis, and no
  ability to segment a page. It has never seen a photograph of text.
- **Not evidence about biology.** A fruit fly does not read. Nothing here should
  be cited as a claim about what an insect brain does. It is a toy that borrows
  anatomical vocabulary to make the architecture concrete.

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
| Digits recognised | **0/300** | 30 sweeps × 10 digits, epoch 400, 4 looks |
| Answers inside vocabulary | 300/300 | every digit answered with a letter |
| Alphabet control, same run | 810/810 | 27 clean glyphs × 30 sweeps, 4 looks |
| Plastic synapses | 13,824 | KC→MBON only: 512 × 27 |
| Train to 100% | 102.5 s | 400 epochs, CPU, `OMP_NUM_THREADS=1`; first within 2 points at epoch 59 |

The two figures that matter together: **0/300 on digits** and **300/300 answers
inside the vocabulary**. The fly is never right about a digit and never invents
a symbol outside its 27 cells.

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
