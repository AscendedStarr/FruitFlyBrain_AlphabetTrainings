# I taught the fruit fly alphabets and made it read

**FlyBrain** is a 512-neuron spiking mushroom body that learned the alphabet
from 27 hand-drawn bitmaps and then read a book it had no business reading.

It has 27 output cells: `A`–`Z` and a blank. There is no digit cell. That
missing digit cell turned out to be the most useful thing in the repository,
because it is the one experiment that cannot be faked: the model is *structurally
incapable* of being right about it, so "0/300" isn't a modest result, it's a
guarantee that the other numbers are coming from somewhere real.

Every figure below is produced by a script in this repo. The interesting script
also hashes its own weights before and after, so you don't have to take any of
this on faith.

---

## The part that actually works

On the alphabet it was trained on, the fly is essentially perfect at four looks
per character:

| Measurement | Result |
| --- | --- |
| Classes | 27 (`A`–`Z` + space) |
| Chance | 3.7% |
| Clean-glyph holdout, 4 looks | **100.0%** |
| Alphabet sweep in the control script | **810/810** |
| Letters read from a real book | **168,985 / 168,985** |
| Trainable synapses | 13,824 (KC→MBON only) |
| Train loop, from scratch | 102.5 s, CPU, one thread |

**On that first 100.0%:** the trainer's own evaluation is only 81 trials (27
classes × 3 repeats), and every trial re-encodes the glyph from a Poisson
generator, so re-running it against the *same* checkpoint can land on 80/81 and
print 98.8% instead. `digit_proof.py` does exactly that: its header reads
`98.8% on holdout` while its alphabet sweep in the same process is a clean
810/810. Nothing differs between those two numbers except the sampling of one
trial. If you want the figure that isn't small-n — **168,985 letters read out of
a book, none wrong** — it is further down. That's a real classifier, trained by
a real (if simplified) three-factor dopamine rule, and it converged in under
two minutes on a laptop.

Which is the least interesting sentence in this document.

**Is it reproducible?** Three independent 400-epoch training runs, same seed,
same hyperparameters, produced a **byte-identical `mbon.w`** (max abs difference
0.0) and the same weight digest, `626bece30a76927e8fd124eef1c3fdc3...`. All three
`.pt` *files* hashed differently — `1e56a34a...`, `d231c8bc...`, `56891cdf...` —
because `torch.save` writes a zip archive with a timestamp inside it; saving one
unchanged in-memory brain twice also gives two different file hashes. So: the
**weights** are the reproducible artifact and the file hash identifies a
particular file on disk, which is exactly what the before/after guard below uses
it for. Full commands at the bottom.

---

## The part that matters: it cannot count

Show it a digit. It has a 5×7 glyph for `0`–`9`, so the pixels can be drawn on
its receptors just fine. But there is no output cell for any digit, so the
answer is guaranteed wrong before the spikes are even generated.

Here is what it says instead, over 30 independent sweeps at epoch 400:

| Digit | Ink | Usually says | Consistency | Mean margin |
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

Pooled over all 300 presentations:

- **digits recognised: 0/300**
- **answers inside the vocabulary: 300/300** — it never invents a symbol it
  doesn't have
- letters: mean winner-minus-runner-up margin **0.359**
- digits: mean margin **0.159**, with **92/300** below 0.05 (near-ties)

So the fly isn't *confidently* wrong on digits. It's confused: the margin
collapses, which is what a pattern does when it matches nothing in the
vocabulary rather than something in it.

**One honest caveat:** 208 of those 300 digit answers are *not* near-ties. The
collapsed margin is a real measured effect, but it is **not a working
"I don't know" signal**, and this repo does not claim it is. It's reported
because it was measured.

### And it isn't learning from the digits

Both scripts that read out-of-vocabulary characters are forward passes only.
Weights are written in exactly one function in the whole project —
`MushroomBody.apply_dopamine()` — and neither script calls it. No `learn=True`,
no `trial()`, no `save()`.

That claim is checked mechanically rather than asserted. `flybrain/proof.py`
hashes all three weight matrices and the checkpoint file before and after:

```
weights sha256     before 626bece30a76927e8fd124eef1c3fdc3...
                   after  626bece30a76927e8fd124eef1c3fdc3...   unchanged
checkpoint sha256  before 1e56a34a4400ad4dc928e5989540d849...
                   after  1e56a34a4400ad4dc928e5989540d849...   unchanged
94,954 bytes on disk, epoch 400 - byte-identical, not rewritten
```

If a weight ever moves during a read-only run, the script does not warn. It
crashes. That is the point.

---

## Made it read the Unabomber Manifesto

One PDF, 132 pages, 669,627 bytes. The fly reads it one character at a time,
four looks each, and answers with its own 27 letters. This is the full text, not
a tidy excerpt — 209,939 characters.

| | |
| --- | --- |
| Document | *Industrial Society and Its Future* (132 pages, 669,627 bytes) |
| Characters extracted | 209,939 |
| Characters presented to the fly | 204,556 |
| Stepped over (no glyph) | 5,383 |
| Wall clock | 1,300 s (21m 40s), one CPU thread, **157 char/s** |
| **Letters** | **168,985 / 168,985 — 100.0%** (mean margin 0.313) |
| **Spaces** | **34,119 / 34,521 — 98.8%** (mean margin 0.329) |
| **Digits** | **0 / 1,050 — 0.0%** (mean margin 0.180) |
| All presented | 203,104 / 204,556 — 99.3% |
| Answerable only (letters + spaces) | **203,104 / 203,506 — 99.8%** |
| Chance | 3.7% |

168,985 letters read out of a real book, in sequence, without a language model,
a word list, or a second pass, every one of them correct. The 402 misses are all
spaces. The blank glyph is the hardest class in the alphabet — zero lit pixels,
so its representation is nothing but baseline Poisson noise — and it is the sole
source of error in the entire run. First one is at character 331.

**Why 169,527 letters become 168,985 presented.** The extractor classifies a
character as a letter with `str.isalpha()`, which is `True` for the PDF's
typographic ligatures: `ﬁ` 245, `ﬀ` 169, `ﬃ` 73, `ﬂ` 54, `ﬄ` 1 — **542 of
them**. The 5×7 font has no glyph for any ligature, so the reader steps over
them, which is why the stepped-over count is 5,383 and not the 4,841 punctuation
characters the extractor counted. Nothing was quietly dropped to make the
denominator prettier; the 542 characters are in neither column and the arithmetic
above accounts for all 209,939. (`str.isalpha()` is not a test for "the fly can
see this" — that mistake and this fix are the same bug.)

### The digits in the book

Handled exactly as in the control above: drawable, unnameable, counted
separately, never folded into the headline number.

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

1,050 digits, 1,050 wrong answers, and the same confusions as the control on 30
sweeps — `1`→`I`, `7`→`Z`, `3`→`Z` — now reproducing over hundreds of instances
of each glyph in a real document rather than 30 synthetic ones.

### And reading a book changed nothing

The same hash guard as the control, run around all 204,556 presentations:

```
weights sha256     before 626bece30a76927e8fd124eef1c3fdc3...
                   after  626bece30a76927e8fd124eef1c3fdc3...   unchanged
checkpoint sha256  before 1e56a34a4400ad4dc928e5989540d849...
                   after  1e56a34a4400ad4dc928e5989540d849...   unchanged
94,954 bytes on disk, epoch 400 - byte-identical
weights are written only by MushroomBody.apply_dopamine(), which
this script never calls. reading a book changed nothing.
```

`runs/manifesto_read.json` holds the whole result: per-bucket counts, the digit
table above, the first errors with their margins, and
`answers_sha256 = d895f5d2429627cc45714b3dc83904091c2f41637676f8faa4c5d890588599af`,
a hash of the fly's 204,556-character answer stream. Two runs that agree on
every answer share that hash, so the run can be checked without shipping a copy
of the book.

**On copyright:** this repository does not contain the text of the manifesto.
`read_document.py` does not write a transcript by default, and the small
`sample_transcript` in the JSON (400 characters, `--sample 0` to omit) is the
fly's own output, included so a reader can see its errors rather than take the
percentages on faith. A full 204,000-character transcript of a book is a copy of
the book; `--save-text` will write one for your own reading, so don't commit it.

---

## The idea, and the disclaimer that goes with it

The architecture borrows its vocabulary from insect neuroanatomy. There is no
connectome here.

**No FlyWire. No hemibrain. No neuPrint. No EM reconstruction of any kind.** The
package docstring says it plainly: *"This is a brain-inspired spiking classifier,
not an emulation of the FlyWire connectome."* There is no wiring diagram in this
repository, and nothing is downloaded at runtime.

What is anatomy-*informed* is the shape and the cell counts:

| Stage | Units | In-vivo correspondence | Reality check |
| --- | --- | --- | --- |
| Receptor sheet | 35 | — | A 5×7 bitmap. A compound eye has ~3,000 ommatidia per eye. This is pixels, not photoreceptors. |
| Projection neurons | 128 | ~150 uniglomerular PNs | Graded logistic, deliberately **not** LIF — a LIF saturates outside a narrow input band and destroys the pattern at stage one. |
| Kenyon cells | 512 | ~2,000 per hemisphere | Sparse top-*k*, *k*=60 (~12% active; in vivo 5–10%). Each KC samples 16 PNs (in vivo ~5–10). |
| MBONs | 27 | Tens, per hemisphere | 26 letters + blank. |
| Dopamine | — | Sugar GRN → SEZ → PAM/DAN → MB | Phasic burst, τ=3 steps, 1-step delay, with baseline adaptation making it a reward-prediction error. |
| Plastic synapses | **13,824** | Sparse and stereotyped | Dense and random. This is the largest gap between this model and an actual fly. |

So: "connectome" appears nowhere in this repository as a claim. The wiring is
invented; the *proportions* are borrowed. If you want a project that runs on a
real connectome, this is not that project, and it says so in its own docstring.

### The learning rule

Three-factor reward-modulated STDP on KC→MBON only:

```
dw = lr · D(t) · e(t)
```

`e(t)` is an eligibility trace (decay 0.97) that bridges the reward delay; `D(t)`
is the dopamine burst. With `credit_mode="pattern"` the credited row's update is
`direction · pattern / norm`, so the burst contributes only its **sign** and the
magnitude comes from the centred KC pattern — a covariance rule, which is why
the rows converge to class centroids.

**Do not remove the `credit=`/`punish=` argument.** Calling `apply_dopamine()`
without naming a row falls back to `delta = gain · eligibility`, where the gain
is the raw burst sum (~5.6) against a weight scale of ~0.053. That's a ~7×
overshoot, and it ran a trained brain down to chance (7%) on a *single click* in
the UI.

---

## Two bugs worth publishing

### 1. The reward rule that starved itself

The `Config` defaults are `reward_mode="operant"`, `lr_schedule="const"`,
`lr=0.004`. That recipe **plateaus at 60.5%** and does not move. It was run again
to confirm, and 400 epochs produced 60.5%, exactly the ceiling the old docstring
quoted.

Under `operant`, the reward depends on the fly's own answer, so an early wrong
answer suppresses the sugar that would have taught it the right one. The rule
starves itself. The fix is `reward_mode="pavlovian"` — the true letter always
precedes the sugar, so sugar is delivered on every training trial and the
*baseline* adaptation turns it into a reward-prediction error. Combined with
`lr_schedule="inv"` (`lr_half_life=300`) so the step anneals instead of
random-walking forever:

```python
Config(
    lr=0.020,
    credit_mode="pattern",
    reward_mode="pavlovian",
    lr_schedule="inv",
    lr_half_life=300,
    epochs=400,
    seed=7,
)
```

400 epochs now reach **100.0%**, and the run reports when it first got within 2
points of that: **epoch 59**. Roughly 85% of the training budget does no
measurable work. `epochs=400` is chosen to be safely past convergence, not
because 400 was needed.

`make_checkpoint.OLD-60pct.py` is the superseded recipe, kept in the repo
because it reproduces the 60.5% plateau on demand.

### 2. The four identical looks

`encode_from_config(cfg, ch)` with no generator builds a **fresh**
`Generator` seeded `cfg.seed + 5` on every call. So this:

```python
counts = brain.mb.present(encode_from_config(cfg, ch))
for _ in range(3):
    counts = counts + brain.mb.present(encode_from_config(cfg, ch))   # same spikes
```

...sums the *identical* spike train four times. The averaging reduces no
variance at all. It showed up as the blank glyph reading 59% in the browser
while `trial()` — which *does* pass the generator — measured 99.5% on the same
code. Passing `gen=brain.gen` makes each look an independent draw.

### The bit where I destroyed a long run

Two `serve.py` processes ended up bound to the same port: one on the project
virtualenv, one on the system Python that happened to be on `PATH`. The rogue
one owned the socket. Because requests alternate between the two, `/api/state`
reported `epoch 2021` while a measurement script reported `epoch 61` — two
different brains answering the same question, and the second one's `POST
/api/save` overwrote `runs/flybrain.pt` with a 60-epoch bootstrap.

The long run is gone. It was not recoverable. The only reason the recipe was
recovered at all is that the *clobbered* checkpoint still carried the config it
had been trained with, so diffing `torch.load(ckpt)["config"]` against
`Config()` named the three fields that mattered.

Two guards came out of it:

- `FlyBrain.save()` refuses to overwrite a longer run with a shorter one; the
  longer file is copied to `<path>.bak` first.
- `serve.main()` refuses to start on a port that is already occupied.

The forensic evidence is still in the repo: `runs/flybrain.CLOBBERED-epoch61.pt`.

And the lesson, which is the reason the README does not say "2000 epochs": the
extra 1,900 epochs were never doing any work. Rebuilding from scratch takes a
102.5-second train loop and converges by epoch 59.

---

## Verify it yourself

Every number above is regenerable. Nothing is a stored constant.
The sequence below was run end to end in a clean copy of this repository — fresh
virtualenv, fresh `pip install`, no inherited state — and it reproduced: **14/14
tests pass**, training gives **100.0%** with first-within-2-points at **epoch 59**
(116.7 s for the whole command, 102.5 s of it the train loop), the digit control
gives **0/300** with **300/300** answers inside the vocabulary, the server answers
`GET /` with 200 at 15,069 bytes, and `read_document.py` reads the book at 100.0%
on letters.

```powershell
python -m venv .venv-flybrain
.\.venv-flybrain\Scripts\python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS=1

# 1. train from scratch (~2 minutes). prints holdout accuracy and the
#    first epoch within 2 points of it.
.\.venv-flybrain\Scripts\python.exe make_checkpoint.py --quiet

# 2. the negative control. 30 sweeps over all ten digits, and it hashes its own
#    weights before and after to prove it taught itself nothing.
.\.venv-flybrain\Scripts\python.exe digit_proof.py --repeats 30 --json runs/digit_proof.json

# 3. the browser demo
.\.venv-flybrain\Scripts\python.exe serve.py        # http://127.0.0.1:8000/

# 4. or read a document headlessly
.\.venv-flybrain\Scripts\python.exe read_document.py <your-book.pdf> --json runs/book_read.json
```

One thing to expect before it surprises you: step 1 overwrites `runs/flybrain.pt`,
and the checkpoint hash you get back **will not match the one printed above**,
even though the weights will. That is `torch.save` timestamping its zip archive,
not a difference in the model. Pass `--out runs/myrun.pt` if you want to keep the
shipped checkpoint alongside your own, and `read_document.py --ckpt <path>` to
read with it.

### Checking the claims rather than believing them

```powershell
# the same weights every time: two independent 400-epoch runs
.\.venv-flybrain\Scripts\python.exe make_checkpoint.py --quiet --out runs/again.pt
# -> holdout accuracy 100.0%, first within 2pt: epoch 59, train loop ~102 s

# a weight digest of both checkpoints comes out identical, mbon.w max diff 0.0;
# only the .pt file hash differs, and only because torch.save stamps the zip
```

| Claim | How to check it | Measured |
| --- | --- | --- |
| Train loop from scratch | time `make_checkpoint.py` | 102.5 s, 400 epochs, one thread |
| Converges long before 400 | the `first within 2pt` line it prints | epoch 59 |
| Reproducible weights | retrain, compare weight digests | `626bece3…`, identical |
| Digits cannot be named | `digit_proof.py --repeats 30` | 0/300, 300/300 in vocabulary |
| Reading changes no weights | before/after hashes in both scripts | unchanged, both |
| A book can be read | `read_document.py` on a 132-page PDF | 99.8% answerable |

Then open the page. The circuit panel shows the receptor sheet, the PN and KC
layers, the 27 MBON membrane traces, and — directly below the MBON readout —
what the fly has actually read so far:

- **the word it is spelling**, one character at a time, with the word it just
  finished kept on screen (dimmed) after its space so a space does not blank
  the panel;
- **the paragraph it has transcribed**, each character coloured by whether it
  was right, amber for a digit, greyscale and struck through for a character
  that has no glyph and was therefore never shown to it;
- **a running tally** — right / wrong / unanswerable / stepped over / words —
  and the accuracy of the characters it could actually be shown.

Every one of those characters is the fly's *answer*, never the document's
character; hover one to see what was on the page. Punctuation is drawn greyed
rather than quietly dropped, and nothing is silently corrected on the fly's
behalf. When it is wrong, you see it be wrong. The panel keeps only the last
1,400 characters in the DOM so a book does not melt the browser; the counters
cover the whole run.

Run the tests with `pytest`.

---

## What's in the box

| Path | What it is |
| --- | --- |
| `flybrain/` | The package: config, encoding, circuit, dopamine, trainer, document parser, non-training proof. |
| `serve.py` | Standard-library HTTP server and the session/API layer. No frameworks, no build step. |
| `web/` | The demo. Vanilla JS canvas, edited and reloaded, nothing to compile. |
| `make_checkpoint.py` | The recipe that converges. Single source of truth for the config. |
| `make_checkpoint.OLD-60pct.py` | The superseded recipe, kept because it reproduces the 60.5% plateau. |
| `digit_proof.py` | The negative control. |
| `read_document.py` | Headless document reader, with the same hash guard. |
| `flybrain/proof.py` | Weight/checkpoint hashing used by both of the above. |
| `docs/methods.md` | Full methods, every modelling choice flagged as a choice. |
| `docs/research-report.md` | The long version: the negative control as the central result, plus threats to validity. |
| `MODEL_CARD.md` | Intended use, out-of-vocabulary behaviour, limitations. |
| `CITATION.cff` | How to cite it. |
| `runs/digit_proof.json` | The measured digit table, with hashes. |
| `runs/manifesto_read.json` | The measured book read, with a hash of the answer stream. |
| `runs/flybrain.pt` | The trained circuit (95 KB, epoch 400). |
| `runs/flybrain.CLOBBERED-epoch61.pt` | The destroyed run. Kept as evidence. |
| `runs/schedule.log`, `runs/sweep.log` | Output of the two long sweeps below. |

### The scripts from the hunt

Kept because they are the measurements the claims rest on, and because most of
them are written to answer one question and then stop. All of them run against
`runs/flybrain.pt` and none of them train.

| Script | The question it was written to answer |
| --- | --- |
| `looks.py` | How many looks before answering? Saturated at 65.4% against the *old* weights — which is what proved the 60.5% ceiling was the weights and not sampling noise. |
| `blank.py` | How reliable is the blank glyph, really? 200 questions per look count, because 10 samples cannot tell 100% from 92%. |
| `classes.py` | All 27 classes, not 26. This is the script that noticed the blank is a class and that "MY NAME IS JEFF" exposes it. |
| `confusions.py` | Are the errors random or structural? They cluster on glyphs that look alike in a 5×7 grid. |
| `repeat_readout.py` | Accuracy against number of presentations of the same glyph — the readout-side fix, nothing retrained. |
| `why_mn.py` | Why do `M` and `N` fail while `O` and `Q` are nearly right? Lit pixels, KC sparsity, and a 35-pixel budget. |
| `why_wrong.py` | Phrase arithmetic: per-letter accuracy raised to 15 letters, and what that costs. |
| `keep_training.py` | The sweep over `trials_per_class`, `decision_repeats` and `n_kc` that ran while the recipe was stuck. |
| `fly_brain.py` | The command-line driver: `demo`, `train`, `say`, `diag`. |

---

## Reading list

Books the fly did **not** read. It read one book, and it was not these.

- *The Fly Who Mistook a Nine for an `O`*
- *Gradient Descent Into Madness: A Memoir*
- *Attention Is All You Need (It Has None)*
- *Great Expectations* — confidence 0.359 on letters, 0.159 on digits
- *All Quiet on the Mushroom Body*
- *One Hundred Years of Solitude, With a Single Co-Author*
- *The Metamorphosis* — Kafka, obviously
- *Brave New World: 13,824 Synapses and a Dream*
- *A Farewell to Legs*

---

## Limitations

Stated here rather than discovered by you later.

1. **27 symbols.** No digits, no punctuation, no lowercase. Text is uppercased
   before presentation. When it meets a digit it returns a letter, and that
   letter is always wrong.
2. **A hand-drawn font.** Every result is a statement about the 5×7 glyphs in
   `flybrain/encoding.py`. Change the font and every number here changes. There
   is no MNIST, no EMNIST, no scraped dataset, and no external font.
3. **It is not OCR.** One character at a time from a rendered bitmap. No word
   model, no language model, no layout analysis, no segmentation. It has never
   seen a photograph of text.
4. **The blank glyph is the hard class.** Zero lit pixels, so its code is
   nothing but baseline Poisson noise. It needs the look-averaging, and in the
   document read it was the *sole* source of error: all 402 misses out of
   204,556 were spaces, and every one of the 168,985 letters was right.
5. **PDF text layer only.** Scanned pages are refused with an explicit message
   rather than silently returning nothing.
6. **Not evidence about biology.** A fruit fly does not read. This borrows
   anatomical vocabulary to make an architecture concrete; it is not a claim
   about what an insect brain does.

---

MIT licensed. See `LICENSE` and `MODEL_CARD.md`.

Repository: **FruitFlyBrain_AlphabetTrainings**. Long-form write-up in
`docs/research-report.md`. Citation metadata in `CITATION.cff`.
