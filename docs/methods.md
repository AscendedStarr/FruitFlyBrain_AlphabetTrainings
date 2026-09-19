# Methods

Everything needed to reproduce the numbers in the README, and every place where
a modelling choice was made rather than a measurement taken.

## 1. The circuit

Five stages, all in `flybrain/circuit.py`. Sizes are in `flybrain/config.py`,
each annotated with the in-vivo structure it stands in for.

| Stage | Units | Substrate | Notes |
| --- | --- | --- | --- |
| Receptor sheet | 35 | `(5, 7)` binary bitmap | Not anatomy. A compound eye has ~3,000 ommatidia per eye; this is a stand-in for "some pixels". Poisson-encoded, rate 0.55 on ink / 0.04 on blank. |
| Antennal lobe (PNs) | 128 | Graded logistic | ~150 uniglomerular PNs in vivo. Deliberately **not** LIF: a LIF's informative input range is ~0.15–0.45 for β=0.85, outside which it saturates to one spike per step and destroys the pattern. Common-mode subtraction (`pn_center`) removes the component every letter shares. |
| Kenyon cells | 512 | Sparse top-k, k=60 | ~2,000 per hemisphere in vivo; in-vivo sparsity 5–10%, here ~12%. Each KC samples `fan_in=16` PNs (in vivo ~5–10), RMS-normalised, then Bernoulli-sampled. |
| MBONs | 27 | Single-compartment LIF, β=0.90 | 26 letters + blank. Bias 0.12 puts the drive near `threshold·(1−β) = 0.1`, the graded operating point. |
| Dopamine | — | Phasic burst, τ=3 steps, 1-step delay | Stands in for sugar GRN → SEZ → PAM/DAN → MB. `dopa_baseline_lr=0.01` makes the burst a reward-prediction error. |

**Plasticity is confined to KC→MBON.** That is `n_kc × n_mbon = 512 × 27 =
13,824` synapses. Every other weight is fixed at construction. A real mushroom
body is far larger and its KC→MBON connectivity is sparse and stereotyped; this
matrix is dense and random. That difference is the single largest gap between
this model and the animal.

## 2. The learning rule

Three-factor reward-modulated STDP (`MushroomBody.apply_dopamine()`, the only
function in the project that writes a weight):

```
dw = lr · D(t) · e(t)
```

- `e(t)` is an eligibility trace accumulated at KC→MBON synapses over the
  stimulus window, decaying at `e_decay = 0.97`, which bridges the reward delay.
- `D(t)` is the phasic dopamine burst.
- With `credit_mode="pattern"`, the credited row's update is `direction ·
  pattern / norm` — the burst contributes only its **sign**, and the magnitude
  comes from the centred KC pattern. This is a covariance rule, and it is why
  the rows converge to class centroids.

A row-norm homeostasis (`target_norm=1.20`, `w_max=0.25`) keeps rows comparable.

### Why this matters: the bug it fixed

Calling `apply_dopamine(trace)` **without naming a row** falls back to
`delta = gain · eligibility`, where the gain is the raw burst sum (~5.6) against
a weight scale of ~0.053. That is a **~7× overshoot**: a single click in the UI
ran a trained brain down to chance (7%). The named-row path is normalised
instead. If you extend this code, do not remove the `credit=`/`punish=` argument.

## 3. The recipe that works

`make_checkpoint.py::build_config()` is the single source of truth:

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

Three fields differ from the `Config` dataclass defaults, and all three matter:

**`reward_mode="pavlovian"`** — the reward is yoked to the *stimulus identity*
rather than to the fly's own guess. Under `operant`, an early wrong answer
suppresses the sugar that would have taught the right one; the rule starves
itself and parks at **60.5%**. Under `pavlovian` the true letter always precedes
the sugar, so the update averages the class's KC patterns into that class's row
on every trial. (A real fly is operant. This is a modelling choice made for
tractability, and it is the difference between a working model and a dead one.)

**`lr_schedule="inv"` with `lr_half_life=300`** — `const` at lr=0.02 takes
full-size steps forever and random-walks around the solution instead of
settling on it.

**`epochs=400`, not 2000** — accuracy reaches ~99% within ~60 epochs and is flat
afterwards. See the convergence note below.

`make_checkpoint.OLD-60pct.py` is the superseded recipe, kept because it
reproduces the 60.5% plateau on demand. If you are writing up a negative result,
that file is the negative result.

## 4. Convergence

`make_checkpoint.py` reports the first epoch that reached within 2 accuracy
points of the final score:

```
holdout accuracy  : 100.0%  (chance 3.7%)
epochs trained    : 400
first within 2pt  : epoch 59
```

So ~85% of the training budget does no measurable work. The 400-epoch default
exists to be safely past convergence, not because 400 was needed. The stale
recipe, for comparison, reaches 60.5% and stops moving.

## 5. Evaluation, and where it can mislead

- `evaluate()` presents each glyph **once** by default. The reported readout
  uses **4 looks** (`LOOKS = 4`), averaged before the argmax.
- **The generator must be passed.** `encode_from_config(cfg, ch, gen=brain.gen)`
  — without `gen`, the function builds a fresh `Generator` seeded `cfg.seed + 5`
  on every call, so all four looks are the *identical* spike train and the
  averaging reduces no variance at all. This bug made the blank glyph read 59%
  in the browser while `trial()` measured 99.5% on the same code.
- Because the receptor sheet is Poisson-sampled from a running generator,
  successive evaluations are independent draws. `evaluate()` returns 100.0%
  immediately after training and ~98.8% later in a long session. Both are real;
  neither is wrong. The clean-glyph alphabet sweep is 810/810.

## 6. The negative control

`digit_proof.py` presents all ten digits and records the answers. It is a
forward-only test: `read_out()` calls `MushroomBody.present()`, which computes
the PN/KC/MBON response and stages an eligibility trace, and **never** calls
`apply_dopamine()`. No `trial()`, no `learn=True`, no `brain.save()`.

Since a digit can never be contained in `CLASSES` — `classes.py` asserts every
answer is a member — the negative control is structural rather than empirical.
The script additionally hashes every weight matrix and the checkpoint file
before and after all 300 presentations and asserts both are unchanged.

See `flybrain/proof.py`, reused by both `digit_proof.py` and
`read_document.py`.

## 7. Document reading

`flybrain/docread.py` extracts text and collapses every whitespace run to a
single space, so a character index is a character index and position is carried
by a page map. `read_document.py` then presents each character:

- **letters and space** — answerable; a right answer is possible.
- **digits** — presentable (a glyph exists) but unanswerable (no output cell),
  so they are always wrong. Recorded separately, not hidden.
- **punctuation** — no glyph; stepped over and excluded from the accuracy, but
  counted.

The step-over test is `is_input(ch)` — membership of the 5×7 font's glyph table,
not `str.isalpha()`. These differ, and the difference is measurable in a real
book: of the 169,527 characters `classify()` calls letters, **542 are PDF
ligatures with no glyph** (`ﬁ` 245, `ﬀ` 169, `ﬃ` 73, `ﬂ` 54, `ﬄ` 1), so only
168,985 letters are presentable. Likewise the stepped-over count is 5,383, not
the 4,841 characters `classify()` marks unreadable — 4,841 + 542. Use the glyph
table for this, never `isalnum()`/`isalpha()`.

Measured on a 132-page, 209,939-character PDF (132 pages, 204,556 presented,
5,383 stepped over, 1,300 s, 157 char/s):

| Bucket | Shown | Right | Accuracy | Mean margin |
| --- | --- | --- | --- | --- |
| letter | 168,985 | 168,985 | 100.0% | 0.313 |
| space | 34,521 | 34,119 | 98.8% | 0.329 |
| digit | 1,050 | 0 | 0.0% | 0.180 |
| all | 204,556 | 203,104 | 99.3% | — |
| answerable (letters + spaces) | 203,506 | 203,104 | 99.8% | — |

All 402 errors are spaces. Weight digest and checkpoint hash were identical
before and after the run.

Scanned PDFs have no text layer and `pypdf` cannot read pixels; `build()` raises
with an explicit message rather than returning an empty document.

## 8. Reproducing everything

```powershell
python -m venv .venv-flybrain
.\.venv-flybrain\Scripts\python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS=1

.\.venv-flybrain\Scripts\python.exe make_checkpoint.py --quiet
.\.venv-flybrain\Scripts\python.exe digit_proof.py --repeats 30 --json runs/digit_proof.json
.\.venv-flybrain\Scripts\python.exe read_document.py <your-book.pdf> --json runs/book_read.json
.\.venv-flybrain\Scripts\python.exe -m pytest -q
```

Everything is seeded from `config.seed` and starts from a naive circuit, so two
runs produce the **same weights**: retraining gives a weight digest of
`626bece30a76927e8fd124eef1c3fdc3…` and `mbon.w` agreeing to 0.0 maximum
absolute difference, converging at epoch 59 with a 102.5 s train loop.

The block above was also run in an empty copy of the repository — no virtualenv,
no caches, no inherited state — as an end-to-end check of the clone-and-run
claim. It passed: `pytest -q` reported 14 passed, `make_checkpoint.py --quiet`
reported 100.0% with first-within-2-point at epoch 59 in 116.7 s wall clock
(102.5 s of it the train loop), `digit_proof.py --repeats 30` reported 0/300
digits with 300/300 in vocabulary and 810/810 on the alphabet control,
`serve.py` answered `GET /` with 200 and 15,069 bytes and `GET /api/state` with
the canonical config, and `read_document.py` read the 132-page PDF at 100.0% on
letters. The install resolved `torch` 2.14.0 and `pypdf` 6.19.0, the latter being
the only dependency used by a single script.

Three separate 400-epoch runs — including that clean copy — produced three
different `.pt` file hashes (`1e56a34a…`, `d231c8bc…`, `56891cdf…`) against the
one weight digest. The saved `.pt` **file** hash is
*not* stable across runs, and it is worth being precise about why, because this
project leans on hashes: `torch.save` writes a zip archive containing a
timestamp, so saving one unchanged in-memory brain twice yields two different
file hashes. A checkpoint hash therefore identifies a particular file on disk —
which is exactly what the before/after guard needs it for, since it compares the
*same path* across a read-only run — while a weight digest is what you compare to
establish reproducibility across runs.
