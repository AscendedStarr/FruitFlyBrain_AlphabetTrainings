# Methods

Everything needed to reproduce the numbers in the README, and every place where
a modelling choice was made rather than a measurement taken.

Section 9 covers the one thing in this repository that is measured rather than
modelled: the `PN → KC` matrix, when `wiring="connectome"`.

## 1. The circuit

Five stages, all in `flybrain/circuit.py`. Sizes are in `flybrain/config.py`,
each annotated with the in-vivo structure it stands in for. The default
configuration is listed first; the `wiring="connectome"` configuration, which
changes only the `PN → KC` matrix and the three dimensions that follow from it,
is shown after the slash.

| Stage | Units | Substrate | Notes |
| --- | --- | --- | --- |
| Receptor sheet | 35 · 35 | `(5, 7)` binary bitmap | Not anatomy. A compound eye has ~3,000 ommatidia per eye; this is a stand-in for "some pixels". Poisson-encoded, rate 0.55 on ink / 0.04 on blank. |
| Antennal lobe (PNs) | 128 · **157** | Graded logistic | 157 is the measured count of hemibrain PNs with traced output to a Kenyon cell (§9). Still **not** LIF: a LIF's informative input range is ~0.15–0.45 for β=0.85, outside which it saturates to one spike per step and destroys the pattern. Common-mode subtraction (`pn_center`) removes the component every letter shares. |
| Kenyon cells | 512 · **1802** | Sparse top-k, k=60 · **k=211** | 1,802 is the measured count of KCs receiving PN input. k is set to hold sparsity at the connectome's 11.7% of the population. Normalised the same way in both configurations, then Bernoulli-sampled. |
| MBONs | 27 · 27 | Single-compartment LIF, β=0.90 | 26 letters + blank. Bias 0.12 puts the drive near `threshold·(1−β) = 0.1`, the graded operating point. |
| Dopamine | — | Phasic burst, τ=3 steps, 1-step delay | Stands in for sugar GRN → SEZ → PAM/DAN → MB. `dopa_baseline_lr=0.01` makes the burst a reward-prediction error. Unchanged by the connectome. |

**Plasticity is confined to KC→MBON.** That is `n_kc × n_mbon`, which is
512 × 27 = **13,824** synapses in the default configuration and
1,802 × 27 = **48,654** in the connectome configuration. Every other weight is
fixed at construction. A real mushroom body's KC→MBON connectivity is sparse and
stereotyped; here it is dense and random **in both configurations**. That
difference is the single largest remaining gap between this model and the animal,
and loading a connectome does not shrink it — see §9.4.

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

Since a digit can never be contained in `CLASSES` — `scripts/classes.py` asserts
every answer is a member — the negative control is structural rather than
empirical.
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
The connectome path needs two more commands, and they are independent of the
block above — no checkpoint has to exist first:

```powershell
.\\.venv-flybrain\Scripts\python.exe tools/build_connectome.py   # ~46 MB public download -> data/hemibrain_mb.npz
.\\.venv-flybrain\Scripts\python.exe connectome_compare.py --seeds 7,11,13 --epochs 400
```

`data/hemibrain_mb.npz` is committed, so the download is only needed to rebuild
the extract from source. `--epochs` and `--seeds` are independent knobs;
`--quick` forces 8 epochs for a plumbing check.
Everything is seeded from `config.seed` and starts from a naive circuit, so two
runs produce the **same weights**: retraining gives a weight digest of
`626bece30a76927e8fd124eef1c3fdc3…` and `mbon.w` agreeing to 0.0 maximum
absolute difference, converging at epoch 59 with a 102.5 s train loop.

The block above was also run in an empty copy of the repository — no virtualenv,
no caches, no inherited state — as an end-to-end check of the clone-and-run
claim. It passed: `pytest -q` reported 14 passed at v0.1.0 and 33 as of v0.2.0
(the 19 connectome tests were added), `make_checkpoint.py --quiet`
reported 100.0% with first-within-2-point at epoch 59 in 116.7 s wall clock
(102.5 s of it the train loop), `digit_proof.py --repeats 30` reported 0/300
digits with 300/300 in vocabulary and 810/810 on the alphabet control,
`serve.py` answered `GET /` with 200 and 15,069 bytes at v0.1.0 and 15,071 as of
v0.2.0 (the demo phrase in `web/index.html` became two characters longer) and
`GET /api/state` with the canonical config, and `read_document.py` read the
132-page PDF at 100.0% on letters. The install resolved `torch` 2.14.0 and `pypdf` 6.19.0, the latter being
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

## 9. The one measured layer

The only material in this repository that was not chosen by its author is the
`PN → KC` matrix in the `wiring="connectome"` configuration. This section is the
whole account of where it comes from and what was done to it.

### 9.1 Source

| | |
| --- | --- |
| Dataset | **hemibrain v1.2** |
| Citation | Scheffer et al. 2020, *eLife* **9**:e57443, doi:10.7554/eLife.57443 |
| Licence | **CC BY 4.0** — permissive, no non-commercial restriction |
| Access | `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz`, no credentials |
| Curated cell types | Schlegel et al. 2021, *eLife* **10**:e66018, doi:10.7554/eLife.66018 |

FlyWire was considered and rejected on licence grounds: it is CC BY-**NC**, which
is a poor fit for a repository published under MIT. Codex was rejected because it
is token-gated. The hemibrain export is public, stable, and CC BY 4.0.

### 9.2 The extraction, and the filter that had to be added

The naive membership rule is a substring test on the cell type:

| Class | Rule |
| --- | --- |
| `PN` | `"PN" in type` |
| `KC` | `type.startswith("KC")` |
| `MBON` | `type.startswith("MBON")` |
| `DAN` | `type.startswith("PPL") or type.startswith("PAM")` |
| `APL` | `type == "APL"` |

Applied alone, that yields 428 PNs and 1,927 KCs — and **271 of those PNs have
exactly zero synapses onto any Kenyon cell.** Three independent checks were run
before that was believed, and only the third explains it:

1. **Edge counts.** Those 271 neurons have 39–780 outgoing edges each (median
   126) and exactly **zero** of them land on a KC. Not a join failure.
2. **Nomenclature.** Their names are legitimate — `DA1_vPN`, `M_lvPNm24`,
   `WEDPN8C`, `MZ_lvPN`, `DP1m_vPN`. A name-based rule cannot exclude them.
3. **Curated tables.** Of 166 `mPN` (multiglomerular) neurons, **140 have no KC
   output**; of 181 `uPN`, 118 do. The `mPN`s and the 91 `WEDPN*` wedge neurons
   project to the **lateral horn**, not the mushroom body calyx. That is correct
   *Drosophila* anatomy.

**The resolution is the calyx filter, and it is the difference between using this
dataset and misusing it:** membership is decided by **connectivity, not
nomenclature**. A PN is kept iff it has at least one traced synapse onto a KC; a
KC is kept iff at least one PN reaches it. Unmatched edges fall from 271 to
**2**.

| | Naive rule | Calyx filter |
| --- | --- | --- |
| PNs | 428 | **157** |
| KCs | 1,927 | **1,802** |
| PNs with no KC output | 271 | 0 (by construction) |
| Unmatched edges | 271 | 2 |

PN composition after filtering: 118 `uPN`, 26 `mPN`, 11 `biPN`, 2 absent from the
curated table.

### 9.3 Measured connectivity, and the three modelled transforms

| Connection | Edges | Synapses |
| --- | --- | --- |
| PN → KC | 12,426 | 185,994 |
| KC → MBON | 28,833 | 299,953 |
| DAN → KC | 61,735 | 120,804 |

KC fan-in after filtering: min 1, median 6, mean 6.90, max 21. Each MBON samples
a median of 317 of the 1,802 KCs. `data/hemibrain_mb.npz` stores only the PN→KC,
KC→MBON and DAN→KC edge lists plus neuron ids and types — 173.9 KB, against ~46 MB
of raw CSVs and tarball, which are gitignored.

The measurement ends at *which cells connect and how many synapses are between
them*. Everything downstream of that is a modelling decision, and there are
exactly three:

| Transform | Why |
| --- | --- |
| Weight magnitude = **√(synapse count)**, not the count | A raw count lets one 400-synapse partner dominate four 4-synapse partners by two orders of magnitude. The square root compresses that range while preserving order. |
| Each KC row **L2-normalised to unit norm** | Puts every KC at the same input magnitude, so the connectome arm cannot win purely by being louder. Exact for both connectome arms. The random arm is only *nominally* at this scale — it draws weights with variance 1/`fan_in`, so its rows average 0.96 over a spread of 0.20–2.15 (median 0.94, sd 0.28 at `fan_in`=6). |
| All synapses **excitatory** | PN→KC is cholinergic, so the sign is right even though the export does not record it. |

**All three are applied identically to the connectome and shuffled arms**, so they
cannot manufacture a difference between them. The random arm shares the √
weighting and the excitatory sign but is only nominally row-normalised, which is
why the shuffled arm — not the random one — carries the weight of the comparison.
They do mean that any "the real wiring is better" claim here is a claim about the
*pattern* of the connectome, not about its conductances: the export records
synapse counts, not synaptic strengths.

### 9.4 The three arms

`connectome_compare.py` holds the seed, the hyperparameters, the encoder and the
learning rule fixed, and changes exactly one thing — the `PN → KC` matrix — across
three arms. Dimensions are taken from the connectome, so all three arms run at
157 PNs, 1,802 KCs, *k*=211 and 48,654 plastic synapses:

| Arm | `PN → KC` | Purpose |
| --- | --- | --- |
| `connectome` | the measured wiring | — |
| `connectome-shuffled` | each KC's partner set replaced by a uniformly random set **of the same size**, keeping that KC's exact degree and weight multiset | isolates partner **identity** while holding fan-in exactly fixed |
| `random` | the synthetic generator at the connectome's dimensions, uniform fan-in | the synthetic baseline at matched scale |

The shuffle is the experiment. It is a degree-matched permutation of the partner
*identities* only, so a difference between `connectome` and `connectome-shuffled`
is attributable to which specific partners the electron microscope detected, and
nothing else. `wiring="random"` is deliberately **not** an accepted value of
`pn_kc_weights()` — passing it raises `ValueError`, so the control cannot be
accidentally conflated with the measured arm.

Results, over three seeds at 400 epochs:

| Arm | seed 7 | seed 11 | seed 13 | pooled |
| --- | --- | --- | --- | --- |
| `connectome` | 81/81 | 81/81 | 81/81 | **243/243** |
| `connectome-shuffled` | 81/81 | 81/81 | 81/81 | **243/243** |
| `random` | 77/81 | 74/81 | 81/81 | 232/243 |

Paired `connectome` − `connectome-shuffled`: **+0 items on every seed.**

**Conclusion, stated plainly:** at this task the specific detected partner
identities contribute nothing measurable over equal-degree random partners. The
`random` arm differs by ~4.5 points, but it differs along two axes at once, since
the two connectome arms carry the real fan-in *distribution* (min 1, median 6,
mean 6.90, max 21) while the `random` arm uses a uniform 6. Because the shuffle
already rules out identity, the residual gap is attributable to the shape of the
fan-in distribution and the 3.5× larger circuit — not to the connectome's
specific partners.

**Two limitations that must travel with that conclusion.** First, the task is too
easy to discriminate the arms: 27 clean glyphs with an 81-item holdout means both
connectome arms sit at the ceiling, and a 100.0% vs 100.0% tie demonstrates that
the test cannot separate them rather than that they are equivalent. Second, the
layer that actually learns — `KC → MBON`, 48,654 plastic synapses — is still
invented, dense and random, so any genuine anatomical advantage would have to
survive that bottleneck to become visible. Nothing here is evidence about
*Drosophila*; it is a result about this model.

