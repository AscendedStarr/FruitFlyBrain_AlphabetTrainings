# The scripts from the hunt

The working scripts behind the numbers in the root `README.md`,
`docs/methods.md` and `docs/research-report.md`. They live here so the
repository root stays readable. Nothing in `flybrain/` or `web/` imports them,
and none of them is needed to run the demo.

Run one with the project virtual environment:

```
.\.venv-flybrain\Scripts\python.exe scripts\looks.py
```

Each script puts the repository root on its own `sys.path`, so `import flybrain`
resolves from here, and resolves the paths it reads and writes from the
repository root as well. The directory you run from therefore does not matter —
but `runs/flybrain.pt` does have to exist, so these need a checkout that
includes the tracked checkpoint.

They are **not** part of the test suite: `pytest.ini` collects only `tests/`.
The scripts here named `test_*.py` are one-off regressions kept next to the code
they check, not pytest tests.

## Entry points

These are the ones the root `README.md` documents and that the demo depends on.
Everything after this section is a one-off measurement.

- `fly_brain.py` — the command-line driver: `demo`, `train`, `say`, `diag`.
- `serve.py` — the local interactive web UI: a stdlib HTTP server plus a handful
  of JSON endpoints, serving `web/`.
- `make_replay.py` — record a real session to `web/replay.json` so the UI can run
  as a static page on GitHub Pages. Inference only (`learn=False`), which is why
  the recording is reusable. Asserts the captured figures against the published
  ones and refuses to write a replay that disagrees with them.
- `build_site.py` — assemble `_site/` for Pages: `web/` at the root, plus the
  two lines that point `index.html` at `replay.js`. Stdlib only, because CI has
  no torch.
- `read_document.py` — read a real document with the trained fly and record
  exactly what it said, with the same hash guard the page uses.
- `digit_proof.py` — the digit negative control: show the fly all ten digits and
  record the answers, so the `0 / 300` in the README can be regenerated.
- `connectome_compare.py` — train the same task on a real connectome and on
  controls, and compare. Same glyphs, same initialisation, same dimensions.
- `make_checkpoint.py` — train one good checkpoint for the web UI (and for
  `digit_proof.py`) to serve.
- `make_checkpoint.OLD-60pct.py` — the same, with the recipe that plateaus at
  60.5%. Kept to show that the ceiling was the recipe.

## Which of these write the checkpoint

`runs/flybrain.pt` — the file `serve.py` loads as the demo's brain — can be
overwritten by **five** scripts here, and in every case the default output path
is the canonical checkpoint itself:

- `keep_training.py` — the long run that produced it. Also streams progress to
  `runs/sweep.log`.
- `fly_brain.py train` and `fly_brain.py demo` — `--out` defaults to it, so a
  plain `demo` replaces the trained brain. `fly_brain.py say --retrain` writes
  it too. Pass `--out` elsewhere, or copy the checkpoint first.
- `make_checkpoint.py` — `--out` defaults to it.
- `make_checkpoint.OLD-60pct.py` — a fixed output path, so it always overwrites.
- `serve.py` — saves back to its `--ckpt` (the canonical checkpoint by default)
  when you train or save from the browser.

Three write throwaway checkpoints instead of the canonical one: `converge.py`
writes `runs/alphabet_<lr>.pt`, `connectome_compare.py` writes
`runs/wiring_<arm>_seed<N>.pt`, and `test_resume.py` writes
`runs/_resume_test.pt`.

## Reading the trained circuit

These load `runs/flybrain.pt` and neither train nor save:

- `looks.py` — accuracy against number of looks, which is what separated the
  weight ceiling from sampling noise.
- `blank.py` — how reliable the blank glyph is, 200 questions per look count.
- `classes.py` — per-class check over all 27 classes, blank included.
- `confusions.py` — what the brain gets wrong, and whether it is random.
- `repeat_readout.py` — whether reading a letter more than once fixes the noise.
- `why_mn.py` — why `M` and `N` fail while `O` and `Q` are almost right.
- `why_wrong.py` — phrase arithmetic: why a 16-letter phrase never comes out
  right, and how long the run that got to 63% actually took.

## Diagnostics and tuning

- `diagnose.py` — the stage-by-stage `ladder` probe; `sweep.py` imports
  `probe_accuracy` from it, so those two belong together.
- `diag_learn.py`, `diag_mbon.py`, `diag_readout.py`, `diag_rule.py` — which of
  the three factors was actually broken.
- `calib_lr.py` — calibrating the three-factor learning rate against the weight
  scale.
- `tune_lr.py`, `tune_learn.py`, `tune_rule.py`, `tune_readout.py`,
  `tune_schedule.py`, `converge.py` — the search for a recipe that converges.
- `keep_training.py` — the sweep over `trials_per_class`, `decision_repeats` and
  `n_kc`, then the long run of the winner.
- `bench_gpu.py` — where the training time goes, on a freshly built circuit.

## Regressions

- `test_docread.py` — the document reader. Stdlib only, except the PDF case.
- `test_resume.py` — that a checkpoint round-trips, and that a brain built from a
  different config refuses to load it. This one trains, into its own file.
- `test_click.py`, `test_doc_http.py` — these drive the live HTTP API and need
  `serve.py` already running on `http://127.0.0.1:8000`.
