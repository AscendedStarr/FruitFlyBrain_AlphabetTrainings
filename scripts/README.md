# The scripts from the hunt

The working scripts behind the numbers in the root `README.md`,
`docs/methods.md` and `docs/research-report.md`. They live here so the
repository root stays readable. Nothing in `flybrain/`, `serve.py` or `web/`
imports them, and none of them is needed to run the demo.

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

## Only one of these writes the checkpoint

`keep_training.py` is the long run that produced `runs/flybrain.pt` — the file
`serve.py` loads as the demo's brain. **Running it retrains and overwrites that
checkpoint.** It also streams progress to `runs/sweep.log`.

Two others write throwaway checkpoints instead of the canonical one:
`converge.py` writes `runs/alphabet_<lr>.pt` and `test_resume.py` writes
`runs/_resume_test.pt`. No other script here writes to `runs/` at all.

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
