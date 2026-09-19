"""Why is it still guessing letters wrong?

Written while the recipe was stuck at 60.5% and the browser showed a naive fly.
Kept because the arithmetic in part 2 is still exactly right, and because the
diagnosis in part 1 is the reason `serve.py` now loads a checkpoint at startup.

Answers two separate questions with numbers rather than opinion.

1. What is actually loaded when you open the page? At the time this was written
   `serve.py` built a brand new `FlyBrain` on every start and never read a
   checkpoint, so whatever the browser trained was discarded on reload and the
   page showed a naive fly. **That is fixed**: `serve.py` now loads
   `runs/flybrain.pt` at startup and says so - `loaded flybrain.pt: epoch 400,
   100.0% accuracy`. The table below is the evidence for the original claim, not
   a description of current behaviour.

2. Even a fully trained brain cannot read a phrase. Each letter is classified
   independently, so the probability of a whole phrase is the per-letter
   accuracy raised to the number of letters. That compounds catastrophically.
"""

from __future__ import annotations

import glob
import os

from flybrain import CLASSES, Config
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
PHRASE = "MY NAME IS JEFF"

print("=" * 68)
print("1. what is on disk, and what does it actually score?")
print("=" * 68)
print(f"{'checkpoint':<26} {'format':>6} {'epoch':>6} {'saved':>7} {'fresh':>7}")
for path in sorted(glob.glob(os.path.join(HERE, "runs", "*.pt"))):
    try:
        brain = FlyBrain.load(path)
    except Exception as exc:                       # noqa: BLE001
        print(f"{os.path.basename(path):<26}  unreadable: {exc}")
        continue
    saved = brain.best_accuracy
    fresh = brain.evaluate()
    print(f"{os.path.basename(path):<26} {'-':>6} {brain.epoch:>6} "
          f"{saved:>6.1%} {fresh:>6.1%}")

print()
print("=" * 68)
print("2. why a phrase can never come out right")
print("=" * 68)
letters = len(PHRASE.replace(" ", ""))
print(f"phrase {PHRASE!r} has {letters} letters, each classified separately")
print("probability of the whole phrase = per-letter accuracy ^ letters\n")
print(f"{'per-letter':>11} {'chance 1/27':>12}  phrase reads correctly")
for acc in (0.037, 0.40, 0.60, 0.63, 0.80, 0.90, 0.95, 0.99, 0.999):
    p = acc ** letters
    label = f"{p:.6%}" if p >= 1e-6 else f"{p:.2e}"
    note = ""
    if acc == 0.63:
        note = "  <- best this rule has reached"
    if acc == 0.962:
        note = "  <- the circuit's proven ceiling"
    print(f"{acc:>10.1%} {label:>12}   {note}")

print()
print("so, how good must one letter be?")
need = 0.5 ** (1 / letters)
print(f"  to read {PHRASE!r} correctly half the time you need "
      f"{need:.1%} per letter")
need90 = 0.9 ** (1 / letters)
print(f"  to read it correctly 90% of the time you need {need90:.1%} per letter")
print(f"  the circuit's measured ceiling is 96.2% per letter, which gives "
      f"{0.962 ** letters:.2%} per phrase")

print()
print("=" * 68)
print("3. how long does training actually take?")
print("=" * 68)
print(f"  measured cost: 0.266 s per epoch (27-class, {len(CLASSES)} classes)")
for epochs in (100, 400, 1000, 3000, 10000):
    print(f"  {epochs:>6} epochs -> {epochs * 0.266 / 60:>6.1f} min")
print()
print("  the 400-epoch run that reached 63% took 106 s. the five hours were")
print("  spent on diagnosis and UI work, not on training.")
