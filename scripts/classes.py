"""Per-class check over ALL 27 classes, including the blank glyph.

The earlier probe iterated "ABC..." - 26 letters - and missed that `CLASSES`
has a 27th entry: the blank/space glyph. The model is perfect on A-Z but the
blank is the one class it still fumbles. That is what reading a whole phrase
exposes: the spaces come out as letters.

Run:  $env:OMP_NUM_THREADS=1; .\\.venv-flybrain\\Scripts\\python.exe classes.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES
from flybrain.trainer import FlyBrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brain = FlyBrain.load(os.path.join(ROOT, "runs", "flybrain.pt"))

print("=" * 62)
print(f"class list ({len(CLASSES)}): {CLASSES}")
print("=" * 62)

for looks in (1, 2, 4, 8):
    brain.cfg.decision_repeats = looks
    brain.cfg.eval_repeats = 10
    print(f"\n{looks} look(s) per answer, 10 questions each:")
    print(f"  {'class':>6} {'correct':>9}   misread as")
    print("  " + "-" * 44)
    bad = 0
    for ch in CLASSES:
        preds = Counter()
        hits = 0
        for _ in range(10):
            res = brain.trial(ch, learn=False, augment=False)
            hits += int(res.correct)
            if not res.correct:
                preds[res.predicted] += 1
        label = "space" if ch == " " else ch
        mark = "" if hits == 10 else "  <--"
        if hits < 10:
            bad += 1
        detail = ", ".join(f"{p} x{n}" for p, n in preds.most_common())
        print(f"  {label:>6} {hits:>7}/10   {detail}{mark}")
    print(f"  -> {len(CLASSES) - bad}/{len(CLASSES)} classes always correct")

# Which MBON row is meant to answer for the blank, and how strong is it?
brain.cfg.decision_repeats = 2
print()
print("why the blank is hard: receptor ink per class")
from flybrain import FONT_5X7  # noqa: E402

ink = {ch: sum(row.count("#") for row in FONT_5X7[ch]) for ch in CLASSES}
for ch in sorted(ink, key=lambda c: ink[c]):
    label = "space" if ch == " " else ch
    if ink[ch] <= 6:
        print(f"  {label:>6}: {ink[ch]:>2} lit pixels of 35")
