"""How reliable is the blank glyph, really?

The blank is the one class with 0 lit pixels of 35. Its Kenyon-cell code is
therefore driven by nothing but the baseline Poisson noise, so how well it can be
read is a question of how many looks are averaged - and it needs far more looks
than a letter does. `classes.py` reported 10/10 at 4 looks, but that is 10
samples, which is not enough to tell 100% from 92%: the demo phrase contains a
single blank, so the whole phrase is only as reliable as the blank class is.

This measures the blank properly, and checks that extra looks do not start
hurting the 26 real letters (they should only help: averaging reduces variance).

Run:  $env:OMP_NUM_THREADS=1; .\\.venv-flybrain\\Scripts\\python.exe blank.py
"""

from __future__ import annotations

import os

from flybrain import CLASSES
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
brain = FlyBrain.load(os.path.join(HERE, "runs", "flybrain.pt"))
N = 200  # questions per look count

print("=" * 70)
print(f"blank-glyph reliability ({N} questions per row, epoch {brain.epoch})")
print("=" * 70)
print(f"{'looks':>6} {'blank':>8} {'letters':>9} {'all 27':>9}   worst letters")
print("-" * 70)

for looks in (1, 2, 4, 6, 8, 12, 16):
    brain.cfg.decision_repeats = looks

    # The blank, measured properly.
    blank_hits = 0
    for _ in range(N):
        res = brain.trial(" ", learn=False, augment=False)
        blank_hits += int(res.correct)

    # Every real letter, to make sure extra looks never cost anything.
    per_letter = {}
    for ch in CLASSES:
        if ch == " ":
            continue
        hits = 0
        for _ in range(N):
            hits += int(brain.trial(ch, learn=False, augment=False).correct)
        per_letter[ch] = hits / N

    letters = sum(per_letter.values()) / len(per_letter)
    # Weight the 27 classes equally, as `evaluate` does.
    overall = (blank_hits / N + sum(per_letter.values())) / len(CLASSES)

    worst = sorted(per_letter.items(), key=lambda kv: kv[1])[:3]
    worst_s = " ".join(f"{c}:{v:.0%}" for c, v in worst if v < 1.0) or "none below 100%"
    print(f"{looks:>6} {blank_hits / N:>7.1%} {letters:>8.1%} {overall:>8.1%}   {worst_s}")

print("-" * 70)
print("'PHILIPPE DELAMBRE' has one blank, so the phrase is exactly as")
print("reliable as the blank row above.")
