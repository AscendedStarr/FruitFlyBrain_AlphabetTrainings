"""Does reading a letter more than once fix the noise?

`M` came back as eight different letters in eight presentations, so a single
presentation is a very small sample. A real fly does not get one look at a
stimulus either. This averages the MBON spike counts over N presentations of
the same glyph and asks how accuracy grows with N.

This is a readout-side change only: nothing is retrained, no extra labels are
used, and the circuit is exactly the one on disk.
"""

from __future__ import annotations

import os

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES
from flybrain.trainer import FlyBrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brain = FlyBrain.load(os.path.join(ROOT, "runs", "flybrain.pt"))
letters = [c for c in CLASSES if c != " "]
print(f"epoch {brain.epoch}, one-shot holdout {brain.evaluate():.1%}\n")
print(f"{'looks':>6} {'accuracy':>9}   phrase 'PHILIPPE DELAMBRE'")


def accuracy(repeats: int, trials: int = 5) -> float:
    hits = total = 0
    for _ in range(trials):
        for ch in letters:
            acc = torch.zeros(len(CLASSES))
            with torch.no_grad():
                for _ in range(repeats):
                    acc += brain.mb.present(brain._encode(ch, augment=False))
            hits += int(CLASSES[int(acc.argmax())] == ch)
            total += 1
    return hits / total


for n in (1, 2, 3, 5, 9, 15, 25):
    a = accuracy(n)
    print(f"{n:>6} {a:>9.1%}   {a ** 12:>10.4%}")
