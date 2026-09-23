"""Can the rule reach the ceiling the circuit has already been shown to have?

A hand-built centroid readout scores 96.2% through this same spiking MBON layer,
so the circuit can represent the alphabet. The trained rule plateaus at ~60%.
The remaining gap is the rule, and the most likely cause is that a *constant*
step never settles: every trial imprints a noisy sample of the class, so the
credited row random-walks around the class mean rather than converging to it.

Averaging schedules shrink the step and should let the row converge onto the
class mean - which is the centroid classifier. This sweeps them.
"""

from __future__ import annotations

import copy
import time

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import Config
from flybrain.trainer import FlyBrain

EPOCHS = 400
RUNS = [
    ("const", 0.02, 40.0),
    ("sqrt", 0.02, 40.0),
    ("inv", 0.02, 40.0),
    ("sqrt", 0.02, 150.0),
    ("inv", 0.02, 150.0),
    ("sqrt", 0.05, 40.0),
]


def main() -> None:
    chance = 1 / 27
    print(f"27-class alphabet, {EPOCHS} epochs, chance {chance:.1%}, "
          f"measured ceiling 96.2%\n")
    print(f"{'sched':>7} {'lr':>6} {'half':>6} {'final':>7} {'best':>7} "
          f"{'s/run':>6}   curve")
    results = []
    for sched, lr, half in RUNS:
        c = copy.deepcopy(Config())
        c.lr_schedule = sched
        c.lr = lr
        c.lr_half_life = half
        c.epochs = EPOCHS
        t0 = time.perf_counter()
        brain = FlyBrain(c)
        brain.train(verbose=False)
        secs = time.perf_counter() - t0
        hist = [h.accuracy for h in brain.history]
        results.append((max(hist), hist[-1], sched, lr, half))
        print(f"{sched:>7} {lr:>6.3f} {half:>6.0f} {hist[-1]:>6.1%} "
              f"{max(hist):>6.1%} {secs:>6.0f}   "
              + " ".join(f"{v:>4.0%}" for v in hist[::40]))

    best_best, best_final, sched, lr, half = max(results)
    print(f"\nbest {sched} lr={lr} half={half:.0f}: "
          f"peak {best_best:.1%}, final {best_final:.1%}")
    print(f"ceiling available: 96.2%   -> gap closed "
          f"{(best_best - 0.60) / (0.962 - 0.60):.0%} of the way from 60%")


if __name__ == "__main__":
    main()
