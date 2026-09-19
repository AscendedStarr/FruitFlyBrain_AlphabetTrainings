"""Sweep the learning rate for the 27-class task now that credit is fixed.

`diag_rule.py` showed the old rule could not learn even 2 classes, and that
naming the credited MBON row fixes it. This re-tunes lr against the real task.
"""

from __future__ import annotations

import copy
import time

from flybrain import Config
from flybrain.trainer import FlyBrain

LRS = [0.004, 0.010, 0.020, 0.040, 0.080]
EPOCHS = 60


def main() -> None:
    print(f"27-class, {EPOCHS} epochs, chance = 3.7%\n")
    print(f"{'lr':>7} {'final':>7} {'best':>7} {'secs':>6}   accuracy every 6 epochs")
    results = []
    for lr in LRS:
        c = copy.deepcopy(Config())
        c.lr = lr
        c.epochs = EPOCHS
        t0 = time.perf_counter()
        brain = FlyBrain(c)
        brain.train(verbose=False)
        secs = time.perf_counter() - t0
        hist = [h.accuracy for h in brain.history]
        results.append((hist[-1], lr, secs, hist))
        print(f"{lr:>7.3f} {hist[-1]:>6.1%} {max(hist):>6.1%} {secs:>6.0f}   "
              + " ".join(f"{v:>4.0%}" for v in hist[::6]))

    best_acc, best_lr, _, best_hist = max(results)
    print(f"\nbest lr={best_lr} -> {best_acc:.1%} "
          f"(first 12 epochs: {' '.join(f'{v:.0%}' for v in best_hist[:12])})")


if __name__ == "__main__":
    main()
