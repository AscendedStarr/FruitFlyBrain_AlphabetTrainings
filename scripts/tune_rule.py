"""Compare the two dopamine credit rules and time a full convergence run.

``credit_mode="pattern"`` impresses the centred KC pattern of the stimulus onto
the credited row, which is what the 96% centroid readout uses.
``credit_mode="outer"`` uses the eligibility outer product, which is signed by
whether that MBON happened to fire. The difference matters most on letters the
fly is currently getting wrong, which is exactly where learning has to happen.

Also reports wall-clock per epoch so the total training time is predictable.
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

MODES = ["pattern"]
LRS = [0.020, 0.035, 0.050, 0.080, 0.120]
EPOCHS = 400


def main() -> None:
    chance = 1 / 27
    print(f"27-class alphabet, {EPOCHS} epochs, chance = {chance:.1%}\n")
    print(f"{'mode':>9} {'lr':>7} {'final':>7} {'best':>7} {'s/epoch':>8}   curve")
    results = []
    for mode in MODES:
        for lr in LRS:
            c = copy.deepcopy(Config())
            c.credit_mode = mode
            c.lr = lr
            c.epochs = EPOCHS
            t0 = time.perf_counter()
            brain = FlyBrain(c)
            brain.train(verbose=False)
            secs = time.perf_counter() - t0
            hist = [h.accuracy for h in brain.history]
            results.append((max(hist), mode, lr, hist[-1], secs))
            print(f"{mode:>9} {lr:>7.3f} {hist[-1]:>6.1%} {max(hist):>6.1%} "
                  f"{secs / EPOCHS:>8.3f}   "
                  + " ".join(f"{v:>4.0%}" for v in hist[::12]))

    best_acc, mode, lr, final, secs = max(results)
    per_epoch = secs / EPOCHS
    print(f"\nbest: {mode} @ lr={lr} -> {best_acc:.1%} (best), {final:.1%} (final)")
    print(f"per epoch     : {per_epoch:.3f} s")
    for target in (300, 1000, 3000):
        print(f"  {target:>5} epochs : {target * per_epoch / 60:6.1f} min")


if __name__ == "__main__":
    main()
