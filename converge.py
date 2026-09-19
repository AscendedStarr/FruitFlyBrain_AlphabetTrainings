"""Train the alphabet to convergence and report the learning curve.

`tune_learn.py` showed the corrected rule climbs steadily and is still rising
when it runs out of epochs. This runs it long enough to see where it plateaus,
and checks whether sugar alone is more stable than sugar-and-bitter.
"""

from __future__ import annotations

import copy
import time

from flybrain import Config
from flybrain.trainer import FlyBrain

RUNS = [
    ("sugar+bitter", 0.008, True),
    ("sugar+bitter", 0.012, True),
    ("sugar+bitter", 0.020, True),
]
EPOCHS = 600


def main() -> None:
    chance = 1 / 27
    print(f"27-class alphabet, {EPOCHS} epochs, chance = {chance:.1%}\n")

    for name, lr, punish in RUNS:
        c = copy.deepcopy(Config())
        c.lr = lr
        c.punish_wrong = punish
        c.epochs = EPOCHS
        t0 = time.perf_counter()
        brain = FlyBrain(c)
        brain.train(verbose=False)
        secs = time.perf_counter() - t0

        hist = [h.accuracy for h in brain.history]
        best, best_ep = max((a, i + 1) for i, a in enumerate(hist))
        tail = sum(hist[-50:]) / 50
        print(f"{name:>12}  lr={lr:<6} {secs:>5.0f}s   "
              f"final {hist[-1]:6.1%}   best {best:6.1%} @ep{best_ep:<4} "
              f"last-50 {tail:6.1%}   x{tail / chance:.1f} chance")
        print("             " + " ".join(f"{v:>4.0%}" for v in hist[::50]) + "\n")

        brain.save(f"runs/alphabet_{lr}.pt")


if __name__ == "__main__":
    main()
