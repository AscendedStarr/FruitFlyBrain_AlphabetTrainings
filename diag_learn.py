"""Is the plasticity rule broken, or just starved of signal?

`diag_readout.py` showed the KC representation is 96% linearly separable and the
MBON layer can express the answer, so any failure to learn is the rule's fault.

This isolates it with the easiest possible task: two letters, two MBONs, a
50%-chance baseline. A working three-factor rule should drive that to ~100%
within a few dozen trials. If it cannot, the problem is the rule itself and no
amount of tuning the 27-class task will fix it.
"""

from __future__ import annotations

import copy
import torch

from flybrain import CLASSES, Config
from flybrain.trainer import FlyBrain

PAIR = [("A", 0), ("B", 1)]
LRS = [0.0005, 0.002, 0.008, 0.03]


def two_class(cfg: Config, epochs: int, verbose: bool) -> tuple[float, list[float]]:
    brain = FlyBrain(cfg)
    curve: list[float] = []

    def evaluate() -> float:
        hits = 0
        for ch, tgt in PAIR:
            counts = brain.mb.present(brain._encode(ch, False))
            hits += int(int(counts[:2].argmax()) == tgt)
        return hits / len(PAIR)

    for ep in range(1, epochs + 1):
        for ch, tgt in PAIR:
            counts = brain.mb.present(brain._encode(ch, True))
            ok = int(counts[:2].argmax()) == tgt
            burst = brain.dopamine.deliver(
                cfg.sucrose if ok else cfg.bitter)
            brain.mb.apply_dopamine(burst.trace)

        if ep % 20 == 0 or ep == 1:
            curve.append(evaluate())
            if verbose:
                # distance from each MBON's row to each letter's KC centroid
                print(f"    epoch {ep:>4}  acc {curve[-1]:5.1%}")
    return curve[-1], curve


def main() -> None:
    print("2-class A/B, chance = 50%")
    for lr in LRS:
        c = copy.deepcopy(Config())
        c.lr = lr
        acc, curve = two_class(c, epochs=200, verbose=False)
        print(f"  lr={lr:<7} final {acc:5.1%}   curve "
              + " ".join(f"{v:.0%}" for v in curve))

    print("\n27-class, chance = 3.7%  (40 epochs each)")
    for lr in LRS:
        c = copy.deepcopy(Config())
        c.lr = lr
        c.epochs = 40
        brain = FlyBrain(c)
        brain.train(verbose=False)
        hist = [h.accuracy for h in brain.history]
        print(f"  lr={lr:<7} final {hist[-1]:6.1%}   curve "
              + " ".join(f"{v:.0%}" for v in hist[::5]))


if __name__ == "__main__":
    main()
