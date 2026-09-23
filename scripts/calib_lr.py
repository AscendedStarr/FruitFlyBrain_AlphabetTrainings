"""Calibrate the three-factor learning rate.

The right `lr` is the one where the weight step per trial is a small but
meaningful fraction of the weight matrix itself. Guessing it is how this
circuit spent a long time learning nothing, so measure the ratio directly and
only then sweep.
"""

from __future__ import annotations

import copy
import time

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES, Config, FlyBrain

LRS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
EPOCHS = 15


def measure_step(cfg: Config) -> tuple[float, float]:
    """||dw|| / ||w|| for a single learning trial."""
    brain = FlyBrain(copy.deepcopy(cfg))
    before = brain.mb.mbon.w.clone()
    brain.trial("A", learn=True)
    after = brain.mb.mbon.w
    return (float((after - before).norm()), float(before.norm()))


def main() -> None:
    cfg = Config()

    dw, w = measure_step(cfg)
    print(f"weight matrix norm      ||w||  = {w:.4f}")
    print(f"one-trial step          ||dw|| = {dw:.6f}")
    print(f"relative step                  = {dw / w:.6f}")
    print(f"  (want roughly 1e-3 .. 2e-2 per trial; {cfg.epochs} epochs x "
          f"{cfg.trials_per_class * (len(CLASSES) - 1)} trials/epoch)")
    print()

    print(f"{'lr':>8} {'acc':>8} {'secs':>7}  reward")
    best = (0.0, 0.0)
    for lr in LRS:
        c = Config()
        c.lr = lr
        c.epochs = EPOCHS
        torch.manual_seed(c.seed)
        t0 = time.perf_counter()
        brain = FlyBrain(c)
        brain.train(verbose=False)
        acc = brain.evaluate()
        secs = time.perf_counter() - t0
        reward = brain.history[-1].reward_mean if brain.history else 0.0
        flag = ""
        if acc > best[1]:
            best = (lr, acc)
            flag = "  <-"
        print(f"{lr:>8.4f} {acc:>7.1%} {secs:>7.1f}  {reward:+.3f}{flag}")

    print(f"\nbest: lr={best[0]} -> {best[1]:.1%}  (chance = "
          f"{1 / len(CLASSES):.1%})")


if __name__ == "__main__":
    main()
