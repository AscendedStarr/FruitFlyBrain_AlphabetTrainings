"""Sweep the learning rate.

The eligibility matrix is 27x512 but only a few hundred entries see a co-active
KC/MBON pair on any given trial, so its mean-absolute value is far smaller than
its max. The 'worst case' figure from ``diagnose.py`` therefore badly overstates
the typical weight step and the useful ``lr`` has to be found empirically.
"""

from __future__ import annotations

import time

from flybrain.config import Config
from flybrain.trainer import FlyBrain

LRS = [0.15, 0.30, 0.60, 1.20]
EPOCHS = 40


def main() -> None:
    print(f"lr sweep, {EPOCHS} epochs each (chance {1 / 27:.1%})")
    print(f"{'lr':>7} {'acc':>7} {'reward':>8} {'dopa-base':>10} {'secs':>6}")
    print("-" * 42)
    for lr in LRS:
        cfg = Config()
        cfg.lr = lr
        cfg.epochs = EPOCHS
        cfg.log_every = 10**9          # silence the per-epoch log
        brain = FlyBrain(cfg)
        t0 = time.time()
        hist = brain.train(verbose=False)
        acc = brain.evaluate()
        secs = time.time() - t0
        print(
            f"{lr:>7.3f} {acc:>6.1%} {hist[-1].reward_mean:>8.3f} "
            f"{hist[-1].dopa_baseline:>+10.3f} {secs:>6.1f}"
        )


if __name__ == "__main__":
    main()
