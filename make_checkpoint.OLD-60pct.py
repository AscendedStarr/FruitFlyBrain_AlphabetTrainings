"""Train one good checkpoint for the web UI to serve.

`serve.py` used to build a naive brain on every start, so the page sat at
chance. Now that it loads a checkpoint, there needs to be a checkpoint worth
loading: the three on disk were written before the credit-assignment fix and
the best of them scores 40.7%.

400 epochs at the measured 0.266 s/epoch is about 1.8 minutes.
"""

from __future__ import annotations

import os

from flybrain import Config
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "flybrain.pt")
EPOCHS = 400

cfg = Config(lr=0.020, credit_mode="pattern", lr_schedule="const",
             epochs=EPOCHS, seed=7)
brain = FlyBrain(cfg)

print(f"training {EPOCHS} epochs, lr={cfg.lr}, credit={cfg.credit_mode}, "
      f"schedule={cfg.lr_schedule}")
brain.train(verbose=True)

acc = brain.evaluate()
print()
print(f"holdout accuracy after {EPOCHS} epochs: {acc:.1%}")
print(f"chance = {100 / cfg.n_mbon:.1f}%   "
      f"phrase 'MY NAME IS JEFF' read correctly: {acc ** 12:.4%}")
brain.save(OUT, note=f"{acc:.1%} after {EPOCHS} epochs (fixed credit rule)")
print(f"saved -> {OUT}")
