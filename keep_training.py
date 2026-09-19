"""Keep training, and find out which knob makes the readout better.

Current checkpoint: 60.5% single-shot on 26 letters, chance 3.7%. The plateau
survived every learning-rate schedule, so this sweeps the three levers that are
still untested:

  * `trials_per_class`  - only 4 samples per class per epoch reach the credited
                          row, which is very little supervision per row.
  * `decision_repeats`  - the reward is an argmax off one Poisson sample, so a
                          share of the punishments are noise rather than error.
  * `n_kc`              - 60 of 512 Kenyon cells are active; more cells is more
                          room to separate glyphs that currently collide.

Every run reports the same held-out accuracy. The winner is then trained long
and written to runs/flybrain.pt, which is what serve.py loads.

Progress is streamed to runs/sweep.log so a long run can be watched.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace

from flybrain import Config
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 200

RUNS = [
    # Pavlovian reward is already proven (95.1% at 300 epochs vs 60.6% for
    # operant). What is left is which schedule settles it best. The burst only
    # contributes its SIGN to a named row (delta = direction * pattern / norm),
    # so `lr * lr_scale` is the entire step size and the schedule is the only
    # thing that can make the averaged update stop orbiting the class mean.
    ("pavlovian + const  ", dict(reward_mode="pavlovian")),
    ("pavlovian + sqrt   ", dict(reward_mode="pavlovian", lr_schedule="sqrt",
                                 lr_half_life=300)),
    ("pavlovian + inv    ", dict(reward_mode="pavlovian", lr_schedule="inv",
                                 lr_half_life=300)),
]

base = Config(lr=0.020, credit_mode="pattern", lr_schedule="const", epochs=EPOCHS)

log_path = os.path.join(HERE, "runs", "sweep.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)


def log(line: str) -> None:
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


log("")
log("=" * 74)
log(f"sweep: {EPOCHS} epochs per run, lr=0.020, credit=pattern")
log("  reward mode (what the sugar is conditioned on) x lr schedule")
log("=" * 74)
log(f"{'run':<20} {'final':>7} {'best':>7} {'last50':>7} {'sec':>6}")

results = []
for name, over in RUNS:
    cfg = replace(base, **over)
    t0 = time.time()
    brain = FlyBrain(cfg)
    brain.train(verbose=False)
    acc = brain.evaluate()
    took = time.time() - t0

    curve = brain.accuracy_curve
    best = max(curve) if curve else 0.0
    tail = curve[-50:]
    last50 = sum(tail) / max(1, len(tail))

    log(f"{name:<20} {acc:>6.1%} {best:>6.1%} {last50:>6.1%} {took:>6.0f}")
    results.append((last50, acc, name, over, brain))

# Rank on the trailing average rather than the last epoch: a single epoch is a
# 108-sample measurement and wobbles by a couple of points on its own.
results.sort(key=lambda r: -r[0])
tail50, acc, name, over, brain = results[0]
log("")
log(f"best run: {name.strip()}  trailing-50 {tail50:.1%}  final {acc:.1%}")
log(f"  overrides: {over or 'defaults'}")

# Train the winner for a long stretch and install it for the web UI.
LONG = 2000
cfg = replace(base, epochs=LONG, **over)
log("")
log(f"training the winner for {LONG} epochs...")
t0 = time.time()
brain = FlyBrain(cfg)
brain.train(verbose=False)
acc = brain.evaluate()
curve = brain.accuracy_curve
tail = curve[-100:]
log(f"  {LONG} epochs in {time.time() - t0:.0f}s -> holdout {acc:.1%}, "
    f"trailing-100 {sum(tail) / len(tail):.1%}, best {max(curve):.1%}")

out = os.path.join(HERE, "runs", "flybrain.pt")
brain.save(out, note=f"{acc:.1%} after {LONG} epochs ({name.strip()})")
log(f"  saved -> {out}")
log(f"  phrase 'MY NAME IS JEFF' reads correctly {acc ** 12:.4%} of the time")
log("")
log("restart serve.py to serve this checkpoint")
