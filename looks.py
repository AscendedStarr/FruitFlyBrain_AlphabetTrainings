"""How many times should the fly look before it answers?

`evaluate()` counts every presentation separately, so the headline accuracy is a
*single-look* number: one Poisson sample of the KC code, one argmax of the MBON
counts. A real fly does not answer off one glance either - it fixates, and the
Kenyon-cell code it sees is stochastic, so averaging a few presentations before
committing is the natural readout.

That makes the number of looks a free lever: no retraining, no extra synapses,
no change to the circuit. It was measured once against the *old* 60.5% weights
and it saturated at 65.4%, which is what proved the old ceiling was the weights
and not sampling noise. Against the trained weights it is worth measuring again,
because near the ceiling averaging should climb much further.

Pavlovian reward is what makes this safe to use during training too: the reward
no longer depends on the fly's own guess, so averaging presentations cannot feed
a wrong answer back into the credit assignment.

Run:  $env:OMP_NUM_THREADS=1; .\\.venv-flybrain\\Scripts\\python.exe looks.py
"""

from __future__ import annotations

import os
import sys
import time

from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "runs", "flybrain.pt")
LOOKS = [1, 2, 3, 5, 9, 15, 25]
PHRASE = "PHILIPPE DELAMBRE"

if not os.path.exists(CKPT):
    print(f"no checkpoint at {CKPT} - train one first (keep_training.py)")
    sys.exit(1)

brain = FlyBrain.load(CKPT)
saved = brain.best_accuracy
print("=" * 66)
print(f"checkpoint: {os.path.basename(CKPT)}")
print(f"  epoch {brain.epoch}   best recorded {saved:.1%}")
print("=" * 66)
print(f"{'looks':>7} {'accuracy':>10} {'phrase (16 letters)':>21} {'sec':>7}")
print("-" * 66)

best = (0.0, 0)
rows = []
for n in LOOKS:
    # One averaged question per letter, repeated a few times for error bars.
    brain.cfg.eval_repeats = 4
    brain.cfg.decision_repeats = n
    t0 = time.time()
    acc = brain.evaluate()
    took = time.time() - t0
    phrase = acc ** len(PHRASE.replace(" ", ""))
    rows.append((n, acc))
    if acc > best[0]:
        best = (acc, n)
    print(f"{n:>7} {acc:>9.1%} {phrase:>20.2%} {took:>6.0f}s")

print("-" * 66)
acc, n = best
print(f"best: {n} looks -> {acc:.1%} per letter, "
      f"'{PHRASE}' correct {acc ** 16:.1%} of the time")

# What it would take to make the phrase reliable.
print()
print("phrase accuracy as a function of per-letter accuracy:")
for p in (0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
    print(f"  {p:>6.1%} per letter -> phrase {p ** 16:>6.1%}")

# Which letters are still unstable, at the best look count.
brain.cfg.decision_repeats = n
print()
print(f"letters still missed at {n} looks (over 20 averaged questions):")
brain.cfg.eval_repeats = 20
bad = []
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    hits = 0
    for _ in range(20):
        res = brain.trial(ch, learn=False, augment=False)
        hits += int(res.correct)
    if hits < 20:
        bad.append((ch, hits))
if bad:
    for ch, hits in bad:
        print(f"  {ch}: {hits}/20")
else:
    print("  none - every letter is now read reliably")
