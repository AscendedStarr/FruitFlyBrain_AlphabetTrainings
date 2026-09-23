"""Find a readout regime the three-factor rule can actually learn in.

Reward-modulated Hebbian learning only has a signal if the readout's winner is
*repeatable*: if argmax jumps between units on repeated presentations of the same
letter, the reward is uncorrelated with the weights and no learning rate helps.

Two knobs set that up, and they have to move together:

* ``kc_rate_gain`` - how faithfully the per-timestep KC spike sample copies the
  graded code. Too low and the sample is a noisy subset of the pattern.
* ``mbon_gain``    - rescales the readout drive back onto the LIF operating
  point, so raising the KC gain does not simply saturate the MBONs.

Stage 1 sweeps for stability, stage 2 trains the most stable candidates.
"""

from __future__ import annotations

import copy
import itertools
import time

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES, Config, FlyBrain, encode_from_config

LETTERS = [c for c in CLASSES if c != " "]
REPEATS = 5

KC_GAINS = [0.12, 0.25, 0.45, 0.70]
MBON_GAINS = [0.10, 0.20, 0.40, 0.80]
BIASES = [0.04, 0.10, 0.18]

TRAIN_EPOCHS = 15
N_CANDIDATES = 4


def stability(cfg: Config, chars: list[str]) -> tuple[int, float, float, float]:
    """(chars with a stable winner, mean margin, spike-count noise, fires/step)."""
    brain = FlyBrain(cfg)
    stable = 0
    margins: list[float] = []
    spreads: list[float] = []
    fires: list[float] = []

    for ch in chars:
        spikes = encode_from_config(cfg, ch, augment=False)
        picks, tops, seconds = [], [], []
        for _ in range(REPEATS):
            counts = brain.mb.present(spikes)
            order = counts.sort(descending=True).values
            picks.append(int(counts.argmax()))
            tops.append(float(order[0]))
            seconds.append(float(order[1]))
            fires.append(float((brain.mb.mbon_raster.sum(1) > 0).float().mean()))
        stable += int(len(set(picks)) == 1)
        margins.append(sum(tops) / len(tops) - sum(seconds) / len(seconds))
        spreads.append(float(torch.tensor(tops).std()))

    return (stable, sum(margins) / len(margins),
            sum(spreads) / len(spreads), sum(fires) / len(fires))


def main() -> None:
    base = Config()
    rows = []

    print("stage 1 - readout stability")
    print(f"{'kc_gain':>8} {'mbon_gain':>10} {'bias':>6} {'stable':>8} "
          f"{'margin':>8} {'noise':>7} {'fire%':>7}")
    for kc_gain, m_gain, bias in itertools.product(KC_GAINS, MBON_GAINS, BIASES):
        c = copy.deepcopy(base)
        c.kc_rate_gain = kc_gain
        c.mbon_gain = m_gain
        c.mbon_bias = bias
        s, margin, noise, fire = stability(c, LETTERS)
        rows.append((margin / (noise + 1e-6), s, margin, kc_gain, m_gain, bias,
                     noise, fire))
        print(f"{kc_gain:>8.2f} {m_gain:>10.2f} {bias:>6.2f} "
              f"{s:>4}/{len(LETTERS):<3} {margin:>8.2f} {noise:>7.2f} "
              f"{fire * 100:>6.1f}%")

    # rank by margin relative to the count noise, but require a fully stable winner
    ranked = sorted(rows, key=lambda r: (-r[1], -r[0]))
    best = [r for r in ranked if r[1] == len(LETTERS)] or ranked
    print(f"\nstage 2 - training the top {N_CANDIDATES} candidates "
          f"({TRAIN_EPOCHS} epochs)")

    for ratio, s, margin, kc_gain, m_gain, bias, _, _ in best[:N_CANDIDATES]:
        c = copy.deepcopy(base)
        c.kc_rate_gain = kc_gain
        c.mbon_gain = m_gain
        c.mbon_bias = bias
        c.epochs = TRAIN_EPOCHS
        # the eligibility scale moves with the KC firing probability, so re-derive
        # a plausible lr rather than reusing the old one
        probe = FlyBrain(copy.deepcopy(c))
        w0 = probe.mb.mbon.w.clone()
        probe.trial("A", learn=True)
        rel = float((probe.mb.mbon.w - w0).norm() / w0.norm())
        c.lr = base.lr * 0.02 / max(rel, 1e-9)

        t0 = time.perf_counter()
        brain = FlyBrain(c)
        brain.train(verbose=False)
        acc = brain.evaluate()
        secs = time.perf_counter() - t0
        print(f"  kc={kc_gain:.2f} gain={m_gain:.2f} bias={bias:.2f} "
              f"lrt={c.lr:.1e} -> acc {acc:6.1%}  ({secs:.0f}s)")

    print(f"\nchance = {1 / len(CLASSES):.1%}")


if __name__ == "__main__":
    main()
