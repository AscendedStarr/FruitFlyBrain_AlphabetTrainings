"""Is the MBON readout reliable enough to learn from?

REINFORCE-style rules learn from the *sign of the reward*, so they need the
readout to pick a repeatable winner. If the spike counts for one character are
so noisy that argmax changes run to run, the reward is uncorrelated with the
weights and no learning rate will help.

This script measures that, and separately checks whether the readout *layer*
could express the right answer at all by installing the class centroids by hand.
"""

from __future__ import annotations

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES, Config, FlyBrain, encode_from_config

REPEATS = 6
LETTERS = [c for c in CLASSES if c != " "]


def readout_stability(cfg: Config, brain: FlyBrain, chars: list[str]) -> None:
    print(f"{'ch':>3} {'top':>4} {'2nd':>4} {'margin':>7} {'winner stable':>14} "
          f"{'counts sd':>10}")
    stable = 0
    margins = []
    for ch in chars:
        picks, tops, seconds = [], [], []
        for _ in range(REPEATS):
            counts = brain.mb.present(encode_from_config(cfg, ch, augment=False))
            order = counts.sort(descending=True).values
            picks.append(int(counts.argmax()))
            tops.append(float(order[0]))
            seconds.append(float(order[1]))
        top, second = sum(tops) / len(tops), sum(seconds) / len(seconds)
        stable += int(len(set(picks)) == 1)
        margins.append(top - second)
        print(f"{ch:>3} {top:>4.1f} {second:>4.1f} {top - second:>7.2f} "
              f"{str(len(set(picks)) == 1):>14} {torch.tensor(tops).std():>10.2f}")

    print(f"\n  winners stable across {REPEATS} repeats: {stable}/{len(chars)}")
    print(f"  mean margin between 1st and 2nd   : "
          f"{sum(margins) / len(margins):.2f} spikes")


def centroid_readout(cfg: Config, brain: FlyBrain, chars: list[str]) -> float:
    """Install perceptron/centroid weights by hand and score the same readout."""
    print("\n  installing class-centroid weights by hand...")
    kc_rates = []
    for ch in chars:
        spikes = encode_from_config(cfg, ch, augment=False)
        brain.mb.reset()
        pn = brain.mb.al.rate_vector(spikes)
        kc_rates.append(brain.mb.kc.encode(pn).clone())
    R = torch.stack(kc_rates)                       # (n_char, n_kc)
    R = R - R.mean(0, keepdim=True)
    # each MBON i (i>=1) gets the centroid of letter i-1
    w = torch.zeros(cfg.n_mbon, cfg.n_kc)
    for i, ch in enumerate(chars[: cfg.n_mbon - 1]):
        w[i + 1] = R[i]
    w = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8) * cfg.target_norm
    brain.mb.mbon.w = w

    hits = 0
    for ch in chars:
        counts = brain.mb.present(encode_from_config(cfg, ch, augment=False))
        pred = CLASSES[int(counts.argmax())]
        hits += int(pred == ch)
    acc = hits / len(chars)
    print(f"  centroid readout accuracy: {acc:.1%}")
    return acc


def main() -> None:
    cfg = Config()
    brain = FlyBrain(cfg)

    print(f"kc_rate_gain={cfg.kc_rate_gain}  n_mbon={cfg.n_mbon}  "
          f"t_stim={cfg.t_stim}\n")
    readout_stability(cfg, brain, LETTERS)
    centroid_readout(Config(), brain, LETTERS)


if __name__ == "__main__":
    main()
