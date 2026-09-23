"""Train one good checkpoint for the web UI (and for `digit_proof.py`) to serve.

    .\\.venv-flybrain\\Scripts\\python.exe make_checkpoint.py
    .\\.venv-flybrain\\Scripts\\python.exe make_checkpoint.py --epochs 800

Why this exact configuration
----------------------------
This script used to build ``Config(lr=0.020, credit_mode="pattern",
lr_schedule="const")`` and rely on the ``operant`` reward mode that ``Config``
defaults to. That recipe plateaus at **60.5%** and does not move: it was run
again to confirm, and 400 epochs produced 60.5%, exactly the number the old
docstring quoted as the ceiling.

The configuration below is the one the working brain was actually trained with,
recovered by diffing the config stored inside a known-good checkpoint against
``Config()``. Three fields differ, and all three matter:

``reward_mode="pavlovian"``
    The reward is yoked to the *stimulus identity*, not to whatever the fly just
    guessed. Under ``operant`` the teaching signal is produced by the fly's own
    action, so an early wrong answer suppresses the sugar that would have taught
    it the right one - the rule starves itself and parks at ~60%.
``lr_schedule="inv"`` with ``lr_half_life=300``
    ``const`` at lr=0.02 keeps making full-size steps forever and never settles.
    The inverse schedule anneals as the run goes on.
``epochs``
    Accuracy reaches ~99% within roughly 60 epochs. Past that it is flat, which
    is why the default here is 400 and not 2000: the extra epochs buy nothing
    measurable. See the README's convergence note.

Everything is seeded by ``config.seed`` and the brain starts naive, so this is
deterministic: the same command produces the same checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys

from flybrain import Config
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "runs", "flybrain.pt")

# Read the look count the UI uses without importing serve.py's torch-heavy body
# until after training. Keep this in sync with serve.LOOKS.
LOOKS = 4


def build_config(epochs: int, seed: int) -> Config:
    """The recipe that actually converges. Kept in one place on purpose."""
    return Config(
        lr=0.020,
        credit_mode="pattern",
        reward_mode="pavlovian",
        lr_schedule="inv",
        lr_half_life=300,
        epochs=epochs,
        seed=seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args.epochs, args.seed)
    brain = FlyBrain(cfg)

    print(f"training {args.epochs} epochs from a naive circuit")
    print(f"  lr={cfg.lr}  schedule={cfg.lr_schedule}"
          f"(half_life={cfg.lr_half_life})  credit={cfg.credit_mode}"
          f"  reward={cfg.reward_mode}  seed={cfg.seed}")
    brain.train(verbose=not args.quiet)

    # Read out the way the UI will, so the number printed here is the number the
    # user sees on the page.
    brain.cfg.decision_repeats = LOOKS
    acc = brain.evaluate()
    brain.best_accuracy = max(brain.best_accuracy, acc)

    # Where did it actually converge? Reporting the epoch that first reached
    # within 2 points of the final score is more honest than implying every one
    # of the epochs was necessary.
    curve = list(brain.accuracy_curve)
    reached = next(
        (i + 1 for i, a in enumerate(curve) if a >= acc - 0.02),
        len(curve),
    ) if curve else 0

    print()
    print(f"holdout accuracy  : {acc:.1%}  (chance {100 / cfg.n_mbon:.1f}%)")
    print(f"epochs trained    : {args.epochs}")
    if reached:
        print(f"first within 2pt  : epoch {reached}"
              f"  ({curve[reached - 1]:.1%} at that point)")
    print(f"phrase 'PHILIPPE DELAMBRE' all 17 characters: {acc ** 17:.4%}")

    brain.save(args.out, note=f"{acc:.1%} after {args.epochs} epochs (pavlovian)")
    print(f"saved -> {args.out}")
    print(f"  written by {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
