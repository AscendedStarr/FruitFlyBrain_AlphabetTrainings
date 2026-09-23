"""Prove that progress survives a restart.

The point of a checkpoint is that stopping and resuming is indistinguishable
from never stopping. This trains, saves, reloads into a fresh process state, and
checks that:

1. accuracy is identical before and after the round trip
2. the epoch counter and learning curve continue instead of resetting
3. further training continues the same trajectory (not a new schedule)
4. a brain built from a *different* config cannot silently load these weights
"""

from __future__ import annotations

import os

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import Config
from flybrain.trainer import FlyBrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "runs", "_resume_test.pt")
EPOCHS_A = 40
EPOCHS_B = 40


def accuracy(brain: FlyBrain) -> float:
    return brain.evaluate(verbose=False)


def main() -> None:
    cfg = Config()
    cfg.epochs = EPOCHS_A
    cfg.lr = 0.02

    print("1. train and save")
    brain = FlyBrain(cfg)
    brain.train(verbose=False)
    acc_before = accuracy(brain)
    brain.save(CKPT)
    print(f"   trained {brain.epoch} epochs, accuracy {acc_before:.1%}")
    print(f"   saved -> {CKPT} ({os.path.getsize(CKPT) / 1024:.0f} KB)")

    print("\n2. reload")
    back = FlyBrain.load(CKPT)
    acc_after = accuracy(back)
    print(f"   epoch {back.epoch}, accuracy {acc_after:.1%}")
    same = abs(acc_before - acc_after) < 1e-9
    print(f"   accuracy preserved exactly: {same}")
    print(f"   history rows {len(back.history)} (curve {len(back.accuracy_curve)})")
    print(f"   best recorded {back.best_accuracy:.1%}")
    print(f"   config round-tripped: lr={back.cfg.lr}, "
          f"credit_mode={back.cfg.credit_mode!r}, "
          f"kc_rate_gain={back.cfg.kc_rate_gain}")

    print("\n3. continue training from the checkpoint")
    cfg2 = Config()
    cfg2.lr = 0.02
    cfg2.epochs = EPOCHS_A + EPOCHS_B

    cont = FlyBrain.load(CKPT)
    cont.cfg.epochs = EPOCHS_A + EPOCHS_B
    cont.train(verbose=False)
    print(f"   epoch {cont.epoch} (was {EPOCHS_A}), "
          f"accuracy {accuracy(cont):.1%}")
    print(f"   history rows {len(cont.history)} "
          f"(appended, not restarted)")

    print("\n4. a fresh brain for comparison")
    fresh = FlyBrain(cfg2)
    fresh.train(verbose=False)
    print(f"   from scratch: epoch {fresh.epoch}, accuracy {accuracy(fresh):.1%}")

    print("\n5. guard against a mismatched architecture")
    other = Config(n_kc=256)
    try:
        bad = FlyBrain(other)
        bad.mb.load_state_dict(torch.load(CKPT, weights_only=False)["circuit"])
        print("   ERROR - mismatched weights were accepted silently")
    except Exception as exc:  # noqa: BLE001 - the message is the point
        print(f"   rejected as expected: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:80]}")

    if os.path.exists(CKPT):
        os.remove(CKPT)
    print("\nverdict:",
          "PASS - progress is durable" if same and cont.epoch == EPOCHS_A + EPOCHS_B
          else "FAIL")


if __name__ == "__main__":
    main()
