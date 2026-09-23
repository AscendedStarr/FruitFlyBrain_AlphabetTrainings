"""Train the same task on a real connectome and on controls, and compare.

    .\\.venv-flybrain\\Scripts\\python.exe connectome_compare.py            # all three arms
    .\\.venv-flybrain\\Scripts\\python.exe connectome_compare.py --seeds 7,11,13
    .\\.venv-flybrain\\Scripts\\python.exe connectome_compare.py --quick      # 8-epoch smoke test

What is being compared
----------------------
All three arms are byte-for-byte the same experiment except for one thing: the
fixed, non-plastic ``PN -> KC`` expansion matrix. Same task, same glyphs, same
seed, same learning rule, same dopamine, same number of trials, same readout
initialisation, same circuit dimensions.

    connectome           the measured hemibrain v1.2 PN -> KC wiring
    connectome-shuffled  the same wiring with every KC's partner set randomised,
                         keeping that KC's exact degree and weight multiset.
                         Holds size, sparsity and input statistics fixed, so it
                         isolates partner *identity* and nothing else.
    random               the seeded random expansion v0.1.0 ships, at the same
                         dimensions and at the connectome's median fan-in

The dimensions are taken from the connectome, not chosen: 157 projection
neurons and 1802 Kenyon cells, which are the neurons that survive the
connectivity filter described in ``tools/build_connectome.py``.

What a result here does and does not mean
-----------------------------------------
It is a claim about the *pattern* of the connectome. The sign and scale of the
synapses are modelling choices (see ``flybrain/connectome.py``), applied
identically to both connectome arms, so they cannot produce a difference between
them. They could in principle shift all arms relative to each other, which is
why the shuffled arm matters more than the random one: it shares every one of
those choices with the connectome arm and differs only in which PN feeds which
KC.

The honest framing, stated up front because this is a published research repo:
a connectome replaces the *wiring*, not the *task*. The 5x7 glyph encoding, the
27 output labels, and reading 27 letters out of an MBON population that the fly
does not label with letters are all inventions of this project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from flybrain import Config
from flybrain.connectome import load
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

#: Read the look count the UI uses without importing serve.py's torch-heavy body.
#: Kept in sync with serve.LOOKS and make_checkpoint.LOOKS.
LOOKS = 4

ARMS = ("connectome", "connectome-shuffled", "random")


def build_config(wiring: str, epochs: int, seed: int,
                 overrides: dict, fan_in: int) -> Config:
    """The v0.1.0 converging recipe, with the wiring swapped out.

    Everything that makes the shipped model converge - pavlovian reward, the
    inverse learning-rate schedule, the pattern credit rule - is held identical
    across arms, because the question is whether the wiring matters given a
    working learner, not whether some other learner could be found.
    """
    return Config(
        lr=0.020,
        credit_mode="pattern",
        reward_mode="pavlovian",
        lr_schedule="inv",
        lr_half_life=300,
        epochs=epochs,
        seed=seed,
        wiring=wiring,
        fan_in=fan_in,
        **overrides,
    )


def run_arm(wiring: str, epochs: int, seed: int, overrides: dict,
            fan_in: int, verbose: bool) -> dict:
    """Train one arm and measure it the way the web UI will."""
    cfg = build_config(wiring, epochs, seed, overrides, fan_in)

    t0 = time.perf_counter()
    brain = FlyBrain(cfg)
    brain.train(verbose=verbose)
    train_s = time.perf_counter() - t0

    # Read out with the same number of looks the UI uses, so the number reported
    # here is the number a user would see on the page.
    brain.cfg.decision_repeats = LOOKS
    acc = brain.evaluate()

    curve = list(brain.accuracy_curve)
    reached = next((i + 1 for i, a in enumerate(curve) if a >= acc - 0.02),
                   len(curve)) if curve else 0

    support = int((brain.mb.kc.w != 0).sum())
    result = {
        "wiring": wiring,
        "seed": seed,
        "epochs": epochs,
        "n_pn": cfg.n_pn,
        "n_kc": cfg.n_kc,
        "n_mbon": cfg.n_mbon,
        "k_active": cfg.k_active,
        "fan_in_config": fan_in,
        "plastic_synapses": int(cfg.n_mbon * cfg.n_kc),
        "wiring_synapses": support,
        "holdout_accuracy": acc,
        "chance": 1.0 / cfg.n_mbon,
        "converged_epoch": reached,
        "train_seconds": train_s,
        "accuracy_curve": curve,
    }

    out = os.path.join(RUNS, "wiring_%s_seed%d.pt" % (wiring, seed))
    brain.save(out, note="%s wiring, seed %d, %.1f%% (%d epochs)"
               % (wiring, seed, 100 * acc, epochs))
    result["checkpoint"] = os.path.relpath(out, HERE)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of: " + ", ".join(ARMS))
    ap.add_argument("--seeds", default="7",
                    help="comma-separated seeds, e.g. 7,11,13")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--quick", action="store_true",
                    help="8 epochs, for checking the harness rather than the science")
    ap.add_argument("--json", default=os.path.join(RUNS, "connectome_compare.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    if args.quick:
        args.epochs = 8

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print("unknown arm(s): %s (expected %s)" % (bad, list(ARMS)))
        return 2
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    conn = load()
    overrides = conn.as_config_overrides()
    # The random arm gets the connectome's median fan-in rather than its own
    # default of 16, so the three arms have the same number of PNs converging on
    # a KC and fan-in is not a confound.
    fan_in = max(1, min(int(round(float(np.median(conn.fan_in)))), conn.n_pn))

    print("=" * 74)
    print("connectome vs controls -- the only variable is the PN -> KC matrix")
    print("=" * 74)
    print(conn.describe())
    print()
    print("dimensions taken from the connectome: %s" % overrides)
    print("random arm fan-in set to the connectome median: %d" % fan_in)
    print("epochs=%d  seeds=%s  arms=%s" % (args.epochs, seeds, arms))
    print("plastic KC -> MBON synapses per arm: %d" % (27 * conn.n_kc))
    print()

    results = []
    for seed in seeds:
        for wiring in arms:
            print("-" * 74)
            print("ARM %s   seed %d" % (wiring, seed))
            print("-" * 74, flush=True)
            r = run_arm(wiring, args.epochs, seed, overrides, fan_in,
                        verbose=not args.quiet)
            results.append(r)
            print("  -> holdout %.1f%%   converged epoch %d   %.1f s"
                  % (100 * r["holdout_accuracy"], r["converged_epoch"],
                     r["train_seconds"]), flush=True)
            print()

    # ---- summary ------------------------------------------------------------
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print("%-22s %8s %8s %10s %10s" %
          ("arm", "mean", "spread", "conv.ep", "seconds"))
    for wiring in arms:
        rs = [r for r in results if r["wiring"] == wiring]
        accs = [100 * r["holdout_accuracy"] for r in rs]
        conv = [r["converged_epoch"] for r in rs]
        secs = [r["train_seconds"] for r in rs]
        spread = (max(accs) - min(accs)) if len(accs) > 1 else 0.0
        print("%-22s %7.1f%% %7.1f%% %10.0f %10.1f" %
              (wiring, sum(accs) / len(accs), spread,
               sum(conv) / len(conv), sum(secs) / len(secs)))
    print()
    # Paired comparison, which is the actual experiment: same seed, same
    # everything, one wiring changed.
    if "connectome" in arms and "connectome-shuffled" in arms:
        print("paired connectome - shuffled, per seed:")
        for seed in seeds:
            a = next((r for r in results
                      if r["wiring"] == "connectome" and r["seed"] == seed), None)
            b = next((r for r in results
                      if r["wiring"] == "connectome-shuffled" and r["seed"] == seed), None)
            if a and b:
                print("  seed %-4d %+.1f points  (%.1f%% vs %.1f%%)"
                      % (seed, 100 * (a["holdout_accuracy"] - b["holdout_accuracy"]),
                         100 * a["holdout_accuracy"], 100 * b["holdout_accuracy"]))
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({
            "connectome": conn.meta,
            "config_overrides": overrides,
            "random_arm_fan_in": fan_in,
            "epochs": args.epochs,
            "decision_repeats_eval": LOOKS,
            "results": results,
        }, fh, indent=2)
    print("wrote %s" % args.json)
    print("checkpoints in %s" % os.path.relpath(RUNS, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
