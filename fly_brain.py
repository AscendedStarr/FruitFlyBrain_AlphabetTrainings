"""Command-line driver.

    python fly_brain.py demo
    python fly_brain.py train --epochs 60
    python fly_brain.py say "my name is jeff"
    python fly_brain.py diag
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from flybrain import CLASSES, Config, FlyBrain, render
from flybrain.encoding import FONT_5X7

PHRASE = "MY NAME IS JEFF"
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "flybrain.pt")


def _banner(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def cmd_diag(args: argparse.Namespace) -> None:
    cfg = Config()
    brain = FlyBrain(cfg)
    _banner("circuit")
    print(f"  receptors {cfg.n_receptor}  ->  PNs {cfg.n_pn}  ->  "
          f"KCs {cfg.n_kc} (k={cfg.k_active})  ->  MBONs {cfg.n_mbon}")
    print(f"  plastic synapses: KC -> MBON  ({cfg.n_kc} x {cfg.n_mbon} = "
          f"{cfg.n_kc * cfg.n_mbon:,})")
    print(f"  PN -> KC fan-in {cfg.fan_in} of {cfg.n_pn}  "
          f"(sparse, non-plastic, matching the fly calyx)")

    _banner("single-character readout, untrained")
    for ch in "JEF":
        print(render(ch))
        spikes = brain.mb.present(brain._encode(ch, augment=False))
        top = int(spikes.topk(3).indices[0])
        print(f"  -> argmax {CLASSES[top]}   (spikes {[int(v) for v in spikes]})")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config(epochs=args.epochs, seed=args.seed)
    train(cfg, save_to=args.out)


def train(cfg: Config, save_to: str | None = None) -> FlyBrain:
    brain = FlyBrain(cfg)
    _banner("training: dopamine-modulated Hebbian plasticity")
    print(f"  {cfg.epochs} epochs x {len(CLASSES) * cfg.trials_per_class} trials "
          f"= {cfg.epochs * len(CLASSES) * cfg.trials_per_class:,} presentations")
    print("  reward = sucrose on a correct readout, bitter on an incorrect one")
    print("  plasticity is gated by the PAM/DAN burst, not applied directly")

    t0 = time.time()
    brain.train(verbose=True)
    dt = time.time() - t0

    print()
    print(f"  finished in {dt:.1f}s")
    brain.evaluate(verbose=True)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        brain.save(save_to)
        print(f"  checkpoint -> {save_to}")
    brain.cfg = cfg
    return brain


def cmd_say(args: argparse.Namespace) -> None:
    if os.path.exists(args.ckpt):
        cfg = Config.from_json(args.config) if args.config else Config()
        brain = FlyBrain(cfg)
        brain.load(args.ckpt)
        print(f"  loaded {args.ckpt}")
    else:
        print(f"  no checkpoint at {args.ckpt}; training first")
        brain = train(Config(epochs=args.epochs), save_to=args.ckpt)

    _banner(f'reading: "{args.phrase}"')
    brain.say(args.phrase, learn=args.learn, verbose=True)


def cmd_demo(args: argparse.Namespace) -> None:
    brain = train(Config(epochs=args.epochs, seed=args.seed), save_to=args.out)

    _banner(f'demo: the fly reads "{PHRASE}"')
    print("  Driving the trained mushroom body one character at a time.")
    print("  No further learning during this pass.\n")
    brain.say(PHRASE, learn=False, verbose=True)

    _banner("demo: teach it the phrase by rewarding the utterance")
    print("  Now learning stays on and the whole-phrase reward is delivered.")
    print("  A correct utterance earns a large sucrose event; anything else")
    print("  earns bitter. Watch the dopamine trace and the accuracy.\n")
    for attempt in range(1, 4):
        print(f"  --- attempt {attempt} ---")
        brain.say(PHRASE, learn=True, verbose=False)
        acc = brain.evaluate(verbose=False)
        print(f"    phrase accuracy after attempt: {acc:.1%}")

    print()
    brain.say(PHRASE, learn=False, verbose=True)

    if args.plot:
        from flybrain.plotting import save_report

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        path = os.path.join(os.path.dirname(args.out), "report.png")
        save_report(brain, path)
        print(f"  plot -> {path}")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Write the figures that show the fly and the circuit."""
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.ckpt) and not args.retrain:
        cfg = Config(seed=args.seed)
        brain = FlyBrain(cfg)
        brain.load(args.ckpt)
        print(f"  loaded {args.ckpt}")
    else:
        print(f"  no checkpoint at {args.ckpt}; training one first")
        brain = train(Config(epochs=args.epochs, seed=args.seed),
                      save_to=args.ckpt)

    from flybrain.visualize import ascii_report, save_activity, save_circuit

    _banner("the fly")
    print(ascii_report(brain, letters=args.letter))

    _banner("writing figures")
    circuit = os.path.join(out_dir, "fly.png")
    save_circuit(brain, circuit)
    print(f"  the animal + circuit + learnt weights + response matrix")
    print(f"    -> {circuit}")

    letters = args.letter.upper() or "JEF"
    for ch in letters:
        path = os.path.join(out_dir, f"activity_{ch}.png")
        save_activity(brain, ch, path)
        print(f"  {ch!r} stage by stage  -> {path}")

    if args.plot:
        from flybrain.plotting import save_report

        report = os.path.join(out_dir, "report.png")
        save_report(brain, report)
        print(f"  training curves       -> {report}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fly_brain.py",
        description="A dopamine-modulated spiking mushroom body that learns A-Z.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("diag", help="print the circuit and an untrained readout")
    d.set_defaults(func=cmd_diag)

    t = sub.add_parser("train", help="train the mushroom body")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--out", default=DEFAULT_CKPT)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("say", help="read a phrase out of the trained circuit")
    s.add_argument("phrase", nargs="?", default=PHRASE)
    s.add_argument("--learn", action="store_true", help="keep plasticity on")
    s.add_argument("--ckpt", default=DEFAULT_CKPT)
    s.add_argument("--config", default=None)
    s.add_argument("--epochs", type=int, default=40, help="used if no checkpoint")
    s.set_defaults(func=cmd_say)

    m = sub.add_parser("demo", help="train, read the phrase, then reward it")
    m.add_argument("--epochs", type=int, default=40)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--out", default=DEFAULT_CKPT)
    m.add_argument("--plot", action="store_true")
    m.set_defaults(func=cmd_demo)

    v = sub.add_parser("visualize", help="draw the fly, the circuit, and one letter")
    v.add_argument("--ckpt", default=DEFAULT_CKPT)
    v.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runs"))
    v.add_argument("--letter", default="JEF", help="characters to trace through")
    v.add_argument("--epochs", type=int, default=40, help="used if no checkpoint")
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--retrain", action="store_true", help="ignore any checkpoint")
    v.add_argument("--plot", action="store_true", help="also write training curves")
    v.set_defaults(func=cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
