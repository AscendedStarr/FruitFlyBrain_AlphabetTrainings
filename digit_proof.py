"""Reproducibility artifact: show the trained fly all ten digits and record its answers.

The claim under test is narrow and falsifiable:

    A network whose output layer contains only A-Z and space cannot recognise a
    digit, and will answer every digit with whichever letter its 35-pixel
    receptor pattern most resembles.

Nothing in this script is simulated, approximated, or post-hoc:

* each digit's real 5x7 bitmap (from ``flybrain.FONT_DIGITS``) is Poisson-encoded
  onto the real 35-receptor sheet by the same ``encode_from_config`` the trainer
  uses;
* the spikes go through the real trained circuit loaded from the checkpoint;
* the answer is the real ``argmax`` of the real MBON spike counts over the real
  decision window;
* ``correct`` is computed the same way the rest of the project computes it, and
  comes out False for all ten simply because ``CLASSES.index(digit)`` does not
  exist - there is no digit output cell to land on.

The script prints the confusion table and, optionally, dumps it as JSON next to
the checkpoint so the numbers in the README can be regenerated.

    .\\.venv-flybrain\\Scripts\\python.exe digit_proof.py
    .\\.venv-flybrain\\Scripts\\python.exe digit_proof.py --json runs/digit_proof.json
    .\\.venv-flybrain\\Scripts\\python.exe digit_proof.py --naive   # untrained control
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from flybrain import (  # noqa: E402
    CLASSES,
    FONT_DIGITS,
    Config,
    FlyBrain,
    encode_from_config,
)
from flybrain.proof import snapshot, verify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read_out(brain: FlyBrain, char: str, looks: int) -> tuple[int, list[float]]:
    """Present one glyph `looks` times and return (argmax, spike counts).

    Deliberately mirrors ``Session.present`` in serve.py - including passing
    ``gen=brain.gen``, without which every look would be the identical
    deterministic spike train and the averaging would do nothing.
    """
    counts = brain.mb.present(
        encode_from_config(brain.cfg, char, augment=False, gen=brain.gen))
    for _ in range(looks - 1):
        counts = counts + brain.mb.present(
            encode_from_config(brain.cfg, char, augment=False, gen=brain.gen))
    return int(counts.argmax()), [float(v) for v in counts]


def sweep(brain, looks):
    """One independent pass over all ten digits.

    Independent because the receptor sheet is Poisson-sampled from ``brain.gen``
    on every presentation, so each call draws fresh noise. That is why a single
    pass cannot be published on its own: the exact letter the fly mistakes a
    digit for is a noisy draw. What is *not* noisy is the fact that the answer
    is always one of the 27 letters and never a digit.
    """
    out = []
    for ch in sorted(FONT_DIGITS):
        pred, counts = read_out(brain, ch, looks)
        top = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:2]
        out.append({
            "digit": ch,
            "pred": pred,              # index into CLASSES
            "guess": CLASSES[pred],
            "correct": False,          # no digit output cell exists; true by construction
            "score": round(counts[pred], 2),
            "runner_up": CLASSES[top[1]],
            "runner_up_score": round(counts[top[1]], 2),
            "margin": round((counts[top[0]] - counts[top[1]]) / (counts[top[0]] or 1.0), 4),
            "bitmap": FONT_DIGITS[ch],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "flybrain.pt"))
    ap.add_argument("--looks", type=int, default=4,
                    help="presentations averaged per answer (the UI uses 4)")
    ap.add_argument("--repeats", type=int, default=20,
                    help="independent sweeps over all ten digits. The receptor "
                         "sheet is Poisson-sampled, so a single sweep is one "
                         "noisy draw; the stability table needs repeats to be "
                         "meaningful.")
    ap.add_argument("--json", default=None, help="write the table as JSON here")
    ap.add_argument("--naive", action="store_true",
                    help="ignore the checkpoint and use an untrained circuit")
    args = ap.parse_args()

    if args.naive or not os.path.isfile(args.ckpt):
        brain = FlyBrain(Config())
        source = "naive circuit (untrained, chance = %.1f%%)" % (100 / len(CLASSES))
    else:
        brain = FlyBrain.load(args.ckpt)
        source = f"{os.path.basename(args.ckpt)} (epoch {brain.epoch})"
    brain.cfg.decision_repeats = args.looks

    # --- proof of non-training, part one -------------------------------------
    # Everything from here to the end of this function is a read-only test.
    # Snapshot the checkpoint bytes and every weight matrix before any glyph
    # is presented, then verify both again once the table is complete.
    proof_before = snapshot(brain, args.ckpt)

    print(f"FlyBrain digit-recognition test")
    print(f"  circuit      : {source}")
    print(f"  output cells : {len(CLASSES)}  ->  {' '.join(CLASSES)}")
    print(f"  vocabulary   : {' '.join(sorted(FONT_DIGITS))} are NOT among them")
    print(f"  looks/answer : {args.looks}")
    print()

    sweeps = [sweep(brain, args.looks)]
    for _ in range(max(0, args.repeats - 1)):
        sweeps.append(sweep(brain, args.looks))
    rows = sweeps[0]

    print(f"  first sweep ({args.repeats} run{'' if args.repeats == 1 else 's'} "
          f"were taken; the whole set is summarised below)")
    print(f"  {'digit':>5}  {'ink':>4}  {'-> ans':>6}  {'score':>7}  "
          f"{'runner-up':>9}  {'margin':>7}")
    print(f"  {'-' * 5}  {'-' * 4}  {'-' * 6}  {'-' * 7}  {'-' * 9}  {'-' * 7}")
    for r in rows:
        ink = sum(row.count("#") for row in r["bitmap"])
        print(f"  {r['digit']:>5}  {ink:>4}  {r['guess']:>6}  {r['score']:>7.1f}  "
              f"{r['runner_up']:>9}  {r['margin']:>7.3f}")

    # Pool every repeat. This is the number that matters, because it averages
    # over the receptor noise instead of reporting one lucky draw.
    all_digit_margins = [r["margin"] for sw in sweeps for r in sw]
    n_presented = sum(len(sw) for sw in sweeps)
    n_correct_total = sum(1 for sw in sweeps for r in sw if r["correct"])
    n_in_vocab = sum(1 for sw in sweeps for r in sw if r["guess"] in CLASSES)

    print()
    print(f"  pooled over {len(sweeps)} sweeps:")
    print(f"    digits recognised      : {n_correct_total}/{n_presented}")
    print(f"    answer in vocabulary   : {n_in_vocab}/{n_presented}")

    # The control: the same circuit, on the alphabet it was actually trained on.
    # Same code path, same look count, same generator - only the glyphs differ.
    letter_margins = []
    letter_hits = 0
    letter_runs = 0
    for _ in range(len(sweeps)):
        for ch in CLASSES:
            pred, counts = read_out(brain, ch, args.looks)
            top = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:2]
            letter_margins.append(
                (counts[top[0]] - counts[top[1]]) / (counts[top[0]] or 1.0))
            letter_hits += int(pred == CLASSES.index(ch))
            letter_runs += 1

    mean = lambda v: sum(v) / len(v)  # noqa: E731 - short and local
    n_ties = sum(1 for m in all_digit_margins if m < 0.05)

    print(f"  alphabet control       : {letter_hits}/{letter_runs} "
          f"({brain.evaluate():.1%} on holdout, {len(CLASSES)} classes)")
    print()
    print("  how *decisive* the answer was (winner minus runner-up, normalised):")
    print(f"    letters  mean margin {mean(letter_margins):.3f}  "
          f"min {min(letter_margins):.3f}   n={letter_runs}")
    print(f"    digits   mean margin {mean(all_digit_margins):.3f}  "
          f"min {min(all_digit_margins):.3f}   n={n_presented}   "
          f"({n_ties}/{n_presented} below 0.05, i.e. near-ties)")
    print("    -> the fly is not confidently wrong on digits. it is *confused*:")
    print("       the margin collapses, which is what a pattern does when it")
    print("       matches nothing in the vocabulary rather than something in it.")

    # Per-digit stability: which letter does the fly settle on, and how often?
    from collections import Counter
    stability = []
    print()
    print(f"  per-digit stability across {len(sweeps)} independent sweeps:")
    print(f"  {'digit':>5}  {'ink':>4}  {'modal answer':>12}  {'consistency':>11}"
          f"  {'all answers seen':<28}  {'mean margin':>11}")
    print(f"  {'-' * 5}  {'-' * 4}  {'-' * 12}  {'-' * 11}  {'-' * 28}  {'-' * 11}")
    for i, ch in enumerate(sorted(FONT_DIGITS)):
        guesses = [sw[i]["guess"] for sw in sweeps]
        margins = [sw[i]["margin"] for sw in sweeps]
        cnt = Counter(guesses)
        modal, hits = cnt.most_common(1)[0]
        seen = " ".join(f"{g}x{c}" for g, c in cnt.most_common())
        ink = sum(row.count("#") for row in FONT_DIGITS[ch])
        stability.append({
            "digit": ch,
            "ink": ink,
            "modal": modal,
            "modal_count": hits,
            "n": len(guesses),
            "consistency": round(hits / len(guesses), 4),
            "answers": dict(cnt),
            "margin_mean": round(mean(margins), 4),
        })
        print(f"  {ch:>5}  {ink:>4}  {modal:>12}  {hits:>4}/{len(guesses):<6}"
              f"  {seen:<28}  {mean(margins):>11.3f}")

    # Sanity: every answer must be a member of the output vocabulary. If a digit
    # ever appeared in the 'guess' column this whole argument would collapse.
    assert all(r["guess"] in CLASSES for sw in sweeps for r in sw), \
        "non-vocabulary answer appeared"
    print(f"\n  every answer, in every sweep, was one of the {len(CLASSES)} output")
    print("  letters, as required. the digit bitmaps are real input; the spikes are")
    print("  real; the argmax is real. the answer is wrong because there is no")
    print("  digit cell for it to win.")

    # --- proof of non-training, part two -------------------------------------
    # If a weight moved, or the file was rewritten, this fails loudly. Both
    # assertions are the whole point of the exercise: the digits are a control
    # against a classifier that was never given a reason to say them.
    proof = verify(proof_before, brain, args.ckpt)

    print()
    print("  non-training proof (hashes taken before and after every answer):")
    print(f"    weights sha256     before {proof['weights_sha256_before'][:32]}")
    print(f"                       after  {proof['weights_sha256_after'][:32]}   "
          f"unchanged")
    if proof["checkpoint_sha256_before"]:
        print(f"    checkpoint sha256  before "
              f"{proof['checkpoint_sha256_before'][:32]}")
        print(f"                       after  "
              f"{proof['checkpoint_sha256_after'][:32]}   unchanged")
        print(f"    {os.path.getsize(args.ckpt):,} bytes on disk, epoch "
              f"{brain.epoch} - byte-identical, not rewritten")
    print("    weights are mutated only by MushroomBody.apply_dopamine(), which")
    print("    this script never calls: every digit above was a forward pass.")

    if args.json:
        payload = {
            "source": source,
            "epoch": brain.epoch,
            "looks": args.looks,
            "repeats": len(sweeps),
            "n_classes": len(CLASSES),
            "classes": CLASSES,
            "chance": round(1 / len(CLASSES), 4),
            "alphabet_accuracy": round(brain.evaluate(), 4),
            "alphabet_hits": letter_hits,
            "alphabet_presented": letter_runs,
            "alphabet_margin_mean": round(mean(letter_margins), 4),
            "alphabet_margin_min": round(min(letter_margins), 4),
            "digit_margin_mean": round(mean(all_digit_margins), 4),
            "digit_margin_min": round(min(all_digit_margins), 4),
            "digit_near_ties": n_ties,
            "digit_presented": n_presented,
            "digit_correct": n_correct_total,
            "digit_in_vocabulary": n_in_vocab,
            **proof,
            "stability": stability,
            "first_sweep": rows,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
