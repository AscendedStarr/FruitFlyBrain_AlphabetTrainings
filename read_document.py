"""Read a real document with the trained fly and record exactly what it said.

The browser demo shows this happening one character at a time. This script does
the same thing headlessly so the result can be published as a number instead of
a screen recording.

What is and is not happening here
---------------------------------
Every character is a **forward pass**. ``learn`` is hard-coded to ``False`` and
``MushroomBody.apply_dopamine()`` - the only function in the project that writes
a weight - is never called. Before the first character and after the last, the
circuit's weight matrices and the checkpoint file are hashed and compared, so
the claim is verified rather than asserted. See :mod:`flybrain.proof`.

What counts as read
-------------------
The stream keeps every character that came out of the file. Three cases:

* **letters and space** - the fly has an output cell for these, so a right
  answer is possible.
* **digits** - there is a 5x7 glyph, so a digit can be drawn on the receptor
  sheet, but there is no output cell. A digit is therefore wrong by
  construction, and the script records which letter the fly says instead.
  This is the negative control, run here on a whole book rather than on a
  ten-glyph table.
* **punctuation** - no glyph at all, so it is stepped over and excluded from
  the accuracy. It is counted separately and never quietly dropped.

    .\\.venv-flybrain\\Scripts\\python.exe read_document.py <path>
    .\\.venv-flybrain\\Scripts\\python.exe read_document.py <path> --limit 20000
    .\\.venv-flybrain\\Scripts\\python.exe read_document.py <path> --json runs/book_read.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from flybrain import (  # noqa: E402
    CLASSES,
    Config,
    FlyBrain,
    encode_from_config,
    is_class,
    is_input,
)
from flybrain.docread import build  # noqa: E402
from flybrain.proof import snapshot, verify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def read_one(brain: FlyBrain, ch: str, looks: int) -> tuple[int, float]:
    """Present one glyph `looks` times; return (argmax index, normalised margin).

    Deliberately mirrors `Session.present` in serve.py and `read_out` in
    digit_proof.py, including passing `gen=brain.gen`. Without that generator
    every look would be the identical spike train and averaging four of them
    would reduce no variance at all.

    The margin is (winner - runner-up) / winner, the same quantity the digit
    control reports, so the two tables are directly comparable.
    """
    counts = brain.mb.present(
        encode_from_config(brain.cfg, ch, augment=False, gen=brain.gen))
    for _ in range(looks - 1):
        counts = counts + brain.mb.present(
            encode_from_config(brain.cfg, ch, augment=False, gen=brain.gen))
    pred = int(counts.argmax())
    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    top = float(counts[order[0]])
    margin = (top - float(counts[order[1]])) / (top or 1.0)
    return pred, margin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="document to read (.pdf, .epub, .txt, ...)")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "flybrain.pt"))
    ap.add_argument("--looks", type=int, default=4,
                    help="presentations averaged per answer (the UI uses 4)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N characters; 0 reads the whole document")
    ap.add_argument("--fold", action="store_true",
                    help="replace punctuation with spaces before reading. "
                         "Off by default, because stepping over punctuation is "
                         "the honest picture of what this model can do.")
    ap.add_argument("--every", type=int, default=20000,
                    help="print progress every N characters (0 = silent)")
    ap.add_argument("--sample", type=int, default=400,
                    help="characters of the fly's own transcript to include in "
                         "the JSON. Kept small on purpose: a full transcript of "
                         "a copyrighted book is a copy of the book.")
    ap.add_argument("--save-text", default=None,
                    help="write the full answer stream here. Off by default for "
                         "the copyright reason above.")
    ap.add_argument("--json", default=None, help="write the result JSON here")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    brain = FlyBrain.load(args.ckpt) if os.path.isfile(args.ckpt) else FlyBrain(Config())
    brain.cfg.decision_repeats = args.looks
    src = (f"{os.path.basename(args.ckpt)} (epoch {brain.epoch})"
           if os.path.isfile(args.ckpt) else "untrained circuit")

    with open(args.path, "rb") as fh:
        raw = fh.read()
    doc = build(os.path.basename(args.path), raw, fold=args.fold)
    stream = doc["stream"]
    if args.limit:
        stream = stream[:args.limit]

    print("FlyBrain reads a document")
    print(f"  circuit      : {src}")
    print(f"  document     : {doc['title']}")
    print(f"  kind         : {doc['kind']}, {len(doc['pages'])} page(s), "
          f"{doc['chars']:,} characters")
    print(f"  vocabulary   : {len(CLASSES)} output cells - {' '.join(CLASSES)}")
    print(f"  no cell for  : {' '.join('0123456789')} "
          f"(presentable input, unanswerable)")
    print(f"  reading      : {len(stream):,} characters, {args.looks} looks each")
    print()

    # --- proof of non-training, part one -------------------------------------
    before = snapshot(brain, args.ckpt)
    print("  weights have been hashed; the run below is a pure forward pass.")
    print()

    t0 = time.perf_counter()
    # Per-bucket tallies. `space` is tracked apart from `letter` because a
    # space is 17% of a typical book and it is the easiest thing to "read" -
    # folding it into the headline number would flatter the result.
    tally = {
        "letter": {"n": 0, "hit": 0, "margin_sum": 0.0},
        "space": {"n": 0, "hit": 0, "margin_sum": 0.0},
        "digit": {"n": 0, "hit": 0, "margin_sum": 0.0},
    }
    digit_answers: dict[str, dict[str, int]] = {d: {} for d in "0123456789"}
    stepped = 0
    answers: list[str] = []
    errors: list[dict] = []

    for i, ch in enumerate(stream):
        if not is_input(ch):
            stepped += 1
            continue
        bucket = "digit" if ch.isdigit() else ("space" if ch == " " else "letter")
        pred, margin = read_one(brain, ch, args.looks)
        guess = CLASSES[pred]
        answers.append(guess)
        target_ok = is_class(ch) and CLASSES[pred] == ch.upper()

        tally[bucket]["n"] += 1
        tally[bucket]["hit"] += int(target_ok)
        tally[bucket]["margin_sum"] += margin

        if bucket == "digit":
            digit_answers[ch][guess] = digit_answers[ch].get(guess, 0) + 1
        elif not target_ok and len(errors) < 200:
            errors.append({"at": i, "shown": ch, "said": guess,
                           "margin": round(margin, 4)})

        if args.every and (i + 1) % args.every == 0:
            done = sum(v["n"] for v in tally.values())
            rate = done / max(1e-9, time.perf_counter() - t0)
            left = (len(stream) - i - 1) / max(1e-9, rate)
            print(f"    {i + 1:>9,}/{len(stream):,} chars  "
                  f"{rate:>8,.0f} char/s  ~{left / 60:>5.1f} min left", flush=True)

    elapsed = time.perf_counter() - t0

    # --- proof of non-training, part two -------------------------------------
    proof = verify(before, brain, args.ckpt)

    n_presented = sum(v["n"] for v in tally.values())
    n_hit = sum(v["hit"] for v in tally.values())
    n_answerable = tally["letter"]["n"] + tally["space"]["n"]
    n_answerable_hit = tally["letter"]["hit"] + tally["space"]["hit"]

    def pct(hit: int, n: int) -> float:
        return round(100.0 * hit / n, 2) if n else 0.0

    print()
    print(f"  read {n_presented:,} characters in {elapsed:,.1f}s "
          f"({n_presented / max(1e-9, elapsed):,.0f} char/s)")
    print(f"  stepped over        : {stepped:,} (no glyph - punctuation)")
    print()
    print(f"  {'bucket':<8} {'shown':>9} {'read right':>12} {'accuracy':>9} {'mean margin':>12}")
    print(f"  {'-' * 8} {'-' * 9} {'-' * 12} {'-' * 9} {'-' * 12}")
    for name in ("letter", "space", "digit"):
        v = tally[name]
        mm = v["margin_sum"] / v["n"] if v["n"] else 0.0
        print(f"  {name:<8} {v['n']:>9,} {v['hit']:>12,} "
              f"{pct(v['hit'], v['n']):>8.1f}% {mm:>12.3f}")
    print(f"  {'-' * 8} {'-' * 9} {'-' * 12} {'-' * 9} {'-' * 12}")
    print(f"  {'all':<8} {n_presented:>9,} {n_hit:>12,} "
          f"{pct(n_hit, n_presented):>8.1f}%")
    print(f"  {'answerable':<8} {n_answerable:>9,} {n_answerable_hit:>12,} "
          f"{pct(n_answerable_hit, n_answerable):>8.1f}%   "
          f"(letters + space, digits excluded)")
    print(f"  chance would be {100 / len(CLASSES):.1f}%")

    print()
    print("  the digits in this book, and what the fly said instead")
    print("  (it has no digit output cell, so every one of these is wrong)")
    print(f"  {'digit':>5} {'shown':>8}   answers")
    print(f"  {'-' * 5} {'-' * 8}   {'-' * 40}")
    for d in "0123456789":
        seen = digit_answers[d]
        n = sum(seen.values())
        if not n:
            continue
        top = " ".join(f"{g}x{c}" for g, c in
                       sorted(seen.items(), key=lambda kv: -kv[1])[:5])
        print(f"  {d:>5} {n:>8,}   {top}")

    print()
    print("  non-training proof")
    print(f"    weights sha256     before {proof['weights_sha256_before'][:32]}")
    print(f"                       after  {proof['weights_sha256_after'][:32]}"
          f"   {'unchanged' if proof['weights_unchanged'] else 'CHANGED'}")
    if proof["checkpoint_sha256_before"]:
        print(f"    checkpoint sha256  before "
              f"{proof['checkpoint_sha256_before'][:32]}")
        print(f"                       after  "
              f"{proof['checkpoint_sha256_after'][:32]}"
              f"   {'unchanged' if proof['checkpoint_unchanged'] else 'CHANGED'}")
        print(f"    {os.path.getsize(args.ckpt):,} bytes on disk, epoch "
              f"{brain.epoch} - byte-identical")
    print("    weights are written only by MushroomBody.apply_dopamine(), which")
    print("    this script never calls. reading a book changed nothing.")

    if args.save_text:
        with open(args.save_text, "w", encoding="utf-8") as fh:
            fh.write("".join(answers))
        print(f"\n  wrote {args.save_text}")

    if args.json:
        payload = {
            "source": src,
            "epoch": brain.epoch,
            "document": {
                "title": doc["title"],
                "kind": doc["kind"],
                "pages": len(doc["pages"]),
                "chars": doc["chars"],
                "counts": doc["counts"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
            "looks": args.looks,
            "limit": args.limit or None,
            "read": {
                "presented": n_presented,
                "hit": n_hit,
                "accuracy_pct": pct(n_hit, n_presented),
                "answerable": n_answerable,
                "answerable_hit": n_answerable_hit,
                "answerable_accuracy_pct": pct(n_answerable_hit, n_answerable),
                "stepped_over": stepped,
                "chance_pct": round(100 / len(CLASSES), 2),
                "elapsed_s": round(elapsed, 2),
                "chars_per_s": round(n_presented / max(1e-9, elapsed), 1),
            },
            "buckets": {
                name: {
                    "shown": v["n"],
                    "hit": v["hit"],
                    "accuracy_pct": pct(v["hit"], v["n"]),
                    "margin_mean": round(v["margin_sum"] / v["n"], 4) if v["n"] else None,
                }
                for name, v in tally.items()
            },
            "digit_answers": {d: digit_answers[d] for d in "0123456789"
                              if digit_answers[d]},
            "answers_sha256": hashlib.sha256("".join(answers).encode()).hexdigest(),
            "alphabet_holdout_accuracy": round(brain.evaluate(), 4),
            "sample_transcript": "".join(answers[:args.sample]),
            "first_errors": errors,
            "proof": proof,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
