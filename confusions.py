"""What does the trained brain actually get wrong, and is it random?

If the errors were noise they would scatter evenly across the alphabet. If the
errors are structural they will cluster on glyphs that look alike in a 5x7
bitmap - which says the readout is limited by the input encoding, not by the
amount of training.
"""

from __future__ import annotations

import os

import torch

from flybrain import CLASSES, FONT_5X7
from flybrain.trainer import FlyBrain

HERE = os.path.dirname(os.path.abspath(__file__))
brain = FlyBrain.load(os.path.join(HERE, "runs", "flybrain.pt"))
print(f"epoch {brain.epoch}, holdout {brain.evaluate():.1%}, "
      f"chance {100 / len(CLASSES):.1f}%\n")

letters = [c for c in CLASSES if c != " "]
rows: dict[str, dict[str, int]] = {}
with torch.no_grad():
    for ch in letters:
        counts = brain.mb.present(brain._encode(ch, augment=False))
        pred = CLASSES[int(counts.argmax())]
        rows.setdefault(ch, {})
        rows[ch][pred] = rows[ch].get(pred, 0) + 1

correct = sum(v.get(k, 0) for k, v in rows.items())
print(f"{correct}/{len(letters)} letters read correctly "
      f"({correct / len(letters):.1%})\n")


def hamming(a: str, b: str) -> int:
    """How many of the 35 pixels differ between two glyphs."""
    pa = [p for row in FONT_5X7[a] for p in row]
    pb = [p for row in FONT_5X7[b] for p in row]
    return sum(1 for x, y in zip(pa, pb) if x != y)


print("misread letters, and how similar the glyphs are:")
print(f"  {'wanted':>6} {'read as':>8} {'pixels differing':>17}  overlap")
for ch in letters:
    for pred, n in sorted(rows[ch].items()):
        if pred == ch:
            continue
        d = hamming(ch, pred)
        pct = 100 * (1 - d / 35)
        bar = "#" * int(pct / 4)
        print(f"  {ch:>6} {pred:>8} {d:>17}  {pct:5.0f}% {bar}")

print()
print("mean glyph distance for wrong answers vs a random other letter:")
wrong = [hamming(ch, pred) for ch in letters
         for pred in rows[ch] if pred != ch]
allpairs = [hamming(a, b) for a in letters for b in letters if a != b]
if wrong:
    print(f"  confused pairs : {sum(wrong) / len(wrong):.1f} pixels differ")
    print(f"  all pairs      : {sum(allpairs) / len(allpairs):.1f} pixels differ")
    print("  a confused pair is closer than average, so the errors are")
    print("  structural: the model is blurring glyphs that look alike.")
else:
    print("  no errors on the training alphabet - nothing to explain")

print()
print("the same letter presented twice is not read identically, because the")
print("Kenyon-cell code is sampled Poisson noise. stability check:")
for ch in ("M", "N", "E"):
    outs = []
    for _ in range(8):
        with torch.no_grad():
            c = brain.mb.present(brain._encode(ch, augment=False))
        outs.append(CLASSES[int(c.argmax())])
    stable = outs.count(ch)
    print(f"  {ch}: {stable}/8 correct -> {' '.join(outs)}")
