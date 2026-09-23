"""Why are M and N read as noise while O and Q are almost right?

Counts the lit pixels per glyph and the resulting KC sparsity. The receptor
sheet is 5x7 = 35 pixels for 26 letters, so dense glyphs push almost every
receptor high at once and the projection-neuron pattern stops being distinctive.
"""

from __future__ import annotations

import os

import torch

# Repo root on sys.path, so this script still finds the package from scripts/.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from flybrain import CLASSES, FONT_5X7
from flybrain.trainer import FlyBrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
brain = FlyBrain.load(os.path.join(ROOT, "runs", "flybrain.pt"))

letters = [c for c in CLASSES if c != " "]
rows = []
with torch.no_grad():
    for ch in letters:
        spikes = brain._encode(ch, augment=False)
        counts = brain.mb.present(spikes)
        pred = CLASSES[int(counts.argmax())]
        lit = sum(row.count("#") for row in FONT_5X7[ch])
        rows.append((ch, lit, float(spikes.mean()), brain.mb.kc_sparsity(),
                     pred == ch))

rows.sort(key=lambda r: -r[3])
print(f"{'ch':>3} {'lit px':>7} {'mean receptor':>14} {'KC active':>10} "
      f"{'ok':>4}")
for ch, lit, rate, spar, ok in rows:
    print(f"{ch:>3} {lit:>7} {rate:>14.3f} {spar:>9.1%} {'yes' if ok else 'no':>4}")

hits = [r for r in rows if r[4]]
miss = [r for r in rows if not r[4]]
print()
print(f"mean KC sparsity for correct letters: {sum(r[3] for r in hits) / len(hits):.1%}")
print(f"mean KC sparsity for wrong letters  : {sum(r[3] for r in miss) / len(miss):.1%}")
print(f"mean lit pixels for correct letters : {sum(r[1] for r in hits) / len(hits):.1f}")
print(f"mean lit pixels for wrong letters   : {sum(r[1] for r in miss) / len(miss):.1f}")

print()
print("the receptor sheet is 5x7 = 35 cells for 26 letters, and the KC layer")
print("keeps a fixed top-k. dense glyphs therefore activate nearly the same")
print("set of Kenyon cells, which is exactly what makes M, N and W collide.")
print()
worst = sorted(rows, key=lambda r: -r[1])[:6]
for ch, lit, _, spar, ok in worst:
    print(f"  {ch}: {lit} of 35 pixels lit ({lit / 35:.0%}), "
          f"KC active {spar:.1%}, {'read correctly' if ok else 'MISREAD'}")
