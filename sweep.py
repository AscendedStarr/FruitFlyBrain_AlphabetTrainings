"""Grid sweep over the representation hyperparameters.

Scores each configuration by the accuracy of a linear probe on the **KC spike
counts** - the actual input the plastic MBON synapses see - plus the KC
sparsity. A configuration that cannot reach high probe accuracy is one where the
expansion has failed to make the letters linearly separable, and no amount of
training on the readout will fix that.
"""

from __future__ import annotations

import itertools
import time

from diagnose import probe_accuracy
from flybrain.config import Config

GRIDS = {
    "kc_norm": ["peak", "rms"],
    "k_active": [8, 16, 32, 60],
    "kc_rate_gain": [0.06, 0.12, 0.30],
    "t_stim": [15, 60],
}


def main() -> None:
    keys = list(GRIDS)
    combos = list(itertools.product(*(GRIDS[k] for k in keys)))
    print(f"Sweeping {len(combos)} configurations on KC spike-count separability")
    header = " | ".join(f"{k:>13}" for k in keys)
    print(f"{header} | {'probe':>7} {'active':>7} {'secs':>6}")
    print("-" * (len(keys) * 16 + 24))

    best = None
    for combo in combos:
        cfg = Config()
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)
        t0 = time.time()
        try:
            acc, sparsity = probe_accuracy(cfg, "kc_spikes", per_class=3, steps=400)
        except Exception as exc:  # noqa: BLE001 - report and keep sweeping
            print(f"{' | '.join(f'{str(v):>13}' for v in combo)} | FAILED: {exc}")
            continue
        secs = time.time() - t0
        vals = " | ".join(f"{str(v):>13}" for v in combo)
        print(f"{vals} | {acc:>6.1%} {sparsity:>6.1%} {secs:>6.1f}")
        if best is None or acc > best[0]:
            best = (acc, dict(zip(keys, combo)), sparsity)

    if best:
        print("\nBest configuration")
        print(f"  probe accuracy : {best[0]:.1%}")
        print(f"  KC sparsity    : {best[2]:.1%}")
        for k, v in best[1].items():
            print(f"  {k:<15}: {v}")


if __name__ == "__main__":
    main()
