"""Is the MBON stage graded, or saturated?

The MBON layer is a leaky integrate-and-fire population, so its usable input
band is narrow: with decay ``beta`` and threshold ``theta`` the steady state is
``drive / (1 - beta)``, which means a drive of ``theta * (1 - beta)`` is exactly
what is needed to sit on threshold. Feed it a drive ten times larger and every
unit with the right sign fires on every single timestep, so the spike *counts*
stop carrying the stimulus and become a constant.

This script measures the real numbers instead of reasoning about them.
"""

from __future__ import annotations

import argparse

import torch

from flybrain import CLASSES, Config, FlyBrain, encode_from_config
from flybrain.encoding import FONT_5X7

LETTERS = [c for c in CLASSES if c != " "]


def firing_profile(cfg: Config, brain: FlyBrain, chars: list[str]) -> dict:
    """Per-timestep drive / spike statistics for a handful of characters."""
    mb = brain.mb
    drives, spikes, kc_rates = [], [], []

    for ch in chars:
        spikes_in = encode_from_config(cfg, ch, augment=False)
        mb.reset()
        mb.pn_rate = mb.al.rate_vector(spikes_in)
        mb.kc.encode(mb.pn_rate)

        d_rows, s_rows = [], []
        for _ in range(cfg.t_stim):
            kc_spk = mb.kc.spike_step(mb.gen)
            drive = mb.mbon.drive(kc_spk)
            spk, mb.mbon.mem = mb.mbon.lif(drive, mb.mbon.mem)
            d_rows.append(drive)
            s_rows.append(spk)
            kc_rates.append(float(kc_spk.sum()))

        drives.append(torch.stack(d_rows))     # (T, n_mbon)
        spikes.append(torch.stack(s_rows))     # (T, n_mbon)

    D = torch.cat(drives)      # (n_char * T, n_mbon)
    S = torch.cat(spikes)

    per_char_spikes = torch.stack([s.sum(0) for s in spikes])   # (n_char, n_mbon)
    # How often does the readout actually change its mind?
    winners = per_char_spikes.argmax(1)

    return {
        "operating_point": cfg.threshold * (1.0 - cfg.beta_mbon),
        "drive_abs_mean": float(D.abs().mean()),
        "drive_sd": float(D.std()),
        "drive_max": float(D.max()),
        "fires_per_step": float(S.sum(1).mean()),
        "units_firing_pct": float((S.sum(0) > 0).float().mean()),
        "kc_spikes_per_step": sum(kc_rates) / len(kc_rates),
        "distinct_winners": int(len(set(winners.tolist()))),
        "n_chars": len(chars),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=len(LETTERS))
    args = ap.parse_args()

    cfg = Config()
    brain = FlyBrain(cfg)
    chars = LETTERS[: args.chars]

    print(f"MBON operating point  theta*(1-beta) = "
          f"{cfg.threshold * (1 - cfg.beta_mbon):.4f}")
    print(f"target_norm={cfg.target_norm}  mbon_bias={cfg.mbon_bias}  "
          f"kc_rate_gain={cfg.kc_rate_gain}  n_kc={cfg.n_kc}  k_active={cfg.k_active}")

    st = firing_profile(cfg, brain, chars)
    print()
    for k, v in st.items():
        print(f"  {k:22s} {v if isinstance(v, int) else round(v, 4)}")

    print()
    n = st["n_chars"]
    print(f"  fires/step of {cfg.n_mbon} MBONs : {st['fires_per_step']:.2f}")
    print(f"  a good readout needs this well below {cfg.n_mbon}")

    if st["fires_per_step"] > cfg.n_mbon * 0.5:
        print("\n  => SATURATED. nearly every MBON fires every step; the counts")
        print("     are near-constant and argmax is reading noise.")
    elif st["fires_per_step"] < 0.5:
        print("\n  => SILENT. too few spikes for the counts to be informative.")
    else:
        print("\n  => graded. spike counts carry the stimulus.")


if __name__ == "__main__":
    main()
