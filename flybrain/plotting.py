"""Plots for the training run. Matplotlib is optional; this module is imported lazily."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .encoding import CLASSES  # noqa: E402


def save_report(brain, path: str) -> None:
    hist = brain.history
    if not hist:
        raise RuntimeError("nothing to plot: train first")

    epochs = [h.epoch for h in hist]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0][0]
    ax.plot(epochs, [h.accuracy for h in hist], color="tab:blue")
    ax.set_title("held-out accuracy over letters")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(epochs, [h.reward_mean for h in hist], color="tab:green")
    ax.set_title("fraction of trials answered correctly (reward rate)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(epochs, [h.dopa_baseline for h in hist], color="tab:red")
    ax.set_title("dopamine baseline (reward-prediction-error adaptation)")
    ax.set_xlabel("epoch")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    burst = brain.dopamine.wave(1.0)
    ax.plot(range(len(burst)), burst, marker="o", color="tab:purple")
    ax.set_title("phasic PAM/DAN burst kernel (tau=%.1f, delay=%d)"
                 % (brain.cfg.dopa_tau, brain.cfg.dopa_delay))
    ax.set_xlabel("timestep after reward")
    ax.set_ylabel("dopamine concentration (a.u.)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"mushroom body: {brain.cfg.n_kc} KCs, k={brain.cfg.k_active}, "
        f"{len(CLASSES)} classes, R-STDP"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
