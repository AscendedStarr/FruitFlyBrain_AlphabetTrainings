"""Pictures of the fly, the circuit, and a letter flowing through it.

Three figures, all written to ``runs/``:

* ``fly.png``      - the animal plus its olfactory circuit, the learnt weight
                     matrix, and the letter x MBON response matrix
* ``activity.png`` - one character traced stage by stage through the circuit
* terminal         - ASCII, for when you cannot open a PNG

The point of the stage-by-stage figure is that you can *see* where information
is lost. A blurred receptor bitmap is the encoding's fault, a flat PN band is
the antennal lobe's fault, an all-or-nothing KC band is the calyx's fault, and a
flat MBON row is the readout's fault.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch  # noqa: E402

from .encoding import CLASSES, FONT_5X7, encode_from_config  # noqa: E402

# Layer geometry used by both the circuit diagram and the activity figure.
LAYERS = ["receptors", "PNs", "KCs", "MBONs"]


# --------------------------------------------------------------------------- fly
def _draw_fly(ax, brain_size: tuple[float, float] = (0.30, 0.30)) -> None:
    """Draw the animal, with the brain picked out."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    wing = dict(facecolor="#cfe3f7", edgecolor="#5b7ea6", alpha=0.75, zorder=1)
    body = dict(facecolor="#3b3b46", edgecolor="#1c1c22", zorder=3)

    # abdomen, thorax, head
    ax.add_patch(Ellipse((0.70, 0.50), 0.40, 0.22, angle=-10, **body))
    ax.add_patch(Ellipse((0.49, 0.50), 0.20, 0.19, **body))
    ax.add_patch(Circle((0.345, 0.50), 0.083, **body))

    # wings
    ax.add_patch(Ellipse((0.63, 0.66), 0.36, 0.11, angle=28, **wing))
    ax.add_patch(Ellipse((0.63, 0.34), 0.36, 0.11, angle=-28, **wing))

    # eyes
    for dy in (0.035, -0.035):
        ax.add_patch(Circle((0.325, 0.50 + dy), 0.030,
                            facecolor="#8e3b3b", edgecolor="#4a1d1d", zorder=4))

    # legs
    for x0, x1, y1 in ((0.46, 0.38, 0.20), (0.50, 0.50, 0.13), (0.54, 0.62, 0.20)):
        ax.plot([x0, x1], [0.44, y1], color="#1c1c22", lw=1.4, zorder=2)
        ax.plot([x0, x1], [0.56, 1 - y1], color="#1c1c22", lw=1.4, zorder=2)

    # antennae
    for dy in (1, -1):
        ax.plot([0.30, 0.16], [0.50, 0.50 + dy * 0.10], color="#1c1c22", lw=1.4)

    # the brain: mushroom body sits at the top of the head
    ax.add_patch(Circle((0.365, 0.575), 0.052, facecolor="#ffe08a",
                        edgecolor="#b8860b", lw=1.6, zorder=5))
    ax.annotate("mushroom body\n(the memory)",
                xy=(0.365, 0.575), xytext=(0.10, 0.86),
                fontsize=8, color="#8a6508", ha="center",
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.2))
    ax.text(0.70, 0.09, "Drosophila melanogaster", fontsize=9,
            style="italic", ha="center", color="#555")


def _band(ax, n: int, x: float, color: str, width: float = 0.09,
          label: str = "", rng=None) -> tuple[np.ndarray, np.ndarray]:
    """A vertical band of ``n`` neurons at position ``x``."""
    rng = rng or np.random.default_rng(0)
    xs = x + rng.uniform(-width / 2, width / 2, n)
    ys = rng.uniform(0.06, 0.94, n)
    ax.scatter(xs, ys, s=9, c=color, alpha=0.85, linewidths=0)
    if label:
        ax.text(x, 1.0, label, ha="center", fontsize=8, color=color)
    return xs, ys


def _sample_edges(ax, src, dst, k: int, color: str, rng, alpha: float = 0.13) -> None:
    a = rng.choice(len(src[0]), size=min(k, len(src[0])), replace=False)
    b = rng.choice(len(dst[0]), size=min(k, len(dst[0])), replace=False)
    for i, j in zip(a, b):
        ax.plot([src[0][i], dst[0][j]], [src[1][i], dst[1][j]],
                color=color, lw=0.45, alpha=alpha, zorder=0)


def save_circuit(brain, path: str) -> None:
    """The fly, its circuit, the learnt weights, and the response matrix."""
    cfg = brain.cfg
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.20)

    # --- (0,0) the animal ---------------------------------------------------
    _draw_fly(fig.add_subplot(gs[0, 0]))

    # --- (0,1) the circuit --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.10)
    ax.axis("off")
    rng = np.random.default_rng(cfg.seed)

    xs = [0.08, 0.36, 0.68, 0.93]
    cols = ["#4c8cbf", "#3f9e6a", "#c98a2b", "#b04a5a"]
    sizes = [cfg.n_receptor, cfg.n_pn, cfg.n_kc, cfg.n_mbon]
    labels = [f"receptors\n{sizes[0]}", f"projection\nneurons {sizes[1]}",
              f"Kenyon cells\n{sizes[2]} (k={cfg.k_active})", f"MBONs\n{sizes[3]}"]
    bands = []
    for x, col, n, lab in zip(xs, cols, sizes, labels):
        # receptor sheet is drawn as an actual 5x7 grid
        if n == cfg.n_receptor:
            gx, gy = np.meshgrid(np.linspace(-0.045, 0.045, 5),
                                 np.linspace(0.30, 0.70, 7))
            xs_b = x + gx.ravel()
            ys_b = gy.ravel()
            ax.scatter(xs_b, ys_b, s=22, c=col, linewidths=0, zorder=3)
            ax.text(x, 1.0, lab, ha="center", fontsize=8, color=col)
            bands.append((xs_b, ys_b))
        else:
            bands.append(_band(ax, n, x, col, label=lab, rng=rng))

    for i, k in enumerate([70, 130, 220]):
        _sample_edges(ax, bands[i], bands[i + 1], k,
                      cols[i + 1], rng, alpha=0.10)

    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.06, 0.02), (x1 - 0.06, 0.02),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#888", lw=1.0))

    ax.text(0.5, -0.005, "APL lateral inhibition: only the top-k KCs stay active",
            ha="center", fontsize=7.5, color="#666")
    ax.text((xs[2] + xs[3]) / 2, 0.06, "PLASTIC", ha="center", fontsize=8,
            color="#b04a5a", weight="bold")
    ax.set_title("olfactory circuit  (only KC -> MBON is plastic)", fontsize=10)

    # --- (1,0) learnt weights ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    w = brain.mb.mbon.w.detach().numpy()
    vmax = float(np.abs(w).max()) or 1.0
    im = ax.imshow(w, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Kenyon cell")
    ax.set_ylabel("MBON  (' ' then A-Z)")
    ax.set_yticks(range(0, len(CLASSES), 3))
    ax.set_yticklabels([repr(c) for c in CLASSES][::3], fontsize=7)
    ax.set_title("KC -> MBON weights after training (the memory)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02,
                 label="synaptic weight")

    # --- (1,1) response matrix ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    resp = np.zeros((len(CLASSES), len(CLASSES)))
    with torch.no_grad():
        for i, ch in enumerate(CLASSES):
            spikes = encode_from_config(cfg, ch, augment=False)
            resp[i] = brain.mb.present(spikes).numpy()
    im = ax.imshow(resp, aspect="auto", cmap="magma")
    ax.plot(range(len(CLASSES)), range(len(CLASSES)), color="#39ff14",
            lw=0.8, ls="--", alpha=0.8, label="correct readout")
    ax.set_xlabel("MBON fired")
    ax.set_ylabel("letter presented")
    ax.set_xticks(range(0, len(CLASSES), 3))
    ax.set_xticklabels([repr(c) for c in CLASSES][::3], fontsize=7)
    ax.set_yticks(range(0, len(CLASSES), 3))
    ax.set_yticklabels([repr(c) for c in CLASSES][::3], fontsize=7)
    n_diag = int(sum(np.argmax(r) == i for i, r in enumerate(resp)))
    ax.set_title(f"MBON spike counts  ({n_diag}/{len(CLASSES)} letters on the diagonal)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="spikes")

    fig.suptitle("FlyBrain - a dopamine-modulated spiking mushroom body",
                 fontsize=13)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- activity
def save_activity(brain, ch: str, path: str) -> None:
    """One character, traced through every stage of the circuit."""
    cfg = brain.cfg
    spikes = encode_from_config(cfg, ch, augment=False)
    out = brain.mb.stage_outputs(spikes)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    shown = ch.upper() if ch.upper() in FONT_5X7 else " "

    # receptor sheet: the clean glyph, and what the Poisson encoding saw
    ax = axes[0]
    ax.imshow(np.array([[1.0 if c == "#" else 0.0 for c in r]
                        for r in FONT_5X7[shown]]),
              cmap="Greys", vmin=0, vmax=1.4)
    ax.set_title(f"receptor sheet: {shown!r}\n{cfg.n_receptor} pixels, {cfg.t_stim} steps")
    ax.set_xticks([])
    ax.set_yticks([])

    # PN band
    ax = axes[1]
    pn = out["pn"].numpy()
    im = ax.imshow(pn.reshape(8, -1), aspect="auto", cmap="viridis")
    ax.set_title(f"antennal lobe: {cfg.n_pn} PNs\nmean firing probability")
    ax.set_yticks([])
    ax.set_xticks([])
    lo, hi = float(pn.min()), float(pn.max())
    ax.text(0.5, -0.10, f"range {lo:.3f} - {hi:.3f}   (a flat band = the "
                        f"pattern is already gone)", transform=ax.transAxes,
            ha="center", fontsize=7, color="#444")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # KC code
    ax = axes[2]
    im = ax.imshow(out["kc_act"].numpy().reshape(1, -1), aspect="auto",
                   cmap="inferno")
    n_active = int((out["kc_act"] > 0).sum())
    ax.set_title(f"Kenyon cells: {cfg.n_kc}\n{n_active} active "
                 f"({n_active / cfg.n_kc:.1%}, target k={cfg.k_active})")
    ax.set_yticks([])
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # MBON readout
    ax = axes[3]
    counts = out["mbon"].numpy()
    colors = ["#39a039" if i == int(counts.argmax()) else "#7a7a8c"
              for i in range(len(CLASSES))]
    ax.barh(range(len(CLASSES)), counts, color=colors)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([repr(c) for c in CLASSES], fontsize=7)
    ax.invert_yaxis()
    win = CLASSES[int(counts.argmax())]
    ax.set_title(f"MBON readout: {counts.sum():.0f} spikes\n"
                 f"decoded as {win!r}  (presented {shown!r})")
    ax.set_xlabel("spikes in the window")
    ax.grid(alpha=0.25, axis="x")

    ok = "correct" if win == shown else "WRONG"
    fig.suptitle(f"{shown!r} through the circuit  -  read out as {win!r}  ({ok})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- ascii
ASCII_FLY = r"""
                       \    |    /
                        \   |   /
                         \  |  /
        .-----------------.   .-----------------.
       /                   \_/                   \
      |   ______      ___   |   ___      ______    |
      |  /      \    /   \  |  /   \    /      \   |
      | |  (o)   |  | (o) |  | | (o) |  |  (o)  |  |
      |  \______/    \___/  |  \___/    \______/   |
       \                    |                    /
        '---------.    .----+----.    .--------'
                  |   |  BRAIN  |   |
                   \   \  () ()  /  /
                    '---'-.___.-'---'
                           |   |
                           |   |
        35 receptors  ->  128 PNs  ->  512 KCs  ->  27 MBONs
"""


def ascii_report(brain, letters: str = "JEF") -> str:
    """A terminal view: the fly, and each letter's readout."""
    lines = [ASCII_FLY]
    cfg = brain.cfg
    lines.append(f"  plastic synapses: KC -> MBON ({cfg.n_kc} x {cfg.n_mbon}"
                 f" = {cfg.n_kc * cfg.n_mbon:,})")
    lines.append("")
    for ch in letters.upper():
        if ch not in FONT_5X7:
            continue
        spikes = encode_from_config(cfg, ch, augment=False)
        counts = brain.mb.present(spikes)
        pred = CLASSES[int(counts.argmax())]
        lines.append(f"  {ch!r}   glyph      ->   read as {pred!r}")
        for row in FONT_5X7[ch]:
            lines.append(f"          {row.replace('#', '#').replace('.', ' ')}")
        lines.append("")
    return "\n".join(lines)
