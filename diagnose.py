"""Diagnostics: find the stage that destroys the letter identity.

`ladder` probes the representation at every stage of the circuit. If accuracy
collapses between two stages, that stage is the one throwing the signal away -
which is far more useful than only seeing the final readout accuracy.

    receptor bitmap -> PN (raw) -> PN (centred) -> KC drive -> KC activation -> KC spikes

The probe is a plain multinomial logistic regression trained to convergence on
standardised features. It measures how *linearly separable* each representation
is; it is a ceiling on what the MBON readout could ever learn, not a model of
the fly.
"""

from __future__ import annotations

import torch

from flybrain.circuit import MushroomBody
from flybrain.config import Config
from flybrain.encoding import CLASSES, encode_from_config

STAGES = ["bitmap", "pn", "pn_centered", "kc_drive", "kc_act", "kc_spikes"]


def make_dataset(cfg: Config, brain: MushroomBody, per_class: int, stage: str):
    xs, ys = [], []
    with torch.no_grad():
        for label, ch in enumerate(CLASSES):
            for _ in range(per_class):
                spikes = encode_from_config(cfg, ch, augment=True)
                out = brain.stage_outputs(spikes)
                xs.append(out[stage])
                ys.append(label)
    return torch.stack(xs), torch.tensor(ys)


def probe_accuracy(
    cfg: Config,
    stage: str,
    per_class: int = 4,
    steps: int = 600,
    lr: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Returns (test accuracy, KC sparsity). Sparsity is 0 for non-KC stages."""
    brain = MushroomBody(cfg)
    x, y = make_dataset(cfg, brain, per_class, stage)
    sparsity = brain.kc_sparsity() if stage.startswith("kc") else 0.0

    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    x = (x - mu) / sd

    torch.manual_seed(seed)
    n_val = max(1, x.shape[0] // 4)
    perm = torch.randperm(x.shape[0])
    tr, va = perm[:-n_val], perm[-n_val:]

    w = torch.zeros(x.shape[1], len(CLASSES), requires_grad=True)
    b = torch.zeros(len(CLASSES), requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    lossf = torch.nn.CrossEntropyLoss()

    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(x[tr] @ w + b, y[tr])
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = (x[va] @ w + b).argmax(1)
        acc = float((pred == y[va]).to(torch.float32).mean())
    return acc, sparsity


def ladder(cfg: Config, per_class: int = 4) -> None:
    chance = 1.0 / len(CLASSES)
    print(f"\n=== representation ladder ({len(CLASSES)} classes, chance {chance:.1%}) ===")
    print(f"    kc_norm={cfg.kc_norm}  k_active={cfg.k_active}  t_stim={cfg.t_stim}")
    print(f"    {'stage':<12} {'dims':>6} {'probe acc':>10} {'kc active':>10}  delta")
    dims = {
        "bitmap": cfg.n_receptor,
        "pn": cfg.n_pn,
        "pn_centered": cfg.n_pn,
        "kc_drive": cfg.n_kc,
        "kc_act": cfg.n_kc,
        "kc_spikes": cfg.n_kc,
    }
    prev = None
    for stage in STAGES:
        acc, sparsity = probe_accuracy(cfg, stage, per_class=per_class)
        drop = "" if prev is None else f"{acc - prev:+.1%}"
        sp = "" if not sparsity else f"{sparsity:>9.1%}"
        print(f"    {stage:<12} {dims[stage]:>6} {acc:>9.1%} {sp:>10}  {drop}")
        prev = acc


def code_stats(cfg: Config, repeats: int = 8) -> None:
    """KC code health: sparsity, and how much the code varies between letters."""
    brain = MushroomBody(cfg)
    print("\n=== KC code stats ===")
    with torch.no_grad():
        codes = []
        for ch in CLASSES:
            act = None
            for _ in range(repeats):
                spikes = encode_from_config(cfg, ch, augment=False)
                brain.reset()
                pn = brain.al.rate_vector(spikes)
                a = brain.kc.encode(pn)
                act = a if act is None else act + a
            codes.append(act / repeats)
        codes = torch.stack(codes)

    active = (codes > 0).to(torch.float32)
    print(f"    KC sparsity        : {active.mean():.1%}  ({cfg.k_active}/{cfg.n_kc} target)")
    print(f"    ever active        : {(active.sum(0) > 0).to(torch.float32).mean():.1%} of KCs used by some letter")
    live = codes[codes > 0]
    print(f"    activation range   : {codes.max():.3f} max, {live.mean():.3f} mean of active")

    # Cosine similarity between letter codes: 1.0 means the code cannot tell the
    # two letters apart at all.
    norm = codes / codes.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = norm @ norm.t()
    off = sim[~torch.eye(len(CLASSES), dtype=torch.bool)]
    print(f"    mean cosine sim    : {off.mean():.3f}  (lower = more distinct)")
    worst = sim.clone()
    worst.fill_diagonal_(-1.0)
    i, j = divmod(int(worst.argmax()), len(CLASSES))
    print(f"    most confusable    : {CLASSES[i]!r} vs {CLASSES[j]!r} ({worst.max():.3f})")


def weight_scale(cfg: Config) -> None:
    """Check the learning rate against the initial weight scale."""
    brain = MushroomBody(cfg)
    init_scale = cfg.target_norm / cfg.n_kc**0.5
    print("\n=== plasticity magnitudes ===")
    with torch.no_grad():
        spikes = encode_from_config(cfg, CLASSES[1], augment=False)
        brain.present(spikes)
        e = brain.eligibility
        eff = e.abs().max() / (1 - cfg.e_decay)
        print(f"    eligibility        : max {e.max():.3f}, mean abs {e.abs().mean():.3f}")
        print(f"    init weight scale  : {init_scale:.4f}")
        print(f"    worst-case dw      : {cfg.lr * eff:.4f}  ({cfg.lr * eff / init_scale:.1f}x init scale)")


def main() -> None:
    cfg = Config()
    print("FlyBrain diagnostics")
    print(f"  receptors={cfg.n_receptor}  PN={cfg.n_pn}  KC={cfg.n_kc}  MBON={cfg.n_mbon}")
    print(f"  only KC->MBON is plastic (w_kc_mbon {tuple(MushroomBody(cfg).mbon.w.shape)})")
    ladder(cfg)
    code_stats(cfg)
    weight_scale(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
