"""Which part of the three-factor rule is broken?

The 2-class A/B task stays at exactly 50% for 400 trials at every learning rate,
so the rule is not merely mis-tuned - it carries no usable signal at all. This
file holds the representation and the LIF fixed and varies only the update rule,
so the fault can be localised instead of guessed at.

Suspects, in order of how likely they are:

1. ``apply_update`` renormalises every weight row to ``target_norm`` after every
   single trial. A row that is uniformly scaled has the same *direction*, and
   argmax over ``w @ x`` depends only on direction - so punishing the winner is
   a no-op, and the only surviving effect is a small rotation. Worth checking.
2. The reward is a sign (+/-1). Early on the readout is noise, so "correct" is a
   coin flip and the updates average to zero. The user's framing - sugar for
   each letter got right - argues for a graded reward.
3. Credit assignment. The eligibility is ``outer(post - post.mean(), kc_act)``,
   which credits whichever MBON happened to fire. In the fly the sugar pathway
   drives the MBON that *predicts the sugar*, so the target row should be the
   one that gets reinforced.
"""

from __future__ import annotations

import copy
import torch

from flybrain import Config
from flybrain.trainer import FlyBrain

PAIR = [("A", 0), ("B", 1)]
LRS = [0.002, 0.02, 0.10]
EPOCHS = 300


def _readout(counts: torch.Tensor) -> int:
    return int(counts[:2].argmax())


def _burst_sum(brain: FlyBrain, reward: float) -> float:
    """Total dopamine delivered by one alpha-function burst."""
    burst = brain.dopamine.deliver(reward)
    return float(burst.trace.sum())


# ---------------------------------------------------------------- the variants
def upd_baseline(brain, ch, tgt, counts, cfg):
    """Current behaviour: signed reward, full eligibility, renormalise."""
    ok = _readout(counts) == tgt
    burst = brain.dopamine.deliver(cfg.sucrose if ok else cfg.bitter)
    brain.mb.apply_dopamine(burst.trace)


def upd_sugar_only(brain, ch, tgt, counts, cfg):
    """Sugar on correct letters, nothing at all on wrong ones."""
    if _readout(counts) == tgt:
        burst = brain.dopamine.deliver(cfg.sucrose)
        brain.mb.apply_dopamine(burst.trace)


def upd_margin(brain, ch, tgt, counts, cfg):
    """Graded sugar: how far the target beat the field, not just a sign.

    This is the reward-prediction-error reading of the dopamine signal - PAM
    firing tracks the size of the surprise, and a graded signal gives the update
    a direction even when the hard readout is still a coin flip.
    """
    post = counts / cfg.t_stim
    r = float(post[tgt] - post.mean())
    if r <= 0.0:
        return
    burst = brain.dopamine.deliver(min(1.0, r * 4.0))
    brain.mb.apply_dopamine(burst.trace)


def upd_target_credit(brain, ch, tgt, counts, cfg):
    """Sugar reaches the MBON that predicts it - the target row, not the winner.

    A naive ``outer(post, pre)`` credits whichever MBON fired, which on a
    mis-trial is exactly the wrong one. Sugar in the fly is delivered to the
    reward-predicting MBON compartment, so the target row is what the dopamine
    gates; a wrong answer additionally suppresses the row that answered.
    """
    ok = _readout(counts) == tgt
    elig = brain.mb.eligibility.clone()
    delta = torch.zeros_like(brain.mb.mbon.w)
    delta[tgt] = elig[tgt]
    if not ok:
        w = _readout(counts)
        delta[w] = delta[w] - elig[w]
    brain.mb.mbon.w = (brain.mb.mbon.w
                       + cfg.lr * delta).clamp(-cfg.w_max, cfg.w_max)


def upd_credit_no_norm(brain, ch, tgt, counts, cfg):
    """Target credit without the post-update row renormalisation."""
    ok = _readout(counts) == tgt
    elig = brain.mb.eligibility.clone()
    delta = torch.zeros_like(brain.mb.mbon.w)
    delta[tgt] = elig[tgt]
    if not ok:
        w = _readout(counts)
        delta[w] = delta[w] - elig[w]
    w_new = brain.mb.mbon.w + cfg.lr * delta
    brain.mb.mbon.w = w_new


VARIANTS = {
    "baseline": upd_baseline,
    "sugar_only": upd_sugar_only,
    "margin": upd_margin,
    "target_credit": upd_target_credit,
    "credit_noreno": upd_credit_no_norm,
}


def run(name: str, lr: float, epochs: int = EPOCHS) -> tuple[float, list[float]]:
    cfg = copy.deepcopy(Config())
    cfg.lr = lr
    brain = FlyBrain(cfg)
    update = VARIANTS[name]
    curve: list[float] = []

    for ep in range(1, epochs + 1):
        for ch, tgt in PAIR:
            counts = brain.mb.present(brain._encode(ch, True))
            update(brain, ch, tgt, counts, cfg)
        if ep % (epochs // 6) == 0:
            hits = 0
            for ch, tgt in PAIR:
                hits += int(_readout(brain.mb.present(brain._encode(ch, False))) == tgt)
            curve.append(hits / len(PAIR))
    return curve[-1], curve


def main() -> None:
    print(f"2-class A/B, {EPOCHS} epochs, chance = 50%\n")
    print(f"{'variant':>15} {'lr':>7} {'final':>7}   curve over training")
    best = (0.0, None, None)
    for name in VARIANTS:
        for lr in LRS:
            acc, curve = run(name, lr)
            print(f"{name:>15} {lr:>7.3f} {acc:>6.1%}   "
                  + " ".join(f"{v:>4.0%}" for v in curve))
            if acc > best[0]:
                best = (acc, name, lr)
    print(f"\nbest: {best[1]} @ lr={best[2]} -> {best[0]:.1%}")
    if best[0] <= 0.5:
        print("STILL AT CHANCE - the fault is in the eligibility, not the reward.")


if __name__ == "__main__":
    main()
