"""Fast checks. The learning test is a smoke test, not a benchmark."""

from __future__ import annotations

import torch

from flybrain import CLASSES, Config, DopamineSystem, FlyBrain
from flybrain.circuit import MushroomBody
from flybrain.encoding import FONT_5X7, N_PIXELS, char_bitmap


def test_font_coverage():
    assert set(CLASSES) == set(FONT_5X7)
    assert len(CLASSES) == 27  # A-Z plus space
    for ch in CLASSES:
        assert char_bitmap(ch).shape == (N_PIXELS,)


def test_bitmap_values_are_binary():
    for ch in "AZ ":
        b = char_bitmap(ch)
        assert torch.all((b == 0) | (b == 1))
    assert char_bitmap(" ").sum() == 0
    assert char_bitmap("I").sum() > 0


def test_encoding_shape_and_rate():
    cfg = Config()
    brain = FlyBrain(cfg)
    spikes = brain._encode("A", augment=False)
    assert spikes.shape == (cfg.t_stim, cfg.n_receptor)
    assert torch.all((spikes == 0) | (spikes == 1))
    # ink pixels must fire more often than blank ones
    ink = char_bitmap("A").bool()
    on = spikes[:, ink].mean()
    off = spikes[:, ~ink].mean()
    assert on > off


def test_circuit_shapes():
    cfg = Config(n_kc=128, k_active=10)
    mb = MushroomBody(cfg)
    counts = mb.present(torch.rand(cfg.t_stim, cfg.n_receptor))
    assert counts.shape == (cfg.n_mbon,)
    assert mb.mbon.w.shape == (cfg.n_mbon, cfg.n_kc)
    assert mb.eligibility.shape == (cfg.n_mbon, cfg.n_kc)


def test_kc_sparsity_is_respected():
    """The APL neuron must cap the active KC population at k."""
    cfg = Config(n_kc=256, k_active=12)
    brain = FlyBrain(cfg)
    with torch.no_grad():
        for ch in "ABCXYZ":
            pn = brain.mb.al.rate_vector(brain._encode(ch, augment=False))
            act = brain.mb.kc.encode(pn)
            active = int((act > 0).sum())
            assert active <= cfg.k_active, f"{ch}: {active} KCs active, max {cfg.k_active}"
            assert active >= cfg.k_active - 1, f"{ch}: only {active} KCs active"


def test_kc_activation_is_normalised_and_graded():
    """Both normalisation modes, because the default is `rms`, not `peak`.

    `peak` divides by the maximum, so the peak lands exactly on 1. `rms`
    divides by the root-mean-square, which keeps the whole pattern instead of
    letting the single strongest unit dominate - which is why it is the
    default. Asserting peak-1 unconditionally would only be testing `peak`.
    """
    for mode in ("peak", "rms"):
        cfg = Config(n_kc=256, k_active=12, kc_norm=mode)
        brain = FlyBrain(cfg)
        brain.mb.present(brain._encode("A", augment=False))
        act = brain.mb.kc.activation
        assert act.min() >= 0.0
        if mode == "peak":
            assert abs(float(act.max()) - 1.0) < 1e-5, f"{mode}: peak is not 1"
        else:
            rms = float((act ** 2).mean().sqrt())
            assert abs(rms - 1.0) < 1e-4, f"{mode}: rms is {rms}, not 1"
        # graded, not binary
        assert 0.0 < float(act.mean()) < float(act.max())


def test_sparse_code_separates_different_letters():
    """Different stimuli must recruit different KC populations."""
    cfg = Config(n_kc=512, k_active=60)
    brain = FlyBrain(cfg)
    with torch.no_grad():
        brain.mb.present(brain._encode("A", augment=False))
        a = (brain.mb.kc.activation > 0).float()
        brain.mb.present(brain._encode("Z", augment=False))
        z = (brain.mb.kc.activation > 0).float()
    overlap = float((a * z).sum()) / max(1.0, float(a.sum()))
    assert overlap < 0.75, f"KC codes for A and Z overlap too much ({overlap:.2f})"


def test_mbon_firing_is_not_saturated():
    """A saturated MBON layer carries no information, so guard the rate."""
    cfg = Config()
    brain = FlyBrain(cfg)
    with torch.no_grad():
        counts = brain.mb.present(brain._encode("A", augment=False))
    frac = float((counts > 0).float().mean())
    assert 0.0 < frac < 1.0, f"MBONs are all-or-nothing (fraction firing = {frac})"
    assert float(counts.max()) < cfg.t_stim, "every MBON fires on every timestep"


def test_weight_normalisation_holds():
    cfg = Config(n_kc=128, n_mbon=27)
    mb = MushroomBody(cfg)
    mb.mbon.apply_update(torch.randn(cfg.n_mbon, cfg.n_kc) * 5.0)
    norms = mb.mbon.w.norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, cfg.target_norm), atol=1e-3)


def test_dopamine_burst_shape_and_polarity():
    dopa = DopamineSystem(n_steps=12, tau=3.0, delay=1, baseline_lr=0.0)
    burst = dopa.sucrose_reward()
    assert burst.trace.shape == (12,)
    assert burst.trace.max().item() > 0
    assert burst.trace[0].item() == 0.0          # conduction delay
    assert burst.peak > 0.9                       # alpha function peaks at 1.0

    punitive = DopamineSystem(n_steps=12, tau=3.0, delay=1, baseline_lr=0.0)
    assert punitive.bitter_punishment().trace.min().item() < 0


def test_reward_prediction_error_adapts():
    dopa = DopamineSystem(n_steps=6, tau=2.0, delay=0, baseline_lr=0.5)
    first = dopa.sucrose_reward().rpe
    for _ in range(20):
        dopa.sucrose_reward()
    later = dopa.sucrose_reward().rpe
    assert later < first
    assert abs(later) < 1e-2


def test_learning_moves_accuracy_above_chance():
    """Real progress above chance (1/27 = 3.7%), with a short run.

    This deliberately uses the converging recipe rather than `Config`'s
    defaults. The defaults are the `operant`/`const`/lr=0.004 combination that
    plateaus at ~60% - correct as a default, but too slow at this tiny size for
    a test that has to finish in seconds.
    """
    cfg = Config(
        epochs=25, trials_per_class=3, n_kc=256, k_active=24, seed=1,
        lr=0.020, credit_mode="pattern", reward_mode="pavlovian",
        lr_schedule="inv", lr_half_life=300,
    )
    brain = FlyBrain(cfg)
    before = brain.evaluate()
    brain.train(verbose=False)
    after = brain.evaluate()
    assert after > max(before, 0.10), f"{before:.3f} -> {after:.3f}"


def test_say_returns_a_fixed_length_string():
    cfg = Config(epochs=1, trials_per_class=1, n_kc=128, k_active=12)
    brain = FlyBrain(cfg)
    out = brain.say("PHILIPPE DELAMBRE", learn=False, verbose=False)
    assert len(out) == len("PHILIPPE DELAMBRE")


def test_checkpoint_round_trip(tmp_path):
    cfg = Config(epochs=1, trials_per_class=1, n_kc=128, k_active=12)
    brain = FlyBrain(cfg)
    p = tmp_path / "ck.pt"
    brain.save(str(p))

    other = FlyBrain(cfg)
    other.load(str(p))
    assert torch.allclose(brain.mb.mbon.w, other.mb.mbon.w)
    assert brain.dopamine.baseline == other.dopamine.baseline
