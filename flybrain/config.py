"""Configuration for the mushroom-body spiking classifier.

Every knob is annotated with the in-vivo structure it stands in for, so it is
always clear what is measured fly anatomy and what is a modelling choice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- circuit size (fly numbers in comments) --------------------------------
    n_receptor: int = 35        # 5x7 letter bitmap ("photoreceptor" sheet)
    n_pn: int = 128             # antennal-lobe projection neurons (~150 uniglomerular PNs)
    n_kc: int = 512             # Kenyon cells (~2000 per hemisphere)
    n_mbon: int = 27            # mushroom-body output neurons (26 letters + space)
    fan_in: int = 16            # PNs sampled per KC (in vivo ~5-10 out of ~150)
    k_active: int = 60          # KC sparsity target, ~12% (in vivo 5-10%)

    # --- where the fixed PN -> KC wiring comes from ----------------------------
    # The Kenyon cell expansion is the one layer a connectome can genuinely
    # supply, because it is genetically determined rather than learned: the
    # fly does not learn which odour channels a Kenyon cell listens to. The
    # plastic synapses stay where the fly keeps them, on KC -> MBON.
    #
    #   "random"              seeded random sampling of `fan_in` PNs per KC.
    #                         This is the exact code path v0.1.0 ships, so its
    #                         published numbers stay reproducible, and it is the
    #                         "no connectome" arm of the comparison - run it at
    #                         the connectome's dimensions with
    #                         Connectome.as_config_overrides().
    #   "connectome"          the measured hemibrain v1.2 PN -> KC wiring
    #                         (Scheffer et al. 2020, CC BY 4.0). See
    #                         flybrain/connectome.py for exactly which part of
    #                         it is measured and which is a modelling choice.
    #   "connectome-shuffled" the same connectome with every KC's partner set
    #                         randomised while keeping that KC's degree. The
    #                         control that separates "these specific partners"
    #                         from "this many partners".
    wiring: str = "random"
    # Path to the compact subgraph. Empty means <repo>/data/hemibrain_mb.npz.
    connectome_path: str = ""

    # --- antennal lobe: graded, probabilistic response -------------------------
    # Each PN fires at a per-timestep probability given by a logistic function of
    # its synaptic drive, which gives a wide dynamic range.
    #
    # This is deliberately *not* a deterministic LIF. With decay beta and
    # threshold theta, a LIF's informative input range is only about
    # theta(1-beta) to 3*theta(1-beta) - for beta=0.85 that is roughly 0.15 to
    # 0.45. Outside it the neuron is either silent or pinned at a constant rate
    # of one spike per step. The latter is fatal here: every PN then emits an
    # identical rate regardless of input, and the letter pattern is destroyed at
    # the very first stage, before anything downstream can see it.
    al_gain: float = 3.0
    al_bias: float = 0.0
    # AL decorrelation: subtract the common mode of the PN rate. Every letter
    # shares a large common component, and if it is left in, the top-k selection
    # downstream is dominated by it and the same KCs win for every stimulus.
    pn_center: bool = True

    # --- calyx: sparse expansion -------------------------------------------------
    # How the KC activation is rescaled before being turned into firing rates:
    #   "rms"  - divide by the RMS of the activation (recommended: keeps the
    #            pattern while removing letter-to-letter scale variation)
    #   "peak" - divide by the maximum, which makes the code dominated by the
    #            single strongest unit and throws away the rest of the pattern
    #   "mean" - divide by the mean of the active units
    #   "none" - raw (drive - threshold) values
    kc_norm: str = "rms"
    # After RMS rescaling the mean active KC sits near 2.9, so the gain has to be
    # well below 1 for the firing probability to land in a graded range. A gain
    # of 3 puts almost every active KC on the kc_max_rate ceiling, which throws
    # away the graded part of the code and keeps only the active/inactive mask.
    kc_rate_gain: float = 0.30  # temperature: activation -> firing probability
    kc_max_rate: float = 0.90   # ceiling on the per-timestep firing probability

    # --- MBON layer: single-compartment LIF with lateral inhibition ------------
    beta_mbon: float = 0.90
    threshold: float = 1.0
    # A LIF settles at drive/(1-beta), so drive must sit near
    # threshold*(1-beta) = 0.1 for the population to fire at graded, informative
    # rates rather than all-or-nothing.
    mbon_bias: float = 0.12
    # Scale on the KC -> MBON synaptic drive. The KC firing probability and the
    # readout gain have to be set independently: the KCs want a *high* firing
    # probability (so the spike pattern is a faithful, low-noise copy of the
    # graded code) while the MBONs want a drive near their 0.1 operating point.
    # Tying the two together through the weight scale forces one of them to be
    # wrong, and a noisy KC sample makes the readout's winner change run to run.
    mbon_gain: float = 0.30

    # --- stimulus presentation -------------------------------------------------
    t_stim: int = 30
    t_reward: int = 12
    rate_on: float = 0.55       # receptor firing rate on an ink pixel
    rate_off: float = 0.04      # receptor firing rate on a blank pixel
    noise_p: float = 0.03       # per-pixel flip probability during training
    shift_prob: float = 0.50    # probability of a +/-1 pixel jitter

    # --- plasticity: three-factor R-STDP on KC -> MBON only --------------------
    # The eligibility trace accumulates over the whole stimulus window, so its
    # entries reach O(1). lr must be small for the weight step to stay comparable
    # to the initial weight scale (target_norm / sqrt(n_kc) ~ 0.05).
    lr: float = 0.004
    e_decay: float = 0.97       # eligibility-trace decay (bridges the reward delay)
    # Learning-rate schedule. A constant step oscillates around the solution
    # instead of settling on it: each trial imprints a noisy sample of the class,
    # so the row random-walks about the class mean forever. An averaging schedule
    # shrinks the step so the row converges to that mean - which is exactly the
    # centroid classifier that measures 96% through this same spiking readout.
    #   "const"  - lr throughout
    #   "inv"    - lr / (1 + epoch / lr_half_life)
    #   "sqrt"   - lr / sqrt(1 + epoch / lr_half_life)
    lr_schedule: str = "const"
    lr_half_life: float = 40.0
    target_norm: float = 1.20   # homeostatic row-norm of the KC->MBON weight matrix
    w_max: float = 0.25

    # --- dopamine (sugar GRN -> SEZ -> PAM/DAN -> MB) --------------------------
    dopa_tau: float = 3.0       # phasic burst time constant, in timesteps
    dopa_delay: int = 1         # conduction delay of the sugar pathway
    dopa_baseline_lr: float = 0.01   # reward-prediction-error baseline adaptation
    sucrose: float = 1.0        # reward delivered on a correct readout
    bitter: float = -1.0        # punishment delivered on an incorrect readout
    # Whether a wrong answer also gets bitter. Punishing the row that answered is
    # what lets the fly reject wrong letters, but early on nearly every trial is
    # wrong, so a held-out evaluation can wobble while the reward signal is still
    # mostly noise. Sugar-only is the gentler regime and matches "sugar for each
    # letter got right"; bitter is what makes it converge faster once it starts.
    punish_wrong: bool = True
    # What the reward is conditioned on:
    #   "operant"   - sugar only when the fly answers correctly, bitter when it
    #                 does not. The teaching signal is therefore a function of
    #                 the fly's own noisy argmax, so the row that gets stamped
    #                 depends on the sampling as much as on the stimulus.
    #   "pavlovian" - the true letter always precedes the sugar, so sugar is
    #                 delivered on every training trial and the *baseline*
    #                 adaptation turns it into a reward prediction error: it is
    #                 large while the letter is a poor predictor and fades as the
    #                 association is learned. The update then averages the KC
    #                 pattern of the class into that class's row on every trial,
    #                 so the row converges to the class centroid - the same
    #                 quantity the centroid readout scores 96% with.
    reward_mode: str = "operant"
    # What dopamine acts on when the credited row is named:
    #   "pattern" - the centred KC pattern of the stimulus, so sugar aligns the
    #               credited row with the letter it was reading (a covariance
    #               rule, and the same quantity the 96% centroid readout uses)
    #   "outer"   - the population-centred eligibility outer product, which is
    #               signed by whether that MBON happened to fire
    credit_mode: str = "pattern"

    # --- run control -----------------------------------------------------------
    seed: int = 7
    epochs: int = 40
    trials_per_class: int = 4
    # How many presentations of the same stimulus are averaged before the reward
    # is decided. The KC code is Poisson-sampled, so a single presentation's
    # argmax flips on noise alone - measured on the trained checkpoint, M came
    # back as eight different letters in eight looks, and averaging saturated at
    # 65% because the weights, not the sampling, set the ceiling. The reward is
    # built from that argmax, so with one look a large share of the "wrong
    # answer" punishments are spurious: they push the credited row away from a
    # stimulus it in fact classifies correctly. More looks cost proportionally
    # more time per trial and buy a reward signal that reflects the stimulus.
    decision_repeats: int = 1
    eval_repeats: int = 3
    log_every: int = 5

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))
