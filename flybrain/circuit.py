"""The mushroom-body circuit.

Three stages, matching the fly's olfactory associative-learning pathway:

    receptor sheet  ->  antennal lobe (PNs)  ->  Kenyon cells  ->  MBONs

Only the **KC -> MBON** synapse is plastic. That is the biologically correct
place for the memory: PN -> KC wiring is genetically determined and not where
reward-modulated learning is observed, so it is held fixed. Reward reaches the
memory through the dopaminergic neurons modelled in ``dopamine.py``.

Because the PN -> KC matrix is fixed rather than learned, it is also the one
layer that can be taken from a real connectome instead of drawn from a random
number generator - which is what ``cfg.wiring`` selects. It changes the wiring,
not the task: the 5x7 glyph encoding, the 27 output labels, and the decision to
read 27 letters out of an MBON population that the fly does not label with
letters remain inventions of this project.

What each stage is for
----------------------
* **Antennal lobe** - a fixed, non-plastic decorrelating expansion with a wide
  dynamic range, so the graded pattern survives.
* **Calyx / Kenyon cells** - a sparse, high-dimensional combinatorial code. Each
  KC samples only a handful of the PNs, and the single giant GABAergic APL
  neuron shifts the population so only the top ``k`` units are active. This is
  the expansion that makes a linear readout separable. In the fly the sampling
  is fixed by development; here it is either a seeded random draw or the
  measured connectome, depending on ``cfg.wiring``.
* **MBON layer** - the readout, with lateral inhibition, and the only plastic
  synapses in the circuit.

Two numerical traps this module is written to avoid, both of which silently
destroy all the information in the signal:

1. A deterministic LIF has a usable input range of only about
   :math:`\\theta(1-\\beta)` to :math:`3\\theta(1-\\beta)`. Outside it the neuron
   is silent or pinned at a constant rate, so the AL uses a logistic response
   instead.
2. Peak-normalising the KC activation makes the code dominated by the single
   strongest unit. The default rescaling is RMS instead.

The MBON layer keeps the LIF, where a spike is what carries the eligibility
trace for plasticity, and its drive is scaled by ``mbon_bias`` to sit on the
:math:`\\theta(1-\\beta)` operating point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config


class _LIF:
    """Leaky integrate-and-fire neuron, reset to zero on spike.

    Implemented locally rather than via snnTorch so the dynamics stay explicit
    and the operating-point arithmetic above is checkable by hand.
    """

    def __init__(self, beta: float, threshold: float) -> None:
        self.beta = beta
        self.threshold = threshold

    def __call__(self, drive: torch.Tensor, mem: torch.Tensor):
        mem = self.beta * mem + drive
        spk = (mem >= self.threshold).to(mem.dtype)
        mem = mem * (1.0 - spk)          # reset-to-zero
        return spk, mem


class AntennalLobe:
    """Receptor sheet -> projection neurons. Fixed, non-plastic, decorrelating."""

    def __init__(self, cfg: Config) -> None:
        gen = torch.Generator().manual_seed(cfg.seed + 11)
        self.w = torch.randn(cfg.n_pn, cfg.n_receptor, generator=gen)
        self.w = self.w / cfg.n_receptor**0.5
        self.spike_gen = torch.Generator().manual_seed(cfg.seed + 12)
        self.rate_history = torch.zeros(0, cfg.n_pn)
        self.cfg = cfg

    def response(self, x: torch.Tensor) -> torch.Tensor:
        """Graded per-timestep firing probability of each PN."""
        drive = F.linear(x, self.w)
        return torch.sigmoid(self.cfg.al_gain * drive + self.cfg.al_bias)

    def spike_step(self, x: torch.Tensor) -> torch.Tensor:
        p = self.response(x)
        u = torch.rand(self.cfg.n_pn, generator=self.spike_gen)
        return (u < p).to(p.dtype)

    def rate_vector(self, spikes: torch.Tensor) -> torch.Tensor:
        """Mean firing probability over a window, ``(T, n_rec)`` -> ``(n_pn,)``.

        The whole window is projected in a single matmul. The PN response is
        purely feedforward, so no timestep depends on the one before it, and
        looping row by row costs 30 dispatches for a 128x35 product whose
        arithmetic is far smaller than the cost of issuing it. ``rate_history``
        keeps the per-timestep rate so the UI can plot the antennal lobe over
        time.
        """
        if spikes.shape[0] == 0:
            self.rate_history = torch.zeros(0, self.cfg.n_pn)
            return torch.zeros(self.cfg.n_pn)
        self.rate_history = self.response(spikes)
        return self.rate_history.mean(0)


class KenyonCells:
    """PNs -> Kenyon cells. A sparse, high-dimensional, graded expansion.

    The expansion matrix is fixed and not learned, because in the fly it is
    genetically determined: a Kenyon cell does not learn which odour channels to
    listen to. ``cfg.wiring`` decides where that fixed matrix comes from - a
    seeded random draw (what v0.1.0 ships) or the measured hemibrain connectome.
    See :mod:`flybrain.connectome` for what is measured and what is a modelling
    choice.
    """

    def __init__(self, cfg: Config) -> None:
        if cfg.wiring == "random":
            gen = torch.Generator().manual_seed(cfg.seed + 23)
            self.w = torch.zeros(cfg.n_kc, cfg.n_pn)
            fan_in = max(1, min(cfg.fan_in, cfg.n_pn))
            scale = 1.0 / fan_in**0.5
            for kc in range(cfg.n_kc):
                idx = torch.randperm(cfg.n_pn, generator=gen)[:fan_in]
                self.w[kc, idx] = torch.randn(fan_in, generator=gen) * scale
        elif cfg.wiring in ("connectome", "connectome-shuffled"):
            from .connectome import load, pn_kc_weights
            conn = load(cfg.connectome_path or None)
            self.w = pn_kc_weights(conn, variant=cfg.wiring, seed=cfg.seed)
            if self.w.shape != (cfg.n_kc, cfg.n_pn):
                raise ValueError(
                    "connectome wiring is %s but the config asks for "
                    "n_kc=%d, n_pn=%d - use "
                    "Connectome.as_config_overrides() to size the circuit from "
                    "the connectome" % (tuple(self.w.shape), cfg.n_kc, cfg.n_pn))
        else:
            raise ValueError("unknown cfg.wiring %r, expected one of %s"
                             % (cfg.wiring, ("random", "connectome",
                                             "connectome-shuffled")))

        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.drive = torch.zeros(self.cfg.n_kc)
        self.activation = torch.zeros(self.cfg.n_kc)
        self.active = torch.zeros(self.cfg.n_kc)

    def encode(self, pn_rate: torch.Tensor) -> torch.Tensor:
        """Sparse graded KC activation for one stimulus."""
        if self.cfg.pn_center:
            pn_rate = pn_rate - pn_rate.mean()

        drive = F.linear(pn_rate, self.w)
        self.drive = drive

        # APL lateral inhibition: shift the population by the k-th largest drive
        # so exactly the top-k units sit above zero.
        k = min(self.cfg.k_active, self.cfg.n_kc)
        kth = drive.topk(k, dim=0).values[-1]
        act = (drive - kth).clamp(min=0.0)

        act = self._rescale(act)
        self.activation = act
        return act

    def _rescale(self, act: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.kc_norm
        if mode == "none":
            return act
        if mode == "peak":
            peak = act.max()
            return act / peak if peak > 0 else act
        if mode == "mean":
            active = act[act > 0]
            mu = active.mean() if active.numel() else torch.tensor(1.0)
            return act / (mu + 1e-6)
        if mode == "rms":
            rms = act.pow(2).mean().sqrt()
            return act / (rms + 1e-6)
        raise ValueError(f"unknown kc_norm {mode!r}")

    def spike_step(self, gen: torch.Generator) -> torch.Tensor:
        """One timestep of KC spikes, Poisson at the graded rate."""
        p = (self.activation * self.cfg.kc_rate_gain).clamp(max=self.cfg.kc_max_rate)
        spk = (torch.rand(self.cfg.n_kc, generator=gen) < p).to(p.dtype)
        self.active = spk
        return spk

    def spike_raster(self, n_steps: int, gen: torch.Generator) -> torch.Tensor:
        """Every timestep of KC spikes sampled in ONE op, as ``(T, n_kc)``.

        The active KCs are independent Bernoulli draws at a rate that is fixed
        for the whole stimulus, so the entire raster is available at once and
        the per-timestep loop buys nothing. It costs a lot, though: profiling
        showed 30 separate ``torch.rand(512)`` calls spend ~96% of their time in
        Python dispatch and only ~4% doing arithmetic, and that loop was ~70% of
        the cost of a trial. One batched draw is ~34x faster for identical
        results.
        """
        p = (self.activation * self.cfg.kc_rate_gain).clamp(max=self.cfg.kc_max_rate)
        raster = (torch.rand(n_steps, self.cfg.n_kc, generator=gen) < p).to(p.dtype)
        self.active = raster[-1]
        return raster


class MushroomBodyOutput:
    """Kenyon cells -> mushroom-body output neurons. The only plastic synapses."""

    def __init__(self, cfg: Config) -> None:
        gen = torch.Generator().manual_seed(cfg.seed + 37)
        self.cfg = cfg
        init_scale = cfg.target_norm / cfg.n_kc**0.5
        self.w = torch.randn(cfg.n_mbon, cfg.n_kc, generator=gen) * init_scale
        self.lif = _LIF(cfg.beta_mbon, cfg.threshold)
        self.reset()

    def reset(self) -> None:
        self.mem = torch.zeros(self.cfg.n_mbon)

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        raw = F.linear(x, self.w)
        # Lateral inhibition: only the population-relative drive matters. The
        # gain sets where that drive lands relative to the LIF's operating point
        # and the bias nudges it near threshold.
        return ((raw - raw.mean()) * self.cfg.mbon_gain
                + self.cfg.mbon_bias)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        spk, self.mem = self.lif(self.drive(x), self.mem)
        return spk

    def run_raster(self, kc_raster: torch.Tensor) -> torch.Tensor:
        """Batched LIF over a whole spike raster, returning ``(T, n_mbon)``.

        A leaky integrator is only sequential in its membrane potential, so
        every timestep's synaptic drive can be computed in a single matmul and
        only the recurrence itself - arithmetic on 27 numbers - has to be
        stepped. Looping and calling ``F.linear`` once per timestep instead
        spends nearly all of its time in dispatch overhead for a 27x512 product
        that takes 3.5 microseconds of actual arithmetic.
        """
        raw = F.linear(kc_raster, self.w)
        # Lateral inhibition, population-relative and per timestep.
        drives = ((raw - raw.mean(dim=1, keepdim=True)) * self.cfg.mbon_gain
                  + self.cfg.mbon_bias)
        mem = torch.zeros(self.cfg.n_mbon, dtype=drives.dtype)
        rows: list[torch.Tensor] = []
        for t in range(drives.shape[0]):
            mem = self.cfg.beta_mbon * mem + drives[t]
            spk = (mem >= self.cfg.threshold).to(drives.dtype)
            # reset-to-zero, matching _LIF exactly
            mem = mem * (1.0 - spk)
            rows.append(spk)
        self.mem = mem
        return torch.stack(rows)

    def eligibility_outer(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Outer product shaped like the weight matrix, ``(n_mbon, n_kc)``.

        Two deliberate choices:

        * ``torch.outer(a, b)`` returns ``(len(a), len(b))``, so the
          post-synaptic vector comes first to line up with ``self.w``.
        * The post-synaptic term is **centred** on the MBON population mean. A
          raw Hebbian product reinforces every co-active pair, including the
          wrong outputs, so reward carries almost no directional information.
          Centring turns this into a covariance rule, which is what makes credit
          assignment work.
        """
        return torch.outer(post - post.mean(), pre)

    def apply_update(self, delta: torch.Tensor) -> None:
        self.w = self.w + delta
        self.w = self.w.clamp(-self.cfg.w_max, self.cfg.w_max)
        # Homeostatic rescaling of each MBON's incoming weight vector. Without
        # this, reward-modulated Hebbian learning runs away.
        norm = self.w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.w = self.w * (self.cfg.target_norm / norm)


class MushroomBody:
    """Composition of the whole circuit plus its eligibility trace."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(cfg.seed + 101)
        self.al = AntennalLobe(cfg)
        self.kc = KenyonCells(cfg)
        self.mbon = MushroomBodyOutput(cfg)
        self.eligibility = torch.zeros(cfg.n_mbon, cfg.n_kc)
        self.lr_scale = 1.0
        self.pn_rate = torch.zeros(cfg.n_pn)
        self.kc_counts = torch.zeros(cfg.n_kc)
        # the centred KC code of the most recent stimulus, used as the Hebbian
        # target when dopamine names a specific MBON row
        self.kc_pattern = torch.zeros(cfg.n_kc)
        # spike trains from the most recent presentation, for visualisation
        self.kc_raster = torch.zeros(0, cfg.n_kc)
        self.mbon_raster = torch.zeros(0, cfg.n_mbon)
        self.mbon_counts = torch.zeros(cfg.n_mbon)

    # -- lifecycle --------------------------------------------------------------
    def reset(self) -> None:
        self.al.reset() if hasattr(self.al, "reset") else None
        self.kc.reset()
        self.mbon.reset()
        self.eligibility = torch.zeros(self.cfg.n_mbon, self.cfg.n_kc)
        self.kc_counts = torch.zeros(self.cfg.n_kc)

    # -- forward ----------------------------------------------------------------
    def present(self, spikes: torch.Tensor) -> torch.Tensor:
        """Run one stimulus and return the per-MBON spike counts.

        ``spikes`` is ``(T, n_receptor)``. The PN stage is run first to produce a
        graded rate vector, the calyx turns that into a sparse code, and only
        then does the spiking MBON stage run - which is the part that carries the
        eligibility trace for plasticity.
        """
        self.reset()
        n_steps = max(1, spikes.shape[0])
        pn_rate = self.al.rate_vector(spikes)
        self.pn_rate = pn_rate
        kc_act = self.kc.encode(pn_rate)

        # All timesteps at once: one batched Bernoulli draw for the Kenyon cells
        # and one matmul for the MBON drives, then only the 27-element membrane
        # recurrence is stepped. See spike_raster / run_raster for why.
        kc_raster = self.kc.spike_raster(n_steps, self.gen)
        mb_raster = self.mbon.run_raster(kc_raster)
        counts = mb_raster.sum(0)

        self.kc_raster = kc_raster
        self.mbon_raster = mb_raster
        self.mbon_counts = counts
        self.kc_counts = kc_raster.sum(0)

        # Three-factor eligibility, evaluated at the timescale the dopamine
        # signal actually operates on: ONE value per stimulus, from the graded
        # KC code (constant across the window) and the trial-level MBON rate.
        #
        # Using the per-step Poisson samples here instead is subtly fatal. The
        # KCs are resampled every timestep, so which MBON crosses threshold in a
        # given step is driven by that step's noise, not by the stimulus. An
        # eligibility built from `outer(step_kc_spikes, step_mbon_spikes)`
        # therefore reinforces whichever KCs happened to coincide with a noisy
        # spike, and it reinforces a *different* random subset on every step -
        # the increments cancel instead of accumulating, so reward carries
        # almost no directional information and the readout stays at chance.
        post = counts / n_steps
        self.kc_pattern = kc_act - kc_act.mean()
        self.eligibility = (self.cfg.e_decay * self.eligibility
                            + self.mbon.eligibility_outer(kc_act, post))
        return counts

    def stage_outputs(self, spikes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every intermediate representation for one stimulus. Diagnostics only."""
        counts = self.present(spikes)
        return {
            "bitmap": spikes.mean(0),
            "pn": self.pn_rate,
            "pn_centered": self.pn_rate - self.pn_rate.mean(),
            "kc_drive": self.kc.drive.clone(),
            "kc_act": self.kc.activation.clone(),
            "kc_spikes": self.kc_counts.clone(),
            "mbon": counts,
        }

    def apply_dopamine(
        self,
        trace: torch.Tensor,
        credit: int | None = None,
        punish: int | None = None,
    ) -> None:
        """Three-factor rule: dw = lr * D(t) * e(t), accumulated over the burst.

        ``credit`` / ``punish`` name the MBON rows the eligibility is attributed
        to, and this is the part that makes the rule work at all.

        The textbook form credits whichever MBON *fired* - ``outer(post, pre)``.
        That is fine when the readout is already correct, but on a mis-trial the
        unit that fired is exactly the one that should not be reinforced, so the
        update strengthens the error. Training then sits at chance forever no
        matter what the learning rate is, because half the updates are pushing
        the wrong way and the two average to zero.

        In the fly, sugar is delivered to the MBON compartment that *predicts*
        the sugar, not to whichever output neuron happened to be active. So the
        target row is what dopamine gates - and on a wrong answer the answering
        row is suppressed instead. With that attribution the same task goes from
        50% to 100% in about twenty trials.
        """
        if not torch.any(trace):
            return
        assert self.eligibility.shape == self.mbon.w.shape, (
            f"eligibility {tuple(self.eligibility.shape)} must match "
            f"weights {tuple(self.mbon.w.shape)}"
        )

        row = credit if credit is not None else punish

        # The decay-weighted burst collapses to a single scalar, and is then
        # reduced to a pure sign. The magnitude of the alpha-function kernel is
        # a property of the dopamine cell, not of the synapse, so letting it
        # scale the weight step makes the effective learning rate depend on the
        # burst shape. With the raw sum (~5.6) and an unnormalised pattern, a
        # single trial moved the credited row by ~0.36 against a weight scale of
        # ~0.053 - a 7x overshoot that erases progress as fast as it makes it.
        decay = self.cfg.e_decay ** torch.arange(trace.shape[0], dtype=trace.dtype)
        gain = float((trace * decay).sum())
        direction = 1.0 if gain > 0 else -1.0

        if row is None:
            # No named row: the outer-product eligibility carries its own sign,
            # so the burst magnitude is what is left to carry.
            delta = gain * self.eligibility
        else:
            delta = torch.zeros_like(self.mbon.w)
            if self.cfg.credit_mode == "pattern":
                # Impress the stimulus pattern onto the credited row, scaled to
                # unit norm so `lr` is a step length in weight units and can be
                # set as a fraction of the initial weight scale.
                #
                # Using the outer-product eligibility here instead re-signs the
                # update by whether that MBON happened to fire: on a letter the
                # fly is currently getting wrong the target row's own
                # eligibility is *negative*, so sugar would push the row away
                # from the very pattern it is meant to learn.
                pattern = self.kc_pattern
                norm = float(pattern.norm())
                if norm > 1e-8:
                    delta[row] = direction * pattern / norm
            else:
                delta[row] = gain * self.eligibility[row]
        self.mbon.apply_update(self.cfg.lr * self.lr_scale * delta)

    # -- introspection ----------------------------------------------------------
    def kc_sparsity(self) -> float:
        """Fraction of Kenyon cells participating in the current code."""
        return float((self.kc.activation > 0).to(torch.float32).mean())

    def state_dict(self) -> dict:
        return {
            "w_kc_mbon": self.mbon.w.detach().clone(),
            "config": dict(self.cfg.__dict__),
        }

    def load_state_dict(self, sd: dict) -> None:
        w = sd["w_kc_mbon"]
        want = tuple(self.mbon.w.shape)
        got = tuple(w.shape)
        if got != want:
            raise ValueError(
                f"checkpoint synapse matrix is {got} but this circuit needs {want}; "
                f"the checkpoint was trained with a different architecture "
                f"(n_kc / n_mbon differ) - retrain with matching config"
            )
        self.mbon.w = w.clone()
