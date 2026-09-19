"""The food-reward pathway, modelled on the real circuit.

Anatomy this module stands in for
---------------------------------
    sugar GRNs (Gr5a / Gr64f, tarsi + labellum)
        -> suboesophageal zone (SEZ)
        -> PAM / DAN dopaminergic neurons innervating the mushroom-body lobes
        -> dopaminergic modulation of the KC -> MBON synapse

The important functional properties, all of which are observed in vivo:

* **Phasic.** DACs/PAMs emit a short, high-amplitude burst followed by a much
  smaller tonic tail. That burst is what gates plasticity.
* **Delayed.** Sugar on a tarsus reaches the SEZ and then the MB, so there is a
  real conduction delay between the behaviour and the neuromodulator arriving.
* **Value-based.** Recent work shows DANs encode a reward-prediction error, not
  raw reward. A well-predicted sugar reward produces a smaller burst. That is
  implemented here as an adaptive baseline, and it is why a trained animal's
  dopamine response to a familiar reward flattens out.
* **Signed.** Bitter ligands (Gr66a) drive the opposite polarity. Punishment is
  a negative-going modulation, not merely an absence of reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Burst:
    """One recorded dopaminergic event, kept for reporting / plotting."""

    reward: float
    rpe: float
    trace: torch.Tensor           # per-timestep dopamine concentration

    @property
    def peak(self) -> float:
        return float(self.trace.abs().max())

    def ascii_trace(self, width: int = 48) -> str:
        """A tiny sparkline, because seeing the burst is the whole point."""
        if not len(self.trace):
            return ""
        vals = self.trace.tolist()
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        span = (hi - lo) or 1.0
        bars = " .:-=+*#%@"

        rows = 5
        lines = []
        for r in range(rows - 1, -1, -1):
            thresh = lo + span * (r / (rows - 1))
            line = "".join(
                bars[min(len(bars) - 1, int((v - lo) / span * (len(bars) - 1)))]
                if (v - lo) / span >= r / (rows - 1) else " "
                for v in vals
            )
            lines.append(f"  {thresh:+.2f} |{line}")
        lines.append("         +" + "-" * len(vals))
        sign = "SUCROSE burst" if self.reward > 0 else "BITTER response"
        return "\n".join(lines) + f"\n         {sign}  (peak {hi:+.2f})"


class DopamineSystem:
    """PAM/DAN population model: sugar-driver -> SEZ -> dopamine -> mushroom body."""

    def __init__(
        self,
        n_steps: int,
        tau: float,
        delay: int,
        baseline_lr: float,
        sucrose: float = 1.0,
        bitter: float = -1.0,
    ) -> None:
        self.n_steps = n_steps
        self.tau = tau
        self.delay = delay
        self.baseline_lr = baseline_lr
        self.sucrose = sucrose
        self.bitter = bitter
        self.baseline = 0.0
        self.last: Burst | None = None

    # -- reward-prediction error ------------------------------------------------
    def rpe(self, reward: float) -> float:
        err = reward - self.baseline
        self.baseline += self.baseline_lr * err
        return err

    # -- phasic waveform --------------------------------------------------------
    def wave(self, amplitude: float) -> torch.Tensor:
        """Alpha-function burst with a conduction delay.

        ``f(t) = (t'/tau) * exp(1 - t'/tau)`` peaks at exactly 1.0 when t' = tau,
        so a modest amplitude never has to be rescaled.
        """
        t = torch.arange(self.n_steps, dtype=torch.float32)
        t = torch.clamp(t - self.delay, min=0.0)
        kernel = (t / self.tau) * torch.exp(1.0 - t / self.tau)
        return amplitude * kernel

    # -- public API -------------------------------------------------------------
    def deliver(self, reward: float) -> Burst:
        """Deliver a reward/punishment and return the resulting dopamine burst."""
        if reward > 0:
            amplitude = self.rpe(reward * self.sucrose)
            label = reward * self.sucrose
        elif reward < 0:
            amplitude = self.rpe(abs(reward) * self.bitter)
            label = reward * abs(self.bitter)
        else:
            amplitude = 0.0
            label = 0.0

        burst = Burst(reward=label, rpe=amplitude, trace=self.wave(amplitude))
        self.last = burst
        return burst

    def sucrose_reward(self, magnitude: float = 1.0) -> Burst:
        """Sugar on the tarsi: the 'happy with food' event."""
        return self.deliver(+magnitude)

    def bitter_punishment(self, magnitude: float = 1.0) -> Burst:
        return self.deliver(-magnitude)

    # -- introspection ----------------------------------------------------------
    def state_dict(self) -> dict:
        return {"baseline": self.baseline}

    def load_state_dict(self, sd: dict) -> None:
        self.baseline = float(sd.get("baseline", 0.0))
