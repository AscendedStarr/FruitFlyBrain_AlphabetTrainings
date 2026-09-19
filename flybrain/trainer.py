"""Training loop, readout, and the "say the phrase" routine."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field

import torch

from .circuit import MushroomBody
from .config import Config
from .dopamine import Burst, DopamineSystem
from .encoding import CLASSES, encode_from_config


@dataclass
class TrialResult:
    target: str
    predicted: str
    correct: bool
    counts: torch.Tensor
    burst: Burst | None = None


@dataclass
class EpochStats:
    epoch: int
    accuracy: float
    reward_mean: float
    dopa_baseline: float
    kc_active: float


class FlyBrain:
    """A mushroom body trained by dopamine-modulated Hebbian plasticity."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.gen = torch.Generator().manual_seed(self.cfg.seed)
        self.mb = MushroomBody(self.cfg)
        self.dopamine = DopamineSystem(
            n_steps=self.cfg.t_reward,
            tau=self.cfg.dopa_tau,
            delay=self.cfg.dopa_delay,
            baseline_lr=self.cfg.dopa_baseline_lr,
            sucrose=self.cfg.sucrose,
            bitter=self.cfg.bitter,
        )
        self.history: list[EpochStats] = []
        self.epoch = 0
        self.best_accuracy = 0.0
        # Trails of the run, so a resumed session can report the whole curve
        # rather than only the epochs it saw itself.
        self.accuracy_curve: list[float] = []

    # ------------------------------------------------------------------ helpers
    def _encode(self, ch: str, augment: bool) -> torch.Tensor:
        return encode_from_config(self.cfg, ch, augment=augment, gen=self.gen)

    def _readout(self, counts: torch.Tensor) -> int:
        return int(torch.argmax(counts).item())

    # -------------------------------------------------------------------- trial
    def trial(
        self,
        ch: str,
        learn: bool,
        augment: bool | None = None,
        teacher: bool = True,
    ) -> TrialResult:
        """Present one character, read out a prediction, deliver dopamine."""
        if augment is None:
            augment = learn

        target_idx = CLASSES.index(ch.upper())
        counts = self.mb.present(self._encode(ch, augment))
        pred_idx = self._readout(counts)

        # Optionally average several presentations before deciding the reward.
        # Each `present` leaves a fresh eligibility trace in the circuit, so the
        # last look supplies the trace that `apply_dopamine` modulates below -
        # it is a valid sample of the same stimulus, just not the one that
        # decided the outcome.
        looks = max(1, int(getattr(self.cfg, "decision_repeats", 1)))
        if looks > 1:
            total = counts.clone()
            for _ in range(looks - 1):
                total += self.mb.present(self._encode(ch, augment))
            pred_idx = self._readout(total)
            counts = total

        correct = pred_idx == target_idx

        burst = None
        if learn:
            # The reward depends on whether the fly produced the right answer.
            # With teacher=False we instead reward only on genuine correctness
            # with no supervision at all, which is the harder RL setting.
            if self.cfg.reward_mode == "pavlovian":
                # Training always presents the true letter, so the letter is a
                # perfect predictor of sugar and the sugar is unconditional.
                # The dopamine baseline tracks the prediction and subtracts it,
                # leaving the prediction error as the teaching signal: strong
                # while the letter does not yet predict sugar, fading to nothing
                # once it does. That makes every trial an increment of the same
                # direction and the row a running mean of its class, instead of
                # a step whose direction is decided by a single noisy argmax.
                reward = self.cfg.sucrose
            else:
                reward = self.cfg.sucrose if correct else self.cfg.bitter
                if not teacher:
                    reward = reward if correct else 0.0
                if self.cfg.punish_wrong is False and not correct:
                    reward = 0.0
            burst = self.dopamine.deliver(reward)
            # Sugar stamps in the letter that was being read - the target row.
            # Bitter stamps out the row that actually answered. Passing the
            # index is the whole trick: crediting whoever fired instead leaves
            # the network reinforcing its own mistakes and stuck at chance.
            if correct or self.cfg.reward_mode == "pavlovian":
                self.mb.apply_dopamine(burst.trace, credit=target_idx)
            else:
                self.mb.apply_dopamine(burst.trace, punish=pred_idx)

        return TrialResult(
            target=ch.upper(),
            predicted=CLASSES[pred_idx],
            correct=correct,
            counts=counts,
            burst=burst,
        )

    # -------------------------------------------------------------------- train
    # -------------------------------------------------------------------- train
    def _lr_scale(self, epochs_done: int) -> float:
        """Step-size multiplier for the schedule, in units of ``cfg.lr``."""
        mode = self.cfg.lr_schedule
        if mode == "const":
            return 1.0
        ratio = 1.0 + epochs_done / max(1e-6, self.cfg.lr_half_life)
        if mode == "inv":
            return 1.0 / ratio
        if mode == "sqrt":
            return 1.0 / ratio**0.5
        raise ValueError(f"unknown lr_schedule {mode!r}")

    def train(self, verbose: bool = True, target_epochs: int | None = None) -> list[EpochStats]:
        """Train until ``cfg.epochs`` total epochs have been run.

        Epoch numbering continues from ``self.epoch``, so calling ``train`` on
        a brain restored from a checkpoint appends to the existing history
        instead of starting the schedule over.
        """
        cfg = self.cfg
        schedule = [c for c in CLASSES for _ in range(cfg.trials_per_class)]
        stop = cfg.epochs if target_epochs is None else target_epochs
        start = self.epoch + 1

        for epoch in range(start, stop + 1):
            self.mb.lr_scale = self._lr_scale(epoch - 1)
            order = torch.randperm(len(schedule), generator=self.gen).tolist()
            rewards: list[float] = []
            for i in order:
                res = self.trial(schedule[i], learn=True)
                rewards.append(1.0 if res.correct else 0.0)

            stats = self.evaluate(verbose=False)
            self.epoch = epoch
            self.accuracy_curve.append(stats)
            self.best_accuracy = max(self.best_accuracy, stats)
            rec = EpochStats(
                epoch=epoch,
                accuracy=stats,
                reward_mean=sum(rewards) / max(1, len(rewards)),
                dopa_baseline=self.dopamine.baseline,
                kc_active=float(self.mb.kc.active.sum()),
            )
            self.history.append(rec)

            if verbose and (epoch % cfg.log_every == 0 or epoch == start):
                print(
                    f"  epoch {epoch:>3}/{stop}  "
                    f"train-reward {rec.reward_mean:.3f}  "
                    f"eval-acc {rec.accuracy:6.1%}  "
                    f"dopa-baseline {rec.dopa_baseline:+.3f}  "
                    f"KC active {rec.kc_active:.0f}",
                    flush=True,
                )
        return self.history

    # ----------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, verbose: bool = False) -> float:
        hits = 0
        total = 0
        confusion: dict[str, list[str]] = {}
        for ch in CLASSES:
            for _ in range(self.cfg.eval_repeats):
                res = self.trial(ch, learn=False, augment=False)
                hits += int(res.correct)
                total += 1
                if not res.correct:
                    confusion.setdefault(ch, []).append(res.predicted)
        acc = hits / max(1, total)
        if verbose:
            print(f"  held-out accuracy: {acc:.1%}  ({hits}/{total})")
            for tgt, preds in sorted(confusion.items()):
                print(f"    {tgt} -> {', '.join(sorted(set(preds)))}")
        return acc

    # ---------------------------------------------------------------------- say
    def say(self, phrase: str, learn: bool = False, verbose: bool = True) -> str:
        """Drive the circuit letter by letter and read the phrase back out.

        If every character is read correctly the fly gets a *large* sucrose
        reward, which is the same dopaminergic event a real fly receives on
        finding food. Any character it fumbles earns a bitter response instead.
        """
        text = phrase.upper()
        emitted: list[str] = []
        misses: list[str] = []

        for ch in text:
            if ch not in CLASSES:
                emitted.append(ch)
                continue
            res = self.trial(ch, learn=learn, augment=False)
            emitted.append(res.predicted)
            if not res.correct:
                misses.append(f"{ch}->{res.predicted}")
            if verbose:
                mark = "ok " if res.correct else "XX "
                print(f"    {mark}{ch}  read as  {res.predicted}")

        out = "".join(emitted)
        said_it = not misses and out.replace(" ", "") == text.replace(" ", "")

        if learn:
            # Whole-phrase contingency: a big sugar event only when the fly
            # actually produces the target utterance.
            if said_it:
                burst = self.dopamine.sucrose_reward(magnitude=3.0)
            else:
                burst = self.dopamine.bitter_punishment(magnitude=0.5)
        else:
            burst = None

        if verbose:
            print()
            print(f"    fly says: {out!r}")
            if misses:
                print(f"    misread:  {', '.join(misses)}")
            if burst is not None:
                print()
                if said_it:
                    print("    >>> correct utterance delivered: sugar on the tarsi")
                    print("    >>> sugar GRN -> SEZ -> PAM/DAN phasic burst")
                else:
                    print("    >>> incorrect utterance: bitter ligand (Gr66a)")
                print(burst.ascii_trace())
        return out

    # ------------------------------------------------------------------- persist
    def save(self, path: str, note: str = "") -> None:
        """Write everything needed to resume, not just the weights.

        The config goes in because the architecture (KC count, gains, thresholds)
        is derived from it at construction - restoring weights into a brain built
        from a different config would silently produce a different circuit. The
        RNG states go in so a resumed run continues the same noise stream rather
        than repeating it, and the epoch and history go in so the learning curve
        survives across sessions and the schedule is not restarted.

        Downgrade guard
        ---------------
        If the file already on disk was trained for *longer* than this brain,
        the old file is copied to ``<path>.bak`` before being overwritten. This
        exists because it actually happened: a second server process, started by
        mistake, loaded a stub checkpoint, ran a 60-epoch bootstrap, saved over
        the real 2000-epoch brain, and every subsequent measurement was silently
        taken against a different circuit. Losing a long run to a short one
        should cost a `.bak` file, not a day.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.isfile(path):
            try:
                existing = torch.load(path, map_location="cpu", weights_only=False)
                old_epoch = int(existing.get("epoch", 0))
            except Exception:  # noqa: BLE001 - an unreadable file is still a file
                old_epoch = 0
            if old_epoch > self.epoch:
                shutil.copy2(path, path + ".bak")
                print(f"  ! saving epoch {self.epoch} over epoch {old_epoch}; "
                      f"the longer run was kept at {os.path.basename(path)}.bak",
                      flush=True)
        torch.save(
            {
                "format": 2,
                "circuit": self.mb.state_dict(),
                "dopamine": self.dopamine.state_dict(),
                "config": self.cfg.__dict__,
                "epoch": self.epoch,
                "best_accuracy": self.best_accuracy,
                "accuracy_curve": list(self.accuracy_curve),
                "history": [asdict(h) for h in self.history],
                "gen_state": self.gen.get_state(),
                "mb_gen_state": self.mb.gen.get_state(),
                "note": note,
                # Provenance: which interpreter produced this file. Two servers
                # on different interpreters sharing one port is exactly how the
                # loss described above happened, and this makes it detectable
                # after the fact instead of only in a terminal that has scrolled.
                "written_by": sys.executable,
                "written_from": os.path.abspath(os.getcwd()),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FlyBrain":
        """Rebuild a brain from a checkpoint, config included."""
        blob = torch.load(path, weights_only=False)
        saved = blob.get("config")
        if saved:
            known = {f for f in Config().__dict__}
            cfg = Config(**{k: v for k, v in saved.items() if k in known})
        else:
            cfg = Config()

        brain = cls(cfg)
        brain.mb.load_state_dict(blob["circuit"])
        brain.dopamine.load_state_dict(blob["dopamine"])
        brain.epoch = int(blob.get("epoch", 0))
        brain.best_accuracy = float(blob.get("best_accuracy", 0.0))
        brain.accuracy_curve = list(blob.get("accuracy_curve", []))
        brain.history = [EpochStats(**h) for h in blob.get("history", [])]
        if blob.get("gen_state") is not None:
            brain.gen.set_state(blob["gen_state"])
        if blob.get("mb_gen_state") is not None:
            brain.mb.gen.set_state(blob["mb_gen_state"])
        return brain
