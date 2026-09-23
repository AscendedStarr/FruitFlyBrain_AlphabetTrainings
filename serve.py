"""Local interactive web UI for FlyBrain.

Run it:

    .\\.venv-flybrain\\Scripts\\python.exe serve.py
    .\\.venv-flybrain\\Scripts\\python.exe serve.py --port 8000 --no-browser

Then open http://127.0.0.1:8000

Stdlib HTTP server plus a handful of JSON endpoints. The browser renders the
circuit live: you present a letter, and the receptor sheet, the antennal lobe,
the sparse Kenyon-cell code and the MBON readout all update, with the
dopamine trace drawn underneath when a reward is delivered.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Single-threaded BLAS keeps the per-request latency predictable and avoids the
# thread-pool thrash torch causes on small tensors.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402

from flybrain import (  # noqa: E402
    CLASSES,
    FONT_5X7,
    FONT_DIGITS,
    INPUT_GLYPHS,
    Config,
    FlyBrain,
    encode_from_config,
    is_class,
    is_input,
)
from flybrain.docread import from_upload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
MIME = {".html": "text/html", ".js": "text/javascript",
        ".css": "text/css", ".png": "image/png", ".ico": "image/x-icon"}

# How many looks the fly averages before it commits to an answer.
#
# The Kenyon-cell code is Poisson-sampled, so one look is one noisy draw from
# the same stimulus, and the MBON counts it produces wobble enough to flip an
# argmax. A real fly does not answer off a single glance either - it fixates.
#
# Four is the number that matters. `CLASSES` has 27 entries: A-Z *plus* the
# blank glyph, and the blank lights 0 of the 35 receptors - an empty sheet, so
# nothing but noise and the prior can separate it from any other near-empty
# input. Measured against the trained checkpoint (see scripts/classes.py):
#
#     1 look   26/27 classes perfect, blank 5/10  (a phrase's spaces come out as letters)
#     2 looks  26/27 classes perfect, blank 9/10
#     4 looks  27/27 classes perfect  <- phrase reads exactly
#
# The cost is 3 extra forward passes on a 13,824-synapse readout, which is the
# cheapest accuracy in the whole project.
LOOKS = 4


class Session:
    """Holds the brain and serialises access to it (the HTTP server is threaded)."""

    def __init__(self, cfg: Config, ckpt: str | None = None) -> None:
        self.lock = threading.Lock()
        self.ckpt = ckpt or os.path.join(HERE, "runs", "flybrain.pt")
        self.log: list[str] = []
        self.brain = self._restore(cfg)
        # The checkpoint's own config wins: its weights only mean anything
        # against the architecture they were trained in.
        self.cfg = self.brain.cfg

    def _restore(self, cfg: Config) -> FlyBrain:
        """Load the checkpoint if there is one, otherwise start from scratch.

        Without this the server built a naive brain on every start, so any
        training done through the browser was thrown away the moment the page
        reloaded or the process restarted - the UI would sit at chance (3.7%)
        no matter how long it was left running.
        """
        if not os.path.isfile(self.ckpt):
            self.log.append(f"no checkpoint at {os.path.basename(self.ckpt)} "
                            "- starting from a naive circuit (3.7% = chance)")
            return FlyBrain(cfg)
        try:
            brain = FlyBrain.load(self.ckpt)
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not
            # stop the server, but it must be loud rather than silent
            self.log.append(f"could not load {os.path.basename(self.ckpt)} "
                            f"({type(exc).__name__}: {exc}) - starting naive")
            return FlyBrain(cfg)
        # Read out the way the UI will, i.e. with the averaged looks, so the
        # number printed here is the number the user actually sees.
        brain.cfg.decision_repeats = LOOKS
        acc = brain.evaluate()
        brain.best_accuracy = max(brain.best_accuracy, acc)
        self.log.append(f"loaded {os.path.basename(self.ckpt)} "
                        f"(epoch {brain.epoch}, {acc:.1%} on holdout, "
                        f"{LOOKS} looks/answer)")
        print(f"  loaded {os.path.basename(self.ckpt)}: epoch {brain.epoch}, "
              f"{acc:.1%} accuracy")
        return brain

    def save(self, note: str = "") -> str:
        """Write the current brain to the session checkpoint."""
        with self.lock:
            self.brain.save(self.ckpt, note=note)
            return os.path.basename(self.ckpt)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _round(values, nd: int = 4) -> list:
        return [round(float(v), nd) for v in values]

    def _flow(self, counts, credited=None, punished=None) -> dict:
        """Who won the argument, and by how much.

        ``counts`` is the spike tally over the whole decision window, so this is
        the fly's actual evidence: the winning MBON, the runner-up it beat, and
        how far apart they are. ``margin`` near 0 means the fly was guessing.
        """
        vals = [float(v) for v in counts]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        win = order[0]
        run = order[1] if len(order) > 1 else win
        return {
            "winner": win,
            "winner_label": CLASSES[win],
            "winner_count": round(vals[win], 2),
            "runner": run,
            "runner_label": CLASSES[run],
            "runner_count": round(vals[run], 2),
            # how decisive: winner minus runner-up, normalised by the winner
            "margin": round((vals[win] - vals[run]) / (vals[win] or 1.0), 4),
            "total": round(sum(vals), 2),
            "top": [{"i": i, "label": CLASSES[i], "count": round(vals[i], 2)}
                    for i in order[:4]],
            "credited": credited,      # row that received the sugar stamp
            "punished": punished,      # row that received the bitter stamp
        }

    def _reward(self, burst, credited, punished, written: bool = False) -> dict:
        """The neuromodulator event, with its sign kept.

        ``regions.dan`` throws the sign away (it is an absolute value), so the
        UI could not tell sugar from bitter - both drove the same magenta fan.
        Here the polarity survives, and `credited` / `punished` say which MBON
        row the burst was actually aimed at.

        ``delivered`` means the taste happened, which it always does.
        ``written`` means plasticity was enabled and a weight row was actually
        stamped. The two are separated on purpose: a real fly's PAM/DAN neurons
        report the taste whether or not a memory is being laid down, and
        conflating the two made it impossible to show the bitter response while
        plasticity was switched off.
        """
        dopa = self.brain.dopamine
        cfg = self.cfg
        if burst is None:
            return {
                "delivered": False, "written": False, "kind": None, "sign": 0,
                "peak": 0.0, "rpe": 0.0, "baseline": round(float(dopa.baseline), 4),
                "credited": None, "punished": None,
                "tau": float(cfg.dopa_tau), "delay": int(cfg.dopa_delay),
                "sucrose": float(cfg.sucrose), "bitter": float(cfg.bitter),
            }
        last = dopa.last
        sign = 1 if (last is not None and last.reward > 0) else (
            -1 if (last is not None and last.reward < 0) else 0)
        return {
            "delivered": True,
            "written": bool(written),
            "kind": "sucrose" if sign > 0 else ("bitter" if sign < 0 else None),
            "sign": sign,
            "peak": round(float(max(abs(v) for v in burst)), 4),
            "rpe": round(float(last.rpe), 4) if last is not None else 0.0,
            "baseline": round(float(dopa.baseline), 4),
            "credited": credited,
            "credited_label": CLASSES[credited] if credited is not None else None,
            "punished": punished,
            "punished_label": CLASSES[punished] if punished is not None else None,
            "tau": float(cfg.dopa_tau),
            "delay": int(cfg.dopa_delay),
            "sucrose": float(cfg.sucrose),
            "bitter": float(cfg.bitter),
        }

    def _stage_payload(self, counts, burst=None,
                       credited=None, punished=None,
                       written: bool = False) -> dict:
        mb = self.brain.mb
        return {
            "mbon": self._round(counts, 2),
            "pn": self._round(mb.pn_rate),
            "pn_centered": self._round(mb.pn_rate - mb.pn_rate.mean()),
            "kc_act": self._round(mb.kc.activation),
            "kc_drive": self._round(mb.kc.drive),
            "kc_spikes": self._round(mb.kc_counts, 1),
            "kc_sparsity": round(mb.kc_sparsity(), 4),
            "burst": self._round(burst, 4) if burst is not None else None,
            "dopa_baseline": round(float(self.brain.dopamine.baseline), 4),
            "t_window": self._window(),
            "regions": self._regions(counts),
            "flow": self._flow(counts, credited=credited, punished=punished),
            "reward": self._reward(burst, credited, punished, written=written),
        }

    def _window(self) -> int:
        """Number of spiking timesteps in the most recent presentation."""
        n = int(self.brain.mb.kc_raster.shape[0])
        return n if n > 0 else int(self.cfg.t_stim)

    def _regions(self, counts) -> dict:
        """Normalised drive for each brain region the model actually simulates.

        Only four regions have real spikes behind them: the antennal lobe
        (projection neurons), the mushroom body (Kenyon-cell somata and their
        axons), its output cells (MBONs), and the dopaminergic PAM/DAN reward
        signal. Everything else drawn in the brain view - the optic lobes, the
        lateral horn, the central complex, the subesophageal zone - is
        anatomical context only and is labelled as such in the UI, because this
        model does not implement them.
        """
        mb = self.brain.mb
        cfg = self.cfg
        t = self._window()
        mbon = torch.as_tensor(counts, dtype=torch.float32)
        pn = mb.pn_rate

        def c01(v: float) -> float:
            return round(max(0.0, min(1.0, v)), 4)

        dan = abs(float(self.brain.dopamine.baseline))
        return {
            "al": c01(float(pn.mean()) / 0.75),
            "kc": c01(float(mb.kc_counts.sum()) / (cfg.n_kc * t) / 0.10),
            "mbon": c01(float(mbon.sum()) / (cfg.n_mbon * t) / 0.12),
            # tonic dopamine baseline is real but should read as a hum, so a
            # delivered reward's burst still dominates the display
            "dan": c01(dan * 0.5),
        }

    # -- endpoints -------------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            return {
                "config": dict(self.cfg.__dict__),
                # `classes` is the fly's vocabulary: 27 output cells, A-Z plus
                # space. It can only ever say one of these, and there is no
                # digit among them.
                "classes": CLASSES,
                "alphabet": CLASSES,
                "n_classes": len(CLASSES),
                "digits": sorted(FONT_DIGITS),
                # Everything that can be drawn on the receptor sheet, which is a
                # strictly larger set than `classes`. Digits live here and only
                # here: presentable, but not answerable.
                "glyphs": {k: v for k, v in INPUT_GLYPHS.items()},
                "input_glyphs": sorted(INPUT_GLYPHS),
                "history": [dict(epoch=h.epoch, accuracy=h.accuracy,
                                 reward_mean=h.reward_mean,
                                 dopa_baseline=h.dopa_baseline,
                                 kc_active=h.kc_active)
                            for h in self.brain.history],
                "accuracy": round(self.brain.evaluate(), 4),
                "plastic_synapses": self.cfg.n_kc * self.cfg.n_mbon,
                "log": self.log[-12:],
            }

    def present(self, char: str, learn: bool) -> dict:
        char = char.upper()
        if not is_input(char):
            raise ValueError(f"no glyph for {char!r}")
        # `known` = the fly has an MBON for this character, so it is *able* to
        # answer correctly. Digits are presentable but not known: there is no
        # output cell to name them, which is why a digit is always wrong no
        # matter how well the fly is trained or how long it is exposed.
        known = is_class(char)
        with self.lock:
            # Average `LOOKS` presentations before answering. The last one still
            # leaves the eligibility trace the reward below modulates, so the
            # learning path is unchanged - only the readout is steadier.
            #
            # `gen=self.brain.gen` is essential, not cosmetic. With no generator
            # `encode_from_config` builds a FRESH one seeded `cfg.seed + 5`, so
            # every call returned the *identical* spike train: the loop below
            # summed the same sample four times, which reduces no variance at
            # all. The blank glyph, whose code is nothing but the baseline
            # Poisson noise, stayed at its single-look 59% in the browser while
            # `trial()` - which does pass `self.gen` - measured 99.5%. Passing
            # the brain's own generator makes each look an independent draw.
            looks = max(1, int(getattr(self.cfg, "decision_repeats", 1)))
            spikes = encode_from_config(self.cfg, char, augment=False,
                                        gen=self.brain.gen)
            counts = self.brain.mb.present(spikes)
            for _ in range(looks - 1):
                counts = counts + self.brain.mb.present(
                    encode_from_config(self.cfg, char, augment=False,
                                       gen=self.brain.gen))
            target = CLASSES.index(char) if known else None
            pred = int(counts.argmax())
            correct = bool(known and pred == target)

            # Taste is delivered on every presentation, whether or not
            # plasticity is on. The dopamine burst is the teaching signal: a
            # real fly's PAM/DAN neurons report sugar or bitter regardless of
            # whether a memory is being laid down, and only the downstream
            # weight write is gated on learning. Gating the *burst* on `learn`
            # (as this used to) made the bitter response invisible in inference
            # mode, which is exactly the regime the digit demonstration runs in.
            reward = self.cfg.sucrose if correct else self.cfg.bitter
            wave = self.brain.dopamine.deliver(reward)
            burst = wave.trace.tolist()
            pulse = 1.0 if correct else -1.0

            credited = None
            punished = None
            if learn:
                # Credit a NAMED row. Calling `apply_dopamine(trace)` without
                # one falls back to `delta = gain * eligibility`, whose gain is
                # the raw burst sum (~5.6) against a weight scale of ~0.053 - a
                # ~7x overshoot that ran a trained brain down to chance (7%) on
                # a single click. The named-row path is normalised instead.
                if correct:
                    self.brain.mb.apply_dopamine(wave.trace, credit=target)
                    credited = target
                else:
                    self.brain.mb.apply_dopamine(wave.trace, punish=pred)
                    punished = pred

            payload = self._stage_payload(counts, burst,
                                          credited=credited, punished=punished,
                                          written=bool(learn))
            payload.update({
                "char": char,
                "predicted": CLASSES[pred],
                "correct": correct,
                # `known` is false for a digit. The UI uses it to say "this
                # glyph is outside the fly's vocabulary" instead of pretending
                # the answer means something it does not.
                "known": known,
                "target_index": target,
                "learned": learn,
                "pulse": pulse,
                "receptor_rate": self._round(spikes.mean(0)),
                "receptor_raster": spikes.to(torch.int8).flatten().tolist(),
                "t_stim": self.cfg.t_stim,
                "history": [h.accuracy for h in self.brain.history],
            })
            return payload

    def say(self, phrase: str, learn: bool) -> dict:
        """Read a phrase out loud, one character at a time.

        Characters with no glyph at all (punctuation, anything the 5x7 font does
        not draw) are reported as ``skipped`` with ``correct: None``. They used
        to be echoed back with ``correct: True``, which made any punctuation and
        every digit score as a hit - the phrase "123" came back 3/3 correct from
        a network that has no digit anywhere in its output layer. That was a
        fabricated success and it is gone.
        """
        text = phrase.upper()
        out = []
        last = None
        for ch in text:
            if not is_input(ch):
                out.append({
                    "char": ch,
                    "predicted": None,
                    "correct": None,
                    "known": False,
                    "skipped": True,
                    "reason": "no glyph in the 5x7 font",
                })
                continue
            res = self.present(ch, learn)
            out.append({"char": ch, "predicted": res["predicted"],
                        "correct": res["correct"], "mbon": res["mbon"],
                        "known": res["known"], "skipped": False})
            last = res

        # Only presented characters contribute to what the fly "said". A skipped
        # character is not a guess and must not be reported as one.
        said = "".join(r["predicted"] for r in out if r["predicted"] is not None)
        graded = [r for r in out if not r["skipped"]]
        skipped = [r for r in out if r["skipped"]]
        hits = sum(1 for r in graded if r["correct"])
        # A phrase only counts as read if every character was both presentable
        # and known to the fly, and all of them came out right.
        ok = bool(graded) and not skipped and hits == len(graded)

        with self.lock:
            # Every letter already earned its own properly-credited update
            # inside `present`, so the phrase-level burst is drawn for the
            # dopamine trace the UI animates and deliberately does NOT touch the
            # weights. It used to call `apply_dopamine` with no row here, which
            # is the same unnormalised 5.6x overshoot - a second, larger blast
            # on top of the per-letter ones.
            burst = self.brain.dopamine.wave(1.0 if ok else -1.0).tolist()
            dopa = self.brain.dopamine
            reward = {
                "delivered": True,
                "written": False,
                "kind": "sucrose" if ok else "bitter",
                "sign": 1 if ok else -1,
                "peak": round(float(max(abs(v) for v in burst)), 4),
                "rpe": round(float(dopa.last.rpe), 4) if dopa.last is not None else 0.0,
                "baseline": round(float(dopa.baseline), 4),
                # Nothing is written at phrase level - the letters already did
                # it - so there is no credited or punished row here, and the UI
                # says so rather than inventing one.
                "credited": None, "credited_label": None,
                "punished": None, "punished_label": None,
                "tau": float(self.cfg.dopa_tau),
                "delay": int(self.cfg.dopa_delay),
                "sucrose": float(self.cfg.sucrose),
                "bitter": float(self.cfg.bitter),
            }
        return {"phrase": text, "results": out, "said": said, "correct": ok,
                "burst": self._round(burst),
                "reward": reward,
                # the last letter's vote, so the brain panel keeps showing a
                # real decision rather than a half-populated one
                "flow": (last or {}).get("flow"),
                "regions": (last or {}).get("regions"),
                "mbon": (last or {}).get("mbon"),
                "letters": len(graded),
                "letters_correct": hits,
                "skipped": len(skipped),
                "skipped_chars": "".join(r["char"] for r in skipped),
                # how many of the presented characters were outside the fly's
                # vocabulary - i.e. guaranteed wrong before it even looked
                "unknown_chars": sum(1 for r in graded if not r["known"])}

    def _readout(self, char: str) -> dict:
        """Encode -> respond -> read the MBONs, with NO dopamine and no state.

        `present()` is the teaching path and deliberately has side effects (it
        delivers a taste and can write weights). The controls in `digit_proof`
        need a pure forward pass, so they use this instead: same encoder, same
        generator, same `decision_repeats`, same argmax, nothing else. Keeping
        the two paths separate is what makes the letter/digit comparison a
        comparison of *stimuli* and not of code.
        """
        with self.lock:
            looks = max(1, int(getattr(self.cfg, "decision_repeats", 1)))
            spikes = encode_from_config(self.cfg, char, augment=False,
                                        gen=self.brain.gen)
            counts = self.brain.mb.present(spikes)
            for _ in range(looks - 1):
                counts = counts + self.brain.mb.present(
                    encode_from_config(self.cfg, char, augment=False,
                                       gen=self.brain.gen))
            pred = int(counts.argmax())
            order = sorted(range(len(counts)),
                           key=lambda i: float(counts[i]), reverse=True)
            top, second = float(counts[order[0]]), float(counts[order[1]])
            # Normalised so the scale of the spike counts cancels out. 0.0 means
            # the winner tied with the runner-up; 1.0 means the runner-up got
            # nothing. This is the "how sure was it" axis, and it is what
            # collapses on out-of-vocabulary input.
            margin = (top - second) / (top or 1.0)
            return {
                "pred": pred,
                "guess": CLASSES[pred],
                "score": round(top, 2),
                "runner": CLASSES[order[1]],
                "runner_score": round(second, 2),
                "margin": round(margin, 6),
            }

    def digit_proof(self, learn: bool = False) -> dict:
        """Show the fly all ten digits and record what it says back.

        This is the reproducibility artifact for the claim "the fly does not
        know digits". Nothing here is abstracted or inferred: each digit's real
        5x7 bitmap is Poisson-encoded onto the real receptor sheet, pushed
        through the real trained circuit, and the real argmax of the real MBON
        spike counts is recorded. Every one of the ten answers is a letter,
        because there is no other kind of output cell to win.

        The alphabet half of the table is measured on the same run, through the
        same `_readout`, so the margin comparison is a comparison of stimuli.
        It used to be a hardcoded 0.37 in the browser - a published number that
        nothing verified. It is now a measurement.
        """
        rows = []
        for ch in sorted(FONT_DIGITS):
            res = self.present(ch, learn)
            rows.append({
                "digit": ch,
                "guess": res["predicted"],
                "correct": res["correct"],
                "known": res["known"],
                "margin": res["flow"]["margin"],
                "runner": res["flow"]["runner_label"],
                "bitmap": FONT_DIGITS[ch],
                "reward": res["reward"],
            })

        # The control: the 27 characters the fly was trained on, same code path.
        letter_rows = [self._readout(c) for c in CLASSES]
        letter_margins = [r["margin"] for r in letter_rows]
        letter_hits = sum(1 for c, r in zip(CLASSES, letter_rows)
                          if r["guess"] == c)

        digit_margins = [r["margin"] for r in rows]
        avg = lambda v: (sum(v) / len(v)) if v else 0.0  # noqa: E731
        with self.lock:
            acc = self.brain.evaluate()
        return {
            "digits": rows,
            "n": len(rows),
            "n_correct": sum(1 for r in rows if r["correct"]),
            # 0 by construction: `correct` requires the argmax row to equal the
            # class index of the presented character, and a digit has no class
            # index. Reported rather than asserted, so the number is measured.
            "chance": round(1.0 / len(CLASSES), 4),
            "n_classes": len(CLASSES),
            "classes": CLASSES,
            "alphabet_accuracy": round(acc, 4),
            "alphabet_hits": letter_hits,
            "alphabet_presented": len(CLASSES),
            "margin_letters": round(avg(letter_margins), 4),
            "margin_letters_min": round(min(letter_margins), 4) if letter_margins else 0.0,
            "margin_digits": round(avg(digit_margins), 4),
            "margin_digits_min": round(min(digit_margins), 4) if digit_margins else 0.0,
            "near_ties": sum(1 for m in digit_margins if m < 0.05),
            "epoch": self.brain.epoch,
            "looks": int(getattr(self.cfg, "decision_repeats", 1)),
            "learn": bool(learn),
        }

    def train(self, epochs: int) -> dict:
        with self.lock:
            # `train` runs from `self.epoch + 1` up to `cfg.epochs`, so the
            # target has to be *current + n*, not n. Passing n directly did
            # nothing at all once a checkpoint had an epoch count in it.
            target = self.brain.epoch + max(1, epochs)
            self.cfg.epochs = target
            self.brain.cfg.epochs = target
            # Train on single looks: with Pavlovian reward the look count cannot
            # bias credit assignment, so averaging here would only double the
            # cost for nothing. Restore the readout setting before evaluating.
            self.brain.cfg.decision_repeats = 1
            self.brain.train(verbose=False)
            self.brain.cfg.decision_repeats = LOOKS
            acc = self.brain.evaluate()
            # Persist automatically: training done in the browser is the only
            # training there is, and it used to die with the page.
            self.brain.save(self.ckpt, note=f">= {acc:.1%} after {epochs} more epochs")
            self.log.append(f"trained {epochs} more epochs -> {acc:.1%} "
                            f"(saved to {os.path.basename(self.ckpt)})")
            return {
                "epochs": epochs,
                "accuracy": round(acc, 4),
                "history": [dict(epoch=h.epoch, accuracy=h.accuracy,
                                 reward_mean=h.reward_mean,
                                 dopa_baseline=h.dopa_baseline)
                            for h in self.brain.history],
            }

    def weights(self) -> dict:
        with self.lock:
            w = self.brain.mb.mbon.w.detach().numpy()
        return {"weights": [[round(float(v), 3) for v in row] for row in w],
                "shape": list(w.shape)}

    def reset(self, seed: int | None) -> dict:
        with self.lock:
            if seed is not None:
                self.cfg.seed = seed
            self.brain = FlyBrain(Config(**self.cfg.__dict__))
            self.log.append(f"reset circuit (seed {self.cfg.seed})")
        return self.state()


SESSION: Session


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:  # quieter console
        if "/api/" in (self.path or ""):
            print(f"  {self.command} {self.path}")

    # -- plumbing --------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _static(self, rel: str) -> None:
        path = os.path.normpath(os.path.join(WEB_DIR, rel.lstrip("/")))
        if not path.startswith(WEB_DIR) or not os.path.isfile(path):
            self._send(404, b"not found", "text/plain")
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as fh:
            self._send(200, fh.read(), MIME.get(ext, "application/octet-stream"))

    # -- routes ----------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route == "/api/state":
                self._json(SESSION.state())
            elif route == "/api/weights":
                self._json(SESSION.weights())
            else:
                self._static(route)
        except Exception as exc:  # noqa: BLE001 - surface errors to the page
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        body = self._read_json()
        try:
            if route == "/api/present":
                self._json(SESSION.present(body.get("char", "A"),
                                           bool(body.get("learn", False))))
            elif route == "/api/digit_proof":
                self._json(SESSION.digit_proof(bool(body.get("learn", False))))
            elif route == "/api/document":
                # Parsing does not touch the brain, so this is a module-level
                # function rather than a Session method: no weights are read,
                # no state is mutated, and nothing is presented.
                try:
                    payload = from_upload(str(body.get("name", "")),
                                          str(body.get("data", "")),
                                          bool(body.get("fold", False)))
                except RuntimeError as exc:
                    # Bad input, not a server fault - too big, encrypted, no text
                    # layer, not really that format. 400, so the page shows the
                    # reason rather than a generic failure.
                    self._json({"error": str(exc)}, 400)
                else:
                    self._json(payload)
            elif route == "/api/say":
                self._json(SESSION.say(body.get("phrase", "PHILIPPE DELAMBRE"),
                                       bool(body.get("learn", False))))
            elif route == "/api/train":
                self._json(SESSION.train(int(body.get("epochs", 5))))
            elif route == "/api/save":
                name = SESSION.save(note="manual save from the browser")
                self._json({"saved": name})
            elif route == "/api/reset":
                seed = body.get("seed")
                self._json(SESSION.reset(int(seed) if seed is not None else None))
            elif route == "/api/weights":
                self._json(SESSION.weights())
            else:
                self._json({"error": "unknown endpoint"}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def _port_in_use(host: str, port: int) -> bool:
    """True if something is already accepting connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web UI for FlyBrain.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=0,
                    help="train this many epochs at startup")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to load and save "
                         "(default: runs/flybrain.pt)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="serve even if something is already on this port")
    args = ap.parse_args()

    # Refuse to share a port. `allow_reuse_address` makes a rebind succeed even
    # when another server is already listening, and on Windows the two then
    # silently split the traffic: some requests go to the old brain, some to the
    # new one, and each can save over the other's checkpoint. That is not
    # hypothetical - it destroyed a 2000-epoch run. One port, one brain.
    if not args.force and _port_in_use(args.host, args.port):
        print(f"\n  ! something is already listening on "
              f"{args.host}:{args.port}."
              f"\n    Two servers on one port silently split requests between"
              f"\n    two different brains, and either can overwrite the other's"
              f"\n    checkpoint. Stop the other one first:"
              f"\n"
              f"\n      Get-CimInstance Win32_Process -Filter \"Name like"
              f"'%python%'\" |"
              f"\n        Where-Object {{ $_.CommandLine -like '*serve.py*' }} |"
              f"\n        ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"
              f"\n"
              f"\n    or pass --port N, or --force to ignore this.",
              flush=True)
        raise SystemExit(2)

    global SESSION
    cfg = Config(seed=args.seed)
    SESSION = Session(cfg, ckpt=args.ckpt)
    if args.epochs:
        print(f"  training {args.epochs} epochs before serving...")
        print(f"  accuracy {SESSION.train(args.epochs)['accuracy']:.1%}")

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print()
    print("=" * 68)
    print("  FlyBrain live  ->  " + url)
    print("=" * 68)
    print("  the fly, the circuit, and a letter moving through it in real time")
    # Print the interpreter: the port-sharing incident above involved one server
    # on the project venv and one on the system Python, and the only place that
    # was visible was here.
    print(f"  interpreter: {sys.executable}")
    print(f"  checkpoint : {os.path.relpath(SESSION.ckpt, HERE)}"
          f"  (epoch {SESSION.brain.epoch})")
    print("  Ctrl+C to stop\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
