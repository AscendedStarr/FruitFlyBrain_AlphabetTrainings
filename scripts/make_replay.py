"""Record a real FlyBrain session to JSON, so the UI can run as a static page.

The browser UI in `web/` is not static: every panel is filled from a `fetch()`
to the Python server, which is the only thing that owns the brain. GitHub Pages
serves files, not torch, so the live simulation cannot be published there.

What *can* be published is a recording of it. This script drives the real
`Session` - the same class `scripts/serve.py` puts behind HTTP, loaded from the
same tracked checkpoint - through a fixed script, and writes every response to
`web/replay.json`. `web/replay.js` then answers the page's `fetch()` calls from
that file instead of from a server.

Two properties make this honest rather than a mock:

  1. **Nothing here is computed by this file.** Each entry is the literal return
     value of a `Session` method. The numbers the page draws are the numbers the
     model produced.
  2. **Inference is side-effect free, so the recording is reusable.** Every
     route is recorded with `learn=False`, which reads the weights and never
     writes them (`MushroomBody.apply_dopamine` is the only writer, and it is
     not called). That is why pressing "A" a second time returns the same
     recorded payload instead of drifting: with plasticity off the response
     depends only on the weights and the glyph.

The recorder asserts the headline figures against the values published in
README.md and runs/*.json, and fails loudly if they have drifted - a stale
recording must not be shipped as though it were current.

Run it with the flybrain virtualenv:

    .\\.venv-flybrain\\Scripts\\python.exe scripts\\make_replay.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

# Must be set before torch is imported; see the comment in scripts/serve.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from flybrain import CLASSES  # noqa: E402

from serve import LOOKS, Session  # noqa: E402

# Phrases recorded for the "say it" box. The input stays editable in the page,
# but only these exact strings resolve; anything else returns a deliberate error
# telling the reader to run the server locally. Recording every phrase is not
# possible, and inventing one would defeat the point.
PHRASES = [
    "PHILIPPE DELAMBRE",   # the value the page ships with
    "FRUIT FLY",
    "HELLO WORLD",
    "123",                 # the negative control: every character is a digit
]

# What the recorder promises the page will contain. Checked after recording.
#
# Note the denominators. The page's digit panel is built by
# `Session.digit_proof()`, which presents each of the ten digits **once** and
# then sweeps the 27 letters once, so it reports `0/10` and `27/27`. The README
# quotes `0/300` and `810/810`, which come from a different, offline artifact:
# `scripts/digit_proof.py --repeats 30`, run over the 10-digit and 27-letter
# alphabets. Same code path, same checkpoint, 30x the samples. Both are true and
# they are not the same table, so they are asserted separately here rather than
# being quietly reconciled in the page.
EXPECT = {
    "plastic_synapses": 13824,   # 512 KC x 27 MBON, the v0.1.0 geometry
    "n_classes": 27,
    "digit_correct": 0,          # page: 0/10  · README: 0/300 (repeats=30)
    "digit_n": 10,
    "alphabet_hits": 27,         # page: 27/27 · README: 810/810 (repeats=30)
    "alphabet_presented": 27,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(out_path: str, ckpt: str | None) -> dict:
    cfg_seed = 7
    from flybrain import Config  # local import: same module serve.py uses

    session = Session(Config(seed=cfg_seed), ckpt=ckpt)
    routes: dict[str, object] = {}

    state = session.state()
    routes["GET /api/state"] = state
    routes["GET /api/weights"] = session.weights()

    # Every glyph the page can press: the 27 output classes, then the 10 digits.
    # Digits are presentable but not answerable - `known` is false and the page
    # says so - which is the whole negative control.
    for ch in list(CLASSES) + [str(d) for d in range(10)]:
        key = f"POST /api/present {ch}"
        routes[key] = session.present(ch, learn=False)

    for phrase in PHRASES:
        routes[f"POST /api/say {phrase}"] = session.say(phrase, learn=False)

    routes["POST /api/digit_proof"] = session.digit_proof(learn=False)

    proof = routes["POST /api/digit_proof"]
    checks = {
        "plastic_synapses": state["plastic_synapses"],
        "n_classes": state["n_classes"],
        "digit_correct": proof["n_correct"],
        "digit_n": proof["n"],
        "alphabet_hits": proof["alphabet_hits"],
        "alphabet_presented": proof["alphabet_presented"],
    }
    mismatched = {k: (checks[k], v) for k, v in EXPECT.items() if checks[k] != v}
    if mismatched:
        for k, (got, want) in mismatched.items():
            print(f"  ! {k}: recorded {got!r}, README says {want!r}")
        raise SystemExit(
            "refusing to write a replay whose numbers disagree with the "
            "published ones - re-read README.md before shipping this"
        )

    meta = {
        "generated_by": "scripts/make_replay.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": (ckpt or
                       os.path.relpath(session.ckpt, ROOT)).replace("\\", "/"),
        "checkpoint_sha256": _sha256(session.ckpt),
        "epoch": int(session.brain.epoch),
        "looks_per_answer": LOOKS,
        "seed": cfg_seed,
        "plasticity": "off - every route recorded with learn=False",
        "class_count": len(CLASSES),
        "routes_recorded": len(routes),
        "phrases": PHRASES,
        "verified": checks,
        "denominators": {
            "digit_panel": f"{checks['digit_correct']}/{checks['digit_n']} "
                           f"(one presentation per digit)",
            "alphabet_panel": f"{checks['alphabet_hits']}/"
                              f"{checks['alphabet_presented']} (one per class)",
            "readme_digits": "0/300 (scripts/digit_proof.py --repeats 30)",
            "readme_alphabet": "810/810 (same script, 30 repeats of 27)",
            "why": ("the page runs the 10- and 27-sample version so it can "
                    "finish in a second; the README quotes the 300-sample "
                    "offline sweep. same code path, same checkpoint."),
        },
        # The one thing a reader must know before believing the page.
        "note": ("every value below is the literal return of a Session method "
                 "in scripts/serve.py, evaluated against the tracked "
                 "checkpoint. nothing is simulated in the browser, and nothing "
                 "was hand-written into this file."),
    }

    payload = {"replay_version": 1, "meta": meta, "routes": routes}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a static replay of FlyBrain.")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "replay.json"))
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to record from (default: runs/flybrain.pt)")
    args = ap.parse_args()

    print(f"  recording from {args.ckpt or 'runs/flybrain.pt'} ...")
    payload = record(args.out, args.ckpt)
    meta = payload["meta"]
    size = os.path.getsize(args.out)
    print(f"  epoch {meta['epoch']}, {meta['routes_recorded']} routes, "
          f"{meta['looks_per_answer']} looks/answer")
    print(f"  digits {meta['verified']['digit_correct']}/"
          f"{meta['verified']['digit_n']}, alphabet "
          f"{meta['verified']['alphabet_hits']}/"
          f"{meta['verified']['alphabet_presented']}")
    print(f"  wrote {os.path.relpath(args.out, ROOT)} "
          f"({size:,} bytes, {size / 1024 / 1024:.2f} MiB)")
    print(f"  checkpoint sha256 {meta['checkpoint_sha256'][:16]}...")

    # A replay that cannot be sliced is not a replay; keep it loadable by a
    # browser on a phone.
    if size > 12 * 1024 * 1024:
        raise SystemExit(f"replay is {size:,} bytes - too large to ship; "
                         f"record fewer routes or trim fields")


if __name__ == "__main__":
    main()
