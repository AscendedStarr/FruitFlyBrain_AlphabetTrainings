"""Regression test: clicking a letter with plasticity on must not wreck the brain.

This is the bug that made the fly look like it "kept guessing wrong". The browser
called `mb.apply_dopamine(trace)` with no row argument, which takes the branch

    delta = gain * self.eligibility          # gain = raw burst sum ~ 5.6

against a weight scale of ~0.053. That is a ~7x overshoot, so a SINGLE click on
a letter with "plasticity on" (the default) drove a 97.5% brain down to 7%
chance - and because the page auto-saves, the wreckage was written to disk.

The fix passes a named row (`credit=target` / `punish=pred`), which normalises
the step to `lr` weight units. This test drives the live HTTP API - the same
path the browser uses - and asserts accuracy survives.

Start the server first, then:

    $env:OMP_NUM_THREADS=1
    .\\.venv-flybrain\\Scripts\\python.exe test_click.py [port]
"""

from __future__ import annotations

import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BASE = f"http://127.0.0.1:{PORT}"

LEARN_AFTER = 8      # how many plastic presentations to fire at it
TOLERANCE = 0.80     # accuracy must stay above this


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=120) as resp:
        return json.loads(resp.read())


before = get("/api/state")["accuracy"]
print(f"accuracy before plastic clicks : {before:.1%}")
if before < TOLERANCE:
    print("  ... server is not holding a trained brain; restart it first")
    sys.exit(1)

print(f"\nclicking {LEARN_AFTER} letters with plasticity ON")
for ch in "ABCDEFGH":
    res = post("/api/present", {"char": ch, "learn": True})
    print(f"  {ch} -> {res['predicted']:>2} "
          f"{'ok' if res['correct'] else 'wrong'}  pulse {res['pulse']:+.0f}")

after = get("/api/state")["accuracy"]
print(f"\naccuracy after  plastic clicks : {after:.1%}")

# A single well-scaled step barely moves a trained readout; a 7x overshoot
# annihilates it. Anything above the tolerance means the credited path is used.
ok = after >= TOLERANCE
print()
print(f"  drop: {before - after:+.1%}")
print("  PASS - credited (normalised) update, brain intact" if ok else
      "  FAIL - weights collapsed, the unnormalised update is back")

# And the phrase should still come out.
said = post("/api/say", {"phrase": "MY NAME IS JEFF", "learn": False})
print(f"\nfly says: {said['said']!r}  {'correct' if said['correct'] else 'MISREAD'}")
final = get("/api/state")["accuracy"]
print(f"accuracy after saying         : {final:.1%}")

sys.exit(0 if (ok and final >= TOLERANCE) else 1)
