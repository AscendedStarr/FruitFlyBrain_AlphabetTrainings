"""Guard the published recording against drifting from the published numbers.

GitHub Pages ships `web/replay.json` - a capture of real `Session` responses -
because CI has no torch and the live server cannot run there. That creates one
specific hazard: if the checkpoint is retrained and nobody re-records, the site
keeps confidently serving yesterday's network. Every assertion below exists to
make that failure loud instead of silent.

Deliberately stdlib-only and torch-free. This has to pass in CI, where the only
thing installed is the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(ROOT, "web", "replay.json")
CKPT = os.path.join(ROOT, "runs", "flybrain.pt")

# The blank is index 0 in flybrain.CLASSES, not last. Checked against the
# recorded state payload rather than assumed.
CLASSES = [" "] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load():
    with open(REPLAY, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_recording_exists_and_is_not_stale():
    assert os.path.isfile(REPLAY), "run scripts/make_replay.py"
    meta = _load()["meta"]
    # The single most important check: a retrained checkpoint invalidates the
    # recording, and nothing else here would notice.
    assert meta["checkpoint_sha256"] == _sha256(CKPT), (
        "runs/flybrain.pt changed since web/replay.json was recorded - "
        "re-run scripts/make_replay.py and re-check the README figures"
    )
    assert meta["generated_by"] == "scripts/make_replay.py"
    assert meta["plasticity"].startswith("off")


def test_recording_matches_the_published_figures():
    meta = _load()["meta"]
    v = meta["verified"]
    # 13,824 = 512 Kenyon cells x 27 MBONs, the geometry this checkpoint uses.
    assert v["plastic_synapses"] == 13824
    assert v["n_classes"] == 27
    # The negative control: structurally impossible to score, so it must be 0.
    assert v["digit_correct"] == 0
    assert v["digit_n"] == 10
    # The positive control, on the same run through the same readout.
    assert v["alphabet_hits"] == v["alphabet_presented"] == 27


def test_the_panels_agree_with_the_headers():
    """`meta.verified` is a summary. It must not be able to disagree with the
    route bodies it summarises."""
    data = _load()
    proof = data["routes"]["POST /api/digit_proof"]
    v = data["meta"]["verified"]
    assert proof["n_correct"] == v["digit_correct"]
    assert proof["n"] == v["digit_n"]
    assert proof["alphabet_hits"] == v["alphabet_hits"]
    assert proof["alphabet_presented"] == v["alphabet_presented"]
    assert proof["n_classes"] == v["n_classes"]
    assert proof["learn"] is False

    state = data["routes"]["GET /api/state"]
    assert state["plastic_synapses"] == v["plastic_synapses"]
    assert state["n_classes"] == v["n_classes"]
    assert state["classes"] == CLASSES


def test_every_glyph_is_recorded_and_labeled():
    routes = _load()["routes"]
    for ch in CLASSES + [str(d) for d in range(10)]:
        payload = routes["POST /api/present " + ch]
        assert payload["char"] == ch
        # Inference only: a payload that claims it learned something could not
        # be replayed again, which is the property the whole page relies on.
        assert payload["learned"] is False
        assert payload["predicted"] in CLASSES
        assert len(payload["receptor_raster"]) > 0


def test_digits_can_never_be_right():
    """The claim the README leads with, re-checked on the recorded table."""
    proof = _load()["routes"]["POST /api/digit_proof"]
    assert len(proof["digits"]) == 10
    for row in proof["digits"]:
        assert row["known"] is False
        assert row["correct"] is False
        # Every answer is a letter: there is no other kind of output cell.
        assert row["guess"] in CLASSES


def test_phrases_are_recorded_whole():
    """`say` cannot be stitched from per-character payloads: `present` advances
    the generator, so a phrase only replays if it was recorded as a sequence.

    Note what `said` is: the fly's *output*, not an echo of the input. "123"
    comes back as "IQZ" because a digit has no output cell to win, which is the
    negative control doing its job. `Session.say` builds it from the predicted
    class of every character it could actually present.
    """
    data = _load()
    meta = data["meta"]
    assert meta["phrases"]
    for phrase in meta["phrases"]:
        res = data["routes"]["POST /api/say " + phrase]
        assert res["phrase"] == phrase
        assert res["said"] == "".join(
            r["predicted"] for r in res["results"] if r["predicted"] is not None)
        # Every recorded phrase is spelled from glyphs the fly can be shown, so
        # nothing should have been stepped over.
        assert res["skipped"] == 0
        if res["correct"]:
            assert res["said"] == phrase.upper()
        assert all(r.get("learned") is not True for r in res["results"])


def test_the_shim_refuses_what_it_cannot_honour():
    """A recording cannot retrain itself. If the shim stops refusing, the page
    would report weight changes that provably did not happen."""
    path = os.path.join(ROOT, "web", "replay.js")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    for endpoint in ("/api/train", "/api/reset", "/api/document"):
        assert "'" + endpoint + "'" in src, endpoint
    assert "body.learn === true" in src
