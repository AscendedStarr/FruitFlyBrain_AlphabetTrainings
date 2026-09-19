"""Byte-level proof that a measurement did not train anything.

The project's central claim about its negative control is narrow and falsifiable:

    Showing the fly a digit cannot teach it anything, because a digit is an
    input glyph with no output cell. The answer is wrong because there is
    nothing to be right with, not because the model declined to guess.

That claim is easy to *state* and easy to get wrong in code - a stray
``learn=True``, a leftover ``trial()`` call, or a second process writing the
checkpoint would all invalidate it silently. This module makes the claim
checkable instead of trustable.

Two independent digests are used:

* :func:`sha256_file` - the checkpoint as it sits on disk. Catches a stray
  process rewriting the file, which is how one long run was lost.
* :func:`weights_digest` - every weight matrix in the live circuit. Catches
  plasticity happening in memory even when nothing is written.

Call :func:`snapshot` before a measurement and :func:`verify` after it. Both are
pure observation.
"""

from __future__ import annotations

import hashlib
from typing import Any


def sha256_file(path: str) -> str | None:
    """sha256 of a file, or ``None`` if it is not there."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def weights_digest(brain) -> str:
    """sha256 over every weight matrix, in a fixed order.

    ``mbon.w`` is the only plastic matrix in the project (13,824 synapses at
    the default config); ``al.w`` and ``kc.w`` are fixed at construction. All
    three go into the digest so it covers the whole circuit and not just the
    part that could move.
    """
    h = hashlib.sha256()
    for name, tensor in (
        ("al.w", brain.mb.al.w),
        ("kc.w", brain.mb.kc.w),
        ("mbon.w", brain.mb.mbon.w),
    ):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(brain, ckpt_path: str | None = None) -> dict[str, Any]:
    """Record the circuit's current identity. Take this *before* measuring."""
    return {
        "weights_sha256": weights_digest(brain),
        "checkpoint_sha256": sha256_file(ckpt_path) if ckpt_path else None,
        "epoch": int(getattr(brain, "epoch", -1)),
    }


def verify(before: dict[str, Any], brain, ckpt_path: str | None = None,
           strict: bool = True) -> dict[str, Any]:
    """Compare the circuit against an earlier :func:`snapshot`.

    Returns a dict of the after-values plus ``weights_unchanged`` and
    ``checkpoint_unchanged``. With ``strict`` set, an unchanged result is
    asserted rather than merely reported - a measurement that silently trained
    is worse than one that crashes.
    """
    after = snapshot(brain, ckpt_path)
    same_weights = after["weights_sha256"] == before["weights_sha256"]
    same_ckpt = after["checkpoint_sha256"] == before["checkpoint_sha256"]

    if strict:
        assert same_weights, (
            "a weight moved during a measurement that must not learn - "
            "check for an apply_dopamine()/trial() call on the inference path"
        )
        assert same_ckpt, "the checkpoint was rewritten during a read-only run"

    return {
        "weights_sha256_before": before["weights_sha256"],
        "weights_sha256_after": after["weights_sha256"],
        "weights_unchanged": same_weights,
        "checkpoint_sha256_before": before["checkpoint_sha256"],
        "checkpoint_sha256_after": after["checkpoint_sha256"],
        "checkpoint_unchanged": same_ckpt,
        "plasticity_applied": not same_weights,
    }
