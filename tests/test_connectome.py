"""Checks for the one measured layer.

These run against the tracked ``data/hemibrain_mb.npz``. They are deliberately
about *plumbing and controls*, not about accuracy: the accuracy experiment is
``connectome_compare.py`` and takes ~40 minutes, which is not a unit test.

The test that matters most here is
``test_shuffled_control_preserves_degree_and_weights``. It is the test that makes
the headline negative result falsifiable: if the shuffle ever stopped preserving
each Kenyon cell's partner count or its multiset of synapse weights, the control
would become a different arm and the comparison would silently stop meaning
anything.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from flybrain import Config
from flybrain.connectome import DEFAULT_PATH, MissingConnectome, load, pn_kc_weights
from flybrain.circuit import KenyonCells

# Documented in the README and in runs/connectome_compare.json. If the extraction
# changes, these change, and that should be a deliberate edit rather than a
# silently absorbed one.
N_PN = 157
N_KC = 1802
N_MBON = 68
N_DAN = 322
PN_TO_KC_EDGES = 12426
PN_TO_KC_SYNAPSES = 185994

# The artifact is tracked, but a checkout without it should skip rather than
# error: every number below is a claim about the extracted subgraph, and there is
# nothing honest to assert without it.
pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_PATH),
    reason="data/hemibrain_mb.npz missing - run tools/build_connectome.py")


def test_connectome_is_present_and_shaped():
    conn = load()
    assert conn.n_pn == N_PN
    assert conn.n_kc == N_KC
    assert conn.n_mbon == N_MBON
    assert conn.n_dan == N_DAN
    # pn_to_kc is a (n_kc, n_pn) synapse-count matrix, not an edge list
    assert conn.pn_to_kc.shape == (N_KC, N_PN)
    assert int(conn.n_synapses) == PN_TO_KC_SYNAPSES
    assert int((conn.pn_to_kc > 0).sum()) == PN_TO_KC_EDGES


def test_connectome_is_cc_by_attributed():
    """The licence and citation ride along inside the artifact.

    The extracted subgraph is redistributed in this repository, so the
    attribution has to travel with the data and not only live in CITATION.cff.
    """
    meta = load().meta
    assert meta["dataset"] == "hemibrain"
    assert meta["licence"] == "CC BY 4.0"
    assert "Scheffer" in meta["citation"]


def test_stored_summary_agrees_with_the_arrays():
    """The `measured` block must match the matrix it summarises.

    A stale block would let the docs quote one number while the model trains on
    a different one, which is the failure mode this whole repository exists to
    avoid.
    """
    conn = load()
    m = conn.meta["measured"]
    f = conn.fan_in
    assert int(m["kc_fan_in_min"]) == int(f.min())
    assert int(m["kc_fan_in_max"]) == int(f.max())
    assert abs(float(m["kc_fan_in_mean"]) - float(f.mean())) < 1e-3
    assert abs(float(m["kc_fan_in_median"]) - float(np.median(f))) < 1e-9
    assert int(m["pn_to_kc_edges"]) == int((conn.pn_to_kc > 0).sum())
    assert int(m["pn_to_kc_synapses"]) == int(conn.n_synapses)


def test_config_overrides_match_the_connectome():
    conn = load()
    ov = conn.as_config_overrides()
    assert ov == {"n_pn": N_PN, "n_kc": N_KC, "k_active": 211}


def test_fan_in_distribution_is_not_uniform():
    """The whole point of the fan-in-distribution caveat.

    If this ever became uniform, the README's explanation for the residual gap
    between the connectome arms and the random arm would be wrong.
    """
    fan_in = load().fan_in
    assert fan_in.min() == 1
    assert int(np.median(fan_in)) == 6
    assert fan_in.max() == 21
    assert fan_in.std() > 0  # not a uniform fan-in
    assert len(set(fan_in.tolist())) > 2


def test_every_kept_pn_reaches_a_kc_and_vice_versa():
    """This is the calyx filter, asserted rather than described.

    The naive `"PN" in type` rule left 271 projection neurons with zero KC
    output. Membership is decided by connectivity, so by construction there can
    be no such neuron in the committed subgraph.
    """
    conn = load()
    occupied = conn.pn_to_kc > 0
    pn_fan_out = occupied.sum(axis=0)   # per PN, summed over KCs
    kc_fan_in = occupied.sum(axis=1)    # per KC, summed over PNs
    assert pn_fan_out.min() >= 1, "a PN with no KC output survived the filter"
    assert kc_fan_in.min() >= 1, "a KC with no PN input survived the filter"


def test_weights_are_unit_l2_per_row():
    conn = load()
    w = pn_kc_weights(conn, variant="connectome")
    norm = w.norm(dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    assert w.shape == (N_KC, N_PN)
    assert torch.all(w >= 0)  # all-excitatory, a documented modelling choice
    assert int((w > 0).sum()) == PN_TO_KC_EDGES


def test_weights_are_deterministic_and_seed_free():
    """The measured arm has no random draw in it at all.

    If `connectome` ever became seed-dependent, each seed would be running a
    different circuit and the paired deltas in the README would be meaningless.
    """
    conn = load()
    a = pn_kc_weights(conn, variant="connectome", seed=0)
    b = pn_kc_weights(conn, variant="connectome", seed=0)
    c = pn_kc_weights(conn, variant="connectome", seed=7)
    assert torch.equal(a, b)
    assert torch.equal(a, c)


def test_shuffled_control_preserves_degree_and_weights():
    """The load-bearing control claim.

    `connectome-shuffled` is only a valid identity control if each KC keeps its
    exact partner count *and* its exact multiset of synapse weights. If this
    fails, the +0 paired result in the README means nothing.
    """
    conn = load()
    real = pn_kc_weights(conn, variant="connectome", count_power=1.0)
    shuf = pn_kc_weights(conn, variant="connectome-shuffled", seed=0, count_power=1.0)

    # same nonzero count per row
    assert torch.equal((real > 0).sum(dim=1), (shuf > 0).sum(dim=1))
    # same multiset of weights per row
    assert torch.allclose(real.sort(dim=1).values, shuf.sort(dim=1).values, atol=1e-6)
    # same total number of edges across the whole matrix
    assert int((real > 0).sum()) == int((shuf > 0).sum()) == PN_TO_KC_EDGES
    # but not the same partners
    assert not torch.equal(real, shuf)
    # and the shuffled matrix is still unit-norm, because it was permuted in place
    norm = shuf.norm(dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_shuffle_differs_across_seeds():
    conn = load()
    a = pn_kc_weights(conn, variant="connectome-shuffled", seed=0)
    b = pn_kc_weights(conn, variant="connectome-shuffled", seed=1)
    assert not torch.equal(a, b)


def test_support_overlap_between_real_and_shuffled_is_near_chance():
    """Sanity: the shuffle really does destroy partner identity.

    The expected overlap is not zero, and it is not meant to be: a KC holding k
    of the 157 PNs keeps k^2 / 157 of them by chance, which over the whole
    matrix predicts ~645 of the 12,426 edges. The measured 604 sits on top of
    that prediction, which is what "same degrees, different partners" looks
    like. An overlap approaching 1.0 would mean the shuffle is a no-op and the
    control is not a control.
    """
    conn = load()
    real = pn_kc_weights(conn, variant="connectome")
    shuf = pn_kc_weights(conn, variant="connectome-shuffled", seed=0)
    overlap = int(((real > 0) & (shuf > 0)).sum())
    total = int((real > 0).sum())
    degrees = (real > 0).sum(dim=1).double()
    expected = float((degrees * degrees).sum() / N_PN)
    assert abs(overlap - expected) / expected < 0.25
    assert overlap / total < 0.10


def test_random_is_not_an_accepted_variant():
    """`random` must go through circuit.py, not through here.

    Accepting it would make it possible to confuse the measured arm with the
    synthetic control in a single call.
    """
    conn = load()
    with pytest.raises(ValueError):
        pn_kc_weights(conn, variant="random")


def test_missing_file_says_how_to_fix_it():
    with pytest.raises(MissingConnectome) as e:
        load("does/not/exist.npz")
    assert "build_connectome" in str(e.value)


def test_connectome_arm_builds_and_is_not_random():
    """Build the real circuit and confirm it did not take the random path.

    `torch.randperm` is the v0.1.0 wiring. The connectome arm must not reach it,
    so this patches it to explode.
    """
    conn = load()
    cfg = Config(wiring="connectome", **conn.as_config_overrides())
    kc = KenyonCells(cfg)
    assert kc.w.shape == (N_KC, N_PN)

    real = pn_kc_weights(conn, variant="connectome")
    assert torch.allclose(kc.w, real)

    original = torch.randperm

    def boom(*a, **k):
        raise AssertionError("the connectome arm called torch.randperm")

    torch.randperm = boom
    try:
        KenyonCells(cfg)
    finally:
        torch.randperm = original


def test_wrong_sized_config_is_rejected_with_a_hint():
    """The guard that stops a 157x1802 matrix being loaded into a 512-KC model.

    Silently broadcasting the wires into the wrong shape would be the easiest
    way to publish a connectome result that is not about the connectome.
    """
    cfg = Config(wiring="connectome")  # default 128 PN / 512 KC
    with pytest.raises(ValueError) as e:
        KenyonCells(cfg)
    assert "as_config_overrides" in str(e.value)


def test_random_arm_still_works_after_the_refactor():
    """v0.1.0's behaviour must be unchanged. This is the backward-compat test.

    Note what is deliberately *not* asserted: the random path never
    row-normalises. It draws `fan_in` weights from N(0, 1/fan_in) per KC and
    stops, so row norms scatter around 1 (measured 0.59-1.29 at fan_in=16). Only
    the connectome path is row-normalised, and that asymmetry is a real
    difference between the arms rather than a bug - so asserting unit norm here
    would be asserting a fiction about the code.
    """
    cfg = Config()  # wiring="random" by default
    kc = KenyonCells(cfg)
    assert kc.w.shape == (cfg.n_kc, cfg.n_pn)
    # exactly fan_in nonzeros per row, because randperm draws distinct indices
    assert int((kc.w != 0).sum()) == cfg.n_kc * cfg.fan_in
    # deterministic for a fixed seed, different for another
    assert torch.equal(kc.w, KenyonCells(cfg).w)
    assert not torch.equal(kc.w, KenyonCells(Config(seed=cfg.seed + 1)).w)
    # scattered around 1, but not equal to 1
    norm = kc.w.norm(dim=1)
    assert float(norm.min()) > 0.3 and float(norm.max()) < 3.0
    assert not torch.allclose(norm, torch.ones_like(norm), atol=1e-3)


def test_unknown_wiring_is_rejected_loudly():
    cfg = Config(wiring="something-else")
    with pytest.raises(ValueError):
        KenyonCells(cfg)


def test_old_checkpoints_still_load():
    """A v0.1.0 checkpoint has no `wiring` key.

    `FlyBrain.load()` filters saved config keys against the current Config, so
    adding fields must not break every checkpoint already on disk.
    """
    from flybrain import FlyBrain

    path = "runs/flybrain.pt"
    try:
        brain = FlyBrain.load(path)
    except FileNotFoundError:
        pytest.skip("no committed checkpoint to load")
    assert brain.cfg.wiring == "random"
    assert brain.cfg.n_kc == 512
    assert brain.cfg.n_pn == 128
    assert brain.cfg.fan_in == 16
