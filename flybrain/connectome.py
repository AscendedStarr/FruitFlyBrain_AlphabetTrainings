"""The measured mushroom-body wiring.

This module loads the compact subgraph built by ``tools/build_connectome.py``
from the hemibrain v1.2 export and turns it into the fixed ``PN -> KC`` weight
matrix the circuit uses. It is the only place in the package that knows anything
about the connectome.

Where the numbers come from
---------------------------
Scheffer et al. 2020, *A connectome and analysis of the adult Drosophila central
brain*, eLife 9:e57443 (doi:10.7554/eLife.57443), data release v1.2 from the
public Janelia bucket. Licence **CC BY 4.0**: unrestricted use and
redistribution provided the original author and source are credited. Attribution
lives in ``CITATION.cff``.

The subgraph itself is 157 projection neurons, 1802 Kenyon cells, 68 MBONs and
322 dopaminergic neurons. Membership is decided by connectivity rather than by
cell-type name - see the module docstring of ``tools/build_connectome.py`` for
why that distinction turned out to matter.

What is measured and what is not
--------------------------------
This is the important part, so it is stated plainly rather than buried.

*Which* Kenyon cell listens to *which* projection neurons, and how many
synapses are between them, is **measured**. That is the wiring, and it is what
this module provides.

The **sign and scale** of those synapses are **not measured here**. The export
records synapse counts, not conductances, and this circuit is not a
biophysical model: it is 35 photoreceptor-like inputs feeding a spiking
classifier, so the units are arbitrary and have to be calibrated to the rest of
the model. The choices are:

    magnitude  proportional to sqrt(synapse count), rather than to the raw
               count, so a Kenyon cell with one 400-synapse partner and one
               4-synapse partner does not end up listening to only the first
    row norm   each KC's weight row is rescaled to unit L2 norm, which is the
               scale the randomly-wired arm already uses - so the two share an
               operating point and any difference between them is attributable
               to *which* PNs each KC hears and not to how loud the input is
    sign       excitation, as PN -> KC is cholinergic and excitatory in the fly

Those three choices are made once, documented here, and applied identically to
every arm of the comparison, so they cannot manufacture a difference between
arms. What they do mean is that a result of the form "the real wiring is better"
is a claim about the *pattern* of the connectome, not about its conductance.

Variants
--------
``pn_kc_weights(conn, variant)`` builds two different fixed matrices from the
same measured dimensions, which is what makes the comparison a controlled one:

    "connectome"          the measured wiring
    "connectome-shuffled" the measured wiring with each KC's partner set replaced
                          by a uniformly random set of the same size, keeping
                          that KC's exact degree and weight multiset. This
                          isolates partner *identity*: it holds the circuit
                          size, the fan-in distribution and the input
                          statistics fixed, so it answers "is it these specific
                          partners, or just having this many of them?"

The third arm of the comparison - a plain random expansion - is *not* built
here. It is the ``wiring="random"`` path in :class:`flybrain.circuit.KenyonCells`,
run at the connectome's dimensions. That is deliberate: it is the same code path
v0.1.0 already ships, so the published v0.1.0 numbers stay reproducible, and
there is only ever one random-wiring implementation in the package.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO_ROOT, "data", "hemibrain_mb.npz")

#: the variants that draw on the connectome
VARIANTS = ("connectome", "connectome-shuffled")


class MissingConnectome(FileNotFoundError):
    """Raised when the wiring is asked for but the artifact is not on disk."""


class Connectome:
    """The extracted mushroom-body subgraph, in full.

    Arrays are ``(n_kc, n_pn)`` / ``(n_mbon, n_kc)`` / ``(n_dan, n_kc)``
    synapse-count matrices, plus the hemibrain body ids and cell types of every
    neuron in them, so any claim made about the circuit can be traced back to a
    specific annotated neuron.
    """

    def __init__(self, z) -> None:
        self.pn_to_kc = z["pn_to_kc"]
        self.kc_to_mbon = z["kc_to_mbon"]
        self.dan_to_kc = z["dan_to_kc"]
        self.pn_id = z["pn_id"]
        self.kc_id = z["kc_id"]
        self.mbon_id = z["mbon_id"]
        self.dan_id = z["dan_id"]
        self.pn_type = z["pn_type"]
        self.kc_type = z["kc_type"]
        self.mbon_type = z["mbon_type"]
        self.dan_type = z["dan_type"]
        self.meta = json.loads(str(z["meta"]))

    # -- shape ---------------------------------------------------------------
    @property
    def n_pn(self) -> int:
        return int(self.pn_to_kc.shape[1])

    @property
    def n_kc(self) -> int:
        return int(self.pn_to_kc.shape[0])

    @property
    def n_mbon(self) -> int:
        return int(self.kc_to_mbon.shape[0])

    @property
    def n_dan(self) -> int:
        return int(self.dan_to_kc.shape[0])

    @property
    def fan_in(self) -> np.ndarray:
        """Number of distinct PNs feeding each KC."""
        return (self.pn_to_kc > 0).sum(axis=1)

    @property
    def n_synapses(self) -> int:
        return int(self.pn_to_kc.sum())

    # -- provenance ----------------------------------------------------------
    def describe(self) -> str:
        f = self.fan_in
        m = self.meta["measured"]
        mb_in = (self.kc_to_mbon > 0).sum(axis=1)
        return (
            "hemibrain %s (CC BY 4.0) - %s\n"
            "  %d PN -> %d KC, %s synapses in %s traced connections\n"
            "  KC fan-in: min %d, median %g, mean %.2f, max %d\n"
            "  %d MBON, each sampling a median of %d of the %d KCs (%s synapses)\n"
            "  %d DAN -> KC, %s synapses"
            % (self.meta["version"], self.meta["citation"],
               self.n_pn, self.n_kc,
               format(m["pn_to_kc_synapses"], ","),
               format(m["pn_to_kc_edges"], ","),
               f.min(), np.median(f), f.mean(), f.max(),
               self.n_mbon, int(np.median(mb_in)), self.n_kc,
               format(m["kc_to_mbon_synapses"], ","),
               self.n_dan, format(m["dan_to_kc_synapses"], ","))
        )

    def as_config_overrides(self, sparsity: float = 0.117) -> dict:
        """Config fields that must follow from the measured circuit.

        ``n_pn`` and ``n_kc`` are sizes, so they are simply the measured ones.
        ``k_active`` is a *fraction* of the KC population in the fly (5-10% of
        Kenyon cells respond to a given odour), so it has to be rescaled to the
        new population rather than held at 60, which was 12% of 512. The default
        keeps the same 11.7% the shipped model uses, so the KC layer is as
        sparse in every arm and sparsity is not a confound.
        """
        return {
            "n_pn": self.n_pn,
            "n_kc": self.n_kc,
            "k_active": max(1, int(round(sparsity * self.n_kc))),
        }

    def __repr__(self) -> str:
        return ("<Connectome hemibrain %s: %d PN -> %d KC, %d MBON>"
                % (self.meta["version"], self.n_pn, self.n_kc, self.n_mbon))


@functools.lru_cache(maxsize=4)
def load(path: str | None = None) -> Connectome:
    """Load the subgraph, cached per path so the three arms share one copy."""
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        raise MissingConnectome(
            "no connectome artifact at %s\n"
            "build it with:\n"
            "    python tools/build_connectome.py\n"
            "(downloads ~46 MB from the public hemibrain bucket on first run)"
            % path)
    with np.load(path, allow_pickle=True) as z:
        return Connectome({k: z[k] for k in z.files})


def _row_normalise(w: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm, leaving all-zero rows at zero."""
    norm = np.linalg.norm(w, axis=1, keepdims=True)
    return np.divide(w, norm, out=np.zeros_like(w), where=norm > 0)


def pn_kc_weights(
    conn: Connectome,
    variant: str = "connectome",
    seed: int = 0,
    count_power: float = 0.5,
) -> torch.Tensor:
    """Build the fixed ``(n_kc, n_pn)`` PN -> KC weight matrix.

    ``count_power`` is the exponent applied to the measured synapse count before
    normalisation. The default 0.5 (square root) is a sub-linear weighting: a
    partner with 400 synapses is not 400 times as loud as a partner with one.
    Section 2.5 of the module docstring explains why, and why it is a modelling
    choice rather than a measurement.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown wiring variant %r, expected one of %s"
                         % (variant, VARIANTS))

    counts = conn.pn_to_kc.astype(np.float64)
    n_kc, n_pn = counts.shape

    w = np.power(counts, count_power)

    if variant == "connectome-shuffled":
        rng = np.random.default_rng(seed + 23)
        out = np.zeros_like(w)
        for row in range(n_kc):
            idx = np.flatnonzero(w[row])
            if idx.size == 0:
                continue
            # keep this KC's exact partner count and weight multiset, throw away
            # which PNs they were
            dest = rng.choice(n_pn, size=idx.size, replace=False)
            out[row, dest] = w[row, idx]
        w = out

    return torch.from_numpy(_row_normalise(w).astype(np.float32))
