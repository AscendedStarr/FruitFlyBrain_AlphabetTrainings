"""Extract the mushroom-body subgraph from the hemibrain connectome.

Source
------
Scheffer et al. 2020, *A connectome and analysis of the adult Drosophila central
brain*, eLife. Data: hemibrain v1.2, downloaded from the public Janelia bucket
(no account or API token required):

    https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz

The archive holds three files, written by neuprint-python's
``fetch_traced_adjacencies``:

    traced-neurons.csv            bodyId, type, instance
    traced-total-connections.csv  bodyId_pre, bodyId_post, weight
    traced-roi-connections.csv    the same, split by brain region

This script keeps only the neurons belonging to the olfactory associative-
learning pathway and writes them as one compressed ``.npz``. It does not touch
the model: it produces a file, nothing more.

Licence
-------
The hemibrain dataset is released under **CC BY 4.0** (verified from the eLife
article metadata for doi:10.7554/eLife.57443: ``copyright.license = CC-BY-4.0``,
"unrestricted use and redistribution provided the original author and source are
credited"). Attribution is in ``CITATION.cff`` and ``README.md``. Note this is
*more* permissive than the FlyWire whole-brain dataset, which is CC BY-NC 4.0.

Extraction rule
---------------
Cell classes are taken from the ``type`` column of ``traced-neurons.csv``, which
carries the community cell-type annotations:

    KC    type startswith "KC"        Kenyon cells (the mushroom-body intrinsic
                                     neurons, and the fan-out of this circuit)
    MBON  type startswith "MBON"      mushroom-body output neurons
    PN    "PN" in type                projection neurons - a *superset*, see below
    DAN   type startswith "PPL"/"PAM" dopaminergic neurons innervating the MB
    APL   type == "APL"               the single giant GABAergic interneuron

Nothing is guessed and nothing is imputed: a neuron is in the subgraph because
its annotated type says it is, and the counts printed below are the counts in
the file.

The one place a name is not enough is the projection neurons, and the export
makes that obvious once you look. The loose ``"PN" in type`` test matches 428
neurons, but 271 of them have **no traced synapse onto any Kenyon cell at all**
- while still having 39 to 780 outgoing synapses elsewhere. They are the
``mPN`` (multiglomerular) and ``WEDPN`` (wedge) neurons, which project to the
lateral horn rather than the mushroom-body calyx. That is correct Drosophila
anatomy, not a tracing gap. Cross-checking against the curated antennal-lobe
table in ``flyconnectome/hemibrain_olf_data`` (Schlegel et al. 2021, S5) agrees:
118 of 181 uPN have calyx output, only 26 of 166 mPN do.

So membership of *this* circuit is decided by connectivity rather than
nomenclature, which needs no judgement call: a projection neuron is kept iff at
least one of its traced synapses lands on a Kenyon cell, and a Kenyon cell is
kept iff at least one projection neuron reaches it. This leaves 157 PNs and
1802 KCs. It is also the only self-consistent choice for the model - a PN with
no KC output cannot change the circuit's behaviour, so wiring one up as a silent
input channel would be manufacturing anatomy the fly does not have.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "_connectome", "raw")
DEFAULT_OUT = os.path.join(ROOT, "data", "hemibrain_mb.npz")

BUCKET = "https://storage.googleapis.com/hemibrain"
ARCHIVE = "v1.2/exported-traced-adjacencies-v1.2.tar.gz"
EXTRACTED = "exported-traced-adjacencies-v1.2"


# --------------------------------------------------------------------- classes
def classify(cell_type: str) -> str | None:
    """Map one annotated cell type onto a circuit class, or None if unrelated."""
    t = cell_type or ""
    if t.startswith("KC"):
        return "KC"
    if t.startswith("MBON"):
        return "MBON"
    if t == "APL":
        return "APL"
    if t.startswith("PPL") or t.startswith("PAM"):
        return "DAN"
    if "PN" in t:
        return "PN"
    return None


def fetch(force: bool = False) -> str:
    """Download and extract the adjacency export, returning its directory."""
    os.makedirs(RAW, exist_ok=True)
    target = os.path.join(RAW, EXTRACTED)
    if os.path.isdir(target) and not force:
        return target

    tgz = os.path.join(RAW, os.path.basename(ARCHIVE))
    if force or not os.path.exists(tgz):
        url = "%s/%s" % (BUCKET, ARCHIVE)
        print("downloading %s" % url)
        urllib.request.urlretrieve(url, tgz)
    print("extracting %s" % tgz)
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(RAW)
    return target


# ------------------------------------------------------------------- subgraph
def load_types(path: str) -> dict[str, tuple[str, str]]:
    """bodyId -> (type, instance)."""
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["bodyId"]] = (row["type"], row["instance"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-extract the raw archive")
    ap.add_argument("--raw-dir", default=None,
                    help="use an already-extracted directory instead of downloading")
    args = ap.parse_args()

    directory = args.raw_dir or fetch(force=args.force)
    neurons_csv = os.path.join(directory, "traced-neurons.csv")
    conn_csv = os.path.join(directory, "traced-total-connections.csv")
    for path in (neurons_csv, conn_csv):
        if not os.path.exists(path):
            print("missing %s" % path)
            return 1

    types = load_types(neurons_csv)
    print("neurons with an annotated type: %d" % len(types))

    # bodyId -> class, and the ordered id list per class (sorted for determinism)
    by_class: dict[str, dict[str, str]] = {c: {} for c in ("PN", "KC", "MBON", "DAN")}
    for body_id, (cell_type, _instance) in types.items():
        cls = classify(cell_type)
        if cls and cls in by_class:
            by_class[cls][body_id] = cell_type

    for cls in ("PN", "KC", "MBON", "DAN"):
        n = len(by_class[cls])
        n_types = len(set(by_class[cls].values()))
        print("  %-5s %5d neurons over %3d types" % (cls, n, n_types))
    apl = [b for b, (t, _) in types.items() if t == "APL"]
    print("  %-5s %5d neuron" % ("APL", len(apl)))

    pn_ids = sorted(by_class["PN"], key=int)
    kc_ids = sorted(by_class["KC"], key=int)
    mbon_ids = sorted(by_class["MBON"], key=int)
    dan_ids = sorted(by_class["DAN"], key=int)

    pn_ix = {b: i for i, b in enumerate(pn_ids)}
    kc_ix = {b: i for i, b in enumerate(kc_ids)}
    mbon_ix = {b: i for i, b in enumerate(mbon_ids)}
    dan_ix = {b: i for i, b in enumerate(dan_ids)}

    pn_to_kc = np.zeros((len(kc_ids), len(pn_ids)), dtype=np.int16)
    kc_to_mbon = np.zeros((len(mbon_ids), len(kc_ids)), dtype=np.int16)
    dan_to_kc = np.zeros((len(dan_ids), len(kc_ids)), dtype=np.int16)

    with open(conn_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pre, post = row["bodyId_pre"], row["bodyId_post"]
            w = int(row["weight"])
            if post in kc_ix:
                k = kc_ix[post]
                if pre in pn_ix:
                    pn_to_kc[k, pn_ix[pre]] += w
                if pre in dan_ix:
                    dan_to_kc[dan_ix[pre], k] += w
            if pre in kc_ix:
                k = kc_ix[pre]
                if post in mbon_ix:
                    kc_to_mbon[mbon_ix[post], k] += w

    # --- the calyx filter ----------------------------------------------------
    # Naming alone is not enough to say a neuron is part of this circuit, and
    # the names in this export prove it: of the 166 `mPN` (multiglomerular)
    # projection neurons, 140 have no traced synapse onto a KC, and the 91
    # `WEDPN*` (wedge) neurons caught by the loose "PN" substring test have
    # none either. They project to the lateral horn, not to the mushroom-body
    # calyx - which is correct Drosophila anatomy, not a gap in the tracing.
    #
    # So membership here is decided by connectivity, not nomenclature: a
    # projection neuron is part of this circuit iff at least one traced synapse
    # of it lands on a Kenyon cell, and a Kenyon cell is part of it iff at least
    # one PN reaches it. That rule is empirical, it needs no judgement call, and
    # it is self-consistent with the model - a PN with no KC output cannot
    # change the circuit's behaviour, so wiring one up as an input would be
    # manufacturing a silent channel that the fly does not have.
    pn_keep = np.flatnonzero(pn_to_kc.sum(axis=0) > 0)
    kc_keep = np.flatnonzero(pn_to_kc.sum(axis=1) > 0)
    n_pn_cand, n_kc_cand = pn_to_kc.shape[1], pn_to_kc.shape[0]

    print()
    print("=== calyx filter (connectivity, not nomenclature) ===")
    print("  PN candidates       : %4d -> %4d kept (%d have no traced KC output)"
          % (n_pn_cand, pn_keep.size, n_pn_cand - pn_keep.size))
    print("  KC candidates       : %4d -> %4d kept (%d receive no traced PN input)"
          % (n_kc_cand, kc_keep.size, n_kc_cand - kc_keep.size))
    print("  KC -> MBON columns kept anyway: all (every KC here feeds >=1 MBON: %s)"
          % bool((kc_to_mbon[:, kc_keep] > 0).any(axis=0).all()))

    pn_to_kc = pn_to_kc[np.ix_(kc_keep, pn_keep)]
    kc_to_mbon = kc_to_mbon[:, kc_keep]
    dan_to_kc = dan_to_kc[:, kc_keep]
    pn_ids = [pn_ids[i] for i in pn_keep]
    kc_ids = [kc_ids[i] for i in kc_keep]

    # Optional: annotate each kept PN with its curated class (uPN = uniglomerular,
    # mPN = multiglomerular, biPN = bilateral) from S5_ALPN_meta.csv, so the
    # composition of the input layer is on the record rather than asserted.
    alpn_class = {}
    s5 = os.path.join(directory, "..", "S5_ALPN_meta.csv")
    if os.path.exists(s5):
        with open(s5, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                alpn_class[row["bodyid"]] = row.get("class") or "NA"
        counts = collections.Counter(alpn_class.get(b, "not-in-ALPN-table") for b in pn_ids)
        print("  PN layer composition: %s"
              % ", ".join("%s=%d" % kv for kv in sorted(counts.items())))
    else:
        counts = collections.Counter()

    fan_in = (pn_to_kc > 0).sum(axis=1)
    meta = {
        "dataset": "hemibrain",
        "version": "v1.2",
        "source": "%s/%s" % (BUCKET, ARCHIVE),
        "licence": "CC BY 4.0",
        "citation": "Scheffer et al. 2020, eLife 9:e57443, doi:10.7554/eLife.57443",
        "extraction_rule": {
            "KC": 'type startswith "KC"',
            "MBON": 'type startswith "MBON"',
            "PN": '"PN" in type',
            "DAN": 'type startswith "PPL" or "PAM"',
            "APL": 'type == "APL"',
        },
        "calyx_filter": (
            "Membership of the circuit is decided by connectivity, not "
            "nomenclature: a PN is kept iff it has >=1 traced synapse onto a KC, "
            "and a KC is kept iff >=1 PN reaches it. The loose '\"PN\" in type' "
            "test over-includes non-calyx projection neurons - 140 of the 166 "
            "mPN (multiglomerular, lateral-horn-projecting) neurons and all 91 "
            "WEDPN* (wedge) neurons have zero KC output."
        ),
        "composition": {
            "n_pn_candidates": int(n_pn_cand),
            "n_pn_kept": int(pn_keep.size),
            "n_kc_candidates": int(n_kc_cand),
            "n_kc_kept": int(kc_keep.size),
            "pn_curated_class": {str(k): int(v) for k, v in counts.items()},
        },
        "n_neurons_typed": len(types),
        "n_apl": len(apl),
        "measured": {
            "pn_to_kc_synapses": int(pn_to_kc.sum()),
            "pn_to_kc_edges": int((pn_to_kc > 0).sum()),
            "kc_fan_in_min": int(fan_in.min()),
            "kc_fan_in_median": float(np.median(fan_in)),
            "kc_fan_in_mean": float(fan_in.mean()),
            "kc_fan_in_max": int(fan_in.max()),
            "kc_to_mbon_synapses": int(kc_to_mbon.sum()),
            "kc_to_mbon_edges": int((kc_to_mbon > 0).sum()),
            "dan_to_kc_synapses": int(dan_to_kc.sum()),
            "dan_to_kc_edges": int((dan_to_kc > 0).sum()),
        },
    }

    print()
    print("=== extracted mushroom-body subgraph ===")
    print("PN  -> KC  : %8s edges %10s synapses, fan-in %d/%g/%d (min/median/max)"
          % (format(meta["measured"]["pn_to_kc_edges"], ","),
             format(meta["measured"]["pn_to_kc_synapses"], ","),
             meta["measured"]["kc_fan_in_min"],
             meta["measured"]["kc_fan_in_median"],
             meta["measured"]["kc_fan_in_max"]))
    print("KC  -> MBON: %8s edges %10s synapses"
          % (format(meta["measured"]["kc_to_mbon_edges"], ","),
             format(meta["measured"]["kc_to_mbon_synapses"], ",")))
    print("DAN -> KC  : %8s edges %10s synapses"
          % (format(meta["measured"]["dan_to_kc_edges"], ","),
             format(meta["measured"]["dan_to_kc_synapses"], ",")))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        pn_to_kc=pn_to_kc,
        kc_to_mbon=kc_to_mbon,
        dan_to_kc=dan_to_kc,
        pn_id=np.array(pn_ids, dtype=np.int64),
        kc_id=np.array(kc_ids, dtype=np.int64),
        mbon_id=np.array(mbon_ids, dtype=np.int64),
        dan_id=np.array(dan_ids, dtype=np.int64),
        pn_type=np.array([by_class["PN"][b] for b in pn_ids]),
        kc_type=np.array([by_class["KC"][b] for b in kc_ids]),
        mbon_type=np.array([by_class["MBON"][b] for b in mbon_ids]),
        dan_type=np.array([by_class["DAN"][b] for b in dan_ids]),
        meta=np.array(json.dumps(meta, indent=2)),
    )
    size = os.path.getsize(args.out)
    print()
    print("wrote %s (%.1f KB)" % (args.out, size / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
