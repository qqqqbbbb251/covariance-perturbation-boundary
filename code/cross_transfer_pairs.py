import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cross_transfer_pairs.py -- cross-dataset specific-residual transfer for additional
pairs (identity vs learned Procrustes vs random), reusing specific_prediction's loader
and cross_predictions.

Usage: cross_transfer_pairs.py
Output: ../results/cross_transfer_pairs.csv
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import specific_prediction as sp  # noqa: E402

DATA = _PERTURB_DATA
OUT = os.path.join(HERE, "..", "results", "cross_transfer_pairs.csv")

PAIRS = [
    ("ReplogleWeissman2022_K562_essential.h5ad", "TianKampmann2021_CRISPRi.h5ad"),
    ("ReplogleWeissman2022_rpe1.h5ad", "TianKampmann2021_CRISPRi.h5ad"),
    ("NadigOConner2024_jurkat.h5ad", "TianKampmann2021_CRISPRi.h5ad"),
    ("FrangiehIzar2021_RNA.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
    ("FrangiehIzar2021_RNA.h5ad", "NadigOConner2024_jurkat.h5ad"),
    ("NormanWeissman2019_filtered.h5ad", "ReplogleWeissman2022_K562_essential.h5ad"),
]


def main():
    cache = {}
    rows = []
    for pa, pb in PAIRS:
        fa, fb = os.path.join(DATA, pa), os.path.join(DATA, pb)
        for f in (fa, fb):
            if f not in cache:
                cache[f] = sp.build_R(f)
                print("loaded", os.path.basename(f), flush=True)
        try:
            o = sp.cross_predictions(*cache[fa], *cache[fb])
            o["pair"] = os.path.basename(pa).replace(".h5ad", "") + "->" + os.path.basename(pb).replace(".h5ad", "")
            rows.append(o)
            print("OK {:<52} n={} identity={:.3f} procrustes={:.3f} random={:.3f}".format(
                o["pair"][:51], o["n_targets"], o["cos2_identity"], o["cos2_procrustes"],
                o["cos2_random"]), flush=True)
        except Exception as e:
            print("SKIP", pa, pb, str(e)[:60], flush=True)
    keys = ["pair", "n_targets", "n_common_genes", "cos2_identity", "cos2_procrustes", "cos2_random"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
