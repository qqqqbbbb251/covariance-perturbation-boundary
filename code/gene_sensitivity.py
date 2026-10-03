import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
gene_sensitivity.py -- robustness of the specific-structure result to the gene panel.

Re-runs the split-half, centred, self-removed RSA reliability while varying the number
of top-expressed genes used (differential_identity.N_TOP): 500 / 1000 / 2000 / 10000
(the last = essentially all expressed genes for these datasets).

Output: ../results/gene_sensitivity.csv
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import differential_identity as di  # noqa: E402
import residual_reproducibility as rr  # noqa: E402

DATA = _PERTURB_DATA
OUT = os.path.join(HERE, "..", "results", "gene_sensitivity.csv")
NTOPS = [500, 1000, 2000, 10000]


def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
    ]
    rows = []
    for f in files:
        for n in NTOPS:
            di.N_TOP = n
            try:
                res = rr.analyze(os.path.join(DATA, f))
            except Exception as e:
                print("SKIP", f, n, str(e)[:50], flush=True)
                continue
            for r in res:
                rows.append({"dataset": r["dataset"], "space": r["space"], "N_TOP": n,
                             "n_genes": r["n_genes"], "rsa_selfrem_spec": r["rsa_selfrem_spec"],
                             "rsa_resid": r["rsa_resid"]})
            got = {r["space"]: r["rsa_selfrem_spec"] for r in res}
            print("OK {:<26} N={:<6} n_genes={} raw={:.3f} cpm={:.3f} pearson={:.3f}".format(
                os.path.basename(f)[:25], n, res[0]["n_genes"] if res else "-",
                got.get("raw", float("nan")), got.get("cpm", float("nan")),
                got.get("pearson", float("nan"))), flush=True)
    with open(OUT, "w") as fh:
        fh.write("dataset,space,N_TOP,n_genes,rsa_selfrem_spec,rsa_resid\n")
        for r in rows:
            fh.write("{dataset},{space},{N_TOP},{n_genes},{rsa_selfrem_spec:.4f},{rsa_resid:.4f}\n".format(**r))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
