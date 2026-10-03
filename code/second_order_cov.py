import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
second_order_cov.py -- C10b: do perturbations reproducibly change gene-gene *covariance*
(not just per-gene variance)?

For each dataset (raw, top-M most-variable genes) split each perturbation's cells into
halves, compute the within-group covariance change dSigma = Sigma_pert - Sigma_control in
each half, vectorise the strict upper triangle, and measure the split-half reliability
(full and after removing the common mode).  Compared with the per-gene-variance analogue.

Output: ../results/second_order_cov.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "second_order_cov.csv")
MAX_PERT, MAX_CELLS, M_GENES = 120, 500, 300
SEED = 0


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ut(v):
    return v[np.triu_indices_from(v, 1)]


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    # top-M genes by control variance
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    varg = Xc.var(0)
    top = np.argsort(-varg)[:M_GENES]
    cmean = Xc[:, top].mean(0)
    Zc = Xc[:, top] - cmean
    Sc = (Zc.T @ Zc) / (Zc.shape[0] - 1)
    del Xc, Zc
    gc.collect()

    D1, D2, V1, V2 = [], [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        if len(r) < 6:
            continue
        if len(r) > MAX_CELLS:
            r = rng.choice(r, MAX_CELLS, replace=False)
        r = rng.permutation(r)
        h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
        if len(h2) < 3:
            continue
        E1 = np.asarray(X[h1].todense(), dtype=np.float64)[:, top]
        E2 = np.asarray(X[h2].todense(), dtype=np.float64)[:, top]
        S1 = np.cov(E1, rowvar=False) - Sc
        S2 = np.cov(E2, rowvar=False) - Sc
        D1.append(_ut(S1)); D2.append(_ut(S2))
        V1.append(E1.var(0) - Sc.diagonal()); V2.append(E2.var(0) - Sc.diagonal())
    if len(D1) < 20:
        raise ValueError("too few")
    D1 = np.asarray(D1); D2 = np.asarray(D2); V1 = np.asarray(V1); V2 = np.asarray(V2)
    n = len(D1)
    perm = np.random.default_rng(SEED).permutation(n)

    def cen(A):
        return A - A.mean(0)

    out = {"dataset": ds, "n_perts": n, "M_genes": M_GENES,
           "cov_rel_full": _veccorr(D1, D2),
           "cov_rel_null_full": _veccorr(D1, D2[perm]),
           "cov_rel_spec": _veccorr(cen(D1), cen(D2)),
           "cov_rel_null_spec": _veccorr(cen(D1), cen(D2[perm])),
           "var_rel_spec": _veccorr(cen(V1), cen(V2)),
           "var_rel_null_spec": _veccorr(cen(V1), cen(V2[perm]))}
    del X, Sc, D1, D2, V1, V2
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "NormanWeissman2019_filtered.h5ad", "NadigOConner2024_jurkat.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "M_genes", "cov_rel_full", "cov_rel_null_full", "cov_rel_spec",
            "cov_rel_null_spec", "var_rel_spec", "var_rel_null_spec"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<24} covSpec={:.3f}(null {:.3f}) varSpec={:.3f} covFull={:.3f}".format(
                r["dataset"][:23], r["cov_rel_spec"], r["cov_rel_null_spec"],
                r["var_rel_spec"], r["cov_rel_full"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
