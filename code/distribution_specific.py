import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
distribution_specific.py -- C11: which moment of the perturbation response is
reproducible?  mean vs variance vs zero-inflation vs (1-D) Wasserstein.

For each gene we compute the control->perturbed change in
  mean, variance, zero-fraction, and 1-D Wasserstein distance (quantile shift),
then measure the split-half reliability of each across perturbations (common mode
removed).  This isolates where the perturbation-specific signal lives.

Output: ../results/distribution_specific.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "distribution_specific.csv")
MAX_PERT, MAX_CELLS = 120, 500
QS = np.linspace(0.05, 0.95, 19)


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stats(E):
    return (E.mean(0), E.var(0, ddof=1), (E == 0).mean(0))


def _w1(E_pert, Q_ctrl):
    Qp = np.quantile(E_pert, QS, axis=0)          # (nq, p)
    return np.mean(np.abs(Qp - Q_ctrl), axis=0)


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Ec = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean, cvar, czero = _stats(Ec)
    Qc = np.quantile(Ec, QS, axis=0)
    del Ec
    gc.collect()

    M1, M2, V1, V2, Z1, Z2, W1, W2 = [], [], [], [], [], [], [], []
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
        E1 = np.asarray(X[h1].todense(), dtype=np.float64)
        E2 = np.asarray(X[h2].todense(), dtype=np.float64)
        m1, v1, z1 = _stats(E1); m2, v2, z2 = _stats(E2)
        M1.append(m1 - cmean); M2.append(m2 - cmean)
        V1.append(v1 - cvar); V2.append(v2 - cvar)
        Z1.append(z1 - czero); Z2.append(z2 - czero)
        W1.append(_w1(E1, Qc)); W2.append(_w1(E2, Qc))
    del X
    gc.collect()
    if len(M1) < 20:
        raise ValueError("too few")
    M1 = np.asarray(M1); M2 = np.asarray(M2)
    V1 = np.asarray(V1); V2 = np.asarray(V2)
    Z1 = np.asarray(Z1); Z2 = np.asarray(Z2)
    W1 = np.asarray(W1); W2 = np.asarray(W2)

    def cen(A):
        return A - A.mean(0)

    out = {"dataset": ds, "n_perts": len(M1),
           "rel_mean": _veccorr(cen(M1), cen(M2)),
           "rel_var": _veccorr(cen(V1), cen(V2)),
           "rel_zero": _veccorr(cen(Z1), cen(Z2)),
           "rel_w1": _veccorr(cen(W1), cen(W2))}
    return out


def main():
    d = _PERTURB_DATA
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "NormanWeissman2019_filtered.h5ad", "NadigOConner2024_jurkat.h5ad",
               "TianKampmann2021_CRISPRi.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "rel_mean", "rel_var", "rel_zero", "rel_w1"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<24} mean={:.3f} var={:.3f} zero={:.3f} W1={:.3f}".format(
                r["dataset"][:23], r["rel_mean"], r["rel_var"], r["rel_zero"], r["rel_w1"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
