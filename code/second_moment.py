import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
second_moment.py -- C10: is there a reproducible perturbation-specific *dispersion*
(variance) response, separate from the mean shift?

For each perturbation, split its cells into halves and compute the change in per-gene
variance relative to control (dvar = var_pert - var_control) in each half.  Then measure
the split-half reliability of dvar (full and after removing the common mode), and compare
it to the analogous reliability of the mean shift.  If dvar is reproducible, distributions
(not just means) carry perturbation-specific information that mean-based models miss.

Output: ../results/second_moment.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "second_moment.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
SEED = 0


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sim_upper(R):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    return (Rn @ Rn.T)[np.triu_indices(R.shape[0], 1)]


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    cvar = Xc.var(0, ddof=1)
    del Xc
    gc.collect()

    V1, V2, M1, M2 = [], [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        if len(r) < 6:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        r = rng.permutation(r)
        h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
        if len(h2) < 3:
            continue
        E1 = np.asarray(X[h1].todense(), dtype=np.float64)
        E2 = np.asarray(X[h2].todense(), dtype=np.float64)
        V1.append(E1.var(0, ddof=1) - cvar)
        V2.append(E2.var(0, ddof=1) - cvar)
        M1.append(E1.mean(0) - cmean)
        M2.append(E2.mean(0) - cmean)
    V1 = np.asarray(V1); V2 = np.asarray(V2); M1 = np.asarray(M1); M2 = np.asarray(M2)
    n = len(V1)

    perm = np.random.default_rng(SEED).permutation(n)

    def cen(A):
        return A - A.mean(0)

    out = {
        "dataset": ds, "n_perts": n,
        "var_rel_full": _veccorr(V1, V2),
        "var_rel_null_full": _veccorr(V1, V2[perm]),
        "var_rel_spec": _veccorr(cen(V1), cen(V2)),
        "var_rel_null_spec": _veccorr(cen(V1), cen(V2[perm])),
        "var_rsa_spec": _veccorr(_sim_upper(cen(V1)), _sim_upper(cen(V2))),
        "mean_rel_spec": _veccorr(cen(M1), cen(M2)),
        "mean_rsa_spec": _veccorr(_sim_upper(cen(M1)), _sim_upper(cen(M2))),
    }
    del X, V1, V2, M1, M2
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "var_rel_full", "var_rel_null_full", "var_rel_spec",
            "var_rel_null_spec", "var_rsa_spec", "mean_rel_spec", "mean_rsa_spec"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<24} varSpec={:.3f}(null {:.3f}) varRSA={:.3f} | meanSpec={:.3f} "
                  "meanRSA={:.3f}".format(
                      r["dataset"][:23], r["var_rel_spec"], r["var_rel_null_spec"],
                      r["var_rsa_spec"], r["mean_rel_spec"], r["mean_rsa_spec"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
