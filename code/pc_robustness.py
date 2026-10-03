import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
pc_robustness.py -- H24: is the reproducible specific structure just deeper nuisance
axes of the control covariance?

Recompute the split-half, centered, self-removed RSA reliability after removing the top-k
control-covariance principal components (k = 1, 3, 5, 10, 20) from each half shift.  If
the structure collapses as k grows, it is nuisance; if it survives, it is genuine
perturbation-specific structure orthogonal to the dominant control axes.

Output: ../results/pc_robustness.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "pc_robustness.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
KS = [1, 3, 5, 10, 20]
SEED = 0


def _sim_upper(R):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Rn @ Rn.T
    return S[np.triu_indices(R.shape[0], 1)]


def _veccorr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)          # ascending
    PCs = V[:, ::-1]                    # descending
    del Xc, Z
    gc.collect()

    D1raw, D2raw, tc = [], [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        r = rng.permutation(r)
        h1, h2 = r[:len(r) // 2], r[len(r) // 2:]
        d1 = np.asarray(X[h1].todense(), dtype=np.float64).mean(0) - cmean
        d2 = np.asarray(X[h2].todense(), dtype=np.float64).mean(0) - cmean
        if d1 @ d1 == 0 or d2 @ d2 == 0:
            continue
        D1raw.append(d1); D2raw.append(d2); tc.append(t)
    D1raw = np.asarray(D1raw); D2raw = np.asarray(D2raw); tc = np.asarray(tc)

    rows = []
    for k in KS:
        P = PCs[:, :k]
        D1 = D1raw - (D1raw @ P) @ P.T
        D2 = D2raw - (D2raw @ P) @ P.T
        for i, t in enumerate(tc):
            if 0 <= t:
                D1[i, t] = 0.0
                D2[i, t] = 0.0
        D1 = D1 - D1.mean(0); D2 = D2 - D2.mean(0)
        rel = _veccorr(_sim_upper(D1), _sim_upper(D2))
        rows.append({"dataset": ds, "k_pcs": k, "rsa_selfrem_spec": rel})
    del X, cov, PCs, D1raw, D2raw
    gc.collect()
    return rows


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
    with open(OUT, "w") as fh:
        fh.write("dataset,k_pcs,rsa_selfrem_spec\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write("{dataset},{k_pcs},{rsa_selfrem_spec:.4f}\n".format(**r))
            byk = {r["k_pcs"]: r["rsa_selfrem_spec"] for r in rows}
            print("OK {:<24} " .format(byk and rows[0]["dataset"][:23]) +
                  " ".join("k%d=%.3f" % (k, byk[k]) for k in KS), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
