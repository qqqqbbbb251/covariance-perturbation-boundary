import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
covariance_estimators.py -- C13: can a better *covariance estimator* let the operator
see the perturbation-specific residual?

For each dataset (raw) we rebuild the forward operator from several estimators of the
control covariance and ask whether its column predicts the (global-axis- and self-removed)
specific residual better than a random column:
  sample        : raw sample covariance
  shrinkI       : convex shrinkage toward mean-diagonal * I
  clipMP        : Marchenko-Pastur-style eigenvalue clipping (bulk -> bulk mean)
  factorK       : keep top-K eigenmodes, replace the rest by their mean
  rank1         : keep only the top eigenmode
Estimators are built from one eigendecomposition (no extra factorisation).

Output: ../results/covariance_estimators.csv
"""

import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from differential_identity import load  # noqa: E402

RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(RESULTS, "covariance_estimators.csv")
MAX_PERT, MAX_PERT_CELLS = 150, 500
SEED = 0
SHRINK = 0.3


def _proj_cos2(r, p):
    pp = float(p @ p); rr = float(r @ r)
    return float(r @ p) ** 2 / (rr * pp) if pp > 0 and rr > 0 else 0.0


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    p, n = cov.shape[0], Z.shape[0]
    w, V = np.linalg.eigh(cov)          # ascending
    w = np.maximum(w, 0.0)
    v = V[:, int(np.argmax(w))]
    del Xc, Z
    gc.collect()

    est = {}
    est["sample"] = w.copy()
    md = float(w.mean())
    est["shrinkI"] = (1 - SHRINK) * w + SHRINK * md
    gamma = md * (1 + np.sqrt(p / max(n, 1))) ** 2
    wc = w.copy(); bulk = w < gamma
    if bulk.sum():
        wc[bulk] = w[bulk].mean()
    est["clipMP"] = wc
    for K in (5, 20):
        wk = w.copy()
        if p - K > 0:
            wk[:p - K] = w[:p - K].mean()
        est["factor%d" % K] = wk
    wr = np.full(p, md); wr[-1] = w[-1]
    est["rank1"] = wr

    # residual responses
    R, tc = [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < p) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        d = d - float(d @ v) * v
        d[t] = 0.0
        R.append(d); tc.append(t)
    R = np.asarray(R); tc = np.asarray(tc)
    del X, cov
    gc.collect()

    rng2 = np.random.default_rng(SEED)
    rows = []
    for name, we in est.items():
        # column estimator: Sigma_est[:,g] = V diag(we) V^T e_g = V (we * V[g,:])
        cs, csr = [], []
        for i in range(len(R)):
            t = tc[i]
            col = V @ (we * V[t, :])
            col = col - float(col @ v) * v
            rc = int(rng2.integers(p))
            colr = V @ (we * V[rc, :])
            colr = colr - float(colr @ v) * v
            cs.append(_proj_cos2(R[i], col)); csr.append(_proj_cos2(R[i], colr))
        rows.append({"dataset": ds, "estimator": name, "n_perts": len(R),
                     "sv1_frac": float(w[-1] / w.sum()),
                     "cos2_off": float(np.mean(cs)), "cos2_off_rand": float(np.mean(csr))})
    return rows


def main():
    d = _PERTURB_DATA
    default = ["ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
               "NormanWeissman2019_filtered.h5ad"]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_perts", "sv1_frac", "cos2_off", "cos2_off_rand"]
    with open(OUT, "w") as fh:
        fh.write("dataset,estimator," + ",".join(keys) + "\n")
    for f in files:
        try:
            rows = analyze(os.path.join(d, f))
            for r in rows:
                with open(OUT, "a") as fh:
                    fh.write(r["dataset"] + "," + r["estimator"] + "," + ",".join(
                        ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
                print("OK {:<24} {:<9} cos2_off={:.3f} rand={:.3f}".format(
                    r["dataset"][:23], r["estimator"], r["cos2_off"], r["cos2_off_rand"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
