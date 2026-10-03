import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
specific_organization.py -- Stage 2c: is the specific structure organised by target
gene function?

For K562 / rpe1 / Norman (raw): build the (global-axis- and self-removed) residual
response R for the perturbation targets, then ask whether target genes that are
co-expressed in control cells have more similar specific residuals:
  * correlation between the target-target expression-similarity matrix (from the
    control covariance) and the residual-similarity matrix (cosine), with a
    permutation p-value;
  * "copy the most co-expressed target's residual" predictor vs a random target.

Output: ../results/specific_organization.csv
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
OUT = os.path.join(RESULTS, "specific_organization.csv")

MAX_PERT = 150
MAX_PERT_CELLS = 500
N_NULL = 200
SEED = 0


def analyze(path):
    X, genes, idx, singles, cidx, nc_all, raw, rng = load(path)
    ds = os.path.basename(path).replace(".h5ad", "")
    Xc = np.asarray(X[cidx].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Xc, Z, V, w
    gc.collect()

    R, tcols = [], []
    for g in singles[:MAX_PERT]:
        r = np.where(raw == g)[0]
        t = idx.get(g, -1)
        if not (0 <= t < cov.shape[0]) or len(r) < 2:
            continue
        if len(r) > MAX_PERT_CELLS:
            r = rng.choice(r, MAX_PERT_CELLS, replace=False)
        d = np.asarray(X[r].todense(), dtype=np.float64).mean(0) - cmean
        if d @ d == 0:
            continue
        d = d - float(d @ v) * v
        d[t] = 0.0
        R.append(d); tcols.append(t)
    R = np.asarray(R); tcols = np.asarray(tcols)
    n = len(R)

    # expression similarity among targets (control covariance -> correlation)
    sub = cov[np.ix_(tcols, tcols)]
    dg = np.sqrt(np.maximum(np.diag(sub), 1e-12))
    Sexp = sub / np.outer(dg, dg)
    np.fill_diagonal(Sexp, 0.0)

    # residual similarity
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    Sres = Rn @ Rn.T
    np.fill_diagonal(Sres, 0.0)

    iu = np.triu_indices(n, 1)
    x = Sexp[iu]; y = Sres[iu]
    m = np.isfinite(x) & np.isfinite(y)
    obs = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 10 else float("nan")
    rs = np.random.default_rng(SEED)
    null = []
    for _ in range(N_NULL):
        p = rs.permutation(n)
        yp = Sres[np.ix_(p, p)][iu]
        mm = np.isfinite(x) & np.isfinite(yp)
        if mm.sum() > 10:
            null.append(np.corrcoef(x[mm], yp[mm])[0, 1])
    null = np.asarray([q for q in null if np.isfinite(q)])
    p_val = float(np.mean(np.abs(null) >= abs(obs))) if null.size else float("nan")

    # copy the most co-expressed target's residual
    best = np.argmax(Sexp, axis=1)
    cs_best = float(np.mean([float(R[i] @ R[best[i]]) ** 2 /
                             ((R[i] @ R[i]) * (R[best[i]] @ R[best[i]]) + 1e-12) for i in range(n)]))
    rand_j = rs.integers(0, n, size=n)
    cs_rand = float(np.mean([float(R[i] @ R[rand_j[i]]) ** 2 /
                             ((R[i] @ R[i]) * (R[rand_j[i]] @ R[rand_j[i]]) + 1e-12) for i in range(n)]))
    cs_allmean = float(np.mean(np.sum(Rn @ Rn.T, axis=1) / (n - 1)))

    del X, cov, R
    gc.collect()
    return {"dataset": ds, "n_targets": n,
            "corr_expr_resid_sim": obs, "corr_null_mean": float(np.nanmean(null)) if null.size else float("nan"),
            "corr_p": p_val,
            "cos2_copy_coexpr": cs_best, "cos2_copy_random": cs_rand,
            "cos2_all_targets": cs_allmean}


def main():
    d = _PERTURB_DATA
    default = [
        "ReplogleWeissman2022_K562_essential.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "NormanWeissman2019_filtered.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["n_targets", "corr_expr_resid_sim", "corr_null_mean", "corr_p",
            "cos2_copy_coexpr", "cos2_copy_random", "cos2_all_targets"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<28} corr(expr,resid)={:.3f} (p={:.3f}) copy_coexpr={:.3f} "
                  "copy_rand={:.3f} all={:.3f}".format(
                      r["dataset"][:27], r["corr_expr_resid_sim"], r["corr_p"],
                      r["cos2_copy_coexpr"], r["cos2_copy_random"], r["cos2_all_targets"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
