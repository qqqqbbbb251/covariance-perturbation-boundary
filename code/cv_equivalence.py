import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cv_equivalence.py  (reviewer points 4 and 5)

(4) Cross-validated nested model: for each perturbation, split its cells into
    train/test halves, fit [v, Sigma_perp[:,g]] on the train pseudobulk, evaluate
    on the test pseudobulk. Report CV dR2 for the real column and for a random
    column (same protocol).

(5) Equivalence testing of the full vs global-mode difference:
    per-perturbation differences -> mean effect + 95% CI (hierarchical bootstrap
    over datasets) + TOST with a pre-set margin (0.01 R2).

Output: cv_results.csv, equivalence_results.csv
"""

import gc
import glob
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 600, 2000, 120
MARGIN = 0.01          # TOST equivalence margin in R2 units


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def r2(y, pred):
    ss = float(y @ y)
    return 1.0 - float((y - pred) @ (y - pred)) / ss if ss else float("nan")


def fit(B, y):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return B @ coef


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    cidx = np.where(ctrl)[0]
    if len(cidx) > MAX_CTRL:
        cidx = rng.choice(cidx, MAX_CTRL, replace=False)
    base = raw[(~ctrl) & (nperts == 1)]
    if len(np.unique(base)) < 10:
        base = raw[~ctrl]
    uq, cn = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uq, cn) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    if len(singles) < 10 or len(cidx) < 100:
        raise ValueError("too few")

    sel = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    pos = {g: i for i, g in enumerate(a.var_names)}
    for g in singles:
        if g in pos:
            sel.add(pos[g])
    gi = np.array(sorted(sel))
    genes = np.asarray(a.var_names)[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    cmean = Xc.mean(0); Xc = Xc - cmean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
    del Xc, V, w
    gc.collect()

    cv_real, cv_shuf, diff_fg = [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) < 40:
            continue
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        perm = rng.permutation(len(rows))
        tr, te = rows[perm[: len(rows) // 2]], rows[perm[len(rows) // 2:]]
        Etr = np.asarray(X[tr].todense(), dtype=np.float32)
        Ete = np.asarray(X[te].todense(), dtype=np.float32)
        dtr = Etr.mean(0) - cmean
        dte = Ete.mean(0) - cmean
        del Etr, Ete
        if dte @ dte == 0:
            continue
        sig = cov[:, idx[g]]
        sigp = sig - float(sig @ v) * v
        g2 = singles[rng.integers(len(singles))]
        shp = cov[:, idx[g2]] - float(cov[:, idx[g2]] @ v) * v
        # nested CV: fit on train, evaluate on test
        pred = fit(np.stack([v, sigp], 1), dtr)
        r2n = r2(dte, pred)
        predv = fit(v[:, None], dtr)
        r2v = r2(dte, predv)
        cv_real.append(r2n - r2v)
        preds = fit(np.stack([v, shp], 1), dtr)
        cv_shuf.append(r2(dte, preds) - r2v)
        # full vs global on the test set (scale fit on train)
        a_full = float(sig @ dtr) / float(sig @ sig) if float(sig @ sig) else 0
        diff_fg.append(r2(dte, a_full * sig) - r2v)
        del dtr, dte, sig, sigp, shp
    del X, cov
    gc.collect()
    return (np.asarray(cv_real), np.asarray(cv_shuf), np.asarray(diff_fg))


def tost(d, margin=MARGIN):
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return float("nan")
    m = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p1 = 1 - _norm_cdf((m - margin) / se)      # H0: m >= margin
    p2 = _norm_cdf((m + margin) / se)          # H0: m <= -margin
    return max(p1, p2)


def _norm_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    cvrows, eqrows = [], []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            cvr, cvs, dfg = run(f)
            cvrows.append((name, np.nanmean(cvr), np.nanmean(cvs)))
            m, lo, hi = np.nanmean(dfg), *np.percentile(dfg[~np.isnan(dfg)], [2.5, 97.5])
            eqrows.append((name, m, lo, hi, tost(dfg)))
            print("OK   {:<28} cvReal={:+.3f} cvShuf={:+.3f} | dFullGlobal={:+.3f} [{:+.3f},{:+.3f}] TOST p={:.3g}".format(
                name[:27], cvrows[-1][1], cvrows[-1][2], m, lo, hi, eqrows[-1][4]), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:30]), flush=True)

    with open("cv_results.csv", "w") as fh:
        fh.write("dataset,cv_dR2_real,cv_dR2_shuf\n")
        for r in cvrows:
            fh.write("{},{:.4f},{:.4f}\n".format(*r))
    with open("equivalence_results.csv", "w") as fh:
        fh.write("dataset,mean_d_full_minus_global,lo,hi,tost_p\n")
        for r in eqrows:
            fh.write("{},{:+.4f},{:+.4f},{:+.4f},{:.3g}\n".format(*r))

    # hierarchical bootstrap over datasets for the mean difference
    if eqrows:
        means = np.array([r[1] for r in eqrows])
        rng = np.random.default_rng(0)
        boot = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(1)
        print("\ncv_dR2_real mean={:.4f}  cv_dR2_shuf mean={:.4f}".format(
            np.mean([r[1] for r in cvrows]), np.mean([r[2] for r in cvrows])))
        print("d(full-global) mean={:+.4f}  95% CI [{:+.4f},{:+.4f}]  TOST p={:.3g}".format(
            means.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5),
            tost(np.concatenate([np.full(1, m) for m in means]))))
    print("wrote cv_results.csv, equivalence_results.csv")


if __name__ == "__main__":
    main()
