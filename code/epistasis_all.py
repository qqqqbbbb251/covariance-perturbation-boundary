import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
epistasis_all.py  (candidate 2, cross-dataset)

Tests, on every dataset that has true double perturbations, whether the
covariance model captures genetic interactions beyond a fairly-fitted additive
model built from the observed single responses.

For each double (A+B) with both singles present:
  add_fixed  = dA + dB
  single_fit = 2-param fit of [dA, dB]        -> observed singles
  cov_fit    = 2-param fit of [SigA, SigB]    -> covariance columns
and the same after removing the global mode from the double.

Usage:
  python epistasis_all.py
"""

import glob
import os

import numpy as np
import anndata as ad


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def wilcoxon_p(x, y):
    """two-sided Wilcoxon signed-rank p-value (normal approximation)."""
    from math import erfc, sqrt
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return float("nan")
    r = _rankdata(np.abs(d))
    W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    return erfc(abs(z) / sqrt(2))


CONTROL_LIKE = {"control", "ctrl", "ntc", "non-targeting", "nontargeting", "safe"}


def load(path, n_top=2000):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pert = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    var_ncounts = a.var["ncounts"].values.astype(float)
    var_names = np.asarray(a.var_names)
    ctrl = np.array([p.lower() in CONTROL_LIKE for p in pert]) | (nperts == 0)
    singles = [g for g in np.unique(pert[nperts == 1]) if g.lower() not in CONTROL_LIKE]
    doubles = [g for g in np.unique(pert[nperts == 2])]
    selected = set(np.argsort(-var_ncounts)[:n_top].tolist())
    pos = {g: i for i, g in enumerate(var_names)}
    for g in singles:
        if g in pos:
            selected.add(pos[g])
    gi = np.array(sorted(selected))
    Xm = a[:, gi].to_memory().X
    import scipy.sparse as sp
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    return X, var_names[gi], pert, nperts, ctrl, singles, doubles


def fit_r2(basis, y):
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    pred = basis @ coef
    ss = float(np.sum(y ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss if ss else float("nan")


def run(path):
    name = os.path.basename(path).replace(".h5ad", "")
    try:
        X, genes, pert, nperts, ctrl, singles, doubles = load(path)
    except Exception as e:
        print("{:<44} SKIP: {}".format(name[:43], str(e)[:30]))
        return
    idx = {g: i for i, g in enumerate(genes)}
    Xc = np.asarray(X[ctrl].todense(), dtype=np.float64)
    cmean = Xc.mean(0)
    cov = np.cov(Xc, rowvar=False)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)]; v = v / np.linalg.norm(v)
    del Xc

    sd = {}
    for g in singles:
        if g in idx:
            sd[g] = np.asarray(X[pert == g].todense(), dtype=np.float64).mean(0) - cmean

    add_f, sf, cf, sf_g, cf_g, epi = [], [], [], [], [], []
    for dd in doubles:
        parts = dd.split("_")
        if len(parts) != 2 or parts[0] not in sd or parts[1] not in sd:
            continue
        A, B = parts
        dA, dB = sd[A], sd[B]
        dAB = np.asarray(X[pert == dd].todense(), dtype=np.float64).mean(0) - cmean
        S = np.stack([dA, dB], 1)
        C = np.stack([cov[:, idx[A]], cov[:, idx[B]]], 1)
        dABg = dAB - (dAB @ v) * v
        add_f.append(1 - np.sum((dAB - (dA + dB)) ** 2) / np.sum(dAB ** 2))
        sf.append(fit_r2(S, dAB)); cf.append(fit_r2(C, dAB))
        sf_g.append(fit_r2(S, dABg)); cf_g.append(fit_r2(C, dABg))
        epi.append(np.linalg.norm(dAB - dA - dB) / np.linalg.norm(dAB))

    n = len(sf)
    if n < 5:
        print("{:<44} too few doubles ({})".format(name[:43], n))
        return
    p = wilcoxon_p(sf, cf)
    print("{:<44} n={:<4} addfix={:.3f} singlefit={:.3f} covfit={:.3f} | "
          "singlefit_g={:.3f} covfit_g={:.3f} | epi={:.3f} | p(sf>cf)={:.1e}".format(
              name[:43], n, np.mean(add_f), np.mean(sf), np.mean(cf),
              np.mean(sf_g), np.mean(cf_g), np.mean(epi), p))


def main():
    import sys
    d = _PERTURB_DATA
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(d, "*.h5ad")))
    print("{:<44}{:>6}{:>9}{:>11}{:>9}{:>13}{:>10}{:>7}{:>12}".format(
        "dataset", "n", "addfix", "singlefit", "covfit",
        "singlefit_g", "covfit_g", "epi", "p(sf>cf)"))
    print("-" * 118)
    for f in files:
        run(f)


if __name__ == "__main__":
    main()
