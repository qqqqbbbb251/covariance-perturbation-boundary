import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
paired_stats.py  (reviewer point 7)

Per-dataset paired tests (over perturbations) for
   full vs global mode
   full vs random covariance column
using a two-sided Wilcoxon signed-rank test (normal approximation), then a
Holm correction across datasets.

Output: paired_stats.csv
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
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def rankdata(a):
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[sorter] = np.arange(len(a))
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    r = rankdata(np.abs(d))
    W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sig
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
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

    full, glob, shuf = [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        Eg = np.asarray(X[rows].todense(), dtype=np.float32)
        d = Eg.mean(0) - cmean
        del Eg
        if d @ d == 0:
            continue
        sig = cov[:, idx[g]]
        g2 = singles[rng.integers(len(singles))]
        sig_shuf = cov[:, idx[g2]]
        if float(sig @ sig) == 0 or float(sig_shuf @ sig_shuf) == 0:
            del d
            continue
        full.append(cos2(d, sig))
        glob.append(float(d @ v) ** 2 / float(d @ d))
        shuf.append(cos2(d, sig_shuf))
        del d
    del X, cov
    gc.collect()
    return full, glob, shuf


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    rows = []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            full, globr, shuf = run(f)
            p_fg = wilcoxon(full, globr)
            p_fs = wilcoxon(full, shuf)
            rows.append((name, np.mean(full), np.mean(globr), np.mean(shuf),
                         np.mean(np.array(full) - np.array(globr)), p_fg, p_fs))
            print("OK   {:<30} full={:.3f} glob={:.3f} shuf={:.3f} d(full-glob)={:+.3f} p_fg={:.2g} p_fs={:.2g}".format(
                name[:29], rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4], p_fg, p_fs), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:30]), flush=True)

    p_fg = np.array([r[5] for r in rows])
    p_fs = np.array([r[6] for r in rows])
    p_fg_h = holm(p_fg); p_fs_h = holm(p_fs)
    with open("paired_stats.csv", "w") as fh:
        fh.write("dataset,full,global,shufcol,d_full_minus_global,p_full_vs_global,p_holm,p_full_vs_shuf,p_holm\n")
        for i, r in enumerate(rows):
            fh.write("{},{:.4f},{:.4f},{:.4f},{:+.4f},{:.3g},{:.3g},{:.3g},{:.3g}\n".format(
                r[0], r[1], r[2], r[3], r[4], r[5], p_fg_h[i], r[6], p_fs_h[i]))
    print("\nMEANS: full={:.3f} global={:.3f} shuf={:.3f}  d(full-global)={:+.3f}".format(
        np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]),
        np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows])))
    print("datasets where full > global (Holm p<0.05): {}".format(
        int(np.sum((p_fg_h < 0.05) & (np.array([r[4] for r in rows]) > 0)))))
    print("datasets where full > random col (Holm p<0.05): {}".format(
        int(np.sum((p_fs_h < 0.05) & (np.array([r[1] for r in rows]) > np.array([r[3] for r in rows]))))))
    print("wrote paired_stats.csv")


if __name__ == "__main__":
    main()
