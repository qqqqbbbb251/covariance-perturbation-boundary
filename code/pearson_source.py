import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
pearson_source.py  (reviewer point 8)

The weak gene-specific signal appears only in Pearson-residual space. Two checks:
  (a) bootstrap CI and a paired test for full vs random column in Pearson space
  (b) is the signal coming from the perturbed gene's OWN entry (its diagonal
      covariance, i.e. the gene changing itself) rather than a transcriptome-wide
      mechanism? We recompute R2 with the perturbed gene's own entry zeroed out.

Output: pearson_source.csv
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
    s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]; d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    r = rankdata(np.abs(d)); W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((W - mu) / sig) / math.sqrt(2))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


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
    nc_c = nc_all[cidx]
    sf_c = nc_c / (nc_c.mean() + 1e-9)
    mu = (Xc / sf_c[:, None]).mean(0)

    def pearson(E, nc):
        sf = nc / (nc.mean() + 1e-9)
        muij = sf[:, None] * mu[None, :]
        return (E - muij) / np.sqrt(muij + 1e-6)

    Yc = pearson(Xc, nc_c)
    cmean = Yc.mean(0); Yc = Yc - cmean
    cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
    del Yc
    gc.collect()

    full, shuf, selfz = [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        Eg = np.asarray(X[rows].todense(), dtype=np.float32)
        d = pearson(Eg, nc_all[rows]).mean(0) - cmean
        del Eg
        if d @ d == 0:
            continue
        sig = cov[:, idx[g]]
        g2 = singles[rng.integers(len(singles))]
        full.append(cos2(d, sig))
        shuf.append(cos2(d, cov[:, idx[g2]]))
        sig0 = sig.copy(); sig0[idx[g]] = 0.0        # remove the perturbed gene's own entry
        selfz.append(cos2(d, sig0))
        del d, sig, sig0
    del X, cov
    gc.collect()
    return np.asarray(full), np.asarray(shuf), np.asarray(selfz)


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    with open("pearson_source.csv", "w") as fh:
        fh.write("dataset,full_mean,full_lo,full_hi,shuf_mean,selfzero_mean,"
                 "d_full_minus_shuf,p_wilcoxon\n")
    full_all, shuf_all, self_all = [], [], []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            full, shuf, selfz = run(f)
            m, lo, hi = boot_ci(full)
            p = wilcoxon(full, shuf)
            with open("pearson_source.csv", "a") as fh:
                fh.write("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:+.4f},{:.3g}\n".format(
                    name, m, lo, hi, np.nanmean(shuf), np.nanmean(selfz),
                    np.nanmean(full) - np.nanmean(shuf), p))
            full_all.append(np.nanmean(full)); shuf_all.append(np.nanmean(shuf))
            self_all.append(np.nanmean(selfz))
            print("OK   {:<28} full={:.3f}[{:.3f},{:.3f}] shuf={:.3f} selfZero={:.3f} p={:.3g}".format(
                name[:27], m, lo, hi, np.nanmean(shuf), np.nanmean(selfz), p), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:30]), flush=True)
    if full_all:
        print("\nMEANS: full={:.4f} shuf={:.4f} selfZero={:.4f}".format(
            np.nanmean(full_all), np.nanmean(shuf_all), np.nanmean(self_all)))


if __name__ == "__main__":
    main()
