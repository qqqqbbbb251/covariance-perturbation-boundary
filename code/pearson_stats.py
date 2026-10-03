import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
pearson_stats.py  (reviewer point 8, strengthened)

The weak gene-specific signal appears only in Pearson-residual space.  We
quantify it with:

  * bootstrap 95% CI for the perturbed gene's column R2 (resampling perturbations),
  * a paired Wilcoxon test (full vs matched random column),
  * Benjamini-Hochberg FDR across datasets,
  * a NEGATIVE CONTROL: the same pipeline run on a "pseudo-perturbation" made of
    a random subset of CONTROL cells (same size as the real perturbation), i.e. a
    delta with no perturbation signal.  If the method is well calibrated the
    negative-control R2 is ~0; the fraction of control draws exceeding the real
    R2 gives a per-perturbation permutation p-value.

Output: ../results/pearson_stats.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "pearson_stats.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
N_BOOT, N_NULL = 10000, 20


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def rankdata(a):
    a = np.asarray(a, float)
    s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int)
    inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def wilcoxon(x, y):
    import math
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
    return math.erfc(abs((W - mu) / sig) / math.sqrt(2))


def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    qv = ranked * m / (np.arange(1, m + 1))
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    out = np.empty(m)
    out[order] = qv
    q[idx] = out
    return q


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
    cmean = Yc.mean(0)
    Yc = Yc - cmean
    cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
    del Yc, Xc
    gc.collect()

    # mean-expression deciles over the selected genes for matched random columns
    dec = np.digitize(cmean, np.quantile(cmean, np.linspace(0, 1, 11)[1:-1]))
    by_dec = {d: np.where(dec == d)[0] for d in np.unique(dec)}

    full, rand, selfz, negc = [], [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        Eg = np.asarray(X[rows].todense(), dtype=np.float32)
        d = pearson(Eg, nc_all[rows]).mean(0) - cmean
        del Eg
        if d @ d == 0:
            continue
        i = idx[g]
        sig = cov[:, i]
        r2f = cos2(d, sig)
        full.append(r2f)
        pool = by_dec[dec[i]]
        pool = pool[pool != i]
        if len(pool) == 0:
            pool = np.arange(len(genes))
        j = int(rng.choice(pool))
        rand.append(cos2(d, cov[:, j]))
        sig0 = sig.copy()
        sig0[i] = 0.0
        selfz.append(cos2(d, sig0))
        # negative control: pseudo-delta from a random control subset
        nn = []
        for _ in range(N_NULL):
            sub = rng.choice(cidx, size=len(rows), replace=(len(rows) > len(cidx)))
            Es = np.asarray(X[sub].todense(), dtype=np.float32)
            dn = pearson(Es, nc_all[sub]).mean(0) - cmean
            nn.append(cos2(dn, sig))
        negc.append(float(np.mean(nn)))
        del d, sig, sig0
    del X, cov
    gc.collect()
    full = np.asarray(full)
    if len(full) < 10:
        raise ValueError("too few usable perturbations")
    m, lo, hi = boot_ci(full)
    return {
        "n": len(full),
        "full_mean": m, "full_lo": lo, "full_hi": hi,
        "rand_mean": float(np.nanmean(rand)),
        "selfzero_mean": float(np.nanmean(selfz)),
        "negctl_mean": float(np.nanmean(negc)),
        "d_full_rand": float(np.nanmean(full) - np.nanmean(rand)),
        "p_full_rand": wilcoxon(full, rand),
        "p_full_negctl": wilcoxon(full, negc),
    }


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["n", "full_mean", "full_lo", "full_hi", "rand_mean", "selfzero_mean",
            "negctl_mean", "d_full_rand", "p_full_rand", "p_full_negctl"]
    rows = []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            rows.append((name, o))
            print("OK   {:<28} n={} full={:.3f}[{:.3f},{:.3f}] rand={:.3f} self0={:.3f} neg={:.3f} p={:.3g}".format(
                name[:27], o["n"], o["full_mean"], o["full_lo"], o["full_hi"],
                o["rand_mean"], o["selfzero_mean"], o["negctl_mean"], o["p_full_rand"]), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:40]), flush=True)
    q = bh_fdr([o["p_full_rand"] for _, o in rows])
    qn = bh_fdr([o["p_full_negctl"] for _, o in rows])
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + ",p_full_rand_BH,p_full_negctl_BH\n")
        for (name, o), qq, qqn in zip(rows, q, qn):
            fh.write(name + "," + ",".join(
                "{:.4f}".format(o[k]) if k != "n" else str(o[k]) for k in keys)
                + ",{:.3g},{:.3g}\n".format(qq, qqn))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
