import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
random_column_stats.py  (reviewer point 3, strengthened)

The "perturbed column ~ random column" claim, made rigorous.

For every single perturbation we compare the perturbed gene's covariance column
against random columns drawn from genes MATCHED on three marginal statistics of
the control cells:
    * mean expression
    * variance
    * detection rate (fraction of nonzero cells)
Matching is by nearest-neighbour distance in the standardised (mean, var,
detection) space, so a random column has the same marginal statistics as the
perturbed gene's column.

Per perturbation:
    R2_full   = cos^2(delta, Sigma[:, g])
    {R2_rand} = cos^2(delta, Sigma[:, j]) for 100 matched random genes j
    percentile of R2_full inside the random distribution
    permutation p = (1 + #{R2_rand >= R2_full}) / (1 + n_draw)

Per dataset we report the full / random means, the random distribution's 5th,
50th and 95th percentiles, the mean percentile of the perturbed column, the
pooled permutation p, and a paired Wilcoxon p (full vs mean random).  The full
per-perturbation distributions are saved to random_column_dist.npz.

Output: ../results/random_column_stats.csv, ../results/random_column_dist.npz
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "random_column_stats.csv")
NPZ = os.path.join(RESULTS, "random_column_dist.npz")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
N_DRAW = 100
N_NEIGH = 50


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def wilcoxon(x, y):
    import math
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    W = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((W - mu) / sig) / math.sqrt(2))


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
    cmean = Xc.mean(0)
    cvar = Xc.var(0)
    cdet = (Xc > 0).mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    del Xc, Z
    gc.collect()

    # standardised (mean, var, detection) space for matching
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else np.zeros_like(x)
    F = np.stack([zscore(cmean), zscore(cvar), zscore(cdet)], 1).astype(np.float32)
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)   # squared distance, n_genes^2
    np.fill_diagonal(D, np.inf)
    neighbours = np.argsort(D, axis=1)[:, :N_NEIGH]      # nearest matched genes
    del D, F
    gc.collect()

    full, rand, pct, pperm = [], [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        Eg = np.asarray(X[rows].todense(), dtype=np.float32)
        d = Eg.mean(0) - cmean
        del Eg
        if d @ d == 0:
            continue
        i = idx[g]
        sig = cov[:, i]
        r2f = cos2(d, sig)
        pool = neighbours[i]
        draws = rng.choice(pool, size=min(N_DRAW, len(pool)), replace=False)
        rr = np.array([cos2(d, cov[:, int(j)]) for j in draws])
        full.append(r2f)
        rand.append(rr.mean())
        pct.append(float((rr < r2f).mean()))
        pperm.append(float((1 + np.sum(rr >= r2f)) / (1 + len(rr))))
        del d, sig, rr
    del X, cov
    gc.collect()
    if len(full) < 5:
        raise ValueError("too few usable perturbations")
    full = np.asarray(full)
    rand = np.asarray(rand)
    all_rand = np.concatenate([np.atleast_1d(rand)])
    return {
        "n_perts": len(full),
        "full_mean": float(np.nanmean(full)),
        "rand_mean": float(np.nanmean(rand)),
        "rand_q05": float(np.nanpercentile(rand, 5)),
        "rand_med": float(np.nanmedian(rand)),
        "rand_q95": float(np.nanpercentile(rand, 95)),
        "pct_mean": float(np.nanmean(pct)),
        "p_perm_pooled": float(np.nanmean(pperm)),
        "p_wilcoxon": wilcoxon(full, rand),
        "_full": full, "_rand": rand,
    }


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["n_perts", "full_mean", "rand_mean", "rand_q05", "rand_med", "rand_q95",
            "pct_mean", "p_perm_pooled", "p_wilcoxon"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    dist = {}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open(OUT, "a") as fh:
                fh.write(name + "," + ",".join(
                    "{:.4f}".format(o[k]) if k != "n_perts" else str(o[k]) for k in keys) + "\n")
            dist[name + "_full"] = o["_full"]
            dist[name + "_rand"] = o["_rand"]
            print("OK   {:<30} n={} full={:.3f} rand={:.3f}[{:.3f},{:.3f}] pct={:.2f} pperm={:.3f}".format(
                name[:29], o["n_perts"], o["full_mean"], o["rand_mean"], o["rand_q05"], o["rand_q95"],
                o["pct_mean"], o["p_perm_pooled"]), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:40]), flush=True)
    np.savez_compressed(NPZ, **dist)
    print("wrote", OUT, NPZ, flush=True)


if __name__ == "__main__":
    main()
