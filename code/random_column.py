import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
random_column.py  (reviewer point 3)

Strengthens the "perturbed column ~ random column" claim:

  * column-column cosine distribution: cos(Sigma[:,g], Sigma[:,g']) over random
    gene pairs (are all covariance columns collinear?)
  * expression-MATCHED random columns: for each perturbation, draw 100 random
    genes from the same mean-expression decile and report the distribution of R2
  * the perturbed gene's R2 relative to that distribution

Output: random_column_results.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
N_DRAW = 100


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


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
    cmean = Xc.mean(0)
    Xc = Xc - cmean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    del Xc
    gc.collect()

    # gene mean-expression deciles (for matching)
    gmean = cmean
    dec = np.digitize(gmean, np.quantile(gmean, np.linspace(0, 1, 11)[1:-1]))
    by_dec = {d: np.where(dec == d)[0] for d in np.unique(dec)}

    # column-column cosine distribution (random pairs)
    ids = rng.integers(0, len(genes), size=(2000, 2))
    pair_cos = np.array([cos(cov[:, int(a)], cov[:, int(b)]) for a, b in ids if a != b])
    print("    [debug] pair_cos n={} nan={}".format(len(pair_cos), int(np.isnan(pair_cos).sum())), flush=True)

    per_pert_full, per_pert_rand, per_pert_ccos = [], [], []
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
        per_pert_full.append(cos(d, sig) ** 2)
        # 100 matched random columns
        pool = by_dec[dec[i]]
        pool = pool[pool != i]
        if len(pool) == 0:
            pool = np.arange(len(genes))
        draws = rng.choice(pool, size=min(N_DRAW, len(pool)), replace=False)
        per_pert_rand.append(np.mean([cos(d, cov[:, j]) ** 2 for j in draws]))
        j = rng.choice(pool)
        per_pert_ccos.append(cos(sig, cov[:, j]))
        del d, sig
    del X, cov
    gc.collect()
    return {
        "pair_cos_mean": float(np.nanmean(pair_cos)),
        "pair_cos_med": float(np.nanmedian(pair_cos)),
        "pair_cos_q05": float(np.nanpercentile(pair_cos, 5)),
        "pair_cos_q95": float(np.nanpercentile(pair_cos, 95)),
        "full": float(np.nanmean(per_pert_full)),
        "rand_matched": float(np.nanmean(per_pert_rand)),
        "col_cos": float(np.nanmean(per_pert_ccos)),
    }


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    keys = ["pair_cos_mean", "pair_cos_med", "pair_cos_q05", "pair_cos_q95",
            "full", "rand_matched", "col_cos"]
    with open("random_column_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    acc = {k: [] for k in keys}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("random_column_results.csv", "a") as fh:
                fh.write(name + "," + ",".join("{:.4f}".format(o[k]) for k in keys) + "\n")
            for k in keys:
                if not np.isnan(o[k]):
                    acc[k].append(o[k])
            print("OK   {:<30} full={:.3f} randMatch={:.3f} colCos(med)={:.3f} pairCos(med)={:.3f}".format(
                name[:29], o["full"], o["rand_matched"], o["col_cos"], o["pair_cos_med"]), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:30]), flush=True)
    print("\nMEANS:", flush=True)
    for k in keys:
        print("  {:<14} {:.4f}".format(k, np.mean(acc[k]) if acc[k] else float("nan")), flush=True)


if __name__ == "__main__":
    main()
