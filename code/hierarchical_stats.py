import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
hierarchical_stats.py  (reviewer point 9)

Hierarchical bootstrap and power analysis for the full-vs-global comparison.
Perturbations are nested within datasets, so we resample datasets and then
perturbations within datasets.

Also saves per-perturbation R2 (full / global / random) for reuse.

Output: per_pert_r2.npz, hierarchical_stats.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 150


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def per_pert(path):
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
    gmean = Xc.mean(0)
    gvar = Xc.var(0)
    gdet = (Xc > 0).mean(0)
    cmean = gmean
    Xc = Xc - cmean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
    del Xc, V, w
    gc.collect()

    # matched-random pools: nearest neighbours in standardised (mean, var, detection)
    def _z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else np.zeros_like(x)
    F = np.stack([_z(gmean), _z(gvar), _z(gdet)], 1).astype(np.float32)
    n_genes = F.shape[0]
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    col_norm = np.einsum("ij,ij->j", cov, cov)
    invalid = col_norm <= 0
    D[:, invalid] = np.inf
    np.fill_diagonal(D, np.inf)
    neighbours = np.argsort(D, axis=1)[:, :50]
    del D, F
    gc.collect()

    full, glob, rand = [], [], []
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
        if float(sig @ sig) == 0:
            continue
        full.append(cos2(d, sig))
        glob.append(float(d @ v) ** 2 / float(d @ d))
        pool = neighbours[idx[g]]
        j = int(pool[rng.integers(len(pool))])
        rand.append(cos2(d, cov[:, j]))
        del d
    del X, cov
    gc.collect()
    return np.asarray(full), np.asarray(glob), np.asarray(rand)


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    store = {}
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            full, globr, rand = per_pert(f)
            if len(full) >= 10:
                store[name] = (full, globr, rand)
                print("OK   {:<30} n={} full={:.3f} glob={:.3f} rand={:.3f}".format(
                    name[:29], len(full), full.mean(), globr.mean(), rand.mean()), flush=True)
        except Exception as e:
            print("SKIP {:<30} {}".format(name[:29], str(e)[:30]), flush=True)

    np.savez_compressed("per_pert_r2.npz", **{k: np.vstack(v) for k, v in store.items()})
    names = list(store)
    ds_mean = np.array([np.nanmean(store[k][0] - store[k][1]) for k in names])   # full - global
    ds_mean_fr = np.array([np.nanmean(store[k][0] - store[k][2]) for k in names])  # full - random

    # hierarchical bootstrap: resample datasets, then perturbations within
    rng = np.random.default_rng(0)
    B = 10000
    boot_fg = np.empty(B); boot_fr = np.empty(B); boot_gr = np.empty(B)
    for b in range(B):
        ds_idx = rng.integers(0, len(names), len(names))
        vals_fg, vals_fr, vals_gr = [], [], []
        for di in ds_idx:
            arr = store[names[di]]
            n = len(arr[0])
            p = rng.integers(0, n, n)
            vals_fg.append(np.nanmean(arr[0][p] - arr[1][p]))
            vals_fr.append(np.nanmean(arr[0][p] - arr[2][p]))
            vals_gr.append(np.nanmean(arr[1][p] - arr[2][p]))
        boot_fg[b] = np.nanmean(vals_fg)
        boot_fr[b] = np.nanmean(vals_fr)
        boot_gr[b] = np.nanmean(vals_gr)

    with open("hierarchical_stats.csv", "w") as fh:
        fh.write("comparison,mean,lo,hi,width\n")
        for name, arr in [("full-global", boot_fg), ("full-random", boot_fr),
                          ("global-random", boot_gr)]:
            fh.write("{},{:+.4f},{:+.4f},{:+.4f},{:.4f}\n".format(
                name, np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5),
                np.percentile(arr, 97.5) - np.percentile(arr, 2.5)))
    print("\nHierarchical bootstrap ({} datasets):".format(len(names)))
    print("  full-global:   mean={:+.4f} 95%CI [{:+.4f},{:+.4f}]".format(
        np.mean(boot_fg), np.percentile(boot_fg, 2.5), np.percentile(boot_fg, 97.5)))
    print("  full-random:   mean={:+.4f} 95%CI [{:+.4f},{:+.4f}]".format(
        np.mean(boot_fr), np.percentile(boot_fr, 2.5), np.percentile(boot_fr, 97.5)))
    print("  global-random: mean={:+.4f} 95%CI [{:+.4f},{:+.4f}]".format(
        np.mean(boot_gr), np.percentile(boot_gr, 2.5), np.percentile(boot_gr, 97.5)))

    # power analysis: to detect |full-global| within margin 0.01 (equivalence),
    # or to detect a nonzero difference, how many perturbations/datasets needed?
    sd_pert = np.mean([np.std(store[k][0] - store[k][1]) for k in names])
    print("\nPower analysis (per-perturbation SD of full-global = {:.3f}):".format(sd_pert))
    for n in [50, 100, 200, 500]:
        se = sd_pert / np.sqrt(n)
        print("  n_pert={:<4} SE={:.4f}  (95% CI half-width ~{:.4f})".format(n, se, 1.96 * se))
    print("wrote per_pert_r2.npz, hierarchical_stats.csv")


if __name__ == "__main__":
    main()
