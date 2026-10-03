import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
double_pert.py  (reviewer point 8)

Extends the single-perturbation critique to DOUBLE perturbations.

For each double (X, Y) with both single responses available:
  full        2-column fit [Sigma[:,X], Sigma[:,Y]]     (CIPHER's double model)
  global      1-column fit [v]
  global2     2-column fit [v, random direction]
  random_pair 2-column fit of a random gene pair
  additive    observed single responses d_X + d_Y
All scored with uncentered R2 on all genes (scale-only per column set).

Output: double_pert_results.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP = 3000, 500, 2000


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def r2(y, pred):
    ss = float(y @ y)
    return 1.0 - float((y - pred) @ (y - pred)) / ss if ss else float("nan")


def fit_r2(B, y):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return r2(y, B @ coef)


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
    singles = [g for g in np.unique(raw[nperts == 1]) if g.lower() not in CONTROL_TOKENS]
    doubles = [d for d in np.unique(raw[nperts == 2])]
    if len(doubles) < 5:
        raise ValueError("no doubles")

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

    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    cmean = Xc.mean(0); Xc = Xc - cmean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
    randdir = rng.normal(size=len(genes)).astype(np.float32)
    randdir = randdir / np.linalg.norm(randdir)
    del Xc, V, w
    gc.collect()

    def delta(mask):
        rows = np.where(mask)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        E = np.asarray(X[rows].todense(), dtype=np.float32)
        return E.mean(0) - cmean

    sd = {g: delta(raw == g) for g in singles if g in idx}

    res = {k: [] for k in ["full", "global", "global2", "random_pair", "additive"]}
    used = 0
    for dd in doubles:
        parts = dd.split("_")
        if len(parts) != 2 or parts[0] not in sd or parts[1] not in sd:
            continue
        A, B = parts
        if A not in idx or B not in idx:
            continue
        dXY = delta(raw == dd)
        if dXY @ dXY == 0:
            continue
        Mfull = np.stack([cov[:, idx[A]], cov[:, idx[B]]], 1)
        g1, g2 = singles[rng.integers(len(singles))], singles[rng.integers(len(singles))]
        Mrand = np.stack([cov[:, idx[g1]], cov[:, idx[g2]]], 1)
        res["full"].append(fit_r2(Mfull, dXY))
        res["global"].append(fit_r2(v[:, None], dXY))
        res["global2"].append(fit_r2(np.stack([v, randdir], 1), dXY))
        res["random_pair"].append(fit_r2(Mrand, dXY))
        res["additive"].append(r2(dXY, sd[A] + sd[B]))
        used += 1
    del X, cov
    gc.collect()
    return res, used


def _rankdata(a):
    a = np.asarray(a, float)
    s = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[s] = np.arange(len(a))
    as_ = a[s]
    obs = np.r_[True, as_[1:] != as_[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def wilcoxon(x, y):
    import math
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]; d = d[d != 0]
    n = len(d)
    if n < 6:
        return float("nan")
    r = _rankdata(np.abs(d)); W = r[d > 0].sum()
    mu = n * (n + 1) / 4.0; sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return math.erfc(abs((W - mu) / sig) / math.sqrt(2))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        os.path.join(d, "NormanWeissman2019_filtered.h5ad"),
        os.path.join(d, "TianKampmann2019_day7neuron.h5ad"),
        os.path.join(d, "TianKampmann2019_iPSC.h5ad"),
    ]
    keys = ["full", "global", "global2", "random_pair", "additive"]
    with open("double_pert_results.csv", "w") as fh:
        fh.write("dataset,n_doubles," + ",".join(keys) +
                 ",d_full_minus_rand,p_wilcoxon,ci_lo,ci_hi\n")
    all_full, all_rand = [], []
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            res, n = run(f)
            m, lo, hi = boot_ci(np.asarray(res["full"]) - np.asarray(res["random_pair"]))
            p = wilcoxon(res["full"], res["random_pair"])
            all_full += res["full"]; all_rand += res["random_pair"]
            with open("double_pert_results.csv", "a") as fh:
                fh.write("{},{},{},{:+.4f},{:.3g},{:+.4f},{:+.4f}\n".format(
                    name, n, ",".join("{:.4f}".format(np.nanmean(res[k])) for k in keys),
                    m, p, lo, hi))
            print("OK   {:<28} n={:<4} full={:.3f} global={:.3f} rand={:.3f} d={:+.4f} p={:.3g} CI[{:+.3f},{:+.3f}]".format(
                name[:27], n, np.nanmean(res["full"]), np.nanmean(res["global"]),
                np.nanmean(res["random_pair"]), m, p, lo, hi), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:30]), flush=True)
    if all_full:
        p = wilcoxon(all_full, all_rand)
        m, lo, hi = boot_ci(np.asarray(all_full) - np.asarray(all_rand))
        print("\nPOOLED (all doubles): full={:.4f} random_pair={:.4f} d={:+.4f} p={:.3g} CI[{:+.4f},{:+.4f}]".format(
            np.mean(all_full), np.mean(all_rand), m, p, lo, hi))
    print("wrote double_pert_results.csv")


if __name__ == "__main__":
    main()
