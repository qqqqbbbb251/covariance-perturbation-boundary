import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
residual_decomposition.py  (reviewer point 8, strengthened)

Explicitly decomposes each perturbation response in Pearson-residual space into
three orthogonal parts:

    d = self + shared + residual

  * self      = d_i * e_i                (the perturbed gene's own coordinate)
  * shared    = proj_{(d-self)} onto v   (the global mode)
  * residual  = what is left

and reports the fraction of ||d||^2 in each part.  It then asks whether the
perturbed gene's covariance column predicts the *residual* better than a matched
random column -- i.e. whether any gene-specific, non-shared, non-self signal
remains.  This separates "real but shared" from "the perturbed gene's own
auto-regulation" (trivial) from "gene-specific mechanism" (the interesting case).

Output: ../results/residual_decomposition.csv
"""

import gc
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "residual_decomposition.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
N_DRAW = 20


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
    w, V = np.linalg.eigh(cov)
    v = V[:, int(np.argmax(w))]
    del Yc, V, w
    gc.collect()

    dec = np.digitize(cmean, np.quantile(cmean, np.linspace(0, 1, 11)[1:-1]))
    by_dec = {d: np.where(dec == d)[0] for d in np.unique(dec)}

    rs, rsh, rf = [], [], []
    full, rand = [], []
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
        dd = float(d @ d)
        self_c = np.zeros_like(d); self_c[i] = d[i]
        rest = d - self_c
        shared_c = (rest @ v) * v
        resid = rest - shared_c
        rs.append(float(self_c @ self_c) / dd)
        rsh.append(float(shared_c @ shared_c) / dd)
        rf.append(float(resid @ resid) / dd)
        if resid @ resid > 0:
            full.append(cos2(resid, cov[:, i]))
            pool = by_dec[dec[i]]; pool = pool[pool != i]
            if len(pool) == 0:
                pool = np.arange(len(genes))
            draws = rng.choice(pool, size=min(N_DRAW, len(pool)), replace=False)
            rand.append(float(np.mean([cos2(resid, cov[:, j]) for j in draws])))
        del d, self_c, rest, shared_c, resid
    del X, cov
    gc.collect()
    if len(rs) < 10:
        raise ValueError("too few")
    full = np.asarray(full); rand = np.asarray(rand)
    return {
        "n_perts": len(rs),
        "R2_self": float(np.nanmean(rs)),
        "R2_shared": float(np.nanmean(rsh)),
        "R2_residual": float(np.nanmean(rf)),
        "resid_full": float(np.nanmean(full)),
        "resid_random": float(np.nanmean(rand)),
        "d_resid": float(np.nanmean(full) - np.nanmean(rand)),
        "p_wilcoxon": wilcoxon(full, rand),
    }


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["n_perts", "R2_self", "R2_shared", "R2_residual", "resid_full",
            "resid_random", "d_resid", "p_wilcoxon"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open(OUT, "a") as fh:
                fh.write(name + "," + ",".join(
                    str(o[k]) if k == "n_perts" else "{:.4f}".format(o[k]) for k in keys) + "\n")
            print("OK   {:<28} self={:.3f} shared={:.3f} resid={:.3f} | residFull={:.3f} rand={:.3f} p={:.3g}".format(
                name[:27], o["R2_self"], o["R2_shared"], o["R2_residual"],
                o["resid_full"], o["resid_random"], o["p_wilcoxon"]), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:40]), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
