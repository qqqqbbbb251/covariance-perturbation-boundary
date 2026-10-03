import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
ci_bootstrap.py

Bootstrap 95% confidence intervals for the key per-dataset metrics:
  * orthogonal specific correlation (raw, CPM, deconfounded)
Resampling is over perturbations (10,000 resamples).

Output: ci_results.csv
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
N_BOOT = 10000


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def cpm(E, nc):
    return E * (1e4 / np.maximum(nc, 1.0))[:, None]


def fit_beta(E, covars):
    C = np.column_stack([np.ones(len(E), dtype=np.float32)] + covars)
    G = C.T @ C + 1e-6 * np.eye(C.shape[1], dtype=np.float32)
    return np.linalg.solve(G, C.T @ E)


def per_pert(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
    ng_all = obs["ngenes"].values.astype(np.float32)
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

    Xc_raw = np.asarray(X[cidx].todense(), dtype=np.float32)
    nc_c, ng_c = nc_all[cidx], ng_all[cidx]
    cmean_raw = Xc_raw.mean(0)

    def collect(space, deconf=False):
        if deconf:
            beta = fit_beta(Xc_raw, [np.log1p(nc_c), np.log1p(ng_c)])
            Xc = Xc_raw - np.column_stack([np.ones(len(Xc_raw), np.float32),
                                           np.log1p(nc_c), np.log1p(ng_c)]) @ beta
            cmean = Xc.mean(0)
            Xc = Xc - cmean
        else:
            Y = Xc_raw if space == "raw" else cpm(Xc_raw, nc_c)
            cmean = Y.mean(0)
            Xc = Y - cmean
        cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
        out = []
        for g in singles:
            rows = np.where(raw == g)[0]
            if len(rows) > MAX_PERT_CELLS:
                rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
            Eg = np.asarray(X[rows].todense(), dtype=np.float32)
            if deconf:
                Cg = np.column_stack([np.ones(len(Eg), np.float32),
                                      np.log1p(nc_all[rows]), np.log1p(ng_all[rows])])
                d = (Eg - Cg @ beta).mean(0) - cmean
            else:
                Yg = Eg if space == "raw" else cpm(Eg, nc_all[rows])
                d = Yg.mean(0) - cmean
            d = d - float(d @ v) * v
            sig = cov[:, idx[g]]
            dd = float(sig @ sig)
            aa = float(sig @ d / dd) if dd else 0
            out.append(pearson(d, aa * sig))
            del Eg, d
        return np.asarray(out)

    return {"raw": collect("raw"), "cpm": collect("cpm"), "deconf": collect("raw", True)}


def boot_ci(x, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    header = "dataset,raw_mean,raw_lo,raw_hi,cpm_mean,cpm_lo,cpm_hi,deconf_mean,deconf_lo,deconf_hi\n"
    with open("ci_results.csv", "w") as fh:
        fh.write(header)
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            pp = per_pert(f)
            cells = []
            for k in ["raw", "cpm", "deconf"]:
                m, lo, hi = boot_ci(pp[k])
                cells += ["{:.4f}".format(v) for v in (m, lo, hi)]
            with open("ci_results.csv", "a") as fh:
                fh.write(name + "," + ",".join(cells) + "\n")
            print("OK   {:<32} raw {:.3f}[{:.3f},{:.3f}] deconf {:.3f}[{:.3f},{:.3f}]".format(
                name[:31], *[float(c) for c in cells[0:3]], *[float(c) for c in cells[6:9]]),
                flush=True)
        except Exception as e:
            print("SKIP {:<32} {}".format(name[:31], str(e)[:30]), flush=True)


if __name__ == "__main__":
    main()
