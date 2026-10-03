import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
deconfound_final.py  (v4 - correct + bounded memory)

Can the perturbation-specific signal be recovered by removing the
cell-complexity confound more aggressively than plain CPM?

Bounded memory:
  * <= 3000 control cells, <= 500 cells per perturbation
  * float32
  * residualisation via 3x3 normal equations
  * the sparse expression matrix is kept; each perturbation group is densified
    per variant and freed immediately (never cached)

Variants: raw, cpm, cpm_ngenes, cpm_cov, raw_cov, cpm_pc1, cpm_pc2
Output  : deconfound_results.csv
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
VARIANTS = ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov", "cpm_pc1", "cpm_pc2"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def cpm(E, ncounts):
    return E * (1e4 / np.maximum(ncounts, 1.0))[:, None]


def fit_beta(E, covars):
    C = np.column_stack([np.ones(len(E), dtype=np.float32)] + covars)
    G = C.T @ C + 1e-6 * np.eye(C.shape[1], dtype=np.float32)
    return np.linalg.solve(G, C.T @ E)


def transform(E, ncounts, ngenes, variant, beta):
    if variant == "raw":
        return E
    Y = cpm(E, ncounts)
    if variant == "cpm":
        return Y
    if variant == "cpm_ngenes":
        C = np.column_stack([np.ones(len(E), dtype=np.float32), ngenes])
        return Y - C @ beta
    if variant in ("cpm_cov", "raw_cov"):
        C = np.column_stack([np.ones(len(E), dtype=np.float32),
                             np.log1p(ncounts), np.log1p(ngenes)])
        return (Y if variant == "cpm_cov" else E) - C @ beta
    if variant in ("cpm_pc1", "cpm_pc2"):
        return Y
    raise ValueError(variant)


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    nc_all = obs["ncounts"].values.astype(np.float32)
    ng_all = obs["ngenes"].values.astype(np.float32)
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    rng = np.random.default_rng(0)
    ctrl_idx = np.where(ctrl)[0]
    if len(ctrl_idx) > MAX_CTRL:
        ctrl_idx = rng.choice(ctrl_idx, MAX_CTRL, replace=False)

    base = raw[(~ctrl) & (nperts == 1)]
    if len(np.unique(base)) < 10:
        base = raw[~ctrl]
    uniq, cnt = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uniq, cnt) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(rng.choice(singles, MAX_PERT, replace=False))
    if len(singles) < 10 or len(ctrl_idx) < 100:
        raise ValueError("too few")

    selected = set(np.argsort(-var_ncounts)[:N_TOP].tolist())
    pos = {g: i for i, g in enumerate(a.var_names)}
    for g in singles:
        if g in pos:
            selected.add(pos[g])
    gi = np.array(sorted(selected))
    genes = np.asarray(a.var_names)[gi]
    Xm = a[:, gi].to_memory().X
    X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
    del a, Xm
    gc.collect()
    idx = {g: i for i, g in enumerate(genes)}
    singles = [g for g in singles if g in idx]

    gidx = {}
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        gidx[g] = rows

    Xc_raw = np.asarray(X[ctrl_idx].todense(), dtype=np.float32)
    nc_c, ng_c = nc_all[ctrl_idx], ng_all[ctrl_idx]

    out = {}
    for variant in VARIANTS:
        beta = None
        if variant == "cpm_ngenes":
            beta = fit_beta(cpm(Xc_raw, nc_c), [ng_c])
        elif variant == "cpm_cov":
            beta = fit_beta(cpm(Xc_raw, nc_c), [np.log1p(nc_c), np.log1p(ng_c)])
        elif variant == "raw_cov":
            beta = fit_beta(Xc_raw, [np.log1p(nc_c), np.log1p(ng_c)])

        Yc = transform(Xc_raw, nc_c, ng_c, variant, beta)
        cmean = Yc.mean(0)
        Yc = Yc - cmean
        cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
        if variant in ("cpm_pc1", "cpm_pc2"):
            k = 1 if variant == "cpm_pc1" else 2
            w, V = np.linalg.eigh(cov)
            o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
            w2 = w.copy(); w2[:k] = 0
            cov = (V * w2) @ V.T
            del V, w
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
        del V, w, Yc
        gc.collect()

        cs = []
        for g in singles:
            rows = gidx[g]
            Eg = np.asarray(X[rows].todense(), dtype=np.float32)
            Yg = transform(Eg, nc_all[rows], ng_all[rows], variant, beta)
            d = Yg.mean(0) - cmean
            d = d - float(d @ v) * v
            sig = cov[:, idx[g]]
            dd = float(sig @ sig)
            a_ = float(sig @ d / dd) if dd else 0.0
            cs.append(pearson(d, a_ * sig))
            del Eg, Yg, d
        out[variant] = float(np.nanmean(cs)) if cs else float("nan")
        del cov, cmean, beta
        gc.collect()
    del X, Xc_raw
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    with open("deconfound_results.csv", "w") as fh:
        fh.write("dataset," + ",".join(VARIANTS) + "\n")
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            r = run(f)
            with open("deconfound_results.csv", "a") as fh:
                fh.write(name + "," + ",".join(
                    "{:.4f}".format(r.get(v, float("nan"))) for v in VARIANTS) + "\n")
            print("OK   {:<38} ".format(name[:37]) + " ".join(
                "{}={:.3f}".format(v, r.get(v, float("nan"))) for v in VARIANTS), flush=True)
        except Exception as e:
            print("SKIP {:<38} {}".format(name[:37], str(e)[:30]), flush=True)


if __name__ == "__main__":
    main()
