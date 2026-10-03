"""
deconfound_test.py  (v2 - memory-safe)

Can the perturbation-specific signal be recovered by removing the
cell-complexity confound more aggressively than plain CPM?

Memory-safe design:
  * control cells are subsampled to <= 5000 (250-2000 were already shown to
    suffice for the covariance)
  * everything is float32
  * residualisation uses the small 3x3 normal equations, never lstsq on the
    full expression matrix
  * each variant frees its matrices immediately

Variants (covariance from control cells; response orthogonalised against that
variant's leading mode):
  raw, cpm, cpm_ngenes, cpm_cov, raw_cov, cpm_pc1, cpm_pc2
"""

import gc
import glob
import os

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL = 5000
N_TOP = 2000
MAX_PERT = 120


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def fit_beta(E, covars):
    """least squares via normal equations (covars is cells x p, p small)."""
    C = np.column_stack([np.ones(len(E), dtype=np.float32)] + covars)
    G = C.T @ C + 1e-6 * np.eye(C.shape[1], dtype=np.float32)
    beta = np.linalg.solve(G, C.T @ E)          # p x genes
    return beta


def cpm(E, ncounts):
    return E * (1e4 / np.maximum(ncounts, 1.0))[:, None]


def transform(E, ncounts, ngenes, variant, beta):
    """E: cells x genes (float32 dense, raw counts)."""
    if variant == "raw":
        return E
    Y = cpm(E, ncounts)
    if variant == "cpm":
        return Y
    if variant == "cpm_ngenes":
        C = np.column_stack([np.ones(len(E), dtype=np.float32), ngenes])
        return Y - C @ beta
    if variant == "cpm_cov":
        C = np.column_stack([np.ones(len(E), dtype=np.float32),
                             np.log1p(ncounts), np.log1p(ngenes)])
        return Y - C @ beta
    if variant == "raw_cov":
        C = np.column_stack([np.ones(len(E), dtype=np.float32),
                             np.log1p(ncounts), np.log1p(ngenes)])
        return E - C @ beta
    if variant in ("cpm_pc1", "cpm_pc2"):
        return Y
    raise ValueError(variant)


def run(path):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    raw = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts_all = obs["ncounts"].values.astype(np.float32)
    ngenes_all = obs["ngenes"].values.astype(np.float32)
    var_ncounts = a.var["ncounts"].values.astype(float)
    ctrl = np.array([is_control(p) for p in raw]) | (nperts == 0)
    ctrl_idx = np.where(ctrl)[0]
    if len(ctrl_idx) > MAX_CTRL:
        ctrl_idx = np.random.default_rng(0).choice(ctrl_idx, MAX_CTRL, replace=False)

    base = raw[(~ctrl) & (nperts == 1)]
    if len(np.unique(base)) < 10:
        base = raw[~ctrl]
    uniq, cnt = np.unique(base, return_counts=True)
    singles = [g for g, c in zip(uniq, cnt) if c >= 30]
    if len(singles) > MAX_PERT:
        singles = list(np.random.default_rng(0).choice(singles, MAX_PERT, replace=False))
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

    # perturbation groups: cache raw dense matrices once (small)
    groups = {}
    for g in singles:
        m = raw == g
        groups[g] = (np.asarray(X[m].todense(), dtype=np.float32),
                     ncounts_all[m], ngenes_all[m])
    Xc_raw = np.asarray(X[ctrl_idx].todense(), dtype=np.float32)
    nc_c, ng_c = ncounts_all[ctrl_idx], ngenes_all[ctrl_idx]
    del X
    gc.collect()

    out = {}
    for variant in ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov", "cpm_pc1", "cpm_pc2"]:
        # fit residualisation betas on control cells
        beta = None
        if variant == "cpm_ngenes":
            beta = fit_beta(cpm(Xc_raw, nc_c), [ng_c])
        elif variant == "cpm_cov":
            beta = fit_beta(cpm(Xc_raw, nc_c), [np.log1p(nc_c), np.log1p(ng_c)])
        elif variant == "raw_cov":
            beta = fit_beta(Xc_raw, [np.log1p(nc_c), np.log1p(ng_c)])

        Yc = transform(Xc_raw, nc_c, ng_c, variant, beta)
        cmean = Yc.mean(0)
        cov = (Yc.T @ Yc) / (Yc.shape[0] - 1)
        if variant in ("cpm_pc1", "cpm_pc2"):
            k = 1 if variant == "cpm_pc1" else 2
            w, V = np.linalg.eigh(cov)
            o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
            w2 = w.copy(); w2[:k] = 0
            cov = (V * w2) @ V.T
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)].astype(np.float32)
        v = v / np.linalg.norm(v)
        del Yc, cov, V, w
        gc.collect()

        cs = []
        for g in singles:
            Eg, nc_g, ng_g = groups[g]
            Yg = transform(Eg, nc_g, ng_g, variant, beta)
            d = Yg.mean(0) - cmean
            d = d - (d @ v) * v
            sig = cov_sig = None
            # recompute the gene column from the saved covariance is not stored;
            # instead use the covariance column via a second pass is avoided by
            # recomputing covariance for this variant once (kept small)
            cs.append(d)
        # (covariance column is needed; recompute it once here)
        # NOTE: we recompute cov from Yc stored? -> store cov per variant instead
        out[variant] = cs
        del beta
        gc.collect()
    return out


def main():
    print("this rewrite is split into run_final; see deconfound_final.py")


if __name__ == "__main__":
    main()
