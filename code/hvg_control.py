import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
hvg_control.py  (A: gene-selection control)

Does the conclusion depend on selecting the most highly expressed genes?
Repeat the key metrics under three gene sets:
  top    - 2000 most highly expressed genes (as in the main analysis)
  hvg    - 2000 most variable genes across control cells
  random - 2000 random genes
plus the perturbed genes in each case.

Output: hvg_results.csv
"""

import gc
import glob
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, N_GENES, MAX_PERT, MAX_PERT_CELLS = 3000, 2000, 120, 500


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")


def metrics(X, genes, idx, raw, ctrl_idx, singles, gidx, ctrl_mean, cov, v):
    cmean = ctrl_mean
    full, spec = [], []
    for g in singles:
        Eg = np.asarray(X[gidx[g]].todense(), dtype=np.float32)
        d = Eg.mean(0) - cmean
        del Eg
        if d @ d == 0:
            continue
        sig = cov[:, idx[g]]
        dd = float(sig @ sig)
        a = float(sig @ d / dd) if dd else 0
        full.append(1 - float((d - a * sig) @ (d - a * sig)) / float(d @ d))
        do = d - float(d @ v) * v
        a2 = float(sig @ do / dd) if dd else 0
        spec.append(pearson(do, a2 * sig))
        del d, do
    return float(np.nanmean(full)), float(np.nanmean(spec))


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

    name_to_pos = {g: i for i, g in enumerate(a.var_names)}
    n_all = a.n_vars
    # HVG: variance across control cells over ALL genes
    Xall = a[cidx, :].to_memory().X
    Xall = (Xall.tocsr() if sp.issparse(Xall) else sp.csr_matrix(Xall)).astype(np.float32)
    var = np.asarray(Xall.power(2).mean(0)).ravel() - np.asarray(Xall.mean(0)).ravel() ** 2
    del Xall
    gc.collect()

    gene_sets = {}
    gene_sets["top"] = set(np.argsort(-var_ncounts)[:N_GENES].tolist())
    gene_sets["hvg"] = set(np.argsort(-var)[:N_GENES].tolist())
    gene_sets["random"] = set(rng.choice(n_all, N_GENES, replace=False).tolist())

    out = {}
    for tag, gset in gene_sets.items():
        gset = set(gset)
        for g in singles:
            if g in name_to_pos:
                gset.add(name_to_pos[g])
        gi = np.array(sorted(gset))
        genes = np.asarray(a.var_names)[gi]
        Xm = a[:, gi].to_memory().X
        X = (Xm.tocsr() if sp.issparse(Xm) else sp.csr_matrix(Xm)).astype(np.float32)
        idx = {g: i for i, g in enumerate(genes)}
        sl = [g for g in singles if g in idx]
        gidx = {}
        for g in sl:
            r = np.where(raw == g)[0]
            if len(r) > MAX_PERT_CELLS:
                r = rng.choice(r, MAX_PERT_CELLS, replace=False)
            gidx[g] = r
        Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
        cmean = Xc.mean(0)
        Xc = Xc - cmean
        cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        v = V[:, np.argmax(w)].astype(np.float32); v = v / np.linalg.norm(v)
        gfrac = float(w.max() / w.sum())
        full, spec = metrics(X, genes, idx, raw, cidx, sl, gidx, cmean, cov, v)
        out[tag] = (gfrac, full, spec)
        del X, Xc, cov, V, w
        gc.collect()
    del a
    gc.collect()
    return out


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(d, "*.h5ad")))
    header = "dataset," + ",".join(
        "{}_{}".format(t, k) for t in ["top", "hvg", "random"]
        for k in ["gfrac", "full", "spec"]) + "\n"
    with open("hvg_results.csv", "w") as fh:
        fh.write(header)
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            o = run(f)
            with open("hvg_results.csv", "a") as fh:
                fh.write(name + "," + ",".join(
                    "{:.4f}".format(o[t][i]) for t in ["top", "hvg", "random"]
                    for i in range(3)) + "\n")
            print("OK   {:<34} ".format(name[:33]) + " ".join(
                "{}:g={:.2f},full={:.3f},spec={:.3f}".format(t, *o[t]) for t in ["top", "hvg", "random"]),
                flush=True)
        except Exception as e:
            print("SKIP {:<34} {}".format(name[:33], str(e)[:30]), flush=True)


if __name__ == "__main__":
    main()
