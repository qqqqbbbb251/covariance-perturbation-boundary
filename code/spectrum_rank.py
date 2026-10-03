import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
spectrum_rank.py  -- eigenvalue spectrum + rank-k prediction

Directly characterises "near rank-one": the fraction of covariance variance in the
top-1/2/5 eigenvectors, and how well a rank-k eigenvector projection predicts the
responses compared with the full covariance column Sigma[:,g].  If rank-1 already
matches the full model, the covariance carries no gene-specific direction beyond
the leading mode.

Output: ../results/spectrum_rank.csv
"""

import gc
import os
import sys

import numpy as np
import scipy.sparse as sp
import anndata as ad

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "spectrum_rank.csv")

CONTROL_TOKENS = ("control", "ctrl", "neg_ctrl", "ntc", "non-targeting", "nontargeting")
MAX_CTRL, MAX_PERT_CELLS, N_TOP, MAX_PERT = 3000, 500, 2000, 120
SPACES = ["raw", "pearson"]


def is_control(l):
    l = l.lower()
    return any(t in l for t in CONTROL_TOKENS)


def cos2(a, b):
    na, nb = float(a @ a), float(b @ b)
    return float(a @ b) ** 2 / (na * nb) if na and nb else float("nan")


def load(path):
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
    return X, idx, singles, cidx, nc_all, raw, rng


def run(path, space):
    X, idx, singles, cidx, nc_all, raw, rng = load(path)
    Xc = np.asarray(X[cidx].todense(), dtype=np.float32)
    if space == "pearson":
        nc_c = nc_all[cidx]
        sf = nc_c / (nc_c.mean() + 1e-9)
        mu = (Xc / sf[:, None]).mean(0)
        muij = sf[:, None] * mu[None, :]
        Xc = (Xc - muij) / np.sqrt(muij + 1e-6)
    cmean = Xc.mean(0)
    Z = Xc - cmean
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]; V = V[:, order]
    wpos = np.clip(w, 0, None)
    tot = wpos.sum() if wpos.sum() > 0 else 1.0
    top = [float(wpos[:k].sum() / tot) for k in (1, 2, 5, 10)]
    del Xc, Z
    gc.collect()

    full, r1, r2, r5 = [], [], [], []
    for g in singles:
        rows = np.where(raw == g)[0]
        if len(rows) > MAX_PERT_CELLS:
            rows = rng.choice(rows, MAX_PERT_CELLS, replace=False)
        E = np.asarray(X[rows].todense(), dtype=np.float32)
        if space == "pearson":
            nc = nc_all[rows]
            sf = nc / (nc.mean() + 1e-9)
            muij = sf[:, None] * mu[None, :]
            E = (E - muij) / np.sqrt(muij + 1e-6)
        d = E.mean(0) - cmean
        del E
        if d @ d == 0:
            continue
        sig = cov[:, idx[g]]
        full.append(cos2(d, sig))
        r1.append(cos2(d, V[:, 0]))
        # rank-k least-squares projection
        for k, store in [(2, r2), (5, r5)]:
            B = V[:, :k]
            coef, *_ = np.linalg.lstsq(B, d, rcond=None)
            store.append(cos2(d, B @ coef))
        del d
    del X, cov, V
    gc.collect()
    if len(full) < 10:
        raise ValueError("too few")
    return {"dataset": os.path.basename(path).replace(".h5ad", ""), "space": space,
            "n": len(full), "top1_frac": top[0], "top2_frac": top[1],
            "top5_frac": top[2], "top10_frac": top[3],
            "full_cos": float(np.nanmean(full)), "rank1_cos": float(np.nanmean(r1)),
            "rank2_cos": float(np.nanmean(r2)), "rank5_cos": float(np.nanmean(r5))}


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    keys = ["n", "top1_frac", "top2_frac", "top5_frac", "top10_frac",
            "full_cos", "rank1_cos", "rank2_cos", "rank5_cos"]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(keys) + "\n")
    for f in files:
        for space in SPACES:
            try:
                r = run(f, space)
                with open(OUT, "a") as fh:
                    fh.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                        r["dataset"], r["space"], r["n"], r["top1_frac"], r["top2_frac"],
                        r["top5_frac"], r["top10_frac"], r["full_cos"], r["rank1_cos"],
                        r["rank2_cos"], r["rank5_cos"]))
                print("OK   {:<28} {:<8} top1={:.3f} top5={:.3f} | full={:.3f} rank1={:.3f} rank2={:.3f} rank5={:.3f}".format(
                    r["dataset"][:27], space, r["top1_frac"], r["top5_frac"],
                    r["full_cos"], r["rank1_cos"], r["rank2_cos"], r["rank5_cos"]), flush=True)
            except Exception as e:
                print("SKIP {:<28} {:<8} {}".format(os.path.basename(f)[:27], space, str(e)[:40]), flush=True)


if __name__ == "__main__":
    main()
