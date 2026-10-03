"""
norman_cipher.py

Faithful reimplementation of the CIPHER forward model (Kuznets-Speck et al.
2025) and validation on the Norman 2019 K562 CRISPRa dataset.

CIPHER forward model
--------------------
    dX = Sigma @ u

  Sigma : gene-gene covariance of UNPERTURBED (control) cells, in the SAME
          expression space as dX (the paper uses raw absolute counts).
  u     : perturbation vector. For a single-gene perturbation of gene i we
          constrain u = a * e_i and fit the scalar a by least squares:
              a = (Sigma[:,i] . dX) / (Sigma[:,i] . Sigma[:,i])
          so the prediction is a scaled copy of the i-th covariance column.
  score : R^2 = 1 - ||dX - a*Sigma[:,i]||^2 / ||dX||^2

This is the important difference from the first attempt, which divided by
Sigma[g,g], forced a negative sign and worked in log space -- all of which are
not what CIPHER does.

The script sweeps three expression spaces (raw / cpm / log) because the space
turns out to matter, and reports a mean-field null (control counts shuffled
column-wise) alongside.

Usage:
  python norman_cipher.py --h5ad <path> --n-top 2000
"""

import argparse
import gc

import numpy as np
import scipy.sparse as sp
import anndata as ad


def load_raw(path, n_top):
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    pert = obs["perturbation"].astype(str).values
    nperts = obs["nperts"].values
    ncounts = obs["ncounts"].values.astype(float)
    var_ncounts = a.var["ncounts"].values.astype(float)
    var_names = np.asarray(a.var_names)

    ctrl = pert == "control"
    singles = np.unique(pert[(nperts == 1) & (~ctrl)])
    print("cells: {}  control: {}  single perturbations: {}".format(
        a.n_obs, int(ctrl.sum()), len(singles)))

    selected = set(np.argsort(-var_ncounts)[:n_top].tolist())
    name_to_pos = {g: i for i, g in enumerate(var_names)}
    for g in singles:
        if g in name_to_pos:
            selected.add(name_to_pos[g])
    gene_idx = np.array(sorted(selected))
    genes = var_names[gene_idx]
    print("genes used: {}".format(len(genes)))

    X = a[:, gene_idx].to_memory().X.tocsr().astype(np.float32)
    del a
    return X, genes, pert, nperts, ctrl, singles, ncounts


def to_space(X, ncounts, space):
    if space == "raw":
        return X
    inv = (1e4 / np.maximum(ncounts, 1.0)).astype(np.float32)
    Y = (sp.diags(inv) @ X).tocsr()
    if space == "cpm":
        return Y
    if space == "log":
        Y = Y.copy()
        Y.data = np.log1p(Y.data)
        return Y
    raise ValueError(space)


def evaluate(X, genes, pert, ctrl, singles, label):
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    Xc = np.asarray(X[ctrl].todense(), dtype=np.float64)
    ctrl_mean = Xc.mean(axis=0)
    Sigma = np.cov(Xc, rowvar=False)

    # mean-field null: shuffle each gene's control values across cells
    rng = np.random.default_rng(0)
    Xs = Xc.copy()
    for j in range(Xs.shape[1]):
        rng.shuffle(Xs[:, j])
    Sigma_mf = np.cov(Xs, rowvar=False)
    del Xc, Xs
    gc.collect()

    r2, r2_mf, r2_zero = [], [], []
    used = []
    for g in singles:
        if g not in gene_to_idx:
            continue
        delta = np.asarray(X[pert == g].todense(), dtype=np.float64).mean(axis=0) - ctrl_mean
        if np.sum(delta ** 2) == 0:
            continue
        i = gene_to_idx[g]
        for S, store in ((Sigma, r2), (Sigma_mf, r2_mf)):
            col = S[:, i]
            denom = float(np.dot(col, col))
            a = float(np.dot(col, delta) / denom) if denom > 0 else 0.0
            pred = a * col
            store.append(1.0 - np.sum((delta - pred) ** 2) / np.sum(delta ** 2))
        r2_zero.append(0.0)  # zero predictor => R2 = 0 by definition
        used.append(g)

    print("  {:<6} n_pert={:<4}  R2(real)={:>7.3f}  R2(mean-field)={:>7.3f}".format(
        label, len(used), np.mean(r2), np.mean(r2_mf)))
    return np.mean(r2), np.mean(r2_mf)


def sweep_control_cells(X, genes, pert, ctrl, singles, sizes):
    """Does R^2 saturate with more control cells, or keep rising?"""
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    ctrl_idx = np.where(ctrl)[0]
    order = np.random.default_rng(0).permutation(len(ctrl_idx))

    # precompute true deltas once
    deltas = {}
    for g in singles:
        if g in gene_to_idx:
            deltas[g] = np.asarray(X[pert == g].todense(), dtype=np.float64).mean(axis=0)

    print("\ndata-sufficiency sweep (raw space):")
    print("{:>10}{:>12}".format("n_control", "mean_R2"))
    for n in sizes:
        n = min(n, len(ctrl_idx))
        Xc = np.asarray(X[ctrl_idx[order[:n]]].todense(), dtype=np.float64)
        ctrl_mean = Xc.mean(axis=0)
        Sigma = np.cov(Xc, rowvar=False)
        r2 = []
        for g, dmean in deltas.items():
            delta = dmean - ctrl_mean
            if np.sum(delta ** 2) == 0:
                continue
            i = gene_to_idx[g]
            col = Sigma[:, i]
            denom = float(np.dot(col, col))
            a = float(np.dot(col, delta) / denom) if denom > 0 else 0.0
            r2.append(1.0 - np.sum((delta - a * col) ** 2) / np.sum(delta ** 2))
        print("{:>10}{:>12.3f}".format(n, np.mean(r2)))
        del Xc, Sigma
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--n-top", type=int, default=2000)
    ap.add_argument("--spaces", nargs="+", default=["raw", "cpm", "log"])
    ap.add_argument("--sweep", action="store_true",
                    help="sweep number of control cells (data sufficiency)")
    ap.add_argument("--sweep-sizes", nargs="+", type=int,
                    default=[250, 500, 1000, 2000, 4000, 8000, 11855])
    args = ap.parse_args()

    X, genes, pert, nperts, ctrl, singles, ncounts = load_raw(args.h5ad, args.n_top)
    print("\nCIPHER forward model, R^2 averaged over single perturbations:")
    for space in args.spaces:
        Y = to_space(X, ncounts, space)
        evaluate(Y, genes, pert, ctrl, singles, space)
        del Y
        gc.collect()

    if args.sweep:
        sweep_control_cells(X, genes, pert, ctrl, singles, args.sweep_sizes)


if __name__ == "__main__":
    main()
