import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cipher_baselines.py -- CIPHER's own forward model + global-mode / random-column
baselines, all under CIPHER's exact preprocessing and gene-holdout scoring.

For each dataset and normalization:
  full    : a_hat * Sigma[:, g]        (CIPHER's model)
  global  : b_hat * v                  (top eigenvector of Sigma)
  random  : a_hat * Sigma[:, g']       (random gene's column)
  meanfield/shuffled nulls from cipher

Metrics: mean r2_uncentered on held-out genes (holdout_frac=0.5).

Output: cipher_baselines.csv
"""

import os
import sys

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)

import numpy as np

from cipher.data import load_dataset
from cipher.normalize import normalize_matrix, library_size
from cipher.covariance import compute_covariance, null_covariance
from cipher.core import forward_fit, forward_metrics, gene_holdout_masks

HOLDOUT = 0.5
COV_MAX = 10000
NORMS = ["raw", "pflog"]


def run(path, norm="raw"):
    ds = load_dataset(path, expression_threshold=1.0, min_samples=100)
    control_raw = ds.control_matrix(dense=True)
    pseudo = ds.pflog_pseudocount if norm == "pflog" else None

    def _norm(X):
        return normalize_matrix(X, norm, libsize=library_size(X), pseudocount=pseudo)

    ctrl_norm = _norm(control_raw)
    control_mean = ctrl_norm.mean(0)
    if ctrl_norm.shape[0] > COV_MAX:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(ctrl_norm.shape[0], COV_MAX, replace=False))
        ctrl_cov = ctrl_norm[sel]
    else:
        ctrl_cov = ctrl_norm
    Sigma = compute_covariance(ctrl_cov)
    w, V = np.linalg.eigh(Sigma)
    v = V[:, np.argmax(w)]
    gfrac = float(w.max() / w.sum())

    n_genes = ds.n_genes
    split_rng = np.random.default_rng(0)
    rng = np.random.default_rng(0)
    full, glob, rand = [], [], []
    for pert, gene_idx in zip(ds.perturbations, ds.target_gene_indices):
        if gene_idx is None or gene_idx < 0:
            continue
        dx = _norm(ds.perturbation_matrix(pert, dense=True)).mean(0) - control_mean
        train, test = gene_holdout_masks(n_genes, gene_idx, HOLDOUT, rng=split_rng)
        if train.sum() < 50 or test.sum() < 50:
            continue
        yt = dx[test]
        for B, store in [(Sigma[:, gene_idx][:, None], full),
                         (v[:, None], glob),
                         (Sigma[:, int(rng.integers(n_genes))][:, None], rand)]:
            a, _ = forward_fit(B[:, 0], dx, mask=train)
            pred = (a if np.isfinite(a) else 0.0) * B[:, 0][test]
            store.append(forward_metrics(yt, pred)["r2_uncentered"])
    return {"dataset": ds.name, "norm": norm, "n_perts": len(full), "global_frac": gfrac,
            "full": float(np.nanmean(full)), "global": float(np.nanmean(glob)),
            "random": float(np.nanmean(rand))}


def main():
    d = _PERTURB_DATA
    import glob as _glob
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(_glob.glob(os.path.join(d, "*.h5ad")))
    with open("cipher_baselines.csv", "w") as fh:
        fh.write("dataset,norm,n_perts,global_frac,full,global,random\n")
    for f in files:
        for norm in NORMS:
            try:
                r = run(f, norm)
                with open("cipher_baselines.csv", "a") as fh:
                    fh.write("{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                        r["dataset"], r["norm"], r["n_perts"], r["global_frac"],
                        r["full"], r["global"], r["random"]))
                print("OK   {:<30} {:<7} full={:.3f} global={:.3f} random={:.3f} (gfrac={:.2f})".format(
                    r["dataset"][:29], r["norm"], r["full"], r["global"], r["random"],
                    r["global_frac"]), flush=True)
            except Exception as e:
                print("SKIP {:<30} {:<7} {}".format(os.path.basename(f)[:29], norm, str(e)[:30]), flush=True)


if __name__ == "__main__":
    main()
