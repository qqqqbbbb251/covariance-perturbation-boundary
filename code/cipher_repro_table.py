import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
cipher_repro_table.py  (reviewer points 1 and 3, reproduction; strengthened)

Per-dataset CIPHER-framework reproduction table, with a MATCHED random-column
baseline, a stable dual metric, and across-split uncertainty.

Protocol (no leakage):
  * control covariance Sigma from up to 10,000 control cells (CIPHER default),
  * scalar a_hat fitted on the TRAIN gene half, scored on the held-out TEST half
    (gene_holdout_masks with holdout_frac=0.5).  The SAME split is used for every
    predictor, so none is evaluated on genes used to fit it.
  * every value is averaged over N_SPLITS independent gene-holdout splits, and the
    across-split SD is reported (uncentered R2 is split-sensitive).

Predictors (all scored identically):
  full            : a_hat * Sigma[:, g]              (CIPHER forward model)
  global          : b_hat * v                        (top eigenvector of Sigma)
  random          : a_hat * Sigma[:, g']             (MATCHED random gene's column:
                    nearest neighbours in standardised (mean, variance, detection)
                    space, N_DRAW draws averaged)
  random_unmatched: a_hat * Sigma[:, g'']            (uniformly random gene column)
  meanfield       : a_hat * Sigma_mf[:, g]           (per-gene-shuffled null)
  shuffled        : a_hat * Sigma_sh[:, g]           (fully permuted null)

Metrics: CIPHER's uncentered R2 (1 - SSE / sum(y^2)) and cosine, both on held-out
genes.  The cosine is the split-stable primary quantity.

Output: ../results/cipher_reproduction_table.csv
"""

import gc
import os
import sys

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)

import numpy as np

from cipher.data import load_dataset
from cipher.normalize import normalize_matrix, library_size
from cipher.covariance import compute_covariance, null_covariance
from cipher.core import forward_fit, gene_holdout_masks

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "cipher_reproduction_table.csv")

HOLDOUT = 0.5
COV_MAX = 10000
N_SPLITS = 10
N_DRAW = 10
N_NEIGH = 50
NORMS = ["raw", "pflog"]
METRIC = "cipher_uncentered_r2_and_cosine_holdout0.5_10splits"
NULLS = ["meanfield", "shuffled"]
PREDS = ["full", "global", "random", "random_unmatched", "meanfield", "shuffled"]
EXCL = ["full_excl", "global_excl", "random_excl"]
ALL_PREDS = PREDS + EXCL


def r2u(yt, yp):
    ss = float(yt @ yt)
    return 1.0 - float((yt - yp) @ (yt - yp)) / ss if ss > 0 else float("nan")


def cosi(yt, yp):
    ny, np_ = float(np.linalg.norm(yt)), float(np.linalg.norm(yp))
    return float(yt @ yp) / (ny * np_) if ny and np_ else float("nan")


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def run(path, norm="raw"):
    ds = load_dataset(path, expression_threshold=1.0, min_samples=100)
    control_raw = ds.control_matrix(dense=True)
    pseudo = ds.pflog_pseudocount if norm == "pflog" else None

    def _norm(X):
        return normalize_matrix(X, norm, libsize=library_size(X), pseudocount=pseudo)

    ctrl_norm = _norm(control_raw)
    control_mean = ctrl_norm.mean(0)
    if ctrl_norm.shape[0] > COV_MAX:
        rng0 = np.random.default_rng(0)
        sel = np.sort(rng0.choice(ctrl_norm.shape[0], COV_MAX, replace=False))
        ctrl_cov = ctrl_norm[sel]
    else:
        ctrl_cov = ctrl_norm
    Sigma = compute_covariance(ctrl_cov)
    w, V = np.linalg.eigh(Sigma)
    v = V[:, int(np.argmax(w))]
    gfrac = float(w.max() / w.sum())
    del V, w
    gc.collect()

    nulls = {k: null_covariance(ctrl_cov, k, seed=0) for k in NULLS}

    # matched-random gene pools: nearest neighbours in standardised (mean, var, detection)
    cmean = ctrl_cov.mean(0)
    cvar = ctrl_cov.var(0)
    cdet = (ctrl_cov > 0).mean(0)
    F = np.stack([_z(cmean), _z(cvar), _z(cdet)], 1).astype(np.float32)
    n_genes = F.shape[0]
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    neighbours = np.argsort(D, axis=1)[:, :N_NEIGH]
    del D, F
    gc.collect()

    items = []
    for pert, gene_idx in zip(ds.perturbations, ds.target_gene_indices):
        if gene_idx is None or gene_idx < 0:
            continue
        dx = _norm(ds.perturbation_matrix(pert, dense=True)).mean(0) - control_mean
        items.append((int(gene_idx), dx))

    rng = np.random.default_rng(0)
    # per predictor: list over splits of the split-mean; store r2 and cosine
    store = {k: {"r2": [], "cos": []} for k in ALL_PREDS}
    for rep in range(N_SPLITS):
        split_rng = np.random.default_rng(rep)
        vals = {k: {"r2": [], "cos": []} for k in ALL_PREDS}
        for gene_idx, dx in items:
            train, test = gene_holdout_masks(n_genes, gene_idx, HOLDOUT, rng=split_rng)
            if train.sum() < 50 or test.sum() < 50:
                continue
            # "excl" removes the perturbed (target) gene from BOTH fit and eval,
            # so no trivial self-prediction remains.
            train_e = train.copy(); train_e[gene_idx] = False
            test_e = test.copy(); test_e[gene_idx] = False
            yt = dx[test]
            yt_e = dx[test_e]

            def score(col, tr=train, te=test, yy=yt):
                a, _ = forward_fit(col, dx, mask=tr)
                pred = (a if np.isfinite(a) else 0.0) * col[te]
                return r2u(yy, pred), cosi(yy, pred)

            a, b = score(Sigma[:, gene_idx])
            vals["full"]["r2"].append(a); vals["full"]["cos"].append(b)
            a, b = score(Sigma[:, gene_idx], train_e, test_e, yt_e)
            vals["full_excl"]["r2"].append(a); vals["full_excl"]["cos"].append(b)

            g_r2, g_cos = score(v)
            vals["global"]["r2"].append(g_r2)
            vals["global"]["cos"].append(g_cos)
            g_r2, g_cos = score(v, train_e, test_e, yt_e)
            vals["global_excl"]["r2"].append(g_r2)
            vals["global_excl"]["cos"].append(g_cos)

            pool = neighbours[gene_idx]
            draws = rng.choice(pool, size=min(N_DRAW, len(pool)), replace=False)
            mr, mc, mr_e, mc_e = [], [], [], []
            for j in draws:
                a, b = score(Sigma[:, int(j)])
                mr.append(a); mc.append(b)
                a, b = score(Sigma[:, int(j)], train_e, test_e, yt_e)
                mr_e.append(a); mc_e.append(b)
            vals["random"]["r2"].append(float(np.nanmean(mr)))
            vals["random"]["cos"].append(float(np.nanmean(mc)))
            vals["random_excl"]["r2"].append(float(np.nanmean(mr_e)))
            vals["random_excl"]["cos"].append(float(np.nanmean(mc_e)))

            a, b = score(Sigma[:, int(rng.integers(n_genes))])
            vals["random_unmatched"]["r2"].append(a)
            vals["random_unmatched"]["cos"].append(b)

            a, b = score(nulls["meanfield"][:, gene_idx])
            vals["meanfield"]["r2"].append(a)
            vals["meanfield"]["cos"].append(b)
            a, b = score(nulls["shuffled"][:, gene_idx])
            vals["shuffled"]["r2"].append(a)
            vals["shuffled"]["cos"].append(b)
        for k in ALL_PREDS:
            store[k]["r2"].append(float(np.nanmean(vals[k]["r2"])) if vals[k]["r2"] else np.nan)
            store[k]["cos"].append(float(np.nanmean(vals[k]["cos"])) if vals[k]["cos"] else np.nan)

    n_perts = len(items)
    del Sigma, nulls, ctrl_cov, ctrl_norm, control_raw, items
    gc.collect()

    def ms(key, m):
        arr = np.asarray(store[key][m], dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    out = {"dataset": ds.name, "norm": norm, "n_perts": n_perts, "global_frac": gfrac}
    for k in ALL_PREDS:
        out[k], out[k + "_sd"] = ms(k, "r2")
        out[k + "_cos"], out[k + "_cos_sd"] = ms(k, "cos")
    return out


def main():
    d = _PERTURB_DATA
    import glob as globr
    files = sys.argv[1:] if len(sys.argv) > 1 else sorted(globr.glob(os.path.join(d, "*.h5ad")))
    cols = ["dataset", "norm", "metric", "n_perts", "global_frac",
            "full", "global", "random", "random_unmatched", "meanfield", "shuffled",
            "full_excl", "global_excl", "random_excl",
            "full_cos", "global_cos", "random_cos",
            "full_excl_cos", "global_excl_cos", "random_excl_cos",
            "full_sd", "global_sd", "random_sd"]
    with open(OUT, "w") as fh:
        fh.write(",".join(cols) + "\n")
    for f in files:
        for norm in NORMS:
            try:
                r = run(f, norm)
                with open(OUT, "a") as fh:
                    fh.write("{},{},{},{},{:.4f},".format(
                        r["dataset"], r["norm"], METRIC, r["n_perts"], r["global_frac"]))
                    fh.write(",".join("{:.4f}".format(r[k]) for k in
                             ["full", "global", "random", "random_unmatched", "meanfield", "shuffled",
                              "full_excl", "global_excl", "random_excl",
                              "full_cos", "global_cos", "random_cos",
                              "full_excl_cos", "global_excl_cos", "random_excl_cos",
                              "full_sd", "global_sd", "random_sd"]))
                    fh.write("\n")
                print("OK   {:<28} {:<6} full={:.3f} global={:.3f} rand={:.3f} | excl full={:.3f} rand={:.3f}".format(
                    r["dataset"][:27], norm, r["full"], r["global"], r["random"],
                    r["full_excl"], r["random_excl"]), flush=True)
            except Exception as e:
                print("SKIP {:<28} {:<6} {}".format(os.path.basename(f)[:27], norm, str(e)[:40]),
                      flush=True)


if __name__ == "__main__":
    main()
