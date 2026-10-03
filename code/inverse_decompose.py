import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
inverse_decompose.py -- Exp4: *why* does the inverse (driver recovery) work?

The official CIPHER forward model is dominated by a shared cell-size axis and
carries no detectable perturbation-specific signal (Exp1/2).  Yet the *inverse*
-- CIPHER's ``fullH_diag`` empirical-Bayes posterior inverse -- recovers the
driven gene with AUC ~0.96-0.99.  This script decomposes that success into the
signals it could be using, on the *same* pipeline as
:func:`cipher.posterior_inverse_prediction` (raw space):

  posterior            official score: covariance + heteroscedastic noise + EB prior
  posterior_selfremoved  zero the target gene's own coordinate of dx  -> kills "self"
  posterior_diagSigma    replace Sigma by diag(Sigma)                 -> kills cross-gene covariance
  posterior_controlvar   use the control variance for every perturbation's noise
  posterior_no_uncinfl   drop the posterior-std inflation (score = |posterior mean|)
  matched_filter         covariance-only linear baseline
  magnitude              rank by |dx| (pure self-effect baseline)

Per perturbation we compute the one-vs-rest ROC-AUC of the true driver over all
genes; per dataset we report the mean.  Aggregated output:
../results/inverse_decompose.csv
"""

import gc
import os
import sys

import numpy as np

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)

from cipher.data import load_dataset
from cipher.normalize import normalize_matrix, library_size, mean_var
from cipher.covariance import compute_covariance
from cipher.inverse import (PosteriorInverseModel, build_model, fit_tau2,
                            posterior_scores_batch, posterior_mean_batch)
from cipher.core import matched_filter_scores, one_vs_rest_auc
from cipher.utils import stable_seed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "inverse_decompose.csv")

COV_MAX_CELLS = 10000
BATCH = 128
MAX_PERT = 300
SEED = 0

VARIANTS = ["posterior", "posterior_selfremoved", "posterior_controlvar",
            "posterior_controlvar_selfremoved", "posterior_diagSigma",
            "posterior_diagSigma_selfremoved", "posterior_no_uncinfl",
            "posterior_stdonly", "matched_filter", "magnitude"]


def batch_scores(model, dx, tau2, use_mean=False):
    """Score every perturbation in ``dx`` (n, p) with the posterior inverse."""
    n, p = dx.shape
    out = np.empty((n, p), dtype=np.float32)
    for a in range(0, n, BATCH):
        b = min(a + BATCH, n)
        if use_mean:
            sc = np.abs(posterior_mean_batch(model, dx[a:b].astype(np.float64), a, b, tau2))
        else:
            sc = posterior_scores_batch(model, dx[a:b].astype(np.float64), a, b, tau2)
        out[a:b] = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def batch_stdonly(model, dx, tau2):
    """Perturbation-independent gene baseline: the posterior std alone."""
    n, p = dx.shape
    out = np.empty((n, p), dtype=np.float32)
    for a in range(0, n, BATCH):
        b = min(a + BATCH, n)
        d_, _ = model.batch_terms(np.asarray(dx[a:b], dtype=np.float64), a, b)
        pv = 1.0 / np.maximum(d_ * d_ + 1.0 / tau2, 1e-12)
        out[a:b] = np.sqrt(np.maximum(pv @ model.V2.T, 0.0))
    return out


def zero_target(dx, tgi, p):
    """Return a copy of dx with each target gene's own coordinate set to 0."""
    out = dx.copy()
    for i, t in enumerate(tgi):
        if 0 <= int(t) < p:
            out[i, int(t)] = 0.0
    return out


def eval_variant(scores, tgi):
    aucs = [one_vs_rest_auc(scores[i], int(t)) for i, t in enumerate(tgi) if 0 <= int(t) < scores.shape[1]]
    aucs = [a for a in aucs if np.isfinite(a)]
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def analyze(path):
    ds = load_dataset(path)
    ds_name = ds.name
    control_raw = ds.control_matrix(dense=True)
    # NOTE: intentionally mirrors cipher.posterior_inverse_prediction (raw, cov_max_cells=10000)
    control_norm = normalize_matrix(control_raw, "raw", libsize=library_size(control_raw))
    control_mean = control_norm.mean(axis=0)
    control_var = control_norm.var(axis=0, ddof=1)
    n0 = float(control_norm.shape[0])
    del control_raw
    gc.collect()

    cov_norm = control_norm
    if control_norm.shape[0] > COV_MAX_CELLS:
        rng = np.random.default_rng(stable_seed(SEED, f"{ds_name}:raw"))
        sel = np.sort(rng.choice(control_norm.shape[0], COV_MAX_CELLS, replace=False))
        cov_norm = control_norm[sel]
    Sigma = compute_covariance(cov_norm)
    p = Sigma.shape[0]
    del control_norm, cov_norm
    gc.collect()

    perts = list(ds.perturbations)
    tgi = np.asarray(ds.target_gene_indices, dtype=np.int64)
    if len(perts) > MAX_PERT:
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(len(perts), MAX_PERT, replace=False))
        perts = [perts[i] for i in sel]
        tgi = tgi[sel]
    dx = np.empty((len(perts), p), dtype=np.float64)
    var_pert = np.empty((len(perts), p), dtype=np.float64)
    nu = np.empty(len(perts), dtype=np.float64)
    for i, pert in enumerate(perts):
        Yp = normalize_matrix(ds.perturbation_matrix(pert, dense=True), "raw")
        nu[i] = Yp.shape[0]
        m, v = mean_var(Yp)
        dx[i] = m - control_mean
        var_pert[i] = v
        del Yp
    gc.collect()

    model = build_model(Sigma, var_pert, n0, nu, control_var=control_var)
    tau2 = fit_tau2(model, dx, batch=BATCH)["tau2_use"]
    gfrac = float(model.eigenvalues.max() / model.eigenvalues.sum())

    results = {}

    sc = batch_scores(model, dx, tau2)
    results["posterior"], n_eval = eval_variant(sc, tgi)
    del sc
    gc.collect()

    dx_sr = zero_target(dx, tgi, p)
    sc = batch_scores(model, dx_sr, tau2)
    results["posterior_selfremoved"], _ = eval_variant(sc, tgi)
    del sc
    gc.collect()

    sc = batch_scores(model, dx, tau2, use_mean=True)
    results["posterior_no_uncinfl"], _ = eval_variant(sc, tgi)
    del sc
    gc.collect()

    sc = batch_stdonly(model, dx, tau2)
    results["posterior_stdonly"], _ = eval_variant(sc, tgi)
    del sc
    gc.collect()

    # control-variance noise model (no per-perturbation / self variance)
    pev_cv = np.broadcast_to(np.maximum(control_var @ model.V2, 0.0), (len(perts), p)).copy()
    model_cv = PosteriorInverseModel(eigenvalues=model.eigenvalues, V=model.V, V2=model.V2,
                                     pert_eigvar=pev_cv, n0=model.n0, nu=model.nu, ridge=model.ridge)
    tau2_cv = fit_tau2(model_cv, dx, batch=BATCH)["tau2_use"]
    sc = batch_scores(model_cv, dx, tau2_cv)
    results["posterior_controlvar"], _ = eval_variant(sc, tgi)
    del sc
    sc = batch_scores(model_cv, dx_sr, tau2_cv)
    results["posterior_controlvar_selfremoved"], _ = eval_variant(sc, tgi)
    del sc, model_cv, pev_cv
    gc.collect()

    # diagonal covariance (kills cross-gene covariance, keeps gene variances)
    model_diag = build_model(np.diag(np.maximum(np.diag(Sigma), 0.0)), var_pert, n0, nu,
                             control_var=control_var)
    tau2_diag = fit_tau2(model_diag, dx, batch=BATCH)["tau2_use"]
    sc = batch_scores(model_diag, dx, tau2_diag)
    results["posterior_diagSigma"], _ = eval_variant(sc, tgi)
    del sc
    sc = batch_scores(model_diag, dx_sr, tau2_diag)
    results["posterior_diagSigma_selfremoved"], _ = eval_variant(sc, tgi)
    del sc, model_diag
    gc.collect()
    del dx_sr
    gc.collect()

    # matched filter + |dx| magnitude
    den = np.einsum("ij,ij->j", Sigma, Sigma) + 1e-8
    sc = (Sigma.T @ dx.T).T / den[None, :]
    results["matched_filter"], _ = eval_variant(sc, tgi)
    del sc
    sc = np.abs(dx)
    results["magnitude"], _ = eval_variant(sc, tgi)
    del sc, Sigma, dx, var_pert, model
    gc.collect()

    row = {"dataset": ds_name, "space": "raw", "n_genes": p, "n_control": int(n0),
           "n_perts_tested": n_eval, "global_frac": round(gfrac, 4), "tau2": round(float(tau2), 4)}
    row.update({k: round(float(results[k]), 4) for k in VARIANTS})
    return row


def main():
    d = _PERTURB_DATA
    default = [
        "NormanWeissman2019_filtered.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad",
        "FrangiehIzar2021_RNA.h5ad",
        "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad",
        "ReplogleWeissman2022_rpe1.h5ad",
        "TianKampmann2021_CRISPRa.h5ad",
        "TianKampmann2021_CRISPRi.h5ad",
    ]
    files = sys.argv[1:] if len(sys.argv) > 1 else default
    keys = ["space", "n_genes", "n_control", "n_perts_tested", "global_frac", "tau2"] + VARIANTS

    done = set()
    if os.path.exists(OUT) and not os.environ.get("INV_DECOMP_OVERWRITE"):
        with open(OUT) as fh:
            next(fh, None)
            for line in fh:
                if line.strip():
                    done.add(line.split(",", 1)[0])
    else:
        with open(OUT, "w") as fh:
            fh.write("dataset," + ",".join(keys) + "\n")

    for f in files:
        name = os.path.basename(f)
        stem = name.replace(".h5ad", "")
        if stem in done:
            print("have {:<30} (skipped)".format(stem[:29]), flush=True)
            continue
        try:
            r = analyze(os.path.join(d, f) if not os.path.isabs(f) else f)
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(str(r[k]) for k in keys) + "\n")
            print("OK   {:<28} post={:.3f} selfRem={:.3f} ctrlVar={:.3f} cvSelfRem={:.3f} "
                  "diag={:.3f} diagSelfRem={:.3f} noUnc={:.3f} stdOnly={:.3f} mf={:.3f} mag={:.3f}".format(
                      r["dataset"][:27], r["posterior"], r["posterior_selfremoved"],
                      r["posterior_controlvar"], r["posterior_controlvar_selfremoved"],
                      r["posterior_diagSigma"], r["posterior_diagSigma_selfremoved"],
                      r["posterior_no_uncinfl"], r["posterior_stdonly"],
                      r["matched_filter"], r["magnitude"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(name[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
