import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
reverse_self_relation.py -- Stage 4 (E10/E11): quantify that reverse driver recovery is
driven by the target gene's own effect, and ask what happens when the self signal is weak.

For each dataset (raw, official-style pipeline):
  * per-perturbation driver-recovery AUC for the fullH posterior and for the
    diagonal-covariance variant (which we showed ~= official);
  * per-perturbation "self" metrics: the target gene's shift z-score and its control
    expression;
  * Spearman correlation of AUC with the self z-score and with target expression;
  * within the bottom-quartile self-z perturbations, compare full vs diagonal AUC
    (does the covariance start to matter when self is weak?).

Output: ../results/reverse_self_relation.csv
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
                            posterior_scores_batch)
from cipher.core import one_vs_rest_auc, matched_filter_scores
from cipher.utils import stable_seed
from scipy.stats import spearmanr

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "reverse_self_relation.csv")

COV_MAX_CELLS = 10000
BATCH = 128
MAX_PERT = 300
SEED = 0


def batch_scores(model, dx, tau2):
    n, p = dx.shape
    out = np.empty((n, p), dtype=np.float32)
    for a in range(0, n, BATCH):
        b = min(a + BATCH, n)
        sc = posterior_scores_batch(model, dx[a:b].astype(np.float64), a, b, tau2)
        out[a:b] = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def analyze(path):
    ds = load_dataset(path)
    name = ds.name
    control = normalize_matrix(ds.control_matrix(dense=True), "raw")
    control_mean = control.mean(0)
    control_var = control.var(0, ddof=1)
    n0 = float(control.shape[0])
    cov_norm = control
    if control.shape[0] > COV_MAX_CELLS:
        rng = np.random.default_rng(stable_seed(SEED, f"{name}:raw"))
        sel = np.sort(rng.choice(control.shape[0], COV_MAX_CELLS, replace=False))
        cov_norm = control[sel]
    Sigma = compute_covariance(cov_norm)
    p = Sigma.shape[0]
    del control, cov_norm
    gc.collect()

    perts = list(ds.perturbations)
    tgi = np.asarray(ds.target_gene_indices, dtype=np.int64)
    if len(perts) > MAX_PERT:
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(len(perts), MAX_PERT, replace=False))
        perts = [perts[i] for i in sel]
        tgi = tgi[sel]

    dx = np.empty((len(perts), p)); var_pert = np.empty((len(perts), p))
    nu = np.empty(len(perts))
    for i, pert in enumerate(perts):
        Yp = normalize_matrix(ds.perturbation_matrix(pert, dense=True), "raw")
        nu[i] = Yp.shape[0]
        m, v = mean_var(Yp)
        dx[i] = m - control_mean; var_pert[i] = v
        del Yp
    gc.collect()

    model = build_model(Sigma, var_pert, n0, nu, control_var=control_var)
    tau2 = fit_tau2(model, dx, batch=BATCH)["tau2_use"]
    sc_full = batch_scores(model, dx, tau2)
    # diagonal covariance variant
    model_diag = build_model(np.diag(np.maximum(np.diag(Sigma), 0.0)), var_pert, n0, nu,
                             control_var=control_var)
    tau2_d = fit_tau2(model_diag, dx, batch=BATCH)["tau2_use"]
    sc_diag = batch_scores(model_diag, dx, tau2_d)

    auc_full, auc_diag, self_z, tgt_expr = [], [], [], []
    for i, t in enumerate(tgi):
        if not (0 <= int(t) < p):
            continue
        auc_full.append(one_vs_rest_auc(sc_full[i], int(t)))
        auc_diag.append(one_vs_rest_auc(sc_diag[i], int(t)))
        se = np.sqrt(control_var[t] / n0 + var_pert[i, t] / max(nu[i], 1)) + 1e-12
        self_z.append(abs(dx[i, t]) / se)
        tgt_expr.append(control_mean[t])
    auc_full = np.asarray(auc_full); auc_diag = np.asarray(auc_diag)
    self_z = np.asarray(self_z); tgt_expr = np.asarray(tgt_expr)

    def sp(a, b):
        if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(spearmanr(a, b).correlation)

    q = np.quantile(self_z, 0.25)
    weak = self_z <= q
    out = {
        "dataset": name, "n_perts": len(auc_full),
        "auc_full": float(np.nanmean(auc_full)), "auc_diag": float(np.nanmean(auc_diag)),
        "spearman_auc_selfz": sp(auc_full, self_z),
        "spearman_auc_targexpr": sp(auc_full, tgt_expr),
        "auc_full_weakself": float(np.nanmean(auc_full[weak])) if weak.sum() else float("nan"),
        "auc_diag_weakself": float(np.nanmean(auc_diag[weak])) if weak.sum() else float("nan"),
        "auc_full_strongself": float(np.nanmean(auc_full[~weak])) if (~weak).sum() else float("nan"),
        "auc_diag_strongself": float(np.nanmean(auc_diag[~weak])) if (~weak).sum() else float("nan"),
        "n_weak": int(weak.sum()),
    }
    del Sigma, dx, var_pert, model, model_diag, sc_full, sc_diag
    gc.collect()
    return out


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
    keys = ["n_perts", "auc_full", "auc_diag", "spearman_auc_selfz", "spearman_auc_targexpr",
            "auc_full_weakself", "auc_diag_weakself", "auc_full_strongself",
            "auc_diag_strongself", "n_weak"]
    with open(OUT, "w") as fh:
        fh.write("dataset," + ",".join(keys) + "\n")
    for f in files:
        try:
            r = analyze(os.path.join(d, f))
            with open(OUT, "a") as fh:
                fh.write(r["dataset"] + "," + ",".join(
                    ("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
            print("OK {:<26} auc={:.3f} rho(selfz)={:.2f} | weakself: full={:.3f} diag={:.3f}".format(
                r["dataset"][:25], r["auc_full"], r["spearman_auc_selfz"],
                r["auc_full_weakself"], r["auc_diag_weakself"]), flush=True)
        except Exception as e:
            import traceback
            print("SKIP {:<30} {}".format(os.path.basename(f)[:29], str(e)[:60]), flush=True)
            traceback.print_exc()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
