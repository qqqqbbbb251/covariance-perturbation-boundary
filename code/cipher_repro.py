import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""cipher_repro.py -- reproduce CIPHER's own forward model using their package.

Runs cipher.forward_prediction on our scPerturb datasets with CIPHER's settings
(raw counts, gene holdout 0.5 as in the paper) and prints the summary metrics.

Usage:
  python cipher_repro.py <h5ad> [normalization ...]
"""
import sys
import os

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)

import numpy as np
import cipher
from cipher import load_dataset, forward_prediction


def run_one(path, norm="raw", holdout=0.5, max_pert=None):
    ds = load_dataset(path, expression_threshold=1.0, min_samples=100)
    print("  dataset={} n_cells={} n_genes={} n_ctrl={} n_perts={}".format(
        ds.name, ds.stats["n_cells"], ds.stats["n_genes"],
        ds.stats["n_control"], ds.stats["n_perturbations"]), flush=True)
    res = forward_prediction(ds, normalization=norm, nulls=("meanfield", "shuffled"),
                             holdout_frac=holdout, max_perturbations=max_pert,
                             cov_max_cells=10000, progress=False)
    s = res.summary
    print("  [{}] mean_r2_uncentered={:.3f} r2_centered={:.3f} pearson={:.3f} "
          "spearman={:.3f} cosine={:.3f}".format(
              norm, s.get("mean_r2_uncentered_real", np.nan),
              s.get("mean_r2_centered_real", np.nan),
              s.get("mean_pearson_real", np.nan),
              s.get("mean_spearman_real", np.nan),
              s.get("mean_cosine_real", np.nan)), flush=True)
    print("       nulls: meanfield={:.3f} shuffled={:.3f}".format(
        s.get("mean_r2_uncentered_meanfield", np.nan),
        s.get("mean_r2_uncentered_shuffled", np.nan)), flush=True)
    return res


if __name__ == "__main__":
    path = sys.argv[1]
    norms = sys.argv[2:] if len(sys.argv) > 2 else ["raw"]
    for n in norms:
        run_one(path, norm=n)
