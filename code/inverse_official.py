import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PERTURB_DATA = _os.environ.get("PERTURB_DATA", _os.path.join(_HERE, "..", "data"))
_CIPHER_ROOT = _os.environ.get("CIPHER_ROOT", _os.path.join(_HERE, "..", "CIPHER"))
_PROJECT_ROOT = _os.path.dirname(_HERE)
"""
inverse_official.py -- Exp3 re-test with CIPHER's OFFICIAL posterior inverse.

Runs cipher.posterior_inverse_prediction (fullH_diag empirical-Bayes posterior)
on each dataset in raw space and records the driver-recovery metrics, so we can
compare the official method against the matched-filter and |dx| baselines from
inverse_false_positive.csv.

Output: ../results/inverse_official.csv
"""

import os
import sys

CIPHER_ROOT = _CIPHER_ROOT
sys.path.insert(0, CIPHER_ROOT)

import numpy as np
from cipher import posterior_inverse_prediction

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "inverse_official.csv")

KEYS = ["n_valid", "mean_per_pert_auc", "median_per_pert_auc", "median_rank",
        "top1", "top5", "top10", "pooled_auc", "pooled_average_precision"]


def main():
    d = _PERTURB_DATA
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "FrangiehIzar2021_RNA.h5ad", "NadigOConner2024_hepg2.h5ad",
        "NadigOConner2024_jurkat.h5ad", "NormanWeissman2019_filtered.h5ad",
        "ReplogleWeissman2022_K562_essential.h5ad", "ReplogleWeissman2022_rpe1.h5ad",
        "TianKampmann2021_CRISPRa.h5ad", "TianKampmann2021_CRISPRi.h5ad"]
    files = [os.path.join(d, f) for f in files]
    with open(OUT, "w") as fh:
        fh.write("dataset,space," + ",".join(KEYS) + "\n")
    for f in files:
        name = os.path.basename(f).replace(".h5ad", "")
        try:
            res = posterior_inverse_prediction(f, normalization="raw",
                                               method="posterior", progress=False)
            s = res.summary
            with open(OUT, "a") as fh:
                fh.write(name + ",raw," + ",".join(
                    ("{:.4f}".format(float(s.get(k, np.nan))) if k != "n_valid" else str(s.get(k, "")))
                    for k in KEYS) + "\n")
            print("OK   {:<28} AUC={:.3f} top10={:.3f} median_rank={}".format(
                name[:27], float(s["mean_per_pert_auc"]), float(s["top10"]),
                s.get("median_rank")), flush=True)
        except Exception as e:
            print("SKIP {:<28} {}".format(name[:27], str(e)[:60]), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
