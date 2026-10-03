"""
inverse_decompose_stats.py -- cross-dataset statistics for the reverse decomposition.

Reads ../results/inverse_decompose.csv (Exp4) and reports the mean driver-recovery
AUC per ablation variant, plus paired Wilcoxon signed-rank tests for the key
contrasts.  Output: ../results/inverse_decompose_stats.csv
"""

import csv
import os

import numpy as np
from scipy.stats import wilcoxon

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
IN = os.path.join(RESULTS, "inverse_decompose.csv")
OUT = os.path.join(RESULTS, "inverse_decompose_stats.csv")

VARIANTS = ["posterior", "posterior_selfremoved", "posterior_controlvar",
            "posterior_controlvar_selfremoved", "posterior_diagSigma",
            "posterior_diagSigma_selfremoved", "posterior_no_uncinfl",
            "posterior_stdonly", "matched_filter", "magnitude"]

CONTRASTS = [
    ("full - diagSigma (drop cross-gene covariance)", "posterior", "posterior_diagSigma"),
    ("full - selfremoved (drop target coordinate)", "posterior", "posterior_selfremoved"),
    ("full - controlvar (drop per-pert variance)", "posterior", "posterior_controlvar"),
    ("full - no_uncertainty_inflation", "posterior", "posterior_no_uncinfl"),
    ("full - matched_filter", "posterior", "matched_filter"),
    ("full - magnitude", "posterior", "magnitude"),
    ("diagSigma - diagSigma_selfremoved", "posterior_diagSigma", "posterior_diagSigma_selfremoved"),
]


def read(name):
    with open(name) as fh:
        return list(csv.DictReader(fh))


def main():
    rows = read(IN)
    n = len(rows)
    means = {k: float(np.mean([float(r[k]) for r in rows])) for k in VARIANTS}

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "name", "value", "n_datasets", "p_value"])
        for k in VARIANTS:
            w.writerow(["mean_auc", k, "%.4f" % means[k], n, ""])
        for label, a, b in CONTRASTS:
            xa = np.array([float(r[a]) for r in rows])
            xb = np.array([float(r[b]) for r in rows])
            d = xa - xb
            try:
                p = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue)
            except Exception:
                p = float("nan")
            w.writerow(["contrast", label, "%.4f" % float(np.mean(d)), n,
                        "" if not np.isfinite(p) else "%.4g" % p])

    print("n_datasets =", n)
    for k in VARIANTS:
        print("  {:<34} {:.4f}".format(k, means[k]))
    print("contrasts (full minus variant, mean difference):")
    for label, a, b in CONTRASTS:
        xa = np.array([float(r[a]) for r in rows])
        xb = np.array([float(r[b]) for r in rows])
        d = xa - xb
        try:
            p = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue)
        except Exception:
            p = float("nan")
        print("  {:<46} d={:+.3f}  p={:.3g}".format(label, float(np.mean(d)), p))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
