"""
exp12_stats.py -- formal statistics for Exp1 (differential) and Exp2 (identity).

Reads results/differential_identity.csv and reports, per space (raw / pearson):
  * mean, median, IQR and 95% CI (bootstrap over datasets) of
    pc_pred, pc_true, RSA, diff_pred, diff_shared, full_cos, shared_frac
  * one-sample Wilcoxon of RSA vs 0 (is there any perturbation-specific structure?)
  * paired Wilcoxon pc_pred vs pc_true (are predictions more collinear than truth?)

Output: ../results/exp12_stats.csv
"""

import csv
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
IN = os.path.join(RESULTS, "differential_identity.csv")
OUT = os.path.join(RESULTS, "exp12_stats.csv")


def num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = list(csv.DictReader(open(IN)))
    out = []
    for space in ["raw", "pearson", "cpm"]:
        sub = [r for r in rows if r["space"] == space and np.isfinite(num(r["rsa"]))]
        if len(sub) < 3:
            continue
        def arr(k):
            return np.array([num(r[k]) for r in sub])
        print("=== {} (n={} datasets) ===".format(space, len(sub)))
        for k in ["full_cos", "shared_frac", "diff_shared", "diff_pred_cos",
                  "paircos_true", "paircos_pred", "rsa"]:
            v = arr(k)
            lo, hi = boot_ci(v)
            print("  {:<14} mean={:+.3f} median={:+.3f} IQR={:.3f} 95%CI[{:+.3f},{:+.3f}]".format(
                k, v.mean(), np.median(v), np.percentile(v, 75) - np.percentile(v, 25), lo, hi))
            out.append((space, k, "mean", v.mean(), lo, hi, len(v)))
        # RSA vs 0
        try:
            from scipy.stats import wilcoxon
            p_rsa = float(wilcoxon(arr("rsa")).pvalue)
        except Exception:
            p_rsa = float("nan")
        print("  RSA vs 0 (Wilcoxon p) = {:.3g}".format(p_rsa))
        out.append((space, "rsa_vs_0", "wilcoxon_p", p_rsa, float("nan"), float("nan"), len(sub)))
        # pc_pred vs pc_true
        try:
            from scipy.stats import wilcoxon
            p_pc = float(wilcoxon(arr("paircos_pred"), arr("paircos_true")).pvalue)
        except Exception:
            p_pc = float("nan")
        print("  pc_pred vs pc_true (Wilcoxon p) = {:.3g}".format(p_pc))
        out.append((space, "pcpred_vs_pctrue", "wilcoxon_p", p_pc, float("nan"), float("nan"), len(sub)))
        print()

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["space", "metric", "stat", "value", "ci_lo", "ci_hi", "n_datasets"])
        for r in out:
            w.writerow([r[0], r[1], r[2], "{:.6g}".format(r[3]),
                        "{:.6g}".format(r[4]) if np.isfinite(r[4]) else "nan",
                        "{:.6g}".format(r[5]) if np.isfinite(r[5]) else "nan", r[6]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
