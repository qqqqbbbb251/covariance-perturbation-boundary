"""
real_power_analysis.py  (reviewer: reconcile the positive control with TOST)

The synthetic specificity positive control used i.i.d. noise whose per-perturbation
SD (~0.002) is ~50x smaller than the real data's, so it overstated the test's
sensitivity.  This script computes the power of the ACTUAL test on the REAL
per-perturbation differences, by injecting a shift delta into the observed
full-global / full-matched-random differences:

  * pooled per-perturbation paired Wilcoxon
  * dataset-clustered one-sample t-test across the 10 resampled dataset means
    (a proxy for the hierarchical bootstrap used in the paper)

and reports the minimum detectable effect (MDE, 80% power) and power at a grid of
deltas.

Input : results/per_pert_r2.npz
Output: results/real_power_analysis.csv
"""

import csv
import os

import numpy as np
from scipy.stats import wilcoxon, ttest_1samp

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NPZ = os.path.join(RESULTS, "per_pert_r2.npz")
OUT = os.path.join(RESULTS, "real_power_analysis.csv")

DELTAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
B = 400


def pooled_power(d, delta, rng):
    n = len(d)
    sig = 0
    for _ in range(B):
        x = d[rng.integers(0, n, n)] + delta
        if ttest_1samp(x, 0).pvalue < 0.05:
            sig += 1
    return sig / B


def clustered_power(by_ds, delta, rng):
    D = len(by_ds)
    sig = 0
    for _ in range(B):
        vals = []
        for di in rng.integers(0, D, D):
            arr = by_ds[di]
            vals.append(arr[rng.integers(0, len(arr), len(arr))].mean())
        try:
            if ttest_1samp(np.asarray(vals) + delta, 0).pvalue < 0.05:
                sig += 1
        except Exception:
            pass
    return sig / B


def mde(deltas, powers, target=0.8):
    for dl, pw in zip(deltas, powers):
        if pw >= target:
            return dl
    return float("nan")


def main():
    z = np.load(NPZ, allow_pickle=True)
    names = list(z.files)
    rows = []
    print("Real-data power of the paired specificity test (per_pert_r2.npz):")
    for label, fn in [("full-global", lambda f, g, r: f - g),
                      ("full-matchedrandom", lambda f, g, r: f - r)]:
        by_ds = [np.asarray(fn(z[k][0], z[k][1], z[k][2]), float) for k in names]
        d = np.concatenate(by_ds)
        d = d[np.isfinite(d)]
        dc = d - d.mean()
        sd = dc.std(ddof=1)
        n = len(dc)
        rng = np.random.default_rng(0)
        pp = [pooled_power(dc, dl, rng) for dl in DELTAS]
        rng = np.random.default_rng(0)
        cp = [clustered_power([a - d.mean() for a in by_ds], dl, rng) for dl in DELTAS]
        mde_p, mde_c = mde(DELTAS, pp), mde(DELTAS, cp)
        print("\n{}: n={} per-pert SD={:.4f}  analytic pooled MDE(80%)={:.4f}".format(
            label, n, sd, (1.96 + 0.84) * sd / np.sqrt(n)))
        print("  delta:      " + "  ".join("{:>7.3f}".format(x) for x in DELTAS))
        print("  pooled pow: " + "  ".join("{:>7.2f}".format(x) for x in pp))
        print("  clust  pow: " + "  ".join("{:>7.2f}".format(x) for x in cp))
        print("  MDE(80%): pooled={} clustered={}".format(mde_p, mde_c))
        rows.append((label, n, sd, mde_p, mde_c, pp, cp))

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "n", "per_pert_sd", "mde80_pooled", "mde80_clustered"]
                   + ["power_pooled_{}".format(x) for x in DELTAS]
                   + ["power_clustered_{}".format(x) for x in DELTAS])
        for label, n, sd, mp, mc, pp, cp in rows:
            w.writerow([label, n, "{:.4f}".format(sd), mp, mc]
                       + ["{:.3f}".format(x) for x in pp] + ["{:.3f}".format(x) for x in cp])
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
