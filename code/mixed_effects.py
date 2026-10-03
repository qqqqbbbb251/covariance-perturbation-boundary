"""
mixed_effects.py  (reviewer point 9, per-dataset effects + mixed model)

Uses the per-perturbation R2 saved by hierarchical_stats.py (per_pert_r2.npz,
shape (3, n_pert) = [full, global, random] for each dataset).

  * per-dataset median / IQR for full, global, random and for the paired
    differences (full - global, full - random),
  * a linear mixed-effects model with a random intercept per dataset for each
    paired difference:  diff ~ 1 + (1 | dataset).  The fixed intercept is the
    overall effect; its 95% CI and p-value test whether the difference differs
    from zero while accounting for the dataset clustering.

Output: ../results/per_dataset_summary.csv, ../results/mixed_effects.csv
"""

import os
import sys

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NPZ = os.path.join(RESULTS, "per_pert_r2.npz")
SUMMARY = os.path.join(RESULTS, "per_dataset_summary.csv")
MIXED = os.path.join(RESULTS, "mixed_effects.csv")


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    if not os.path.exists(NPZ):
        sys.exit("missing {} (run hierarchical_stats.py first)".format(NPZ))
    z = np.load(NPZ, allow_pickle=True)
    names = list(z.files)

    with open(SUMMARY, "w") as fh:
        fh.write("dataset,n_pert,full_median,full_iqr,global_median,global_iqr,"
                 "random_median,random_iqr,dFG_median,dFG_iqr,dFR_median,dFR_iqr\n")
    import pandas as pd
    long_rows = []
    for name in names:
        arr = z[name]
        full, globr, rand = arr[0], arr[1], arr[2]
        fg, fr = full - globr, full - rand
        with open(SUMMARY, "a") as fh:
            fh.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:+.4f},{:.4f},{:+.4f},{:.4f}\n".format(
                name, len(full), np.median(full), iqr(full), np.median(globr), iqr(globr),
                np.median(rand), iqr(rand), np.median(fg), iqr(fg), np.median(fr), iqr(fr)))
        for v, g, r in zip(full, globr, rand):
            long_rows.append({"dataset": name, "full": float(v), "global": float(g),
                              "random": float(r), "dFG": float(v - g), "dFR": float(v - r),
                              "dGR": float(g - r)})

    df = pd.DataFrame(long_rows)
    print("per-dataset summary ->", SUMMARY)
    print(df.groupby("dataset")[["full", "global", "random", "dFG", "dFR"]].median().round(3).to_string())

    import statsmodels.formula.api as smf
    with open(MIXED, "w") as fh:
        fh.write("model,effect,estimate,se,ci_lo,ci_hi,z,p,n_groups,n_obs,"
                 "group_var,resid_var,icc,converged\n")
        labels = {"dFG": "full_minus_global", "dFR": "full_minus_random",
                  "dGR": "global_minus_random"}
        for outcome in ["dFG", "dFR", "dGR"]:
            sub = df[np.isfinite(df[outcome])].copy()
            md = smf.mixedlm("{} ~ 1".format(outcome), sub, groups=sub["dataset"]).fit(reml=False)
            est = float(md.params["Intercept"])
            se = float(md.bse["Intercept"])
            z = est / se if se else float("nan")
            ci = md.conf_int().loc["Intercept"].values.astype(float)
            p = float(md.pvalues["Intercept"])
            gvar = float(md.cov_re.iloc[0, 0])
            rvar = float(md.scale)
            icc = gvar / (gvar + rvar) if (gvar + rvar) > 0 else float("nan")
            conv = bool(getattr(md, "converged", False))
            fh.write("{},{},{:+.4f},{:.4f},{:+.4f},{:+.4f},{:.3f},{:.3g},{},{},"
                     "{:.4f},{:.4f},{:.3f},{}\n".format(
                labels[outcome], "intercept(fixed mean)", est, se, ci[0], ci[1], z, p,
                sub["dataset"].nunique(), len(sub), gvar, rvar, icc, conv))
            print("\nMixedLM  {} ~ 1 + (1|dataset):".format(outcome))
            print("  fixed mean = {:+.4f} (SE {:.4f})  95% CI [{:+.4f}, {:+.4f}]  p = {:.3g}".format(
                est, se, ci[0], ci[1], p))
    print("\nwrote", SUMMARY, MIXED)


if __name__ == "__main__":
    main()
