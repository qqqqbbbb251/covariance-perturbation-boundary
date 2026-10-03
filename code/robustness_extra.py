"""
robustness_extra.py  (reviewer points 2 and 3, strengthened)

Two robustness checks requested by reviewers:

  1. Is the "global mode is strongest" advantage driven by a few datasets?
     * per-dataset mean full-global / full-random / global-random
     * leave-one-dataset-out means (does the sign survive dropping each dataset?)
     * dataset-level paired Wilcoxon (n = 10)

  2. Significance of the global advantage in pflog (the reproduction table only
     reported means).  Dataset-level paired tests for global-full, global-random
     and full-random in both raw and pflog.

Inputs : results/per_pert_r2.npz, results/cipher_reproduction_table.csv
Output : results/robustness_extra.csv
"""

import csv
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NPZ = os.path.join(RESULTS, "per_pert_r2.npz")
TABLE = os.path.join(RESULTS, "cipher_reproduction_table.csv")
OUT = os.path.join(RESULTS, "robustness_extra.csv")


def wilcoxon(x, y):
    from scipy.stats import wilcoxon as _w
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.allclose(d, 0):
        return float("nan")
    return float(_w(d).pvalue)


def main():
    rows = []
    # ---- 1. per-dataset and leave-one-out (scale-only cosine, matched random) ----
    if os.path.exists(NPZ):
        z = np.load(NPZ, allow_pickle=True)
        names = list(z.files)
        fg = np.array([np.nanmean(z[k][0] - z[k][1]) for k in names])
        fr = np.array([np.nanmean(z[k][0] - z[k][2]) for k in names])
        gr = np.array([np.nanmean(z[k][1] - z[k][2]) for k in names])
        print("Per-dataset means (scale-only cosine, matched random):")
        for i, k in enumerate(names):
            print("  {:<32} full-global={:+.4f} full-random={:+.4f} global-random={:+.4f}".format(
                k[:31], fg[i], fr[i], gr[i]))
            rows.append(("per_dataset", k, "full-global", fg[i]))
            rows.append(("per_dataset", k, "full-random", fr[i]))
            rows.append(("per_dataset", k, "global-random", gr[i]))
        print("\nLeave-one-dataset-out (mean over remaining 9):")
        for arr, name in [(fg, "full-global"), (fr, "full-random"), (gr, "global-random")]:
            loo = np.array([np.delete(arr, i).mean() for i in range(len(arr))])
            frac_pos = float(np.mean(arr > 0))
            print("  {:<14} mean={:+.4f}  LOO range [{:+.4f},{:+.4f}]  per-dataset frac>0={:.2f}".format(
                name, arr.mean(), loo.min(), loo.max(), frac_pos))
            rows.append(("leave_one_out", name, "mean", float(arr.mean())))
            rows.append(("leave_one_out", name, "loo_min", float(loo.min())))
            rows.append(("leave_one_out", name, "loo_max", float(loo.max())))
            rows.append(("leave_one_out", name, "frac_pos", frac_pos))
        print("  dataset-level paired Wilcoxon (n=10):")
        for arr, name in [(fg, "full-global"), (fr, "full-random"), (gr, "global-random")]:
            p = wilcoxon(arr, np.zeros_like(arr))
            print("    {:<14} p={:.3g}".format(name, p))
            rows.append(("wilcoxon_dataset", name, "p", p))

    # ---- 2. pflog / raw significance from the reproduction table ----
    if os.path.exists(TABLE):
        tab = list(csv.DictReader(open(TABLE)))
        print("\nReproduction table, dataset-level paired tests (n = 10):")
        for norm in ["raw", "pflog"]:
            r = [x for x in tab if x["norm"] == norm and x["full"] not in ("nan", "")]
            if len(r) < 6:
                continue
            full = np.array([float(x["full"]) for x in r])
            glob = np.array([float(x["global"]) for x in r])
            rand = np.array([float(x["random"]) for x in r])
            for a, b, nm in [(glob, full, "global-full"), (glob, rand, "global-random"),
                             (full, rand, "full-random")]:
                d = a - b
                p = wilcoxon(a, b)
                print("  {:<6} {:<14} mean={:+.4f}  p={:.3g}".format(norm, nm, d.mean(), p))
                rows.append(("repro_" + norm, nm, "mean", float(d.mean())))
                rows.append(("repro_" + norm, nm, "p", p))

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "key", "stat", "value"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], "{:.6g}".format(r[3]) if isinstance(r[3], float) else r[3]])
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
