"""
exploration_stats.py -- consolidate the exploratory claims with bootstrap CIs and
one-sample / paired tests across datasets.

Reads the result CSVs and writes ../results/exploration_stats.csv with one row per
claim: label, n, mean, 95% bootstrap CI, test, p-value.
"""

import csv
import os

import numpy as np
from scipy.stats import wilcoxon, spearmanr

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "exploration_stats.csv")


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def col(rows, k):
    return np.array([f(r[k]) for r in rows], float)


def boot_ci(x, n=5000, seed=0):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, x.size, x.size)].mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def one_sample(label, x, alternative="two-sided"):
    x = x[np.isfinite(x)]
    if x.size < 5:
        return None
    try:
        p = float(wilcoxon(x, alternative=alternative).pvalue)
    except Exception:
        p = float("nan")
    lo, hi = boot_ci(x)
    return {"label": label, "n": x.size, "mean": float(x.mean()), "ci_lo": lo, "ci_hi": hi,
            "test": "wilcoxon_vs_0(%s)" % alternative, "p": p}


def paired(label, a, b, alternative="two-sided"):
    a = np.array(a, float); b = np.array(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 5:
        return None
    d = a - b
    try:
        p = float(wilcoxon(a, b, alternative=alternative).pvalue)
    except Exception:
        p = float("nan")
    lo, hi = boot_ci(d)
    return {"label": label, "n": a.size, "mean": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "test": "wilcoxon_paired(%s)" % alternative, "p": p}


def main():
    rows = []

    rr = [r for r in read("residual_reproducibility.csv") if r["space"] == "raw"]
    rows.append(one_sample("specific RSA reliability (raw)", col(rr, "rsa_selfrem_spec"), "greater"))

    cds = read("cross_dataset_specific.csv")
    if cds:
        rows.append(one_sample("cross-dataset cos_specific", col(cds, "cos_specific"), "greater"))
        rows.append(paired("cross-dataset cos_specific - null", col(cds, "cos_specific"),
                           col(cds, "cos_specific_null"), "greater"))

    tes = read("transform_estimator_scan.csv")
    if tes:
        rows.append(paired("operator off-axis - random (all spaces/estimators)",
                           col(tes, "cos2_sigma_off"), col(tes, "cos2_sigma_off_rand")))

    sn = [r for r in read("specific_network.csv") if r["network"] == "PPI"]
    if sn:
        rows.append(paired("PPI edge - non-edge residual cos", col(sn, "sim_edge"),
                           col(sn, "sim_none"), "greater"))

    sp = [r for r in read("specific_pathway.csv") if "Reactome" in r["network"]]
    if sp:
        rows.append(paired("Reactome edge - non-edge residual cos", col(sp, "sim_edge"),
                           col(sp, "sim_none"), "greater"))

    pvd = [r for r in read("program_vs_direction.csv") if str(r["k"]) == "20"]
    if pvd:
        rows.append(one_sample("target-specific direction reliability (after k=20)",
                               col(pvd, "dir_reliab"), "greater"))

    sm = read("second_moment.csv")
    if sm:
        rows.append(paired("variance response - null", col(sm, "var_rel_spec"),
                           col(sm, "var_rel_null_spec"), "greater"))

    ds = read("distribution_specific.csv")
    for k, lab in [("rel_mean", "mean response"), ("rel_var", "variance response"),
                   ("rel_zero", "zero-fraction response"), ("rel_w1", "W1 distributional response")]:
        if ds:
            rows.append(one_sample("%s reliability" % lab, col(ds, k), "greater"))

    tc = read("timecourse.csv")
    if tc:
        rows.append(paired("time-course cos_specific - null", col(tc, "cos_specific"),
                           col(tc, "cos_null"), "greater"))

    spred = read("specific_prediction.csv")
    if spred:
        rows.append(paired("cross-dataset Procrustes - random",
                           col(spred, "cos2_procrustes"), col(spred, "cos2_random"), "greater"))

    rows = [r for r in rows if r]
    # Benjamini-Hochberg FDR across the consolidated exploratory claims
    ps = np.array([r["p"] for r in rows], float)
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, ps[idx] * m / k)
        q[idx] = prev
    for r, qv in zip(rows, q):
        r["q_BH"] = float(qv)
    keys = ["label", "n", "mean", "ci_lo", "ci_hi", "test", "p", "q_BH"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            print("%-52s n=%-3d mean=%+.3f [%+.3f,%+.3f] p=%.3g q=%.3g" % (
                r["label"][:51], r["n"], r["mean"], r["ci_lo"], r["ci_hi"], r["p"], r["q_BH"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
