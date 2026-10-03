"""
merge_results.py

Builds ONE canonical results table under the unified definition:
  global mode  = leading eigenvector of the control covariance
  specific     = delta orthogonalised against the global mode
and merges the outputs of scan_final (global fraction, full R2),
deconfound_final (specific by variant), robustness_test, ci_bootstrap.

Output: unified_results.csv and a printed table.
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    for p in [name, os.path.join(HERE, name),
              os.path.join(HERE, "..", "results", name)]:
        if os.path.exists(p):
            with open(p) as fh:
                return {r["dataset"]: r for r in csv.DictReader(fh)}
    return {}


def f(d, k):
    try:
        v = d[k]
        return float(v) if v not in ("", "nan", None) else float("nan")
    except Exception:
        return float("nan")


scan = read("scan_results.csv")          # global_frac_raw, full_r2_raw
dec = read("deconfound_results.csv")     # raw, cpm, raw_cov, cpm_cov
rob = read("robustness_results.csv")     # full, pc1_raw, counts_dir, pc12_raw
ci = read("ci_results.csv")              # raw_mean, cpm_mean, deconf_mean + CI

names = sorted(set(scan) | set(dec) | set(rob) | set(ci))
rows = []
for n in names:
    s, d, r, c = scan.get(n, {}), dec.get(n, {}), rob.get(n, {}), ci.get(n, {})
    rows.append({
        "dataset": n,
        "global_frac": f(s, "global_frac_raw"),
        "full_R2": f(s, "full_r2_raw"),
        "full_rob": f(r, "full"),
        "pc1_rob": f(r, "pc1_raw"),
        "counts_rob": f(r, "counts_dir"),
        "spec_raw": f(d, "raw"),
        "spec_cpm": f(d, "cpm"),
        "spec_deconf": f(d, "raw_cov"),
        "ci_raw_lo": f(c, "raw_lo"),
        "ci_raw_hi": f(c, "raw_hi"),
        "ci_deconf_lo": f(c, "deconf_lo"),
        "ci_deconf_hi": f(c, "deconf_hi"),
    })

cols = list(rows[0].keys())
with open("unified_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("{:<34}{:>8}{:>8}{:>9}{:>9}{:>10}{:>10}{:>11}".format(
    "dataset", "glob", "fullR2", "rob_full", "rob_pc1", "spec_raw", "spec_cpm", "spec_deconf"))
print("-" * 99)
for r in rows:
    def g(k):
        v = r[k]
        return "{:.3f}".format(v) if v == v else "  nan"
    print("{:<34}{:>8}{:>8}{:>9}{:>9}{:>10}{:>10}{:>11}".format(
        r["dataset"][:33], g("global_frac"), g("full_R2"), g("full_rob"),
        g("pc1_rob"), g("spec_raw"), g("spec_cpm"), g("spec_deconf")))

print("\nMEANS (over datasets with values):")
for k in ["global_frac", "full_R2", "full_rob", "pc1_rob", "counts_rob",
          "spec_raw", "spec_cpm", "spec_deconf"]:
    vals = [r[k] for r in rows if r[k] == r[k]]
    print("  {:<12} {:.3f}  (n={})".format(k, sum(vals) / len(vals) if vals else float("nan"), len(vals)))
print("wrote unified_results.csv")
