"""fig10: specific signal by deconfounding variant."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov", "cpm_pc1", "cpm_pc2"]
LAB = {"raw": "raw", "cpm": "CPM", "cpm_ngenes": "CPM +\nresid ngenes",
       "cpm_cov": "CPM +\nresid covars", "raw_cov": "raw +\nresid covars",
       "cpm_pc1": "CPM\n- PC1", "cpm_pc2": "CPM\n- PC1,2"}

rows = list(csv.DictReader(open("deconfound_results.csv")))
mean = []
for v in V:
    vals = [float(r[v]) for r in rows if r[v] not in ("", "nan")]
    mean.append(np.mean(vals) if vals else np.nan)

plt.figure(figsize=(9, 5))
bars = plt.bar(range(len(V)), mean, color=["tab:gray", "tab:gray", "tab:blue",
                                           "tab:blue", "tab:blue", "tab:green", "tab:green"])
plt.axhline(0, color="k", lw=0.8)
plt.xticks(range(len(V)), [LAB[v] for v in V], fontsize=8)
plt.ylabel("mean orthogonal specific correlation")
plt.title("Removing the cell-complexity confound recovers the specific signal")
for i, m in enumerate(mean):
    plt.text(i, m + 0.005, "{:.2f}".format(m), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig10_deconfound.png", dpi=150)
print("wrote figures/fig10_deconfound.png")
