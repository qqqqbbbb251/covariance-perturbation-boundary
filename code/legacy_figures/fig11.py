"""fig11: robustness to the definition of the global mode + graphical abstract."""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- fig11 ----
rows = list(csv.DictReader(open("results/robustness_results.csv")))
P = ["full", "pc1_raw", "counts_dir", "pc1_cpm", "pc12_raw"]
LAB = ["full\n(covariance)", "PC1\n(raw)", "counts\ndirection", "PC1\n(CPM)", "PC1-2\n(raw)"]
mean = []
for p in P:
    vals = [float(r[p]) for r in rows if r[p] not in ("", "nan")]
    mean.append(np.mean(vals) if vals else np.nan)

plt.figure(figsize=(8, 5))
plt.bar(range(len(P)), mean, color=["#2b6cb0"] + ["#dd6b20"] * 4)
plt.xticks(range(len(P)), LAB, fontsize=9)
plt.ylabel("mean $R^2$")
plt.title("A single global direction matches the full model,\nregardless of how it is defined")
for i, m in enumerate(mean):
    plt.text(i, m + 0.005, "{:.2f}".format(m), ha="center", fontsize=9)
plt.ylim(0, max(mean) * 1.25)
plt.tight_layout()
plt.savefig("figures/fig11_robustness.png", dpi=150)
plt.close()
print("wrote figures/fig11_robustness.png")
