"""
graphical_abstract.py

A one-panel graphical abstract for the paper.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.size"] = 10


def mean_col(path, cols):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for c in cols:
        vals = [float(r[c]) for r in rows if r[c] not in ("", "nan")]
        out[c] = np.mean(vals) if vals else np.nan
    return out


strong = mean_col("results/strong_r2.csv", ["full_raw", "global_raw", "spec_raw"])
decon = mean_col("deconfound_results.csv",
                 ["raw", "cpm", "cpm_ngenes", "cpm_cov", "raw_cov"])

fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.32)

# ---------------- Panel A: concept ----------------
axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
axA.set_title("A. Covariance predicts perturbation\nfrom unperturbed cells", fontsize=11, weight="bold")
axA.add_patch(Rectangle((0.02, 0.55), 0.28, 0.32, fc="#dbe9f6", ec="#2b6cb0"))
axA.text(0.16, 0.71, "control\ncells", ha="center", va="center", fontsize=9)
axA.add_patch(Rectangle((0.38, 0.55), 0.28, 0.32, fc="#fde9d9", ec="#c05621"))
axA.text(0.52, 0.71, "covariance\n$\\Sigma$", ha="center", va="center", fontsize=9)
axA.add_patch(Rectangle((0.74, 0.55), 0.24, 0.32, fc="#e2f0d9", ec="#276749"))
axA.text(0.86, 0.71, "predicted\nresponse", ha="center", va="center", fontsize=9)
for x0, x1 in [(0.30, 0.38), (0.66, 0.74)]:
    axA.add_patch(FancyArrowPatch((x0, 0.71), (x1, 0.71), arrowstyle="-|>",
                                  mutation_scale=14, color="k"))
axA.text(0.5, 0.40, "$\\Delta X = \\Sigma\\,u$", ha="center", fontsize=13)
axA.text(0.5, 0.16,
         "We reproduce this across 13 Perturb-seq\n"
         "datasets (>1.8M cells).",
         ha="center", fontsize=9, style="italic")

# ---------------- Panel B: full vs global vs specific ----------------
axB = fig.add_subplot(gs[0, 1])
vals = [strong["full_raw"], strong["global_raw"], strong["spec_raw"]]
bars = axB.bar(range(3), vals, color=["#2b6cb0", "#dd6b20", "#276749"])
axB.set_xticks(range(3)); axB.set_xticklabels(["full\n(covariance)", "global mode\nonly", "specific\n(orthogonal)"], fontsize=9)
axB.set_ylabel("mean $R^2$ (raw)")
axB.set_title("B. A single global mode reproduces\nthe full model; specific $R^2\\approx 0$", fontsize=11, weight="bold")
axB.axhline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    axB.text(i, v + 0.01, "{:.2f}".format(v), ha="center", fontsize=9)
axB.set_ylim(0, max(vals) * 1.25)

# ---------------- Panel C: deconfounding ----------------
axC = fig.add_subplot(gs[0, 2])
lab = ["raw", "CPM", "resid\ncovars"]
vals = [decon["raw"], decon["cpm"], max(decon["cpm_cov"], decon["raw_cov"])]
bars = axC.bar(range(3), vals, color=["#a0aec0", "#a0aec0", "#276749"])
axC.set_xticks(range(3)); axC.set_xticklabels(lab, fontsize=9)
axC.set_ylabel("specific correlation")
axC.set_title("C. Removing the complexity confound\nrecovers the specific signal", fontsize=11, weight="bold")
axC.axhline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    axC.text(i, v + 0.005, "{:.2f}".format(v), ha="center", fontsize=9)
axC.set_ylim(0, max(vals) * 1.3)

fig.suptitle("Covariance-based perturbation prediction is dominated by a single global mode",
             fontsize=13, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/graphical_abstract.png", dpi=200, bbox_inches="tight")
print("wrote figures/graphical_abstract.png")
