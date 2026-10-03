"""
fig25_meta_epistasis.py -- meta-analysis and combination-perturbation summary.

A. structure strength vs off-axis fraction across datasets (Spearman rho).
B. Norman double-perturbation decomposition: additive fraction, interaction magnitude,
   interaction off-axis fraction, and additivity for PPI vs non-PPI pairs.
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def short(s):
    for a, b in [("ReplogleWeissman2022_", "Repl."), ("NadigOConner2024_", "Nadig."),
                 ("Weissman2019_filtered", "Norman"), ("Kampmann2021_", "Tian "),
                 ("_essential", ""), ("_filtered", ""), ("_perturbseq", ""),
                 ("akana_etal_2026_", "akana-"), ("schemidt_etal_2022_", "schemidt-")]:
        s = s.replace(a, b)
    return s[:14]


def main():
    sm = read("structure_meta.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.array([float(r["mean_offaxis_frac"]) for r in sm])
    y = np.array([float(r["rsa_selfrem_spec"]) for r in sm])
    lab = [short(r["dataset"]) for r in sm]
    ax[0].scatter(x, y, color="#2b6cb0")
    for xi, yi, li in zip(x, y, lab):
        ax[0].annotate(li, (xi, yi), fontsize=7, xytext=(2, 2), textcoords="offset points")
    if np.std(x) > 0:
        rho, p = spearmanr(x, y)
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax[0].plot(xs, b[0] * xs + b[1], "r--", lw=1)
        ax[0].set_title("A. Stronger specific structure where responses are\nless on the depth axis  (rho=%.2f, p=%.2f)" % (rho, p))
    ax[0].set_xlabel("mean off-axis fraction of ΔX"); ax[0].set_ylabel("centred RSA reliability (raw)")

    ep = read("epistasis.csv")
    row = next((r for r in ep if "Norman" in r["dataset"]), None)
    if row:
        labels = ["additive\ncos²", "|I|/|d$_{AB}$|", "I off-axis\nfraction"] 
        vals = [float(row["mean_additive_cos2"]), float(row["mean_I_over_dAB"]),
                float(row["mean_Ioff_over_I"])]
        ax[1].bar([0, 1, 2], vals, 0.5, color=["#276749", "#dd6b20", "#2b6cb0"])
        ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(labels, fontsize=8)
        ax[1].set_ylim(0, 1.0)
        ax[1].set_title("B. Norman doubles: additive, but interactions\nare mostly off-axis (n=%s)" % row["n_doubles"])
        for xi, v in enumerate(vals):
            ax[1].text(xi, v + 0.02, "%.2f" % v, ha="center", fontsize=9)

    fig.suptitle("Meta-analysis and combination perturbations", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "fig25_meta_epistasis.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
