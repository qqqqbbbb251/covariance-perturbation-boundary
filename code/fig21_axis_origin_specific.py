"""
fig21_axis_origin_specific.py -- Stage 1+2 exploratory figure.

A. top-1 covariance share before vs after equal-depth resampling (axis_origin.csv)
B. PC1 nuisance R^2 and |corr(PC1, gene-mean)| (axis_origin.csv)
C. split-half RSA reliability of the residual specific structure, centered and
   self-removed (residual_reproducibility.csv, rsa_selfrem_spec)
D. cross-cell-line agreement of the specific residual vs shuffled null
   (cross_dataset_specific.csv)
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def short(s, n=12):
    return s.replace("ReplogleWeissman2022_", "Replogle_").replace("NadigOConner2024_", "Nadig_") \
            .replace("Weissman2019_filtered", "").replace("Kampmann2021_", "") \
            .replace("_essential", "")[:n]


def main():
    ao = read("axis_origin.csv")
    rr = read("residual_reproducibility.csv")
    cs = read("cross_dataset_specific.csv")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # A
    names = [short(r["dataset"]) for r in ao]
    x = np.arange(len(ao)); w = 0.38
    ax[0, 0].bar(x - w / 2, [float(r["top1_frac"]) for r in ao], w, label="raw", color="#2b6cb0")
    ax[0, 0].bar(x + w / 2, [float(r["top1_frac_eq"]) for r in ao], w, label="equal-depth", color="#dd6b20")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0, 0].set_ylabel("top-1 covariance variance share"); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("A. Equal-depth resampling removes most of the\ndominant axis"); ax[0, 0].legend(fontsize=8)

    # B
    ax[0, 1].bar(x - w / 2, [float(r["R2_nuisance"]) for r in ao], w, label="R$^2$ (depth/#genes/mito/ribo)", color="#2b6cb0")
    ax[0, 1].bar(x + w / 2, [abs(float(r["r_v_geneMean"])) for r in ao], w, label="|corr(PC1, gene mean)|", color="#dd6b20")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_title("B. PC1 is a technical scaling direction")
    ax[0, 1].legend(fontsize=7, loc="lower right")

    # C
    spaces = ["raw", "cpm", "pearson"]
    dss = sorted({r["dataset"] for r in rr})
    xs = np.arange(len(dss)); ws = 0.26
    cols = {"raw": "#2b6cb0", "cpm": "#dd6b20", "pearson": "#276749"}
    for k, sp in enumerate(spaces):
        vals = []
        for ds in dss:
            row = [r for r in rr if r["dataset"] == ds and r["space"] == sp]
            vals.append(float(row[0]["rsa_selfrem_spec"]) if row else np.nan)
        ax[1, 0].bar(xs + (k - 1) * ws, vals, ws, label=sp, color=cols[sp])
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels([short(d) for d in dss], rotation=45, ha="right", fontsize=8)
    ax[1, 0].set_ylabel("split-half RSA reliability (centered, self-removed)")
    ax[1, 0].set_ylim(-0.1, 1.0)
    ax[1, 0].set_title("C. Real perturbation-specific trans structure\nin several datasets (not self, not one common axis)")
    ax[1, 0].legend(fontsize=8)

    # D
    xn = np.arange(len(cs)); wn = 0.38
    ax[1, 1].bar(xn - wn / 2, [float(r["cos_specific"]) for r in cs], wn, label="same target", color="#276749")
    ax[1, 1].bar(xn + wn / 2, [float(r["cos_specific_null"]) for r in cs], wn, label="shuffled targets", color="#a0aec0")
    ax[1, 1].axhline(0, color="k", lw=0.8)
    ax[1, 1].set_xticks(xn)
    ax[1, 1].set_xticklabels([r["pair"].replace("ReplogleWeissman2022_", "Repl.").replace("NadigOConner2024_", "Nadig.")
                              .replace("TianKampmann2021_", "Tian ").replace(".h5ad", "").replace("_essential", "")
                              for r in cs], rotation=20, ha="right", fontsize=7)
    ax[1, 1].set_ylabel("cross-dataset specific-residual cosine")
    ax[1, 1].set_ylim(-0.1, 0.5)
    ax[1, 1].set_title("D. Specific structure transfers across cell lines\n(but not across CRISPRa vs CRISPRi)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("The dominant axis is a depth artifact; a real, transferable perturbation-specific structure exists beneath it",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIG, "fig21_axis_origin_specific.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
