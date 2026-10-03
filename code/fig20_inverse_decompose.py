"""
fig20_inverse_decompose.py -- Exp4 figure: why does the inverse (driver recovery) work?

Panel A: mean driver-recovery AUC across datasets for each ablation of the official
         fullH_diag posterior inverse.
Panel B: per-dataset posterior vs diagSigma (cross-gene covariance removed) vs
         diagSigma+self-removed (target's own coordinate also removed).
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

VARIANTS = ["posterior", "posterior_selfremoved", "posterior_controlvar",
            "posterior_controlvar_selfremoved", "posterior_diagSigma",
            "posterior_diagSigma_selfremoved", "posterior_no_uncinfl",
            "posterior_stdonly", "matched_filter", "magnitude"]

PRETTY = {
    "posterior": "official posterior (all signals)",
    "posterior_selfremoved": "drop target's own coordinate",
    "posterior_controlvar": "drop per-perturbation variance",
    "posterior_controlvar_selfremoved": "drop per-pert var + target coord",
    "posterior_diagSigma": "drop cross-gene covariance",
    "posterior_diagSigma_selfremoved": "drop covariance + target coord",
    "posterior_no_uncinfl": "drop uncertainty inflation",
    "posterior_stdonly": "std-only (no observed dx)",
    "matched_filter": "matched filter",
    "magnitude": "|dx| (self magnitude only)",
}


def read(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def main():
    rows = read("inverse_decompose.csv")
    n = len(rows)
    means = {k: float(np.mean([float(r[k]) for r in rows])) for k in VARIANTS}
    order = sorted(VARIANTS, key=lambda k: means[k])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    colors = []
    for k in order:
        if k == "posterior":
            colors.append("#1a365d")
        elif k == "posterior_diagSigma":
            colors.append("#2b6cb0")
        elif k in ("posterior_diagSigma_selfremoved", "posterior_stdonly"):
            colors.append("#c53030")
        else:
            colors.append("#a0aec0")
    y = np.arange(len(order))
    ax[0].barh(y, [means[k] for k in order], color=colors)
    ax[0].axvline(0.5, color="k", ls="--", lw=1)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([PRETTY[k] for k in order], fontsize=8)
    ax[0].set_xlabel("mean driver-recovery AUC (8 datasets)")
    ax[0].set_xlim(0.3, 1.0)
    ax[0].set_title("A. Driver recovery survives removing covariance,\ncollapses when the target's own coordinate is removed")
    for yi, k in zip(y, order):
        ax[0].text(means[k] + 0.005, yi, "%.3f" % means[k], va="center", fontsize=7)

    names = [r["dataset"][:13] for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    ax[1].bar(x - w, [float(r["posterior"]) for r in rows], w, label="official posterior", color="#1a365d")
    ax[1].bar(x, [float(r["posterior_diagSigma"]) for r in rows], w, label="diag($\\Sigma$) (no cross-gene cov)", color="#2b6cb0")
    ax[1].bar(x + w, [float(r["posterior_diagSigma_selfremoved"]) for r in rows], w,
              label="diag($\\Sigma$) + target coordinate removed", color="#c53030")
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("driver-recovery AUC")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("B. Same picture in every dataset")
    ax[1].legend(fontsize=7, loc="lower right")

    fig.suptitle("The inverse identifies the driver from its own expression change, not from covariance structure", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(FIG, "fig20_inverse_decomposition.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p, "n=", n)


if __name__ == "__main__":
    main()
