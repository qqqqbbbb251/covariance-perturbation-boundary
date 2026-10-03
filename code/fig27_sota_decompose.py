"""
fig27_sota_decompose.py -- model predictions under the program/amplitude/direction
decomposition (CIPHER, linear_mean, ... as available).

Reads ../results/sota_decompose.csv (written by code/sota_decompose.py).  Skips
gracefully if absent.
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
IN = os.path.join(RES, "sota_decompose.csv")


def main():
    if not os.path.exists(IN):
        print("skip: no sota_decompose.csv"); return
    with open(IN) as fh:
        rows = list(csv.DictReader(fh))
    models = [r["model"] for r in rows]
    x = np.arange(len(models)); w = 0.2
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for i, (k, lab, c) in enumerate([("fwd_cos2", "forward cos²", "#2b6cb0"),
                                     ("rsa_full", "RSA (full)", "#dd6b20"),
                                     ("specific_cos2", "specific cos²", "#276749"),
                                     ("pc_pred", "pred collinearity", "#805ad5")]):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                vals.append(np.nan)
        ax[0].bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
    ax[0].set_xticks(x); ax[0].set_xticklabels(models, fontsize=9)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("value")
    ax[0].set_title("A. Model predictions vs the decomposition")
    ax[0].legend(fontsize=8)

    ax[1].bar(x - w / 2, [float(r["pc_true"]) for r in rows], w, label="true", color="#718096")
    ax[1].bar(x + w / 2, [float(r["pc_pred"]) for r in rows], w, label="predicted", color="#dd6b20")
    ax[1].set_xticks(x); ax[1].set_xticklabels(models, fontsize=9)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("mean pairwise |cos|")
    ax[1].set_title("B. Predictions collapse to the shared direction")
    ax[1].legend(fontsize=8)
    fig.suptitle("SOTA predictions under the program/amplitude/direction decomposition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "fig27_sota_decompose.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
