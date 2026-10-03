"""
fig15_16.py

fig15: residual decomposition (self / shared / residual) and the residual
       full-vs-matched-random test.
fig16: cross-dataset transfer of the covariance model.
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


def num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def fig15():
    rows = [r for r in read("residual_decomposition.csv") if r["R2_self"] not in ("", "nan")]
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows))
    self_ = np.array([num(r["R2_self"]) for r in rows])
    shared = np.array([num(r["R2_shared"]) for r in rows])
    resid = np.array([num(r["R2_residual"]) for r in rows])
    rf = np.array([num(r["resid_full"]) for r in rows])
    rr = np.array([num(r["resid_random"]) for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x, self_, label="self (own gene)", color="#2b6cb0")
    ax[0].bar(x, shared, bottom=self_, label="shared (global mode)", color="#dd6b20")
    ax[0].bar(x, resid, bottom=self_ + shared, label="residual", color="#a0aec0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("fraction of $||\\Delta X||^2$ (Pearson)")
    ax[0].set_title("A. Response is mostly residual; self is trivial")
    ax[0].legend(fontsize=8)

    w = 0.38
    ax[1].bar(x - w / 2, rf, w, label="perturbed column", color="#2b6cb0")
    ax[1].bar(x + w / 2, rr, w, label="matched random", color="#dd6b20")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("$R^2$ on the residual")
    ax[1].set_title("B. Covariance column does not predict the residual")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "fig15_residual_decomposition.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig16():
    rows = read("cross_dataset_transfer.csv")
    names = [r["pair"].replace("NadigOConner2024_", "").replace("ReplogleWeissman2022_", "")
             .replace("TianKampmann2021_", "").replace("->", " -> ")[:22] for r in rows]
    keys = ["within_full", "cross_full", "cross_global", "cross_random"]
    lab = ["within full", "cross full", "cross global", "cross random"]
    col = ["#2b6cb0", "#dd6b20", "#276749", "#a0aec0"]
    x = np.arange(len(rows)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (k, l, c) in enumerate(zip(keys, lab, col)):
        ax.bar(x + (i - 1.5) * w, [num(r[k + "_cos"]) for r in rows], w, label=l, color=c)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("mean cosine on target dataset")
    ax.set_title("Cross-dataset transfer matches the within-dataset model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "fig16_cross_dataset_transfer.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig15()
    fig16()
