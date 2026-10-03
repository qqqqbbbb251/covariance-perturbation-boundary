"""
fig_new.py -- figures for the new forward-indistinguishability / reverse-boundary
results.

fig17: per-dataset prediction collinearity (pc_pred vs pc_true) and RSA (raw).
fig18: official posterior inverse AUC / top-10 per dataset.
fig19: eigenvalue spectrum (top1/top5 fraction) and rank-k prediction R2.
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
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def fig17():
    rows = [r for r in read("differential_identity.csv") if r["space"] == "raw" and num(r["rsa"]) == num(r["rsa"])]
    if not rows:
        return
    rows.sort(key=lambda r: num(r["paircos_pred"]))
    names = [r["dataset"][:14] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x - w / 2, [num(r["paircos_true"]) for r in rows], w, label="true responses", color="#2b6cb0")
    ax[0].bar(x + w / 2, [num(r["paircos_pred"]) for r in rows], w, label="CIPHER predictions", color="#dd6b20")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("mean pairwise |cos|")
    ax[0].set_title("A. Predictions collapse to one direction")
    ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.05)

    ax[1].bar(x, [num(r["rsa"]) for r in rows], color="#4a5568")
    ax[1].axhline(0, color="k", lw=0.9)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("RSA (true vs predicted structure)")
    ax[1].set_title("B. No perturbation-specific structure preserved")
    fig.tight_layout()
    p = os.path.join(FIG, "fig17_forward_indistinguishability.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig18():
    rows = [r for r in read("inverse_official.csv") if num(r["mean_per_pert_auc"]) == num(r["mean_per_pert_auc"])]
    if not rows:
        return
    rows.sort(key=lambda r: num(r["mean_per_pert_auc"]))
    names = [r["dataset"][:16] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x, [num(r["mean_per_pert_auc"]) for r in rows], color="#276749")
    ax[0].axhline(0.5, color="r", ls="--", lw=1, label="chance")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("driver-recovery AUC"); ax[0].set_ylim(0.4, 1.02)
    ax[0].set_title("A. Inverse (driver ID) is accurate"); ax[0].legend(fontsize=8)
    ax[1].bar(x, [num(r["top10"]) for r in rows], color="#2b6cb0")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("top-10 accuracy"); ax[1].set_ylim(0, 1)
    ax[1].set_title("B. Top-10 driver recovery")
    fig.tight_layout()
    p = os.path.join(FIG, "fig18_inverse_driver.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


def fig19():
    rows = [r for r in read("spectrum_rank.csv") if r["space"] == "raw" and num(r["full_cos"]) == num(r["full_cos"])]
    if not rows:
        return
    rows.sort(key=lambda r: num(r["top1_frac"]))
    names = [r["dataset"][:14] for r in rows]
    x = np.arange(len(rows)); w = 0.28
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x - w, [num(r["top1_frac"]) for r in rows], w, label="top-1 eig", color="#2b6cb0")
    ax[0].bar(x, [num(r["top5_frac"]) for r in rows], w, label="top-5 eig", color="#dd6b20")
    ax[0].bar(x + w, [num(r["top10_frac"]) for r in rows], w, label="top-10 eig", color="#a0aec0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("fraction of covariance variance"); ax[0].set_ylim(0, 1.05)
    ax[0].set_title("A. Covariance is near rank-one"); ax[0].legend(fontsize=8)
    ax[1].bar(x - 1.5 * w, [num(r["full_cos"]) for r in rows], w, label="full", color="#2b6cb0")
    ax[1].bar(x - 0.5 * w, [num(r["rank1_cos"]) for r in rows], w, label="rank-1", color="#dd6b20")
    ax[1].bar(x + 0.5 * w, [num(r["rank2_cos"]) for r in rows], w, label="rank-2", color="#276749")
    ax[1].bar(x + 1.5 * w, [num(r["rank5_cos"]) for r in rows], w, label="rank-5", color="#a0aec0")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("mean $R^2$ (cos$^2$)"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("B. Rank-1 matches the full model"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, "fig19_spectrum_rank.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    for fn in [fig17, fig18, fig19]:
        try:
            fn()
        except Exception as e:
            print("skip", fn.__name__, str(e)[:60])
