"""
fig22_exploration_summary.py -- summary of the exploratory analysis.

A. program_vs_direction: leftover per-target direction reliability vs number of shared
   population programs removed.
B. transform_estimator_scan: operator predicts the specific residual ~ random in every
   data space.
C. pc_robustness: specific-structure reliability vs number of control PCs removed.
D. second_moment: reproducible perturbation-specific variance response vs mean response.
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
    for a, b in [("ReplogleWeissman2022_", "Repl."), ("NadigOConner2024_", "Nadig."),
                 ("Weissman2019_filtered", ""), ("Kampmann2021_", ""), ("_essential", "")]:
        s = s.replace(a, b)
    return s[:n]


def main():
    pvd = read("program_vs_direction.csv")
    tes = read("transform_estimator_scan.csv")
    pr = read("pc_robustness.csv")
    sm = read("second_moment.csv")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # A
    ks = [0, 1, 5, 20]
    dss = sorted({r["dataset"] for r in pvd})
    x = np.arange(len(dss)); w = 0.2
    cols = ["#c53030", "#dd6b20", "#2b6cb0", "#276749"]
    for i, k in enumerate(ks):
        vals = []
        for ds in dss:
            row = [r for r in pvd if r["dataset"] == ds and int(r["k"]) == k]
            vals.append(float(row[0]["dir_reliab"]) if row else np.nan)
        ax[0, 0].bar(x + (i - 1.5) * w, vals, w, label="k=%d" % k, color=cols[i])
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([short(d) for d in dss], rotation=45, ha="right", fontsize=8)
    ax[0, 0].set_ylabel("per-target direction reliability (split-half)")
    ax[0, 0].set_title("A. Structure = shared program x amplitude + residual\ntarget-specific direction (survives removing 20 programs)")
    ax[0, 0].legend(fontsize=8, title="population programs removed", title_fontsize=7)

    # B
    dss2 = sorted({r["dataset"] for r in tes})
    x2 = np.arange(len(dss2)); w2 = 0.38
    off = []; rnd = []
    for ds in dss2:
        rows = [r for r in tes if r["dataset"] == ds and r["space"] == "raw"]
        off.append(float(rows[0]["cos2_sigma_off"]) if rows else np.nan)
        rnd.append(float(rows[0]["cos2_sigma_off_rand"]) if rows else np.nan)
    ax[0, 1].bar(x2 - w2 / 2, off, w2, label="operator (off-axis)", color="#2b6cb0")
    ax[0, 1].bar(x2 + w2 / 2, rnd, w2, label="random column", color="#a0aec0")
    ax[0, 1].set_xticks(x2); ax[0, 1].set_xticklabels([short(d) for d in dss2], rotation=45, ha="right", fontsize=8)
    ax[0, 1].set_ylabel("cos$^2$ with specific residual (raw)")
    ax[0, 1].set_title("B. The operator cannot see the specific structure\n(no transform/estimator helps)")
    ax[0, 1].legend(fontsize=8)

    # C
    kk = [1, 3, 5, 10, 20]
    for ds in sorted({r["dataset"] for r in pr}):
        rows = sorted([r for r in pr if r["dataset"] == ds], key=lambda r: int(r["k_pcs"]))
        ax[1, 0].plot(kk, [float(r["rsa_selfrem_spec"]) for r in rows], marker="o", label=short(ds))
    ax[1, 0].axhline(0, color="k", lw=0.8)
    ax[1, 0].set_xlabel("control PCs removed"); ax[1, 0].set_ylabel("centered RSA reliability")
    ax[1, 0].set_ylim(-0.1, 1.0)
    ax[1, 0].set_title("C. Structure is not the control covariance's nuisance axes")
    ax[1, 0].legend(fontsize=7)

    # D
    dss3 = [r["dataset"] for r in sm]
    x3 = np.arange(len(dss3)); w3 = 0.38
    ax[1, 1].bar(x3 - w3 / 2, [float(r["mean_rel_spec"]) for r in sm], w3, label="mean shift", color="#2b6cb0")
    ax[1, 1].bar(x3 + w3 / 2, [float(r["var_rel_spec"]) for r in sm], w3, label="variance (2nd moment)", color="#276749")
    ax[1, 1].set_xticks(x3); ax[1, 1].set_xticklabels([short(d) for d in dss3], rotation=45, ha="right", fontsize=8)
    ax[1, 1].set_ylabel("split-half reliability (specific part)")
    ax[1, 1].set_ylim(0, 1.0)
    ax[1, 1].set_title("D. A reproducible perturbation-specific variance response\n(separate from the mean)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Exploration: below the depth axis lies a real, transferable specific structure that the covariance operator cannot see",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIG, "fig22_exploration_summary.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
