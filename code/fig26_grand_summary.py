"""
fig26_grand_summary.py -- one figure summarising the whole exploration.

A. dominant covariance axis vs gene mean + equal-depth collapse
B. cross-lab/cell/library transfer of the specific residual
C. organisation by PPI / pathway
D. program-vs-direction: leftover target-specific direction reliability
E. specific structure persists over time (Marson D1)
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


def read(n):
    with open(os.path.join(RES, n)) as fh:
        return list(csv.DictReader(fh))


def short(s):
    for a, b in [("ReplogleWeissman2022_", "Repl."), ("NadigOConner2024_", "Nadig."),
                 ("Weissman2019_filtered", "Norman"), ("_essential", ""), ("_filtered", ""),
                 (".h5ad", "")]:
        s = s.replace(a, b)
    return s[:16]


def main():
    ao = read("axis_origin.csv")
    cds = read("cross_dataset_specific.csv")
    snet = read("specific_network.csv")
    spw = read("specific_pathway.csv")
    pvd = read("program_vs_direction.csv")
    tc = read("timecourse.csv")

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # A
    x = np.arange(len(ao)); w = 0.38
    ax[0, 0].bar(x - w / 2, [float(r["top1_frac"]) for r in ao], w, label="raw", color="#2b6cb0")
    ax[0, 0].bar(x + w / 2, [float(r["top1_frac_eq"]) for r in ao], w, label="equal-depth", color="#dd6b20")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([short(r["dataset"]) for r in ao], rotation=45, ha="right", fontsize=7)
    ax[0, 0].set_ylabel("top-1 covariance share"); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("A. Dominant axis is a depth artifact\n(PC1 vs gene-mean |r|=%.2f)" %
                       np.mean([abs(float(r["r_v_geneMean"])) for r in ao])); ax[0, 0].legend(fontsize=8)

    # B
    cds2 = sorted(cds, key=lambda r: float(r["cos_specific"]))
    x = np.arange(len(cds2)); w = 0.38
    ax[0, 1].bar(x - w / 2, [float(r["cos_specific"]) for r in cds2], w, color="#276749", label="same target")
    ax[0, 1].bar(x + w / 2, [float(r["cos_specific_null"]) for r in cds2], w, color="#cbd5e0", label="shuffled")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([short(r["pair"]) for r in cds2], rotation=60, ha="right", fontsize=6)
    ax[0, 1].set_ylabel("specific cosine"); ax[0, 1].set_title("B. Specific structure transfers across\nlabs / cell lines / libraries"); ax[0, 1].legend(fontsize=8)

    # C
    ds = sorted({r["dataset"] for r in snet if r["network"] == "PPI"})
    ppi = {r["dataset"]: float(r["p_perm"]) for r in snet if r["network"] == "PPI"}
    rct = {r["dataset"]: float(r["p_perm"]) for r in spw if "Reactome" in r["network"]}
    x = np.arange(len(ds)); w = 0.38
    ax[0, 2].bar(x - w / 2, [-np.log10(max(ppi.get(d, 1), 1e-300)) for d in ds], w, label="PPI", color="#276749")
    ax[0, 2].bar(x + w / 2, [-np.log10(max(rct.get(d, 1), 1e-300)) for d in ds], w, label="Reactome", color="#dd6b20")
    ax[0, 2].axhline(-np.log10(0.05), color="r", ls="--", lw=1)
    ax[0, 2].set_xticks(x); ax[0, 2].set_xticklabels([short(d) for d in ds], rotation=45, ha="right", fontsize=7)
    ax[0, 2].set_ylabel("-log10 p"); ax[0, 2].set_title("C. Organised by PPI / pathways"); ax[0, 2].legend(fontsize=8)

    # D
    ks = [0, 1, 5, 20]
    dss = sorted({r["dataset"] for r in pvd})
    x = np.arange(len(dss)); w = 0.2
    for i, k in enumerate(ks):
        vals = [float([r for r in pvd if r["dataset"] == d and int(r["k"]) == k][0]["dir_reliab"]) for d in dss]
        ax[1, 0].bar(x + (i - 1.5) * w, vals, w, label="k=%d" % k)
    ax[1, 0].axhline(0, color="k", lw=0.8); ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels([short(d) for d in dss], rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("target direction reliability"); ax[1, 0].set_title("D. Shared program x amplitude +\nresidual target-specific direction"); ax[1, 0].legend(fontsize=7, title="programs removed")

    # E
    tc1 = [r for r in tc if r["donor"] == "D1"] if tc and "donor" in tc[0] else tc
    ax[1, 1].bar(np.arange(len(tc1)), [float(r["cos_specific"]) for r in tc1], 0.5, color="#2b6cb0")
    ax[1, 1].bar(np.arange(len(tc1)), [float(r["cos_null"]) for r in tc1], 0.5, color="#cbd5e0")
    ax[1, 1].set_xticks(np.arange(len(tc1))); ax[1, 1].set_xticklabels([r["pair"] for r in tc1], fontsize=8)
    ax[1, 1].set_ylabel("specific cosine"); ax[1, 1].set_ylim(0, 0.5)
    ax[1, 1].set_title("E. Specific structure persists over\ntime (Marson D1)")

    # F: text summary
    ax[1, 2].axis("off")
    txt = ("Boundary:\n  cov matrix identifies the driver (self)\n  but cannot predict the specific response\n\n"
           "Beneath the depth axis:\n  a real, broad, transferable\n  perturbation-specific structure\n\n"
           "  - cross-lab/cell/library (strong data) ~0.4-0.5\n"
           "  - organised by PPI / pathways / TF\n"
           "  - stable over 48h\n  - present in mean/var/cov/distribution\n\n"
           "  - NOT seen by any covariance operator\n  - partly guide-level (0.06-0.19)")
    ax[1, 2].text(0.02, 0.98, txt, va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("The covariance matrix is a driver detector, not a response predictor; a real specific structure exists beneath the depth axis",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "fig26_grand_summary.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
